"""Extra analyses for the TOPS paper. Uses saved SMoRFFI model outputs (same configs as pair_resid_compare.py)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, json, pickle, numpy as np
sys.path.insert(0, str(_R / 'audit'))
from prc import *
out = {}
sets, n, sha = data_windows(str(DATA_FILE))
sc = StandardScaler().fit(sets['train'][0]); sets = {k: (sc.transform(x), y) for k, (x, y) in sets.items()}
(xdev, ydev), (xcal, ycal), (xte, yte) = [sets[k] for k in ('dev', 'cal', 'test')]
out['sha256'] = sha; out['n_devices'] = int(n)
out['windows'] = {k: int(len(v[1])) for k, v in sets.items()}
out['test_windows_per_source'] = int(np.bincount(yte).min())
rng = np.random.default_rng(0)
idx_by_s = [np.flatnonzero(yte == s) for s in range(n)]

def pairmat(acc):
    a = acc.copy(); a[np.arange(len(yte)), yte] = False
    P = np.stack([a[yte == s].mean(0) for s in range(n)]); np.fill_diagonal(P, np.nan); return P

def boot_worst(acc, keep=None, B=500):
    a = acc.copy(); a[np.arange(len(yte)), yte] = False
    W = []
    for _ in range(B):
        P = np.stack([a[rng.choice(i, len(i))].mean(0) for i in idx_by_s]); np.fill_diagonal(P, np.nan)
        W.append(np.nanmax(P if keep is None else P[:, keep]))
    return [float(np.quantile(W, .025)), float(np.quantile(W, .975))]

def pertar(acc):
    g = acc[np.arange(len(yte)), yte]
    return np.array([g[yte == t].mean() for t in range(n)])

for m in ('RF128', 'MLP64'):
    d = np.load(f'probs_{m}.npz'); pdev, pcal, pte = d['pdev'], d['pcal'], d['pte']
    ver, rivals = pickle.load(open(f'ver_{m}.pkl', 'rb')); r = np.load(f'resid_{m}.npz'); rcal, rte = r['rcal'], r['rte']
    R = {}
    thr0 = genuine_thr(pcal, ycal, n, 0.9)
    acc0 = pte >= thr0[None, :]
    P0 = pairmat(acc0); f = P0[np.isfinite(P0)]
    R['baseline'] = dict(TAR=float(pertar(acc0).mean()), pooled=float(f.mean()), worst=float(f.max()),
        n_pairs=int(f.size), pairs_gt0=int((f > 0).sum()), pairs_ge10=int((f >= .1).sum()), pairs_ge50=int((f >= .5).sum()),
        pairs_eq100=int((f >= 1 - 1e-12).sum()), targets_with_source_ge50=int((np.nanmax(P0, 0) >= .5).sum()),
        mean_max_source=float(np.nanmean(np.nanmax(P0, 0))), worst_CI=boot_worst(acc0))
    R['baseline']['cdf'] = np.sort(f).tolist()
    # dev-chosen source & retry (sequential windows in time order within each source)
    _, Pd = accept_matrix(pdev, ydev, thr0)
    sel = np.array([np.nanargmax(Pd[:, t]) for t in range(n)])
    devfar = np.array([P0[sel[t], t] for t in range(n)])
    R['baseline']['dev_chosen_mean'] = float(devfar.mean())
    R['baseline']['oracle_mean'] = float(np.nanmean(np.nanmax(P0, 0)))
    retry = {}
    for k in (1, 2, 5, 10, 20):
        vals = []
        for t in range(n):
            ii = idx_by_s[sel[t]]  # already in chronological order
            a = acc0[ii, t]
            nb = len(a) // k
            vals.append(np.mean([a[b * k:(b + 1) * k].any() for b in range(nb)]))
        retry[k] = dict(sequential=float(np.mean(vals)), iid=float(np.mean(1 - (1 - devfar) ** k)),
                        targets_ge50=int((np.array(vals) >= .5).sum()))
    R['retry'] = retry
    # gate
    gate = np.nanmax(Pd, 0) <= 1e-12
    accg = acc0 & gate[None, :]
    Pg = pairmat(accg); tt = pertar(accg)
    R['gate'] = dict(stated=int(gate.sum()), eff0=int((tt > 0).sum()), eff50=int((tt >= .5).sum()), TAR=float(tt.mean()),
                     worst=float(np.nanmax(Pg[:, gate])), worst_CI=boot_worst(accg, gate))
    # farcap
    curves = {'baseline': [], 'farcap': [], 'resid_cal': []}
    for q in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
        th = genuine_thr(pcal, ycal, n, q); a = pte >= th[None, :]; P = pairmat(a)
        curves['baseline'].append([float(pertar(a).mean()), float(np.nanmax(P))])
    for gm in (0.0, 0.05, 0.1, 0.2):
        th = far_capped_thr(pdev, ydev, pcal, ycal, n, gm); a = pte >= th[None, :]; P = pairmat(a); tt = pertar(a)
        curves['farcap'].append([float(tt.mean()), float(np.nanmax(P))])
        if gm == 0.0:
            R['farcap0'] = dict(stated=122, forced_one=int((th >= 1 - 1e-12).sum()), eff0=int((tt > 0).sum()), eff50=int((tt >= .5).sum()),
                                TAR=float(tt.mean()), worst=float(np.nanmax(P)), worst_CI=boot_worst(a),
                                worst_eff=float(np.nanmax(P[:, tt > 0])))
    for keep in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.3):
        b = np.array([np.quantile(rcal[ycal == t, t], 1 - keep, method='higher') for t in range(n)])
        a = acc0 & (rte >= b[None, :]); P = pairmat(a); tt = pertar(a)
        curves['resid_cal'].append([float(tt.mean()), float(np.nanmax(P))])
        if keep == 0.45:
            R['resid_cal045'] = dict(stated=122, eff0=int((tt > 0).sum()), eff50=int((tt >= .5).sum()), TAR=float(tt.mean()), worst=float(np.nanmax(P)))
    # residual with in-sample beta
    bi = np.array([v[1] for v in ver]); okr = rte >= bi[None, :]
    a = acc0 & okr; P = pairmat(a); tt = pertar(a)
    R['resid_insample'] = dict(stated=122, eff0=int((tt > 0).sum()), eff50=int((tt >= .5).sum()), TAR=float(tt.mean()), worst=float(np.nanmax(P)),
                               worst_CI=boot_worst(a), worst_eff=float(np.nanmax(P[:, tt > 0])),
                               resid_alone_genuine_pass=float(okr[np.arange(len(yte)), yte].mean()),
                               beta_median=float(np.median(bi)), beta_min=float(bi.min()))
    R['curves'] = curves
    out[m] = R
    print(m, 'done', flush=True)

# statistical requirement curve
P_full, P_strict, delta = 14762, 3146, 0.05
def need(eps, P): return int(np.ceil(np.log(delta / P) / np.log(1 - eps)))
out['stats'] = dict(U0_full=1 - (delta / P_full) ** (1 / 37), U0_strict=1 - (delta / P_strict) ** (1 / 37),
                    need_full_1pct=need(0.01, P_full), need_strict_1pct=need(0.01, P_strict), need_single_1pct=need(0.01, 1),
                    curve=[[e, need(e, P_full), need(e, 1)] for e in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)])
from scipy.stats import beta as B
out['stats']['cp_upper_3of37'] = float(B.ppf(0.95, 4, 34))
json.dump(out, open(str(_R / 'audit' / 'numbers.json'), 'w'), indent=1)
print(json.dumps({m: {k: v for k, v in out[m].items() if k not in ('curves',)} for m in ('RF128', 'MLP64')}, default=lambda o: '...', indent=1)[:6000])
print(out['stats'])
