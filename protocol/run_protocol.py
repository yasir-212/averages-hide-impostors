"""Path-A-protocol evaluation on SMoRFFI (frozen rules): equal A/B blocks, 2 role orders x 3 seeds,
4 verifiers x 6 mitigations, every threshold on the calibration role, effective coverage reported."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, json, time, numpy as np, hashlib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(_R / 'protocol'))
from cosface import CosFace
DATA = str(DATA_FILE)
SPLITS = {'train': (0, 600), 'A': (600, 725), 'B': (725, 850), 'test': (850, 999)}

def windows(path):
    d = np.load(path); x, y, row = d['x'], d['y'], d['row']
    valid = np.array([j for j in range(len(d['labels'])) if (row[y == j] >= 998).any()])
    g = np.isin(y, valid); x, y, row = x[g], np.searchsorted(valid, y[g]), row[g]; n = len(valid); out = {}
    for seg, (lo, hi) in SPLITS.items():
        xs, ys = [], []
        for j in range(n):
            ix = np.flatnonzero((y == j) & (row >= lo) & (row < hi)); ix = ix[np.argsort(row[ix])]
            for p in range(0, len(ix) - 3, 4):
                ids = ix[p:p + 4]
                if not np.all(np.diff(row[ids]) == 1): continue
                v = np.r_[x[ids].mean(0), x[ids].std(0)]
                if np.all(np.isfinite(v)): xs.append(v); ys.append(j)
        out[seg] = (np.asarray(xs), np.asarray(ys))
    return out, n, hashlib.sha256(Path(path).read_bytes()).hexdigest()

def q_thr(S, y, n, q=0.9):
    return np.array([np.quantile(S[y == t, t], 1 - q, method='higher') for t in range(n)])

def conformal_thr(S, y, n, alpha=0.1):
    th = np.empty(n)
    for t in range(n):
        s = np.sort(S[y == t, t]); k = int(np.floor(alpha * (len(s) + 1)))
        th[t] = -np.inf if k < 1 else s[k - 1]
    return th

def farcap0(Sd, yd, Sc, yc, n):
    floor = q_thr(Sc, yc, n); th = floor.copy()
    for t in range(n):
        imp = Sd[yd != t, t]; mx = imp.max()
        cands = np.unique(np.r_[Sc[yc == t, t], imp, floor[t]])
        ok = cands[(cands >= floor[t] - 1e-15) & (cands > mx)]
        th[t] = ok.min() if ok.size else np.inf          # no feasible threshold -> identity effectively denied
    return th

def tnorm(S, cohort):
    out = np.empty_like(S)
    for t in range(S.shape[1]):
        c = S[:, cohort[t]]; out[:, t] = (S[:, t] - c.mean(1)) / (c.std(1) + 1e-6)
    return out

def evaluate(S, y, th, keep, n):
    acc = (S >= th[None, :]) & keep[None, :]
    g = acc[np.arange(len(y)), y]; tt = np.array([g[y == t].mean() for t in range(n)])
    a = acc.copy(); a[np.arange(len(y)), y] = False
    P = np.stack([a[y == s].mean(0) for s in range(n)]); np.fill_diagonal(P, np.nan)
    return tt, P, acc

def summary(tt, P, keep):
    return dict(stated=int(keep.sum()), eff0=int((tt > 0).sum()), eff50=int((tt >= .5).sum()), TAR=float(tt.mean()),
                worst=float(np.nanmax(P[:, keep])) if keep.any() else 0.0, pooled=float(np.nanmean(P)))

def make(name, seed):
    if name == 'RF128': return RandomForestClassifier(n_estimators=128, min_samples_leaf=2, random_state=31 + 100 * seed, n_jobs=-1)
    if name == 'MLP64': return MLPClassifier((64,), alpha=0.1, max_iter=120, early_stopping=True, validation_fraction=0.12, n_iter_no_change=10, random_state=67 + 100 * seed, batch_size=256)
    if name == 'DeepMLP': return MLPClassifier((256, 256), alpha=1e-3, max_iter=200, early_stopping=True, validation_fraction=0.12, n_iter_no_change=10, random_state=7 + 100 * seed, batch_size=256)
    if name == 'CosFace': return None

def main():
    sets, n, sha = windows(DATA)
    sc = StandardScaler().fit(sets['train'][0]); sets = {k: (sc.transform(x), y) for k, (x, y) in sets.items()}
    xtr, ytr = sets['train']; xte, yte = sets['test']
    info = dict(sha=sha, n=n, windows={k: int(len(v[1])) for k, v in sets.items()}, per_source={k: int(np.bincount(v[1]).min()) for k, v in sets.items()})
    print(info, flush=True)
    rows, curves = [], {}
    for vname in ('RF128', 'MLP64', 'DeepMLP', 'CosFace'):
        for seed in (0, 1, 2):
            t0 = time.time()
            if vname == 'CosFace':
                m = CosFace(xtr.shape[1], n, seed=seed).fit(xtr, ytr); score = m.scores
            else:
                m = make(vname, seed).fit(xtr, ytr); score = m.predict_proba
            S = {k: score(sets[k][0]) for k in ('train', 'A', 'B', 'test')}
            fit_s = time.time() - t0
            # T-norm cohort from TRAINING data: identities whose score is highest on t's training windows
            Mbar = np.stack([S['train'][ytr == t].mean(0) for t in range(n)])
            K = min(20, n - 1); cohort = [np.argsort(-np.where(np.arange(n) == t, -np.inf, Mbar[t]))[:K] for t in range(n)]
            ST = {k: tnorm(v, cohort) for k, v in S.items()}
            for order in ('orig', 'swap'):
                D, C = ('A', 'B') if order == 'orig' else ('B', 'A')
                yd, yc = sets[D][1], sets[C][1]
                allk = np.ones(n, bool)
                res = {}
                th0 = q_thr(S[C], yc, n); res['M0 reference'] = (S, th0, allk)
                res['M4 conformal'] = (S, conformal_thr(S[C], yc, n), allk)
                res['M1 FAR-cap'] = (S, farcap0(S[D], yd, S[C], yc, n), allk)
                # gate: dev worst-source FAR == 0 at reference threshold
                _, Pd, _ = evaluate(S[D], yd, th0, allk, n); gate = np.nanmax(Pd, 0) <= 1e-12
                res['M2 gate'] = (S, th0, gate)
                thT = q_thr(ST[C], yc, n); res['M3 T-norm'] = (ST, thT, allk)
                _, PdT, _ = evaluate(ST[D], yd, thT, allk, n); gateT = np.nanmax(PdT, 0) <= 1e-12
                res['M3+M2 T-norm+gate'] = (ST, thT, gateT)
                for mit, (SS, th, keep) in res.items():
                    tt, P, acc = evaluate(SS['test'], yte, th, keep, n)
                    r = dict(verifier=vname, seed=seed, order=order, mitigation=mit, fit_s=round(fit_s, 1), **summary(tt, P, keep))
                    # development-chosen attacker under this mitigation
                    _, PdM, _ = evaluate(SS[D], yd, th, keep, n)
                    sel = [np.nanargmax(np.nan_to_num(PdM[:, t], nan=-1)) for t in range(n)]
                    r['dev_chosen'] = float(np.mean([P[sel[t], t] for t in range(n) if keep[t]])) if keep.any() else 0.0
                    rows.append(r)
                if seed == 0 and order == 'orig':
                    cv = {}
                    for lab, SS in (('M0 reference', S), ('M3 T-norm', ST)):
                        pts = []
                        for q in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
                            tt, P, _ = evaluate(SS['test'], yte, q_thr(SS[C], yc, n, q), allk, n); pts.append([float(tt.mean()), float(np.nanmax(P))])
                        cv[lab] = pts
                    curves[vname] = cv
            print(vname, seed, 'done', round(time.time() - t0, 1), 's', flush=True)
    json.dump(dict(info=info, rows=rows, curves=curves), open(str(_R / 'protocol' / 'protocol_results.json'), 'w'), indent=1)
    print('saved', len(rows))

if __name__ == '__main__':
    main()
