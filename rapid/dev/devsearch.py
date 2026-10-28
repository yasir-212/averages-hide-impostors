"""Model selection on DEVELOPMENT data only: fit on train, threshold on block B, evaluate on block A.
The locked test block (frames 850-998) is never loaded here."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, json, time, numpy as np
sys.path.insert(0, str(_R / 'rapid')); sys.path.insert(0, str(_R / 'protocol'))
from rapid import RAPID, window_stats
from cosface import CosFace
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
SPL = {'train': (0, 600), 'A': (600, 725), 'B': (725, 850)}   # NO test block
def build(kind):
    d = np.load(str(DATA_FILE)); x, y, row = d['x'], d['y'], d['row']
    valid = np.array([j for j in range(len(d['labels'])) if (row[y == j] >= 998).any()])
    g = np.isin(y, valid); x, y, row = x[g], np.searchsorted(valid, y[g]), row[g]; n = len(valid); out = {}
    for seg, (lo, hi) in SPL.items():
        xs, ys = [], []
        for j in range(n):
            ix = np.flatnonzero((y == j) & (row >= lo) & (row < hi)); ix = ix[np.argsort(row[ix])]
            for p in range(0, len(ix) - 3, 4):
                ids = ix[p:p + 4]
                if not np.all(np.diff(row[ids]) == 1): continue
                v = window_stats(x[ids], kind)
                if np.all(np.isfinite(v)): xs.append(v); ys.append(j)
        out[seg] = (np.asarray(xs), np.asarray(ys))
    return out, n
def evalset(S_thr, y_thr, S_ev, y_ev, n, q=0.9):
    th = np.array([np.quantile(S_thr[y_thr == t, t], 1 - q, method='higher') for t in range(n)])
    acc = S_ev >= th[None, :]; g = acc[np.arange(len(y_ev)), y_ev]; tt = np.array([g[y_ev == t].mean() for t in range(n)])
    a = acc.copy(); a[np.arange(len(y_ev)), y_ev] = False
    P = np.stack([a[y_ev == s].mean(0) for s in range(n)]); np.fill_diagonal(P, np.nan); f = P[np.isfinite(P)]
    return dict(TAR=float(tt.mean()), eff50=int((tt >= .5).sum()), pooled=float(f.mean()), worst=float(f.max()),
                ge50=int((f >= .5).sum()), eq100=int((f >= 1 - 1e-9).sum()), oracle=float(np.nanmean(np.nanmax(P, 0))))
res = []
cache = {}
for kind in ('ms', 'rich', 'rich+'):
    sets, n = build(kind); sc = StandardScaler().fit(sets['train'][0]); sets = {k: (sc.transform(a), b) for k, (a, b) in sets.items()}
    (xtr, ytr), (xa, ya), (xb, yb) = sets['train'], sets['A'], sets['B']
    cfgs = [('RF128', None)]
    cfgs += [('CosFace', dict(k_rivals=0, score='cos')), ('CosFace+ratio', dict(k_rivals=0, score='ratio')),
             ('RAPID k5 mr.15', dict(k_rivals=5, mr=0.15)), ('RAPID k10 mr.25', dict(k_rivals=10, mr=0.25))]
    for name, cfg in cfgs:
        t0 = time.time()
        if name == 'RF128':
            m = RandomForestClassifier(n_estimators=128, min_samples_leaf=2, random_state=31, n_jobs=-1).fit(xtr, ytr); f = m.predict_proba
        else:
            m = RAPID(xtr.shape[1], n, seed=0, **cfg).fit(xtr, ytr); f = m.scores
        r = evalset(f(xb), yb, f(xa), ya, n); r.update(rep=kind, model=name, fit_s=round(time.time() - t0, 1)); res.append(r)
        print(f"{kind:6s} {name:16s} TAR {100*r['TAR']:5.1f} eff50 {r['eff50']:3d} pooled {100*r['pooled']:5.2f} worst {100*r['worst']:5.1f} pairs>=50% {r['ge50']:4d} =100% {r['eq100']:3d} oracle {100*r['oracle']:5.1f}  ({r['fit_s']}s)", flush=True)
json.dump(res, open(str(_HERE / 'devsearch.json'), 'w'), indent=1)
