"""Development-only selection for the extended baselines (inner split; frames >= 850 are never loaded).
kNN: number of neighbours chosen by inner identification accuracy. Gradient boosting: library defaults (timed only)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R / 'rapid' / 'v2', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, time, json, gc, numpy as np
from devh import load_inner, SPLIT
from evalkit import full_metrics
from core2 import margin
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

def post(m, sc, W, n):
    P = m.predict_proba(sc.transform(W.reshape(-1, 15))); Q = np.zeros((len(P), n), np.float32); Q[:, m.classes_] = P
    return Q.reshape(len(W), W.shape[1], -1).mean(1)

def main():
    x, y, row, n, ftr, ytr, W = load_inner(); res = []
    for cal, att in (('B', 'A'), ('A', 'B')):
        lo, hi = SPLIT[cal]; inb = (row >= lo) & (row < hi); X1, Y1 = np.r_[ftr, x[inb]], np.r_[ytr, y[inb]]
        sc = StandardScaler().fit(X1); Z1 = sc.transform(X1)
        models = [(f'kNN{k}', lambda k=k: KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=2)) for k in (1, 5, 15, 45)]
        models.append(('HGB', lambda: HistGradientBoostingClassifier(random_state=0)))
        for name, mk in models:
            t0 = time.time(); m = mk().fit(Z1, Y1); ft = time.time() - t0
            P = {k: post(m, sc, W[k][0], n) for k in (att, 'test')}; M = {k: margin(v) for k, v in P.items()}
            r = full_metrics(np.full(n, 1e-12), M[att], W[att][1], M['test'], W['test'][1], n)
            r.update(name=name, cal=cal, fit_s=ft, n_iter=int(getattr(m, 'n_iter_', 0))); res.append(r)
            print(name, cal, 'acc %.3f TAR %.3f devch %.3f EER %.4f fit %.0fs' % (r['acc'], r['TAR'], r['devchosen'], r['EER'], ft), flush=True)
            del m; gc.collect()
    json.dump(res, open(str(_HERE / 'dev_ext.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
