"""Development-only tuning grid for the gradient-boosted-trees baseline (inner split; frames >= 850 never loaded).
Selection criterion: mean inner identification accuracy over both role orders."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R / 'rapid' / 'v2', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, time, json, numpy as np
from devh import load_inner, SPLIT
from evalkit import global_roc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
GRID = [dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, max_iter=200),
        dict(learning_rate=0.1, max_leaf_nodes=7, min_samples_leaf=200, l2_regularization=1.0, max_iter=200),
        dict(learning_rate=0.02, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, max_iter=200, max_features=0.5)]
def main():
    x, y, row, n, ftr, ytr, W = load_inner(); res = []
    for gi, g in enumerate(GRID):
        for cal in ('B', 'A'):
            lo, hi = SPLIT[cal]; inb = (row >= lo) & (row < hi); X1, Y1 = np.r_[ftr, x[inb]], np.r_[ytr, y[inb]]
            sc = StandardScaler().fit(X1); t0 = time.time()
            m = HistGradientBoostingClassifier(early_stopping=False, random_state=0, **g).fit(sc.transform(X1), Y1); ft = time.time() - t0
            Wt, yt = W['test']; Zt = sc.transform(Wt.reshape(-1, 15))
            for it, P in enumerate(m.staged_predict_proba(Zt), 1):
                if it % 50 and it != g['max_iter']: continue
                Q = np.zeros((len(P), n), np.float32); Q[:, m.classes_] = P; S = Q.reshape(len(Wt), Wt.shape[1], -1).mean(1)
                r = dict(grid=gi, cal=cal, iter=it, acc=float((S.argmax(1) == yt).mean()), **global_roc(S, yt), fit_s=ft, **g); res.append(r)
                print(gi, cal, it, 'acc %.3f EER %.4f fit %.0fs' % (r['acc'], r['EER'], ft), flush=True)
    json.dump(res, open(str(_HERE / 'dev_gbt2.json'), 'w'), indent=1)
if __name__ == '__main__':
    main()
