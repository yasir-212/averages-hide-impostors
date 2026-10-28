"""Development-only tuning of the gradient-boosted-trees baseline (inner split; frames >= 850 never loaded):
one fit with many iterations, inner identification accuracy and EER read off at stages (staged_predict_proba)."""
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

def main(lr=float(sys.argv[1]) if len(sys.argv) > 1 else 0.1, iters=int(sys.argv[2]) if len(sys.argv) > 2 else 300):
    x, y, row, n, ftr, ytr, W = load_inner(); res = []
    for cal in ('B', 'A'):
        lo, hi = SPLIT[cal]; inb = (row >= lo) & (row < hi); X1, Y1 = np.r_[ftr, x[inb]], np.r_[ytr, y[inb]]
        sc = StandardScaler().fit(X1); t0 = time.time()
        m = HistGradientBoostingClassifier(learning_rate=lr, max_iter=iters, early_stopping=False, random_state=0).fit(sc.transform(X1), Y1)
        ft = time.time() - t0; Wt, yt = W['test']; Zt = sc.transform(Wt.reshape(-1, 15))
        for it, P in enumerate(m.staged_predict_proba(Zt), 1):
            if it % 25 and it != iters: continue
            Q = np.zeros((len(P), n), np.float32); Q[:, m.classes_] = P; S = Q.reshape(len(Wt), Wt.shape[1], -1).mean(1)
            r = dict(cal=cal, lr=lr, iter=it, acc=float((S.argmax(1) == yt).mean()), **global_roc(S, yt), fit_s=ft); res.append(r)
            print(cal, lr, it, 'acc %.3f EER %.4f' % (r['acc'], r['EER']), 'fit %.0fs' % ft, flush=True)
    json.dump(res, open(str(_HERE / f'dev_gbt_lr{lr}.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
