import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, gc, numpy as np
sys.path.insert(0, str(_R / 'rapid'))
from common import load_frames, kalman_seq
from evalkit import windows, thr, accept, global_roc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
x, y, row, n = load_frames()
WA, yA = windows(x, y, row, n, 600, 725); WB, yB = windows(x, y, row, n, 725, 850)
for r_ in (0.05, 0.2, 1.0):
    Z, Y = [], []
    for j in range(n):
        ix = np.flatnonzero((y == j) & (row < 600)); ix = ix[np.argsort(row[ix])]; Z.append(kalman_seq(x[ix], r_)); Y.append(np.full(len(ix), j))
    Z = np.concatenate(Z); Y = np.concatenate(Y); sc = StandardScaler().fit(Z)
    rf = RandomForestClassifier(128, random_state=31, n_jobs=2).fit(sc.transform(Z), Y)
    f = lambda W: rf.predict_proba(sc.transform(np.stack([kalman_seq(w, r_)[-1] for w in W])))
    SA, SB = f(WA), f(WB); del rf; gc.collect()
    tar, P, _ = accept(SA, yA, n, thr(SB, yB, n, .9)); fl = P[np.isfinite(P)]; g = global_roc(SA, yA)
    print(f'KF r={r_}: TAR {100*tar.mean():.1f} pooled {100*fl.mean():.2f} p99 {100*np.quantile(fl,.99):.1f} ge50 {(fl>=.5).sum()} EER {100*g["EER"]:.2f} acc {100*(SA.argmax(1)==yA).mean():.1f}', flush=True)
