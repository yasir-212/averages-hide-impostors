"""Single-session latency of the equal-data forest with one thread (seed 0, enrolment block B), on an otherwise idle host."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, json, numpy as np
from common import load_frames
from evalkit import windows
from final_runner import make_base, post, frames_in, single_latency, BLK
from sklearn.preprocessing import StandardScaler
x, y, row, n = load_frames(); W = windows(x, y, row, n, *BLK['test'])[0]
ftr, ytr, _ = frames_in(x, y, row, 0, 600); fc, yc, _ = frames_in(x, y, row, *BLK['B']); F1, Y1 = np.r_[ftr, fc], np.r_[ytr, yc]
sc = StandardScaler().fit(F1); m = make_base('RF', 0).fit(sc.transform(F1), Y1); m.n_jobs = 1
med, p95 = single_latency(lambda w: post(m, sc, w, n), W, 400); json.dump(dict(ms_single=med, ms_single_p95=p95), open('rf_eq_latency.json', 'w')); print(med, p95)
