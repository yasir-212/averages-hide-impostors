import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, json, numpy as np
sys.path.insert(0, str(_R / 'rapid'))
from common import load_frames
from evalkit import windows, SMoRFFIBaseline
from ablate_cost import single_latency
from sklearn.preprocessing import StandardScaler
if __name__ == '__main__':
    x, y, row, n = load_frames(); W, _ = windows(x, y, row, n, 850, 999); ftr, yf = x[row < 600], y[row < 600]; sc = StandardScaler().fit(ftr)
    b = SMoRFFIBaseline(n, seed=0).fit(ftr, yf, sc); out = {}
    for nj in (1, 2):
        b.m.n_jobs = nj; out[nj] = single_latency(b.scores, W, 200)
    print(out); json.dump(out, open('rf_latency.json', 'w'))
