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
from common import *
from cosface import CosFace
from sklearn.preprocessing import StandardScaler
x, y, row, n = load_frames()
WA, yA = windows(x, y, row, n, 600, 725); WB, yB = windows(x, y, row, n, 725, 850)
ftr, yf = x[row < 600], y[row < 600]; scf = StandardScaler().fit(ftr); Ftr = scf.transform(ftr)
pf = lambda f, W: f(scf.transform(W.reshape(-1, 15))).reshape(len(W), W.shape[1], -1).mean(1)
def lam(S, l):
    srt = np.sort(S, 1); top, sec = srt[:, -1:], srt[:, -2:-1]; bo = np.where(S >= top, sec, top); return S - l * bo
def show(name, r):
    print(f"{name:30s} TAR {100*r['TAR']:5.1f} eff50 {r['eff50']:3.0f} pooled {100*r['pooled']:5.2f} worst {100*r['worst']:5.1f} p99 {100*r['p99']:5.1f} >=50% {r['ge50']:4.0f} =100% {r['eq100']:3.0f} devch {100*r['devchosen']:5.1f} EER {100*r['EER']:5.2f} acc {100*r['acc']:5.1f}", flush=True)
for cfg in (dict(h=(128, 128), epochs=30, batch=256, E=3), dict(h=(128, 128), epochs=15, batch=512, lr=4e-3, E=3), dict(h=(96, 96), epochs=20, batch=512, lr=4e-3, E=3)):
    E = cfg.pop('E'); t0 = time.time(); M = [CosFace(15, n, seed=s, **cfg).fit(Ftr, yf) for s in range(E)]; ft = time.time() - t0
    cA = np.mean([pf(m.scores, WA) for m in M], 0); cB = np.mean([pf(m.scores, WB) for m in M], 0)
    np.savez(f"cos_{cfg['h'][0]}_{cfg['epochs']}.npz", cA=cA, cB=cB)
    print(cfg, 'E', E, 'fit_s %.1f' % ft)
    for l in (0, .5, .75, .9, 1.0):
        show(f'  lambda={l}', metrics(lam(cB, l), yB, lam(cA, l), yA, n, S_dev=lam(cB, l), y_dev=yB))
