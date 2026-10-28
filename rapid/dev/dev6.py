import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import os, sys, json, time, numpy as np

sys.path.insert(0, str(_R / 'rapid'))
from common import *
from rapid_core import CosFace32, rival
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
x, y, row, n = load_frames()
WA, yA = windows(x, y, row, n, 600, 725); WB, yB = windows(x, y, row, n, 725, 850)
ftr, yf = x[row < 600], y[row < 600]; scf = StandardScaler().fit(ftr); Ftr = scf.transform(ftr).astype(np.float32)
FA = scf.transform(WA.reshape(-1, 15)).astype(np.float32); FB = scf.transform(WB.reshape(-1, 15)).astype(np.float32)
def member(args):
    seed, kw = args; m = CosFace32(15, n, seed=seed, **kw).fit(Ftr, yf)
    return m.cos(FA).reshape(len(WA), 4, -1).mean(1), m.cos(FB).reshape(len(WB), 4, -1).mean(1)
def roc_stats(S, y):
    G = S[np.arange(len(y)), y]; M = np.ones_like(S, bool); M[np.arange(len(y)), y] = False; I = np.sort(S[M])
    out = {}
    for f in (1e-2, 1e-3):
        t = I[int(np.ceil((1 - f) * len(I))) - 1]; out[f'TAR@FAR{f:g}'] = float((G > t).mean())
    return out
def tar_at_pooled(SB, yB_, SA, yA_, target):
    pts = []
    for q in np.linspace(.5, .995, 60):
        r = metrics(SB, yB_, SA, yA_, n, q=q); pts.append((r['pooled'], r['TAR']))
    pts = np.array(sorted(pts)); return float(np.interp(target, pts[:, 0], pts[:, 1]))
cfgs = {'h128 e30 b256': dict(h=(128, 128), epochs=30, batch=256), 'h128 e20 b512': dict(h=(128, 128), epochs=20, batch=512, lr=3e-3),
        'h128 e40 b512': dict(h=(128, 128), epochs=40, batch=512, lr=3e-3), 'h192 e20 b512': dict(h=(192, 192), epochs=20, batch=512, lr=3e-3)}
out = {}
if True:
    for name, kw in cfgs.items():
        for E in (3,):
            t0 = time.time(); R = [member((s, kw)) for s in range(E)]; ft = time.time() - t0
            cA = np.mean([r[0] for r in R], 0); cB = np.mean([r[1] for r in R], 0)
            for lam in (0.75, 0.9, 1.0):
                SA, SB = rival(cA, lam), rival(cB, lam); r = metrics(SB, yB, SA, yA, n, S_dev=SB, y_dev=yB); r.update(roc_stats(SA, yA))
                r['TAR@pooled0.87'] = tar_at_pooled(SB, yB, SA, yA, 0.0087); r['fit_s'] = ft; out[f'{name} E{E} lam{lam}'] = r
                print(f"{name} E{E} lam{lam}: fit {ft:5.1f}s TAR {100*r['TAR']:.1f} pooled {100*r['pooled']:.2f} p99 {100*r['p99']:.1f} ge50 {r['ge50']:.0f} eq100 {r['eq100']:.0f} devch {100*r['devchosen']:.1f} EER {100*r['EER']:.2f} acc {100*r['acc']:.1f} TAR@pooled.87% {100*r['TAR@pooled0.87']:.1f} TAR@FAR1% {100*r['TAR@FAR0.01']:.1f}", flush=True)
json.dump(out, open('dev6.json', 'w'), indent=1)
# baseline reference numbers on the same functions
d = np.load('dev_scores.npz'); PA, PB = d['PA'], d['PB']
r = metrics(PB, yB, PA, yA, n, S_dev=PB, y_dev=yB); r.update(roc_stats(PA, yA)); r['TAR@pooled0.87'] = tar_at_pooled(PB, yB, PA, yA, 0.0087)
print('BASELINE', {k: round(v, 4) for k, v in r.items()})
