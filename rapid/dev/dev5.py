import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, json, time, gc, numpy as np
sys.path.insert(0, str(_R / 'rapid'))
from common import *
from rapid_core import RAPID, rival
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
x, y, row, n = load_frames()
WA, yA = windows(x, y, row, n, 600, 725); WB, yB = windows(x, y, row, n, 725, 850)
ftr, yf = x[row < 600], y[row < 600]; scf = StandardScaler().fit(ftr)
t0 = time.time(); m = RAPID(n, E=3, lam=0.9).fit(ftr, yf, scf); print('RAPID f32 fit', round(m.fit_s, 1), 'params', m.n_params())
cA, cB = m.session_cos(WA), m.session_cos(WB)
rf = RandomForestClassifier(128, random_state=31, n_jobs=2).fit(scf.transform(ftr), yf)
pf = lambda W: rf.predict_proba(scf.transform(W.reshape(-1, 15))).reshape(len(W), W.shape[1], -1).mean(1)
PA, PB = pf(WA), pf(WB); del rf; gc.collect()
np.savez('dev_scores.npz', cA=cA, cB=cB, PA=PA, PB=PB, yA=yA, yB=yB)
curves = {}
for name, SA, SB in (('SMoRFFI RF128', PA, PB), ('RAPID', rival(cA, .9), rival(cB, .9))):
    pts = []
    for q in np.r_[0.5:0.96:0.025, 0.97, 0.98, 0.99]:
        r = metrics(SB, yB, SA, yA, n, q=q, S_dev=SB, y_dev=yB); r['q'] = float(q); pts.append(r)
    curves[name] = pts
    r = [p for p in pts if abs(p['q'] - .9) < 1e-9][0]
    print(name, {k: round(v, 4) for k, v in r.items()})
json.dump(curves, open('dev_curves.json', 'w'), indent=1)
# dominance check at matched TAR (interpolate baseline metric at RAPID's TAR)
B = curves['SMoRFFI RF128']; R = curves['RAPID']
for key in ('pooled', 'worst', 'p99', 'ge50', 'devchosen', 'eq100'):
    bt = np.array([p['TAR'] for p in B]); bv = np.array([p[key] for p in B]); o = np.argsort(bt)
    rows = []
    for p in R:
        if bt.min() <= p['TAR'] <= bt.max(): rows.append((round(p['TAR'], 3), round(p[key], 4), round(float(np.interp(p['TAR'], bt[o], bv[o])), 4)))
    print(key, 'RAPID<=BL at matched TAR in', sum(r[1] <= r[2] for r in rows), 'of', len(rows), rows[-6:])
