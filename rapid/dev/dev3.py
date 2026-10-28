import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, json, time, gc, pickle, numpy as np
sys.path.insert(0, str(_R / 'rapid')); sys.path.insert(0, str(_R / 'protocol'))
from common import *
from cosface import CosFace
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
x, y, row, n = load_frames()
Wtr, ytr = windows(x, y, row, n, 0, 600); WA, yA = windows(x, y, row, n, 600, 725); WB, yB = windows(x, y, row, n, 725, 850)
ftr, yf = x[row < 600], y[row < 600]; scf = StandardScaler().fit(ftr); Ftr = scf.transform(ftr)
res = json.load(open('dev2.json')) if False else {}
def run(name, SA, SB, extra=None):
    r = metrics(SB, yB, SA, yA, n, S_dev=SB, y_dev=yB); r.update(extra or {}); res[name] = r
    print(f"{name:28s} TAR {100*r['TAR']:5.1f} eff50 {r['eff50']:3.0f} pooled {100*r['pooled']:5.2f} worst {100*r['worst']:5.1f} p99 {100*r['p99']:5.1f} >=50% {r['ge50']:4.0f} =100% {r['eq100']:3.0f} devch {100*r['devchosen']:5.1f} EER {100*r['EER']:5.2f} acc {100*r['acc']:5.1f} {extra or ''}", flush=True)
def per_frame(f, W):
    return f(scf.transform(W.reshape(-1, 15))).reshape(len(W), W.shape[1], -1).mean(1)
t0 = time.time(); rff = RandomForestClassifier(128, random_state=31, n_jobs=2).fit(Ftr, yf); ft = time.time() - t0
size = sum(e.tree_.node_count for e in rff.estimators_); mb = sum(e.tree_.value.nbytes + e.tree_.node_count * 56 for e in rff.estimators_) / 2**20
PA, PB = per_frame(rff.predict_proba, WA), per_frame(rff.predict_proba, WB)
t0 = time.time(); _ = rff.predict_proba(scf.transform(WA[:200].reshape(-1, 15))); lat = (time.time() - t0) / 200 * 1e3
del rff; gc.collect()
run('BL SMoRFFI RF128 frame', PA, PB, dict(nodes=size, MB=round(mb, 1), fit_s=round(ft, 1), ms_per_window=round(lat, 3)))
run('  RF frame +ratio (abl.)', rival_ratio(PA), rival_ratio(PB))
for h, E, ep in (((64, 64), 1, 30), ((128, 128), 1, 30), ((128, 128), 3, 30)):
    t0 = time.time(); M = [CosFace(15, n, h=h, seed=s, epochs=ep).fit(Ftr, yf) for s in range(E)]; ft = time.time() - t0
    cA = np.mean([per_frame(m.scores, WA) for m in M], 0); cB = np.mean([per_frame(m.scores, WB) for m in M], 0)
    t0 = time.time(); _ = [per_frame(m.scores, WA[:200]) for m in M]; lat = (time.time() - t0) / 200 * 1e3
    npar = sum(v.size for m in M for v in m.P.values())
    ex = dict(params=int(npar), MB=round(npar * 4 / 2**20, 3), fit_s=round(ft, 1), ms_per_window=round(lat, 4))
    run(f'fCos h{h[0]} E{E} cos', cA, cB, ex); run(f'fCos h{h[0]} E{E} ratio', rival_ratio(cA), rival_ratio(cB), ex)
json.dump(res, open('dev3.json', 'w'), indent=1)
