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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
x, y, row, n = load_frames()
Wtr, ytr = windows(x, y, row, n, 0, 600); WA, yA = windows(x, y, row, n, 600, 725); WB, yB = windows(x, y, row, n, 725, 850)
res = {}
def run(name, fA, fB):
    SA, SB = fA, fB
    r = metrics(SB, yB, SA, yA, n, S_dev=SB, y_dev=yB); res[name] = r
    print(f"{name:28s} TAR {100*r['TAR']:5.1f} eff50 {r['eff50']:3.0f} pooled {100*r['pooled']:5.2f} worst {100*r['worst']:5.1f} p99 {100*r['p99']:5.1f} >=50% {r['ge50']:4.0f} =100% {r['eq100']:3.0f} devch {100*r['devchosen']:5.1f} EER {100*r['EER']:5.2f} acc {100*r['acc']:5.1f}", flush=True)
# --- SMoRFFI baselines ---
sc = StandardScaler().fit(ms(Wtr)); XA, XB, Xtr = sc.transform(ms(WA)), sc.transform(ms(WB)), sc.transform(ms(Wtr))
t0 = time.time(); rf = RandomForestClassifier(128, min_samples_leaf=2, random_state=31, n_jobs=-1).fit(Xtr, ytr); print('rf fit', time.time()-t0)
run('BL RF128 window-ms', rf.predict_proba(XA), rf.predict_proba(XB))
run('  +ratio (ablation)', rival_ratio(rf.predict_proba(XA)), rival_ratio(rf.predict_proba(XB)))
# per-frame RF (as published), posterior averaged over the window
ftr = x[(row < 600)]; ytrf = y[row < 600]; scf = StandardScaler().fit(ftr)
rff = RandomForestClassifier(128, random_state=31, n_jobs=-1).fit(scf.transform(ftr), ytrf)
pf = lambda W: rff.predict_proba(scf.transform(W.reshape(-1, 15))).reshape(len(W), W.shape[1], -1).mean(1)
run('BL RF128 frame, avg post', pf(WA), pf(WB))
# Kalman-smoothed (SMoRFFI optional step): train on KF-smoothed training sequences, KF over each test window
for r_ in (0.01, 0.05, 0.2):
    xs = np.empty_like(ftr); yy = []
    Ztr = []; Ytr = []
    for j in range(n):
        ix = np.flatnonzero((y == j) & (row < 600)); ix = ix[np.argsort(row[ix])]
        Ztr.append(kalman_seq(x[ix], r_)); Ytr.append(np.full(len(ix), j))
    Ztr = np.concatenate(Ztr); Ytr = np.concatenate(Ytr); sck = StandardScaler().fit(Ztr)
    rfk = RandomForestClassifier(128, random_state=31, n_jobs=-1).fit(sck.transform(Ztr), Ytr)
    pk = lambda W: rfk.predict_proba(sck.transform(np.stack([kalman_seq(w, r_)[-1] for w in W])))
    run(f'BL RF128+KF r={r_}', pk(WA), pk(WB))
# --- candidate scheme ---
for E in (1, 3, 5):
    t0 = time.time(); ms_ = [CosFace(Xtr.shape[1], n, seed=s, epochs=60).fit(Xtr, ytr) for s in range(E)]
    cA = np.mean([m.scores(XA) for m in ms_], 0); cB = np.mean([m.scores(XB) for m in ms_], 0)
    print('cos fit', E, time.time() - t0)
    run(f'CosFace E={E} cos', cA, cB); run(f'CosFace E={E} ratio', rival_ratio(cA), rival_ratio(cB))
json.dump(res, open('dev2.json', 'w'), indent=1)
