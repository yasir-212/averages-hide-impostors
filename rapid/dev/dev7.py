import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, json, time, numpy as np
sys.path.insert(0, str(_R / 'rapid'))
from common import load_frames
from evalkit import windows, full_metrics, thr, accept
from ens import CosFaceEnsemble
from rapid_core import rival
from sklearn.preprocessing import StandardScaler
x, y, row, n = load_frames()
WA, yA = windows(x, y, row, n, 600, 725); WB, yB = windows(x, y, row, n, 725, 850)
ftr, yf = x[row < 600], y[row < 600]; scf = StandardScaler().fit(ftr); Ftr = scf.transform(ftr).astype(np.float32)
sess = lambda m, W: m.cos(scf.transform(W.reshape(-1, 15))).reshape(len(W), W.shape[1], -1).mean(1)
def show(name, r):
    print(f"{name:34s} TAR {100*r['TAR']:.1f} eff50 {r['eff50']:.0f} pooled {100*r['pooled']:.2f} p99 {100*r['p99']:.1f} ge50 {r['ge50']:.0f} eq100 {r['eq100']:.0f} devch {100*r['devchosen']:.1f} r20 {100*r['retry20']:.1f} EER {100*r['EER']:.2f} AUC {r['AUC']:.4f} T@1% {100*r['TAR@FAR0.01']:.1f} T@.1% {100*r['TAR@FAR0.001']:.1f} acc {100*r['acc']:.1f} excl {100*r['excl']:.1f}", flush=True)
d = np.load('dev_scores.npz'); PA, PB = d['PA'], d['PB']
show('BASELINE', full_metrics(PB, yB, PB, yB, PA, yA, n))
out = {}
for E, ep, bs, lr in ((3, 30, 256, 2e-3), (5, 30, 256, 2e-3), (5, 40, 256, 2e-3)):
    t0 = time.time(); m = CosFaceEnsemble(15, n, E=E, epochs=ep, batch=bs, lr=lr, seed=0).fit(Ftr, yf); ft = time.time() - t0
    cA, cB = sess(m, WA), sess(m, WB); print(f'--- E{E} e{ep} b{bs} fit {ft:.1f}s params {m.n_params()}')
    np.savez(f'ens_E{E}_e{ep}.npz', cA=cA, cB=cB)
    for lam in (0.75, 1.0):
        SA, SB = rival(cA, lam), rival(cB, lam)
        r = full_metrics(SB, yB, SB, yB, SA, yA, n); show(f'  lam{lam} quantile', r); out[f'E{E}e{ep} lam{lam} q'] = r
        # floor rule: tau_t = min(genuine quantile, 0)  (accept every top-1 identification of t)
        for floor in (0.0,):
            th = np.minimum(thr(SB, yB, n, .9), floor)
            # emulate by shifting scores so the quantile rule reproduces th: evaluate directly
            tar, P, acc = accept(SA, yA, n, th); f = P[np.isfinite(P)]
            _, Pd, _ = accept(SB, yB, n, th); sel = np.nanargmax(np.where(np.isfinite(Pd), Pd, -1), 0)
            print(f"  lam{lam} floor{floor}: TAR {100*tar.mean():.1f} eff50 {(tar>=.5).sum()} pooled {100*f.mean():.2f} p99 {100*np.quantile(f,.99):.1f} ge50 {(f>=.5).sum()} eq100 {(f>=1-1e-9).sum()} devch {100*P[sel,np.arange(n)].mean():.1f} excl {100*np.mean(acc.sum(1)<=1):.1f}")
json.dump(out, open('dev7.json', 'w'), indent=1)
