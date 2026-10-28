import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, time, json, numpy as np
sys.path.insert(0, str(_R / 'rapid'))
from partrain import train_members
from common import load_frames
from evalkit import windows, thr, accept, global_roc
from rapid_core import rival
from sklearn.preprocessing import StandardScaler
if __name__ == '__main__':
    x, y, row, n = load_frames()
    WA, yA = windows(x, y, row, n, 600, 725); WB, yB = windows(x, y, row, n, 725, 850)
    ftr, yf = x[row < 600], y[row < 600]; scf = StandardScaler().fit(ftr); F = scf.transform(ftr).astype(np.float32)
    def pt(SA, SB, th):
        tar, P, acc = accept(SA, yA, n, th); f = P[np.isfinite(P)]; _, Pd, _ = accept(SB, yB, n, th)
        sel = np.nanargmax(np.where(np.isfinite(Pd), Pd, -1), 0)
        return f"TAR {100*tar.mean():.1f} eff50 {(tar>=.5).sum()} pooled {100*f.mean():.2f} p99 {100*np.quantile(f,.99):.1f} ge50 {(f>=.5).sum()} eq100 {(f>=1-1e-9).sum()} devch {100*P[sel,np.arange(n)].mean():.1f}"
    for E, kw, st in ((3, dict(epochs=30), 2), (4, dict(epochs=30), 2), (4, dict(epochs=40), 3), (6, dict(epochs=30), 3)):
        sub = (row[row < 600] % st) == 0
        t0 = time.time(); ms = train_members(F[sub], yf[sub], n, range(E), kw); ft = time.time() - t0
        sess = lambda W: np.mean([m.cos(scf.transform(W.reshape(-1, 15))) for m in ms], 0).reshape(len(W), 4, -1).mean(1)
        cA, cB = sess(WA), sess(WB); RA, RB = rival(cA, 1.0), rival(cB, 1.0); g = global_roc(RA, yA)
        print(f"E{E} {kw} stride{st} fit {ft:.1f}s | floor: {pt(RA, RB, np.minimum(thr(RB, yB, n, .9), 0))} | EER {100*g['EER']:.2f} T@1% {100*g['TAR@FAR0.01']:.1f} T@.1% {100*g['TAR@FAR0.001']:.1f} acc {100*(RA.argmax(1)==yA).mean():.1f}", flush=True)
