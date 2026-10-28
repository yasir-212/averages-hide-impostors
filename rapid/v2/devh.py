"""Development harness on a drift-faithful inner split (the locked test block, frames 850-998, is never loaded).
inner-train 0-349 | inner-A 350-474 | inner-B 475-599 | inner-test 600-849 (gap to training mirrors the real protocol)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, os, json, time, gc, numpy as np
from common import load_frames
from evalkit import windows, full_metrics, thr
from sklearn.preprocessing import StandardScaler
from core2 import train_members, class_means, session_cos, margin, unit

SPLIT = dict(train=(0, 350), A=(350, 475), B=(475, 600), test=(600, 850))

def load_inner():
    x, y, row, n = load_frames(); assert row.max() >= 998
    keep = row < 850; x, y, row = x[keep], y[keep], row[keep]          # hard guard: nothing from the locked test block
    W = {k: windows(x, y, row, n, *SPLIT[k]) for k in ('A', 'B', 'test')}
    tr = (row >= SPLIT['train'][0]) & (row < SPLIT['train'][1])
    return x, y, row, n, x[tr], y[tr], W

def fold_ids(yc, K):
    f = np.zeros(len(yc), int)
    for j in np.unique(yc):
        ix = np.flatnonzero(yc == j); f[ix] = (np.arange(len(ix)) * K) // len(ix)
    return f

def embed_sessions(members, sc, W):
    S, L, d = W.shape; Z = sc.transform(W.reshape(-1, d)).astype(np.float32)
    return np.stack([m.embed(Z).reshape(S, L, -1) for m in members])

def show(name, r):
    print(f"{name:40s} TAR {100*r['TAR']:5.1f} e50 {r['eff50']:4.0f} pool {100*r['pooled']:.2f} worst {100*r['worst']:5.1f} p99 {100*r['p99']:5.1f} "
          f"ge10 {r['ge10']:4.0f} ge50 {r['ge50']:3.0f} eq100 {r['eq100']:2.0f} devch {100*r['devchosen']:5.1f} r5 {100*r['retry5']:5.1f} r20 {100*r['retry20']:5.1f} "
          f"multi {100*r['multi']:5.1f} EER {100*r['EER']:.2f} AUC {r['AUC']:.4f} T1 {100*r['TAR@FAR0.01']:.1f} T.1 {100*r['TAR@FAR0.001']:.1f} acc {100*r['acc']:.1f}", flush=True)
