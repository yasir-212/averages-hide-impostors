"""Train embedding ensembles on inner-train and cache per-member session embeddings (dev only)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, time, numpy as np
from devh import load_inner, embed_sessions
from core2 import train_members, class_means
from sklearn.preprocessing import StandardScaler
import argparse
ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--E', type=int, default=3); ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--aug', type=float, default=0.0)
ap.add_argument('--h', default='128,128'); ap.add_argument('--demb', type=int, default=64); ap.add_argument('--m', type=float, default=0.25)
ap.add_argument('--s', type=float, default=16.0); ap.add_argument('--cos', type=int, default=0); ap.add_argument('--drop', default=''); ap.add_argument('--addcal', default='')
def main():
    a = ap.parse_args()
    x, y, row, n, ftr, ytr, W = load_inner()
    cols = [c for c in range(15) if str(c) not in a.drop.split(',')]
    if a.addcal:                                           # equal-data variant: the calibration block's frames join training
        from devh import SPLIT
        lo, hi = SPLIT[a.addcal]; inb = (row >= lo) & (row < hi); ftr = np.r_[ftr, x[inb]]; ytr = np.r_[ytr, y[inb]]
    sc = StandardScaler().fit(ftr[:, cols]); Ftr = sc.transform(ftr[:, cols]).astype(np.float32)
    aug = None
    if a.aug > 0:
        # drift scale per feature: spread of per-device mean shift between the first and last third of inner-train (standardised units)
        r = row[(row < 350)]; yy = y[row < 350]; Z = sc.transform(x[row < 350][:, cols]).astype(np.float32)
        d = np.array([Z[(yy == j) & (r >= 233)].mean(0) - Z[(yy == j) & (r < 117)].mean(0) for j in range(n)])
        aug = a.aug * d.std(0); print('aug sigma', np.round(aug, 3))
    kw = dict(h=tuple(int(v) for v in a.h.split(',')), d_emb=a.demb, epochs=a.epochs, m=a.m, s=a.s, aug=aug, cosine_lr=bool(a.cos))
    t0 = time.time(); ms = train_members(Ftr, ytr, n, [1000 * a.seed + e for e in range(a.E)], kw, workers=2); ft = time.time() - t0
    Utr = np.stack([m.embed(Ftr) for m in ms]); P0 = class_means(Utr, ytr, n)
    out = dict(P0=P0, fit_s=ft, cols=np.array(cols))
    for k, (Wk, yk) in W.items():
        out[f'U_{k}'] = embed_sessions(ms, type('S', (), {'transform': staticmethod(lambda X: sc.transform(X[:, cols]))})(), Wk).astype(np.float16); out[f'y_{k}'] = yk
    np.savez(str(_HERE) + f'/cache/emb_{a.tag}{a.addcal}_s{a.seed}.npz', **out)
    print(a.tag, a.seed, 'fit %.1fs' % ft, flush=True)

if __name__ == '__main__':
    main()
