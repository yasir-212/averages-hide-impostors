"""Dev evaluation of local (nearest-exemplar) prototypes on equal-data embedding caches; identification rule (tau=0)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, os, glob, time, numpy as np
from evalkit import global_roc, full_metrics
from dev_eval import ORD, n, avg
from devh import show
from core2 import margin, pad_refs, local_cos, centroids

def score_order(z, c, a, k, cent=0, mem=slice(None), sub=1, delta=0.0):
    U = {kk: z[f'U_{kk}'][mem].astype(np.float32) for kk in (a, 'test', c)}; Y = {kk: z[f'y_{kk}'] for kk in ('A', 'B', 'test')}
    E, S, L, d = U[c].shape; R = U[c].reshape(E, -1, d)[:, ::sub]; yr = np.repeat(Y[c], L)[::sub]
    if cent: R, yr = centroids(R, yr, n, cent)
    Rp, mask = pad_refs(R, yr, n); kk = min(k, mask.sum(1).min())
    Sa, St = margin(local_cos(U[a], Rp, mask, kk)), margin(local_cos(U['test'], Rp, mask, kk))
    r = full_metrics(np.full(n, 1e-12 + delta), Sa, Y[a], St, Y['test'], n); r.update(global_roc(St, Y['test'])); return r

if __name__ == '__main__':
    tag = sys.argv[1]; ks = [int(v) for v in sys.argv[2].split(',')]; cent = int(os.environ.get('CENT', 0)); sub = int(os.environ.get('SUB', 1))
    mem = [int(v) for v in os.environ['EM'].split(',')] if os.environ.get('EM') else slice(None)
    seeds = sorted({f.split('_s')[-1][:-4] for f in glob.glob(f'cache/emb_{tag}B_s*.npz')})
    for k in ks:
        rs = []
        for s in seeds:
            for o, (c, a) in ORD.items():
                rs.append(score_order(np.load(f'cache/emb_{tag}{c}_s{s}.npz'), c, a, k, cent, mem, sub, float(os.environ.get('DELTA', 0))))
        show(f'{tag} k{k} d{os.environ.get("DELTA",0)} sub{sub} EM{os.environ.get("EM","all")} (n={len(rs)})', avg(rs))
