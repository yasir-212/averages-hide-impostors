"""Dev evaluation v2: decision score = rival margin (identification / clip rules); native score for global-threshold
metrics = session posterior softmax(beta * c). Supports equal-data embeddings (one cache per calibration block)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, os, glob, numpy as np
from evalkit import full_metrics, thr, global_roc
from devh import fold_ids, show
from core2 import class_means, session_cos, margin, unit
from dev_eval import ORD, n, protos_from, avg

def softmax_post(C, beta):
    Z = beta * C; Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)

def rapid_cos(z, cal, w=0.0, K=4, mem=slice(None)):
    P0 = z['P0'][mem].astype(np.float32); U = {k: z[f'U_{k}'][mem].astype(np.float32) for k in ('A', 'B', 'test')}; Y = {k: z[f'y_{k}'] for k in ('A', 'B', 'test')}
    if w >= 1.0: return {k: session_cos(U[k], P0) for k in U}, Y
    Pf = protos_from(P0, U[cal], Y[cal], w); C = {k: session_cos(U[k], Pf) for k in U if k != cal}
    fo = fold_ids(Y[cal], K); Cc = np.zeros((len(Y[cal]), n), np.float32)
    for k in range(K):
        Pk = protos_from(P0, U[cal], Y[cal], w, mask=fo != k); Cc[fo == k] = session_cos(U[cal][:, fo == k], Pk)
    C[cal] = Cc; return C, Y

def run(files_for_order, w, rule, beta, delta=0.0, q=0.9, mem=slice(None)):
    rs = []
    for files in files_for_order:            # list of dicts order -> npz path (one dict per seed)
        for o, (c, a) in ORD.items():
            z = np.load(files[o]); C, Y = rapid_cos(z, c, w, mem=mem); S = {k: margin(v) for k, v in C.items()}
            if rule == 'ident': th = np.full(n, 1e-12)
            elif rule == 'clip': th = np.clip(thr(S[c], Y[c], n, q), 1e-12, delta)        # 0 <= tau_t <= delta, never above the genuine quantile
            elif rule == 'quantile': th = thr(S[c], Y[c], n, q)
            r = full_metrics(th, S[a], Y[a], S['test'], Y['test'], n)
            r.update(global_roc(softmax_post(C['test'], beta) if beta else S['test'], Y['test'])); rs.append(r)
    return avg(rs)

if __name__ == '__main__':
    tag = sys.argv[1]; eq = tag.startswith('eq')
    seeds = sorted({f.split('_s')[-1][:-4] for f in glob.glob(f'cache/emb_{tag}*_s*.npz')})
    files = [{'orig': f'cache/emb_{tag}B_s{s}.npz', 'swap': f'cache/emb_{tag}A_s{s}.npz'} if eq else
             {'orig': f'cache/emb_{tag}_s{s}.npz', 'swap': f'cache/emb_{tag}_s{s}.npz'} for s in seeds]
    WS = [float(v) for v in os.environ.get('WS', '0.0').split(',')]
    BETAS = [float(v) for v in os.environ.get('BETAS', '0,32,64,128').split(',')]
    for w in WS:
        for beta in BETAS:
            show(f'{tag} w{w} ident beta{beta:g} (n={2*len(files)})', run(files, w, 'ident', beta))
        for d in [float(v) for v in os.environ.get('DELTAS', '').split(',') if v]:
            show(f'{tag} w{w} clip{d:g} beta{BETAS[-1]:g}', run(files, w, 'clip', BETAS[-1], delta=d))
