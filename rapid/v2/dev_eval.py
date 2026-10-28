"""Evaluate RAPID variants and baselines from cached inner-split scores (dev only)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, os, glob, numpy as np
from evalkit import full_metrics, thr
from devh import fold_ids, show
from core2 import class_means, session_cos, margin, unit

ORD = {'orig': ('B', 'A'), 'swap': ('A', 'B')}       # (calibration block, attacker development block)
n = 122

def rule_th(rule, Sc, yc, q=0.9, delta=0.0):
    if rule == 'quantile': return thr(Sc, yc, n, q)
    if rule == 'floor': return np.minimum(thr(Sc, yc, n, q), 0.0)
    if rule == 'ident': return np.full(n, 1e-12 + delta)
    raise ValueError(rule)

def protos_from(P0, U, yu, w, mask=None):
    E, S, L, d = U.shape
    if mask is not None: U, yu = U[:, mask], yu[mask]
    M = class_means(U.reshape(E, -1, d), np.repeat(yu, L), n)
    return unit(w * P0 + (1 - w) * M)

def rapid_scores(z, cal, w=1.0, K=4, agg='margin'):
    mem = [int(v) for v in os.environ['EM'].split(',')] if os.environ.get('EM') else slice(None)
    P0 = z['P0'][mem].astype(np.float32); U = {k: z[f'U_{k}'][mem].astype(np.float32) for k in ('A', 'B', 'test')}; Y = {k: z[f'y_{k}'] for k in ('A', 'B', 'test')}
    f = (lambda C: margin(C)) if agg == 'margin' else agg
    if w >= 1.0:
        S = {k: f(session_cos(U[k], P0)) for k in U}; return S, Y
    Pf = protos_from(P0, U[cal], Y[cal], w); S = {k: f(session_cos(U[k], Pf)) for k in U if k != cal}
    fo = fold_ids(Y[cal], K); Sc = np.zeros((len(Y[cal]), n), np.float32)
    for k in range(K):
        Pk = protos_from(P0, U[cal], Y[cal], w, mask=fo != k); Sc[fo == k] = f(session_cos(U[cal][:, fo == k], Pk))
    S[cal] = Sc; return S, Y

def evaluate(S, Y, cal, att, rule, q=0.9, native=None, delta=0.0):
    th = rule_th(rule, S[cal], Y[cal], q, delta)
    r = full_metrics(th, S[att], Y[att], S['test'], Y['test'], n)
    if native is not None:                                  # global-threshold metrics on the system's native score
        from evalkit import global_roc; r.update(global_roc(native['test'], Y['test']))
    return r

def avg(rs):
    return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}

if __name__ == '__main__':
    tags = sys.argv[1:] if len(sys.argv) > 1 else []
    for model in ('rf', 'mlp'):
        fs = sorted(glob.glob(f'cache/base_{model}_s*.npz'))
        if not fs: continue
        for rule in ('quantile', 'ident'):
            rs = []
            for fn in fs:
                z = np.load(fn); P = {k: z[f'P_{k}'] for k in ('A', 'B', 'test')}; Y = {k: z[f'y_{k}'] for k in ('A', 'B', 'test')}
                S = P if rule == 'quantile' else {k: margin(v) for k, v in P.items()}
                for o, (c, a) in ORD.items(): rs.append(evaluate(S, Y, c, a, rule, native=P))
            show(f'{model.upper()} {rule} (n={len(rs)})', avg(rs))
    for model in ('rf', 'mlp'):
        fs = sorted(glob.glob(f'cache/baseeq_{model}_s*.npz'))
        for rule in ('quantile', 'ident'):
            rs = []
            for fn in fs:
                z = np.load(fn)
                for o, (c, a) in ORD.items():
                    P = {c: z[f'{c}_cal'], a: z[f'{c}_att'], 'test': z[f'{c}_test']}; Y = {k: z[f'y_{k}'] for k in ('A', 'B', 'test')}
                    S = P if rule == 'quantile' else {k: margin(v) for k, v in P.items()}
                    rs.append(evaluate(S, Y, c, a, rule, native=P))
            if rs: show(f'{model.upper()}-eq {rule} (n={len(rs)})', avg(rs))
    WS = [float(v) for v in os.environ.get('WS', '1.0,0.5,0.0').split(',')]
    RULES = os.environ.get('RULES', 'quantile,floor,ident').split(',')
    for tag in tags:
        fs = sorted(glob.glob(f'cache/emb_{tag}_s*.npz'))
        for w in WS:
            for rule in RULES:
                rs = []
                for fn in fs:
                    z = np.load(fn)
                    for o, (c, a) in ORD.items():
                        S, Y = rapid_scores(z, c, w); rs.append(evaluate(S, Y, c, a, rule))
                show(f'{tag} w{w} {rule} (n={len(rs)})', avg(rs))
