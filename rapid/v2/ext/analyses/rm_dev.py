"""Rival-margin rule applied to the baselines: development selection of each baseline's margin on the inner split
(train 0-349, blocks 350-474 / 475-599, inner evaluation 600-849; 2 seeds x 2 orders), exactly the data on which RAPID's
margin was frozen. Never reads frames 850-998.

Systems (all with the decision R_t = s_t - max_{j!=t} s_j >= beta on their own session score s):
  RF-eq   forest posterior (128 trees), equal data            cache/baseeq_rf_s*.npz
  MLP-eq  MLP posterior, equal data                            cache/baseeq_mlp_s*.npz
  CF-eq   CosFace member 0, one mean prototype per identity   cache/emb_eq{cal}_s*.npz
  RAPID   3 members, k=3 nearest recent exemplars              cache/emb_eq{cal}_s*.npz
Two selection criteria, fixed before any test score is read:
  (T) availability-matched: the largest beta whose development TAR loss relative to the system's own identification rule
      (beta=0) does not exceed RAPID's development TAR loss at its frozen beta=0.02;
  (A) attacker-matched: the smallest beta whose development-chosen attacker success is at most RAPID's at beta=0.02.
"""
import sys, json, numpy as np
import pathlib as _pl
R = str(next(p for p in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents] if (p / 'paths.py').exists()))
for p in (R + '/rapid/v2', R + '/rapid', R): sys.path.insert(0, p)
from evalkit import accept
from core2 import margin, class_means, session_cos, pad_refs, local_cos
C = R + '/rapid/v2/cache'   # inner-split caches written by dev_base_eq.py and dev_emb.py
n = 122; ORD = {'orig': ('B', 'A'), 'swap': ('A', 'B')}

def light(S_att, y_att, S_te, y_te, beta):
    th = np.full(n, 1e-12 + beta); tar, P, _ = accept(S_te, y_te, n, th); f = P[np.isfinite(P)]
    _, Pd, _ = accept(S_att, y_att, n, th); sel = np.nanargmax(np.where(np.isfinite(Pd), Pd, -1), 0)
    return dict(TAR=float(tar.mean()), pooled=float(f.mean()), devchosen=float(P[sel, np.arange(n)].mean()), ge50=int((f >= .5).sum()))

def scores(seed, cal, att):
    out = {}
    for m, tag in (('RF-eq', 'rf'), ('MLP-eq', 'mlp')):
        z = np.load(f'{C}/baseeq_{tag}_s{seed}.npz'); out[m] = (margin(z[f'{cal}_att']), z[f'y_{att}'], margin(z[f'{cal}_test']), z['y_test'])
    e = np.load(f'{C}/emb_eq{cal}_s{seed}.npz'); U = {k: e[f'U_{k}'].astype(np.float32) for k in (att, 'test', cal)}; Y = {k: e[f'y_{k}'] for k in (att, 'test', cal)}
    E, S, L, d = U[cal].shape
    P0 = class_means(U[cal][:1].reshape(1, -1, d), np.repeat(Y[cal], L), n)          # member 0, mean prototypes of the recent block
    out['CF-eq'] = (margin(session_cos(U[att][:1], P0)), Y[att], margin(session_cos(U['test'][:1], P0)), Y['test'])
    Rp, mask = pad_refs(U[cal].reshape(E, -1, d), np.repeat(Y[cal], L), n)
    out['RAPID'] = (margin(local_cos(U[att], Rp, mask, 3)), Y[att], margin(local_cos(U['test'], Rp, mask, 3)), Y['test'])
    return out

GRIDS = {'RF-eq': np.round(np.arange(0, 0.6001, 0.005), 4), 'MLP-eq': np.round(np.arange(0, 0.6001, 0.005), 4),
         'CF-eq': np.round(np.arange(0, 0.2001, 0.0025), 4), 'RAPID': np.round(np.arange(0, 0.2001, 0.0025), 4)}

if __name__ == '__main__':
    curves = {s: {float(b): [] for b in g} for s, g in GRIDS.items()}
    for seed in (0, 1):
        for o, (cal, att) in ORD.items():
            sc = scores(seed, cal, att)
            for s, g in GRIDS.items():
                Sa, ya, St, yt = sc[s]
                for b in g: curves[s][float(b)].append(light(Sa, ya, St, yt, float(b)))
            print('dev', seed, o, flush=True)
    mean = {s: {b: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} for b, rs in c.items()} for s, c in curves.items()}
    rap0, rap2 = mean['RAPID'][0.0], mean['RAPID'][0.02]
    dT = rap0['TAR'] - rap2['TAR']; aRef = rap2['devchosen']
    sel = {}
    for s in ('RF-eq', 'MLP-eq', 'CF-eq'):
        c = mean[s]; t0 = c[0.0]['TAR']
        bT = max(b for b in c if c[b]['TAR'] >= t0 - dT - 1e-12)
        okA = [b for b in sorted(c) if c[b]['devchosen'] <= aRef + 1e-12]
        bA = okA[0] if okA else None
        sel[s] = dict(beta_T=bT, dev_T=c[bT], beta_A=bA, dev_A=(c[bA] if bA is not None else None), dev_0=c[0.0])
    res = dict(rapid_dev=dict(beta0=rap0, beta002=rap2, dTAR=dT), selection=sel, curves=mean)
    json.dump(res, open(str(_pl.Path(__file__).resolve().parent / 'rm_dev.json'), 'w'), indent=1)
    print('RAPID dev beta=0', rap0, '\nRAPID dev beta=0.02', rap2, '\ndTAR', dT)
    for s, v in sel.items(): print(s, 'beta_T', v['beta_T'], v['dev_T'], '| beta_A', v['beta_A'], v['dev_A'], '| beta0', v['dev_0'])
