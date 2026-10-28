"""Component ablation of RAPID v2 on the test block (seed 0, both role orders), run after the design freeze.
Rows: v1 design (training-only embedding, one mean prototype per identity), + prototypes from the recent enrolment block,
+ embedding trained on the same enrolment data, + local prototypes (k nearest exemplars), + identification margin delta (= RAPID)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, json, numpy as np
from common import load_frames
from evalkit import windows, full_metrics, global_roc
from core2 import train_members, class_means, session_cos, margin, pad_refs, local_cos
from final_runner import CFG, BLK, ORDERS, frames_in
from sklearn.preprocessing import StandardScaler

def emb_sessions(ms, sc, W):
    S, L, d = W.shape; Z = sc.transform(W.reshape(-1, d)).astype(np.float32); return np.stack([m.embed(Z).reshape(S, L, -1) for m in ms])

if __name__ == '__main__':
    x, y, row, n = load_frames(); W = {k: windows(x, y, row, n, *v) for k, v in BLK.items()}; Y = {k: W[k][1] for k in W}
    ftr, ytr, _ = frames_in(x, y, row, 0, 600); kw = dict(h=tuple(CFG['h']), d_emb=CFG['d_emb'], epochs=CFG['epochs'], m=CFG['m'], s=CFG['s'])
    rows = []
    sc0 = StandardScaler().fit(ftr); ms0 = train_members(sc0.transform(ftr).astype(np.float32), ytr, n, [0, 1, 2], kw, workers=2)
    P0 = class_means(np.stack([m.embed(sc0.transform(ftr).astype(np.float32)) for m in ms0]), ytr, n)
    U0 = {k: emb_sessions(ms0, sc0, W[k][0]) for k in W}
    def ev(name, Sa, St, att, order, delta=0.0):
        r = full_metrics(np.full(n, 1e-12 + delta), Sa, Y[att], St, Y['test'], n); r.update(global_roc(St, Y['test'])); r.update(variant=name, order=order); rows.append(r)
        print(name, order, {k: round(r[k], 4) for k in ('TAR', 'pooled', 'worst', 'ge50', 'devchosen', 'retry20', 'EER', 'AUC', 'TAR@FAR0.001')}, flush=True)
    for order, (cal, att) in ORDERS.items():
        ev('v1: training-only embedding, mean prototype', margin(session_cos(U0[att], P0)), margin(session_cos(U0['test'], P0)), att, order)
        E, S, L, d = U0[cal].shape; Pc = class_means(U0[cal].reshape(E, -1, d), np.repeat(Y[cal], L), n)
        ev('+ prototypes from recent enrolment', margin(session_cos(U0[att], Pc)), margin(session_cos(U0['test'], Pc)), att, order)
        fc, yc, _ = frames_in(x, y, row, *BLK[cal]); F1, Y1 = np.r_[ftr, fc], np.r_[ytr, yc]
        sc1 = StandardScaler().fit(F1); ms1 = train_members(sc1.transform(F1).astype(np.float32), Y1, n, [0, 1, 2], kw, workers=2)
        U1 = {k: emb_sessions(ms1, sc1, W[k][0]) for k in (att, 'test', cal)}; Pc1 = class_means(U1[cal].reshape(E, -1, d), np.repeat(Y[cal], L), n)
        ev('+ embedding trained on the same enrolment data', margin(session_cos(U1[att], Pc1)), margin(session_cos(U1['test'], Pc1)), att, order)
        R = np.stack([m.embed(sc1.transform(fc).astype(np.float32)) for m in ms1]); Rp, mask = pad_refs(R, yc, n)
        Sa, St = margin(local_cos(U1[att], Rp, mask, CFG['k'])), margin(local_cos(U1['test'], Rp, mask, CFG['k']))
        ev('+ local prototypes (k nearest exemplars)', Sa, St, att, order)
        ev('+ identification margin delta (RAPID)', Sa, St, att, order, CFG['delta'])
    json.dump(rows, open(str(_HERE / 'ablation_v2.json'), 'w'))
