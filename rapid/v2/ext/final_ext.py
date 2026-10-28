"""Extended frozen-protocol evaluation on SMoRFFI (post-freeze; RAPID's design is unchanged).
For every run (3 seeds x 2 role orders) this script refits RAPID and the equal-data baselines exactly as final_runner.py does,
adds further baselines (kNN, gradient-boosted trees, CosFace prototype verifiers), and stores PER-FRAME scores of the attacker
block and the test block. Session scores for any session length L are the mean of per-frame scores over the session's frames,
which is exactly how every system scores a session (RAPID averages local similarities over frames and members; the
posterior systems average per-frame posteriors), so L = 4 reproduces final_results.json.
Outputs: ext/frames_s{seed}_{order}.npz and ext/ext_costs.json."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R / 'rapid' / 'v2', _R, _HERE):
    _sys.path.insert(0, str(_p))
import os, sys, json, time, gc, pickle, argparse, numpy as np
from common import load_frames
from final_runner import RAPID2, CFG, BLK, ORDERS, frames_in, make_base, latency, single_latency
from core2 import train_members, class_means, local_cos, unit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

OUT = str(_HERE)
KNN_K = int(json.load(open(f'{OUT}/dev_choice.json'))['knn_k'])

def frame_index(x, y, row, n, lo, hi):
    """Indices (into x) of all finite frames in rows [lo,hi), ordered by device then row (the order windows() uses)."""
    out = []
    for j in range(n):
        ix = np.flatnonzero((y == j) & (row >= lo) & (row < hi)); ix = ix[np.argsort(row[ix])]
        out.append(ix[np.all(np.isfinite(x[ix]), 1)])
    return np.concatenate(out)

def frame_post(m, sc, F, n):
    P = m.predict_proba(sc.transform(F)); Q = np.zeros((len(P), n), np.float32); Q[:, m.classes_] = P; return Q

def sess_post(m, sc, n):
    def f(W):
        return frame_post(m, sc, W.reshape(-1, W.shape[2]), n).reshape(len(W), W.shape[1], -1).mean(1)
    return f

def emb(members, sc, F):
    Z = sc.transform(F).astype(np.float32); return np.stack([mm.embed(Z) for mm in members])      # (E, F, d)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--base', default='RF-eq,MLP-eq,kNN-eq'); a = ap.parse_args()
    x, y, row, n = load_frames(); ftr, ytr, _ = frames_in(x, y, row, 0, 600)
    FI = {k: frame_index(x, y, row, n, *BLK[k]) for k in BLK}
    cpath = f'{OUT}/ext_costs.json'; costs = json.load(open(cpath)) if os.path.exists(cpath) else []
    for seed in [int(s) for s in a.seeds.split(',')]:
        for order, (cal, att) in ORDERS.items():
            path = f'{OUT}/frames_s{seed}_{order}.npz'
            if os.path.exists(path): continue
            t_run = time.time(); fc, yc, rc = frames_in(x, y, row, *BLK[cal]); F1, Y1 = np.r_[ftr, fc], np.r_[ytr, yc]
            blocks = (att, 'test'); out = {f'idx_{b}': FI[b] for b in blocks}
            # ---------------- RAPID (refit exactly as in final_runner.py)
            m = RAPID2(n, seed, CFG).fit(F1, Y1, fc, yc)
            for b in blocks:
                U = emb(m.members, m.sc, x[FI[b]])[:, :, None, :]                                       # (E, F, 1, d)
                for e in range(U.shape[0]):
                    out[f'RAPIDm{e}_{b}'] = local_cos(U[e:e + 1], m.Rp[e:e + 1], m.mask, m.k)
                for kk in (1, 5, 10):
                    out[f'RAPIDk{kk}_{b}'] = local_cos(U, m.Rp, m.mask, kk)
                Ur = emb(m.members, m.sc, fc); P = class_means(Ur, yc, n)                               # mean prototypes of recent frames
                cm = np.einsum('efd,end->efn', U[:, :, 0], P, optimize=True)
                out[f'CFmean3_{b}'] = cm.mean(0); out[f'CFeq_{b}'] = cm[0]
            del m; gc.collect(); print(seed, order, 'RAPID', flush=True)
            # ---------------- conventional metric-learning verifier: one CosFace member on the training block, mean prototypes
            sc0 = StandardScaler().fit(ftr); Z0 = sc0.transform(ftr).astype(np.float32)
            kw = dict(h=tuple(CFG['h']), d_emb=CFG['d_emb'], epochs=CFG['epochs'], m=CFG['m'], s=CFG['s'])
            t0 = time.time(); mem = train_members(Z0, ytr, n, [1000 * seed + 7], kw, workers=1); ft = time.time() - t0
            P0 = class_means(np.stack([mem[0].embed(Z0)]), ytr, n)
            for b in blocks:
                out[f'CFtrain_{b}'] = (emb(mem, sc0, x[FI[b]])[0] @ P0[0].T).astype(np.float32)
            def cf_score(W, mem=mem, sc0=sc0, P0=P0):
                S, L, d = W.shape; return (emb(mem, sc0, W.reshape(-1, d))[0] @ P0[0].T).reshape(S, L, -1).mean(1)
            Wt = np.stack([x[FI['test'][i:i + 4]] for i in range(0, 4000, 4)])
            p = sum(v.size for k, v in mem[0].P.items() if k != 'C') + P0[0].size
            sl = single_latency(cf_score, Wt)
            costs.append(dict(system='CF-train', seed=seed, order=order, fit_s=ft, size_MB=4 * p / 2**20, params=int(p),
                              ms_batched=latency(cf_score, Wt[:1000]), ms_single=sl[0], ms_single_p95=sl[1]))
            del mem; gc.collect(); print(seed, order, 'CF-train', flush=True)
            # ---------------- equal-data posterior baselines
            for name in a.base.split(','):
                sc = StandardScaler().fit(F1); t0 = time.time()
                if name == 'RF-eq': mdl = make_base('RF', seed)
                elif name == 'MLP-eq': mdl = make_base('MLP', seed)
                elif name == 'kNN-eq': mdl = KNeighborsClassifier(n_neighbors=KNN_K, weights='distance', n_jobs=2)
                else: mdl = HistGradientBoostingClassifier(random_state=11 + seed)
                mdl.fit(sc.transform(F1), Y1); ft = time.time() - t0
                for b in blocks: out[f'{name}_{b}'] = frame_post(mdl, sc, x[FI[b]], n)
                if name in ('kNN-eq', 'GBT-eq'):
                    size = len(pickle.dumps(mdl)) / 2**20; f = sess_post(mdl, sc, n)
                    sl = single_latency(f, Wt, 200)
                    costs.append(dict(system=name, seed=seed, order=order, fit_s=ft, size_MB=size, ms_batched=latency(f, Wt[:1000]),
                                      ms_single=sl[0], ms_single_p95=sl[1], n_iter=int(getattr(mdl, 'n_iter_', 0))))
                del mdl; gc.collect(); print(seed, order, name, 'fit %.0fs' % ft, flush=True)
            np.savez_compressed(path, **out); json.dump(costs, open(cpath, 'w'), indent=1)
            print(seed, order, 'run done in %.0fs' % (time.time() - t_run), flush=True)

if __name__ == '__main__':
    main()
