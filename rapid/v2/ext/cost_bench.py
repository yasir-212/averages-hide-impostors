"""Host-cost benchmark of every system on one machine, run alone (seed 0, enrolment block B): training time, model size,
single-session latency (median and 95th percentile of 400 calls) and batched cost per session (1,000 sessions)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R / 'rapid' / 'v2', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, json, time, gc, pickle, platform, numpy as np
from common import load_frames
from evalkit import windows
from final_runner import RAPID2, CFG, BLK, frames_in, make_base, post, base_size, latency, single_latency
from core2 import train_members, class_means
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
EXT = str(_HERE)

def main():
    gb = json.load(open(f'{EXT}/dev_choice.json')); x, y, row, n = load_frames()
    Wt = windows(x, y, row, n, *BLK['test'])[0]; ftr, ytr, _ = frames_in(x, y, row, 0, 600); fc, yc, _ = frames_in(x, y, row, *BLK['B'])
    F1, Y1 = np.r_[ftr, fc], np.r_[ytr, yc]; out = dict(host=open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].strip(': '), rows=[])
    def rec(name, fit_s, size_b, params, fn, k=400):
        with threadpool_limits(1):                     # scoring on one thread for every system
            sl = single_latency(fn, Wt, k); bt = latency(fn, Wt[:1000])
        r = dict(system=name, fit_s=fit_s, size_MB=size_b / 2**20, params=params, ms_batched=bt, ms_single=sl[0], ms_single_p95=sl[1])
        out['rows'].append(r); print(r, flush=True)
        json.dump(out, open(f'{EXT}/cost_bench.json', 'w'), indent=1)
    m = RAPID2(n, 0, CFG).fit(F1, Y1, fc, yc); size, npar, nref = m.size(); rec('RAPID', m.fit_s, size, int(npar), m.scores); del m; gc.collect()
    sc0 = StandardScaler().fit(ftr); Z0 = sc0.transform(ftr).astype(np.float32)
    kw = dict(h=tuple(CFG['h']), d_emb=CFG['d_emb'], epochs=CFG['epochs'], m=CFG['m'], s=CFG['s'])
    t0 = time.time(); mem = train_members(Z0, ytr, n, [7], kw, workers=1); ft = time.time() - t0; P0 = class_means(np.stack([mem[0].embed(Z0)]), ytr, n)[0]
    def cf(W):
        S, L, d = W.shape; return (mem[0].embed(sc0.transform(W.reshape(-1, d)).astype(np.float32)) @ P0.T).reshape(S, L, -1).mean(1)
    p = sum(v.size for k, v in mem[0].P.items() if k != 'C') + P0.size; rec('CFtrain', ft, 4 * p, int(p), cf); del mem
    for name in ('MLP-eq', 'kNN-eq', 'GBT-eq', 'RF-eq'):
        sc = StandardScaler().fit(F1); t0 = time.time()
        if name == 'RF-eq': mdl = make_base('RF', 0)
        elif name == 'MLP-eq': mdl = make_base('MLP', 0)
        elif name == 'kNN-eq': mdl = KNeighborsClassifier(n_neighbors=int(gb['knn_k']), weights='distance', n_jobs=2)
        else: mdl = HistGradientBoostingClassifier(random_state=11, early_stopping=False, **gb['gbt'])
        mdl.fit(sc.transform(F1), Y1); ft = time.time() - t0
        if 'n_jobs' in mdl.get_params(): mdl.set_params(n_jobs=1)
        if name in ('RF-eq', 'MLP-eq'): size, npar = base_size(mdl)
        else: size, npar = len(pickle.dumps(mdl)), 0
        rec(name, ft, size, int(npar), lambda w: post(mdl, sc, w, n), 200 if name == 'RF-eq' else 400); del mdl; gc.collect()

if __name__ == '__main__':
    main()
