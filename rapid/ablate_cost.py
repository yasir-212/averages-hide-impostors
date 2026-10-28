"""(1) Component ablation of RAPID on the protocol (seed 0, both role orders, test block).
(2) Host-cost benchmark: single-session latency (median/p95 of 400 calls), batched throughput, model size."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, json, time, gc, numpy as np
sys.path.insert(0, str(_R / 'rapid'))
from common import load_frames
from evalkit import windows, SMoRFFIBaseline, rule_thresholds, full_metrics
from rapid_core import rival, RAPID
from sklearn.preprocessing import StandardScaler

def single_latency(f, W, calls=400):
    ts = []
    for i in range(calls):
        w = W[i % len(W)][None]; t0 = time.perf_counter(); f(w); ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1e3; return float(np.median(ts)), float(np.quantile(ts, .95))

if __name__ == '__main__':
    x, y, row, n = load_frames(); B = {'A': (600, 725), 'B': (725, 850), 'test': (850, 999)}
    W = {k: windows(x, y, row, n, *v) for k, v in B.items()}; ftr, yf = x[row < 600], y[row < 600]; sc = StandardScaler().fit(ftr)
    m = RAPID(n, seed=0, E=3).fit(ftr, yf, sc)
    def member_cos(mem, Wx):
        return mem.cos(sc.transform(Wx.reshape(-1, 15)).astype(np.float32)).reshape(len(Wx), Wx.shape[1], -1).mean(1)
    C1 = {k: member_cos(m.members[0], W[k][0]) for k in W}; C3 = {k: m.session_cos(W[k][0]) for k in W}
    variants = [('CosFace, 1 member, cosine, quantile', C1, 'quantile'), ('+ ensemble (E=3)', C3, 'quantile'),
                ('+ rival margin', {k: rival(v, 1.0) for k, v in C3.items()}, 'quantile'),
                ('+ floor rule (RAPID)', {k: rival(v, 1.0) for k, v in C3.items()}, 'floor'),
                ('RAPID without ensemble', {k: rival(v, 1.0) for k, v in C1.items()}, 'floor')]
    abl = []
    for name, S, rule in variants:
        for order in ('orig', 'swap'):
            Cb, D = ('B', 'A') if order == 'orig' else ('A', 'B')
            th = rule_thresholds(rule, S[Cb], W[Cb][1], n, 0.9); r = full_metrics(th, S[D], W[D][1], S['test'], W['test'][1], n)
            r.update(variant=name, order=order); abl.append(r)
        print(name, {k: round(np.mean([a[k] for a in abl if a['variant'] == name]), 4) for k in ('TAR', 'pooled', 'p99', 'ge50', 'devchosen', 'EER', 'acc')}, flush=True)
    cost = {}
    Wt = W['test'][0]
    cost['RAPID'] = dict(single=single_latency(m.scores, Wt), batch_ms=None, params=m.n_params(), bytes=4 * m.n_params())
    t0 = time.perf_counter(); m.scores(Wt); cost['RAPID']['batch_ms'] = 1e3 * (time.perf_counter() - t0) / len(Wt)
    m2 = RAPID(n, seed=0, E=2).fit(ftr, yf, sc)
    cost['RAPID-Lite'] = dict(single=single_latency(m2.scores, Wt), params=m2.n_params(), bytes=4 * m2.n_params(), fit_s=m2.fit_s)
    t0 = time.perf_counter(); m2.scores(Wt); cost['RAPID-Lite']['batch_ms'] = 1e3 * (time.perf_counter() - t0) / len(Wt)
    cost['RAPID']['fit_s'] = m.fit_s; del m, m2; gc.collect()
    b = SMoRFFIBaseline(n, seed=0).fit(ftr, yf, sc); size, nodes = b.size_bytes()
    cost['SMoRFFI-RF128'] = dict(single=single_latency(b.scores, Wt), nodes=nodes, bytes=size, fit_s=b.fit_s)
    t0 = time.perf_counter(); b.scores(Wt); cost['SMoRFFI-RF128']['batch_ms'] = 1e3 * (time.perf_counter() - t0) / len(Wt)
    # per-session work: RF visits 128 trees x 4 frames; RAPID 3 MLP passes x 4 frames + 122-prototype products
    cost['RAPID']['flops_per_session'] = int(3 * 4 * 2 * (15 * 128 + 128 * 128 + 128 * 64 + 64 * 122))
    cost['RAPID-Lite']['flops_per_session'] = int(2 * 4 * 2 * (15 * 128 + 128 * 128 + 128 * 64 + 64 * 122))
    depth = [e.tree_.max_depth for e in b.m.estimators_]; cost['SMoRFFI-RF128']['mean_tree_depth'] = float(np.mean(depth))
    print(cost, flush=True)
    json.dump(dict(ablation=abl, cost=cost), open('ablate_cost.json', 'w'), indent=1)
