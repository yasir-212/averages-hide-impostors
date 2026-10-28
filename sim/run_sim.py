"""SYNTHETIC sensitivity study (simulated 802.11g preambles; not measured data).
Each configuration: draw a device population, simulate captures, extract SMoRFFI-style features, then run the
2026 SMoRFFI baseline and RAPID under the same protocol (2 role orders) as on measured data."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, os, json, time, gc, itertools, numpy as np
sys.path.insert(0, str(_R / 'rapid')); sys.path.insert(0, str(_R / 'rapid' / 'sim'))
import wifi_sim as WS
from evalkit import windows, blocks_for, SMoRFFIBaseline, RAPIDSystem, rule_thresholds, full_metrics, global_roc, latency_ms
from rapid_core import rival
from sklearn.preprocessing import StandardScaler

DEFAULT = dict(n_dev=60, snr_db=20.0, spread=1.0, twins=0.05, twin_delta=0.05, drift_ppm=0.6)
SWEEPS = {'n_dev': [20, 40, 80, 120], 'snr_db': [5.0, 10.0, 15.0, 25.0, 30.0], 'spread': [0.5, 0.75, 1.5, 2.0],
          'twins': [0.0, 0.1, 0.2], 'twin_delta': [0.0, 0.005, 0.02, 0.2, 0.5], 'drift_ppm': [0.2, 1.0, 1.5]}
NF = 500

def run_config(cfg, data_seed, model_seed=0, Ls=(4,)):
    t0 = time.time(); kw = dict(cfg); n = kw.pop('n_dev')
    x, y, row = WS.dataset(n, NF, seed=data_seed, **kw); bl = blocks_for(NF); tgen = time.time() - t0
    tr = row < bl['train'][1]; ftr, yf = x[tr], y[tr]; sc = StandardScaler().fit(ftr); res = []
    for sname, make in (('SMoRFFI-RF128', lambda: SMoRFFIBaseline(n, seed=model_seed)), ('RAPID', lambda: RAPIDSystem(n, seed=model_seed, d_in=15, E=3))):
        sysm = make().fit(ftr, yf, sc)
        for L in Ls:
            W = {k: windows(x, y, row, n, *bl[k], L=L) for k in ('A', 'B', 'test')}
            S = {k: sysm.scores(W[k][0]) for k in W}
            M = S if sname == 'RAPID' else {k: rival(v, 1.0) for k, v in S.items()}
            g = global_roc(S['test'], W['test'][1]); lat = latency_ms(sysm, W['test'][0][:500])
            for rule, SS in ((('floor', S), ('ident', S)) if sname == 'RAPID' else (('quantile', S), ('ident', M))):
                for order in ('orig', 'swap'):
                    C, D = ('B', 'A') if order == 'orig' else ('A', 'B')
                    th = rule_thresholds(rule, SS[C], W[C][1], n, 0.9)
                    r = full_metrics(th, SS[D], W[D][1], SS['test'], W['test'][1], n); r.update(g)
                    r.update(system=sname, rule=rule, order=order, L=L, data_seed=data_seed, fit_s=sysm.fit_s,
                             size_MB=sysm.size_bytes()[0] / 2**20, ms_per_session=lat, **cfg); res.append(r)
        sysm.free(); del sysm; gc.collect()
    print(cfg, 'seed', data_seed, 'gen %.1fs total %.1fs' % (tgen, time.time() - t0), flush=True)
    return res

def main():
    out = 'sim_results.json'; rows = json.load(open(out)) if os.path.exists(out) else []
    done = {(json.dumps({k: r[k] for k in DEFAULT}, sort_keys=True), r['data_seed']) for r in rows}
    plan = [(dict(DEFAULT), s, (1, 2, 4, 8, 16)) for s in (11, 12)]
    for key, vals in SWEEPS.items():
        for v in vals:
            c = dict(DEFAULT); c[key] = v
            if key == 'twin_delta': c['twins'] = 0.1
            for s in (11, 12): plan.append((c, s, (4,)))
    for c, s, Ls in plan:
        k = (json.dumps(c, sort_keys=True), s)
        if k in done: continue
        rows += run_config(c, s, Ls=Ls); done.add(k); json.dump(rows, open(out, 'w'))

if __name__ == '__main__':
    main()
