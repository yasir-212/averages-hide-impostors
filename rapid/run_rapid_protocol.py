"""Frozen-protocol comparison on SMoRFFI: RAPID / RAPID-Lite vs the 2026 SMoRFFI framework baseline.
3 seeds x 2 role orders. The test block is scored once per fitted model; thresholds come from the calibration role."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, json, time, gc, hashlib, numpy as np
sys.path.insert(0, str(_R / 'rapid'))
from common import load_frames, FEAT
from evalkit import windows, SMoRFFIBaseline, RAPIDSystem, rule_thresholds, full_metrics, curve, latency_ms, global_roc
from rapid_core import rival
from sklearn.preprocessing import StandardScaler

def margin(S):
    return rival(S, 1.0)

def main(out='protocol_rapid.json'):
    x, y, row, n = load_frames(); sha = hashlib.sha256(open(FEAT, 'rb').read()).hexdigest()
    B = {'A': (600, 725), 'B': (725, 850), 'test': (850, 999)}
    W = {k: windows(x, y, row, n, *v) for k, v in B.items()}
    ftr, yf = x[row < 600], y[row < 600]; sc = StandardScaler().fit(ftr)
    info = dict(sha=sha, n=n, train_frames=int(len(ftr)), windows={k: int(len(v[1])) for k, v in W.items()},
                min_per_source={k: int(np.bincount(v[1]).min()) for k, v in W.items()})
    print(info, flush=True)
    runs, curves, costs = [], {}, []
    systems = [('SMoRFFI-RF128', lambda s: SMoRFFIBaseline(n, seed=s)), ('RAPID', lambda s: RAPIDSystem(n, seed=s, E=3)),
               ('RAPID-Lite', lambda s: RAPIDSystem(n, seed=s, E=2))]
    for seed in (0, 1, 2):
        for sname, make in systems:
            sysm = make(seed).fit(ftr, yf, sc)
            S = {k: sysm.scores(W[k][0]) for k in W}
            lat = latency_ms(sysm, W['test'][0][:1000]); size, npar = sysm.size_bytes()
            costs.append(dict(system=sname, seed=seed, fit_s=sysm.fit_s, size_MB=size / 2**20, params_or_nodes=npar, ms_per_session=lat))
            sysm.free(); del sysm; gc.collect()
            native = S if sname.startswith('RAPID') else {k: v for k, v in S.items()}
            marg = {k: margin(v) for k, v in S.items()} if sname == 'SMoRFFI-RF128' else S   # RAPID scores are already margins
            rules = {'SMoRFFI-RF128': [('quantile', native), ('ident', marg), ('floor', marg)],
                     'RAPID': [('floor', S), ('quantile', S), ('ident', S)], 'RAPID-Lite': [('floor', S), ('quantile', S), ('ident', S)]}[sname]
            g = global_roc(native['test'], W['test'][1])
            for order in ('orig', 'swap'):
                C, D = ('B', 'A') if order == 'orig' else ('A', 'B')
                for rule, SS in rules:
                    th = rule_thresholds(rule, SS[C], W[C][1], n, 0.9)
                    r = full_metrics(th, SS[D], W[D][1], SS['test'], W['test'][1], n)
                    r.update({k: v for k, v in g.items()}); r.update(system=sname, rule=rule, seed=seed, order=order); runs.append(r)
                    curves[f'{sname}|{rule}|{seed}|{order}'] = curve(SS[C], W[C][1], SS[D], W[D][1], SS['test'], W['test'][1], n, rule=rule)
            np.savez_compressed(f'scores_{sname}_s{seed}.npz', **{k: v.astype(np.float32) for k, v in S.items()}, **{f'y_{k}': W[k][1] for k in W})
            print(sname, seed, 'fit %.1fs' % costs[-1]['fit_s'], 'size %.2f MB' % costs[-1]['size_MB'], 'lat %.4f ms' % lat, flush=True)
            del S; gc.collect()
    json.dump(dict(info=info, runs=runs, costs=costs, curves=curves), open(out, 'w'))
    print('saved', len(runs), 'rows')

if __name__ == '__main__':
    main()
