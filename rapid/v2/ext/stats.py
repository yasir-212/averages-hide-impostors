"""Statistical significance of RAPID's improvements (post-freeze, from saved session scores): per-target genuine acceptance and
per-target development-chosen attacker success for RAPID and the equal-data baselines in each of the 6 runs; 95% confidence
intervals of the mean paired difference by a cluster bootstrap over the 122 targets (all runs of a target resampled together),
and an exact sign test over targets."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R / 'rapid' / 'v2', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, json, numpy as np
from common import load_frames
from evalkit import windows, accept
from core2 import margin
from final_runner import BLK, ORDERS, CFG
from scipy.stats import binomtest
SC = str(_HERE.parent / 'scores')

def per_target(sysn, seed, order, Y, n):
    cal, att = ORDERS[order]; z = np.load(f'{SC}/{sysn}_s{seed}_{order}.npz'); S = {k: margin(z[k]) for k in (att, 'test')}
    th = np.full(n, 1e-12 + (CFG['delta'] if sysn == 'RAPID' else 0.0))
    tar, P, _ = accept(S['test'], Y['test'], n, th); _, Pd, _ = accept(S[att], Y[att], n, th)
    sel = np.nanargmax(np.where(np.isfinite(Pd), Pd, -1), 0); return tar, P[sel, np.arange(n)], np.nanmax(P, 0)

def main():
    x, y, row, n = load_frames(); Y = {k: windows(x, y, row, n, *v)[1] for k, v in BLK.items()}; rng = np.random.default_rng(2026)
    D = {s: [per_target(s, seed, o, Y, n) for seed in (0, 1, 2) for o in ('orig', 'swap')] for s in ('RAPID', 'RF-eq', 'MLP-eq')}
    out = {}
    for b in ('RF-eq', 'MLP-eq'):
        for k, name in ((0, 'TAR'), (1, 'devchosen'), (2, 'worst_per_target')):
            d = np.stack([D['RAPID'][r][k] - D[b][r][k] for r in range(6)], 1)           # (targets, runs)
            tm = d.mean(1); boots = [tm[rng.integers(0, n, n)].mean() for _ in range(10000)]
            better = int((tm > 0).sum()) if name == 'TAR' else int((tm < 0).sum()); worse = int((tm < 0).sum()) if name == 'TAR' else int((tm > 0).sum())
            p = binomtest(better, better + worse, 0.5).pvalue if better + worse else 1.0
            out[f'{b}.{name}'] = dict(mean=float(tm.mean()), lo=float(np.quantile(boots, 0.025)), hi=float(np.quantile(boots, 0.975)),
                                       better=better, worse=worse, ties=int(n - better - worse), sign_p=float(p))
            print(b, name, {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in out[f'{b}.{name}'].items()})
    json.dump(out, open(str(_HERE / 'stats.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
