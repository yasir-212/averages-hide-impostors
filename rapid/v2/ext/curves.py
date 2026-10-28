"""Post-freeze analyses from the saved session scores of final_runner.py (no refitting):
retry success for k = 1..20 attempts, the margin (beta) sensitivity of RAPID, per-pair FAR tails, per-target attacker success.
Identification rule for the baselines (their strongest rule on the risk metrics), RAPID's frozen rule unless beta is varied."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R / 'rapid' / 'v2', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, json, numpy as np
from common import load_frames
from evalkit import windows, accept, full_metrics
from core2 import margin
from final_runner import BLK, ORDERS, CFG

SC = str(_HERE.parent / 'scores')
SYS = ['RF', 'MLP', 'RF-eq', 'MLP-eq', 'RAPID']

def load_scores(sysn, seed, order, cal, att):
    if sysn == 'RAPID': z = np.load(f'{SC}/RAPID_s{seed}_{order}.npz'); return {k: margin(z[k]) for k in (att, 'test')}
    if sysn in ('RF-eq', 'MLP-eq'): z = np.load(f'{SC}/{sysn}_s{seed}_{order}.npz')
    elif sysn == 'MLP': z = np.load(f'{SC}/MLP_s{seed}.npz')
    else: z = np.load(str(_R / 'rapid' / f'scores_SMoRFFI-RF128_s{seed}.npz'))
    return {k: margin(z[k]) for k in (att, 'test')}

def retry_curve(acc, ye, sel, n, kmax=20):
    out = []
    for k in range(1, kmax + 1):
        vals = []
        for t in range(n):
            a = acc[ye == sel[t], t]; nb = len(a) // k
            if nb: vals.append(np.mean([a[b * k:(b + 1) * k].any() for b in range(nb)]))
        out.append(float(np.mean(vals)))
    return out

def main():
    x, y, row, n = load_frames(); Y = {k: windows(x, y, row, n, *v)[1] for k, v in BLK.items()}
    res = dict(retry={}, beta={}, tail={}, target={})
    grid = np.logspace(-3, 0, 61)
    betas = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
    for sysn in SYS:
        R, T, G = [], [], []
        for seed in (0, 1, 2):
            for order, (cal, att) in ORDERS.items():
                S = load_scores(sysn, seed, order, cal, att); b = CFG['delta'] if sysn == 'RAPID' else 0.0
                th = np.full(n, 1e-12 + b)
                tar, P, acc = accept(S['test'], Y['test'], n, th); _, Pd, _ = accept(S[att], Y[att], n, th)
                sel = np.nanargmax(np.where(np.isfinite(Pd), Pd, -1), 0)
                R.append(retry_curve(acc, Y['test'], sel, n)); f = P[np.isfinite(P)]
                T.append([int((f >= g - 1e-12).sum()) for g in grid]); G.append(P[sel, np.arange(n)].tolist())
                if sysn == 'RAPID':
                    for bb in betas:
                        r = full_metrics(np.full(n, 1e-12 + bb), S[att], Y[att], S['test'], Y['test'], n)
                        res['beta'].setdefault(str(bb), []).append(r)
        res['retry'][sysn] = np.median(np.array(R), 0).tolist()
        res['tail'][sysn] = np.median(np.array(T), 0).tolist()
        res['target'][sysn] = G
        print(sysn, 'retry k=1,5,20:', [round(100 * res['retry'][sysn][i], 1) for i in (0, 4, 19)], flush=True)
    res['tail_grid'] = grid.tolist(); res['betas'] = betas
    res['beta_median'] = {b: {m: float(np.median([r[m] for r in rs])) for m in rs[0]} for b, rs in res['beta'].items()}
    json.dump(res, open(str(_HERE / 'curves.json'), 'w'))
    for b in betas:
        r = res['beta_median'][str(b)]
        print('beta %.3f TAR %.1f pooled %.3f worst %.1f ge50 %.1f devch %.1f r20 %.1f' % (b, 100 * r['TAR'], 100 * r['pooled'], 100 * r['worst'], r['ge50'], 100 * r['devchosen'], 100 * r['retry20']))

if __name__ == '__main__':
    main()
