"""Crossed (two-way) bootstrap for the main comparison (Owen's pigeonhole bootstrap): source radios and target radios are resampled
independently with replacement, the development-chosen attacker picks, for every resampled target, the resampled source with the
highest development FAR, and every metric is recomputed; the same resample is applied to RAPID and the baseline in every run, and
the six runs are averaged before differencing. Uses the stored session scores of the frozen evaluation only."""
import sys, json, numpy as np
import pathlib as _pl
R = str(next(p for p in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents] if (p / 'paths.py').exists()))
for p in (R + '/rapid/v2', R + '/rapid', R): sys.path.insert(0, p)
import os; os.environ.setdefault('SMORFFI_FEATURES', 'features.npz')   # set to the SMoRFFI feature file
from common import load_frames
from evalkit import windows, accept
from core2 import margin
from final_runner import BLK, ORDERS, CFG
SC = R + '/rapid/v2/scores'; n = 122; B = 4000

def mats(S, Y, att, beta):
    th = np.full(n, 1e-12 + beta); tar, P, _ = accept(S['test'], Y['test'], n, th); _, Pd, _ = accept(S[att], Y[att], n, th)
    return tar, P, Pd

def main():
    x, y, row, _ = load_frames(); Y = {k: windows(x, y, row, n, *v)[1] for k, v in BLK.items()}
    sysrule = {'RAPID': CFG['delta'], 'MLP-eq': 0.0, 'RF-eq': 0.0}         # identification rule for the baselines (Table 8 best rule)
    M = {s: [] for s in sysrule}
    for seed in (0, 1, 2):
        for order, (cal, att) in ORDERS.items():
            for s, b in sysrule.items():
                d = np.load(f'{SC}/{s}_s{seed}_{order}.npz'); S = {k: margin(d[k]) for k in (att, 'test')}
                M[s].append(mats(S, Y, att, b))
    rng = np.random.default_rng(2026); out = {s: {'TAR': [], 'pooled': [], 'devchosen': []} for s in sysrule}
    def stat(tar, P, Pd, src, tgt):
        Pt = P[np.ix_(src, tgt)]; Pdt = Pd[np.ix_(src, tgt)]
        same = src[:, None] == tgt[None, :]; Pt = np.where(same, np.nan, Pt); Pdt = np.where(same, -1, np.nan_to_num(Pdt, nan=-1))
        sel = Pdt.argmax(0); dc = Pt[sel, np.arange(len(tgt))]
        return tar[tgt].mean(), np.nanmean(Pt), np.nanmean(dc)
    for b in range(B + 1):
        src = np.arange(n) if b == 0 else rng.integers(0, n, n); tgt = np.arange(n) if b == 0 else rng.integers(0, n, n)
        for s in sysrule:
            v = np.array([stat(*M[s][r], src, tgt) for r in range(6)]).mean(0)
            for k, val in zip(('TAR', 'pooled', 'devchosen'), v): out[s][k].append(float(val))
    res = {}
    for base in ('MLP-eq', 'RF-eq'):
        for k in ('TAR', 'pooled', 'devchosen'):
            d = np.array(out['RAPID'][k]) - np.array(out[base][k])
            res[f'{base}.{k}'] = dict(point=float(d[0]), lo=float(np.quantile(d[1:], .025)), hi=float(np.quantile(d[1:], .975)),
                                       frac_better=float(np.mean(d[1:] > 0) if k == 'TAR' else np.mean(d[1:] < 0)))
            print(base, k, {kk: (round(100 * vv, 3) if kk != 'frac_better' else vv) for kk, vv in res[f'{base}.{k}'].items()}, flush=True)
    json.dump(res, open(str(_pl.Path(__file__).resolve().parent / 'crossed_boot.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
