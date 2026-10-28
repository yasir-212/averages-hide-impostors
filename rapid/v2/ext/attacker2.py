"""Stronger development-chosen attacker (post-freeze, from saved session scores): for every target t the attacker ranks sources
by their development FAR and breaks ties (in particular all-zero development FAR) by the mean development score of claim t
(each system's own decision score), instead of by source index. Also reports how many targets had all-zero development FAR."""
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
SC = str(_HERE.parent / 'scores')

def load(sysn, seed, order, att):
    if sysn == 'RAPID': z = np.load(f'{SC}/RAPID_s{seed}_{order}.npz')
    elif sysn == 'RF': z = np.load(str(_R / 'rapid' / f'scores_SMoRFFI-RF128_s{seed}.npz'))
    else: z = np.load(f'{SC}/{sysn}_s{seed}_{order}.npz')
    return {k: margin(z[k]) for k in (att, 'test')}

def main():
    x, y, row, n = load_frames(); Y = {k: windows(x, y, row, n, *v)[1] for k, v in BLK.items()}; out = {}
    for sysn in ('RF', 'RF-eq', 'MLP-eq', 'RAPID'):
        rows = []
        for seed in (0, 1, 2):
            for order, (cal, att) in ORDERS.items():
                S = load(sysn, seed, order, att); th = np.full(n, 1e-12 + (CFG['delta'] if sysn == 'RAPID' else 0.0))
                _, P, acc = accept(S['test'], Y['test'], n, th); _, Pd, _ = accept(S[att], Y[att], n, th)
                ms = np.stack([S[att][Y[att] == s].mean(0) for s in range(n)]); np.fill_diagonal(ms, -np.inf)   # (source, target)
                Pd2 = np.where(np.isfinite(Pd), Pd, -1)
                sel_idx = np.nanargmax(Pd2, 0)
                key = Pd2 * 1e6 + np.where(np.isfinite(ms), ms, -1e6)                                     # FAR first, then mean score
                sel = np.argmax(key, 0)
                zero = int((Pd2.max(0) <= 0).sum())
                k20 = []
                for t in range(n):
                    a = acc[Y['test'] == sel[t], t]; nb = len(a) // 20
                    if nb: k20.append(np.mean([a[b * 20:(b + 1) * 20].any() for b in range(nb)]))
                rows.append(dict(seed=seed, order=order, devchosen_index=float(P[sel_idx, np.arange(n)].mean()), devchosen_score=float(P[sel, np.arange(n)].mean()),
                                 retry20_score=float(np.mean(k20)), zero_dev_targets=zero))
        out[sysn] = dict(runs=rows, median={k: float(np.median([r[k] for r in rows])) for k in ('devchosen_index', 'devchosen_score', 'retry20_score', 'zero_dev_targets')})
        print(sysn, {k: round(v, 4) for k, v in out[sysn]['median'].items()})
    json.dump(out, open(str(_HERE / 'attacker2.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
