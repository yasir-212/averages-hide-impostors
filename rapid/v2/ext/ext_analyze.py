"""Evaluates the per-frame scores written by final_ext.py (and gbt_ext.py): extended baselines, session-length sweep, ensemble
size and exemplar count of RAPID, and an exact-reproduction check of final_runner.py's stored session scores."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R / 'rapid' / 'v2', _R, _HERE):
    _sys.path.insert(0, str(_p))
import os, sys, json, glob, numpy as np
from common import load_frames
from evalkit import full_metrics
from core2 import margin
from final_runner import BLK, ORDERS, CFG
EXT = str(_HERE); SC = str(_HERE.parent / 'scores')

def sessions(x, y, row, n, lo, hi, idx, L):
    pos = -np.ones(len(x), int); pos[idx] = np.arange(len(idx)); S, Y = [], []
    for j in range(n):
        ix = np.flatnonzero((y == j) & (row >= lo) & (row < hi)); ix = ix[np.argsort(row[ix])]
        for p in range(0, len(ix) - L + 1, L):
            ids = ix[p:p + L]
            if np.all(np.diff(row[ids]) == 1) and np.all(pos[ids] >= 0): S.append(pos[ids]); Y.append(j)
    return np.array(S), np.array(Y)

def agg(F, S):
    return F[S].mean(1)

def systems(z, b):
    """name -> (per-frame score matrix, kind) for block b."""
    out = {'RAPID': ((z[f'RAPIDm0_{b}'] + z[f'RAPIDm1_{b}'] + z[f'RAPIDm2_{b}']) / 3, 'rapid'),
           'RAPID-E1': (z[f'RAPIDm0_{b}'], 'rapid'), 'RAPID-E2': ((z[f'RAPIDm0_{b}'] + z[f'RAPIDm1_{b}']) / 2, 'rapid')}
    for kk in (1, 5, 10): out[f'RAPID-k{kk}'] = (z[f'RAPIDk{kk}_{b}'], 'rapid')
    for nm in ('CFmean3', 'CFeq', 'CFtrain'): out[nm] = (z[f'{nm}_{b}'], 'cos')
    for nm in ('RF-eq', 'MLP-eq', 'kNN-eq', 'GBT-eq'):
        if f'{nm}_{b}' in z: out[nm] = (z[f'{nm}_{b}'], 'post')
    return out

def main():
    x, y, row, n = load_frames(); rows = []; checks = []
    for path in sorted(glob.glob(f'{EXT}/frames_s*_*.npz')):
        import re; mm = re.search(r'frames_s(\d+)_(orig|swap)\.npz$', path); seed, order = int(mm.group(1)), mm.group(2); cal, att = ORDERS[order]
        z = dict(np.load(path)); g = f'{EXT}/gbt_s{seed}_{order}.npz'
        if os.path.exists(g): z.update(dict(np.load(g)))
        sysA, sysT = systems(z, att), systems(z, 'test')
        for L in (1, 2, 4, 8, 16):
            Sa, Ya = sessions(x, y, row, n, *BLK[att], z[f'idx_{att}'], L); St, Yt = sessions(x, y, row, n, *BLK['test'], z['idx_test'], L)
            for name in sysT:
                if L != 4 and name not in ('RAPID', 'RF-eq', 'MLP-eq', 'kNN-eq', 'GBT-eq', 'CFeq', 'CFtrain'): continue
                Fa, kind = sysA[name]; Ft, _ = sysT[name]; A, T = agg(Fa, Sa), agg(Ft, St)
                Ma, Mt = margin(A), margin(T); th = np.full(n, 1e-12 + (CFG['delta'] if kind == 'rapid' else 0.0))
                r = full_metrics(th, Ma, Ya, Mt, Yt, n)
                if kind != 'rapid':                       # global-threshold metrics on each system's native score
                    from evalkit import global_roc; r.update(global_roc(T, Yt))
                r.update(system=name, seed=seed, order=order, L=L); rows.append(r)
                if L == 4 and name in ('RAPID', 'RF-eq', 'MLP-eq'):
                    ref = np.load(f'{SC}/{name}_s{seed}_{order}.npz')['test']
                    checks.append(dict(system=name, seed=seed, order=order, max_abs_diff=float(np.abs(ref - T).max())))
        print(path, 'done', flush=True)
    json.dump(dict(rows=rows, checks=checks), open(f'{EXT}/ext_results.json', 'w'))
    for c in checks: print(c)
    names = sorted({r['system'] for r in rows})
    for L in (1, 2, 4, 8, 16):
        for nm in names:
            rr = [r for r in rows if r['system'] == nm and r['L'] == L]
            if not rr: continue
            md = {m: float(np.median([r[m] for r in rr])) for m in ('TAR', 'pooled', 'worst', 'ge50', 'devchosen', 'retry20', 'EER', 'AUC', 'TAR@FAR0.001', 'acc')}
            print(f'L={L:2d} {nm:10s} runs={len(rr)} ' + ' '.join(f'{k} {100*v:.2f}' if k not in ('ge50', 'AUC') else f'{k} {v:.4g}' for k, v in md.items()))

if __name__ == '__main__':
    main()
