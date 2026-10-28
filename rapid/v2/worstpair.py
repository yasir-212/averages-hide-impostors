"""Worst pair per run with exact 95% Clopper-Pearson intervals (37 test sessions per source) and the pair's identity."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, json, numpy as np
from scipy.stats import beta
from common import load_frames
from evalkit import windows, accept
from core2 import margin
n = 122; x, y, row, _ = load_frames(); Y = {k: windows(x, y, row, n, *v)[1] for k, v in {'A': (600, 725), 'B': (725, 850), 'test': (850, 999)}.items()}
def cp(k, m):
    lo = beta.ppf(0.025, k, m - k + 1) if k > 0 else 0.0; hi = beta.ppf(0.975, k + 1, m - k) if k < m else 1.0; return lo, hi
out = {}
for name, beta_m in (('RAPID', 0.02), ('RF-eq', 0.0), ('MLP-eq', 0.0)):
    for seed in (0, 1, 2):
        for order in ('orig', 'swap'):
            try: d = np.load(f'scores/{name}_s{seed}_{order}.npz')
            except FileNotFoundError: continue
            S = margin(d['test']); tar, P, acc = accept(S, Y['test'], n, np.full(n, 1e-12 + beta_m))
            i = np.nanargmax(P); s, t = np.unravel_index(i, P.shape); m = int((Y['test'] == s).sum()); k = int(round(P[s, t] * m))
            lo, hi = cp(k, m); out[f'{name}|{seed}|{order}'] = dict(src=int(s), tgt=int(t), far=float(P[s, t]), k=k, m=m, lo=lo, hi=hi, tar_src=float(tar[s]), tar_tgt=float(tar[t]))
            print(f'{name:7s} {seed} {order}: worst {s}->{t} {100*P[s,t]:.1f}% ({k}/{m}) CI [{100*lo:.1f}, {100*hi:.1f}]  TAR of source {100*tar[s]:.0f}%')
json.dump(out, open('worstpair.json', 'w'), indent=1)
