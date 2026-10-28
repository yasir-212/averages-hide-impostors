"""Summary of the unenrolled-attacker experiment (3 device partitions x 2 role orders)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import json, sys, numpy as np
rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else str(_HERE / 'openset_test.json')))
M = ['TAR', 'eff50', 'unenr_pooled', 'unenr_worst', 'unenr_ge50', 'unenr_devchosen', 'unenr_retry20', 'enr_pooled', 'enr_worst', 'enr_ge50']
HI = ('TAR', 'eff50'); out = {}
for s in ('RF-eq', 'MLP-eq', 'RAPID'):
    for r in ('ident', 'ident+abs'):
        rr = [x for x in rows if x['system'] == s and x['rule'] == r]
        if rr: out[f'{s}|{r}'] = {m: float(np.median([x[m] for x in rr])) for m in M} | {'n': len(rr)}
wins = {}
for r in ('ident', 'ident+abs'):
    rp = {(x['part'], x['order']): x for x in rows if x['system'] == 'RAPID' and x['rule'] == r}
    for m in M:
        w = 0
        for k, v in rp.items():
            bs = [x[m] for x in rows if (x['part'], x['order']) == k and x['rule'] == r and x['system'] != 'RAPID']
            w += bool(bs) and all((v[m] > b + 1e-12) if m in HI else (v[m] < b - 1e-12) for b in bs)
        wins[f'{r}|{m}'] = [w, len(rp)]
json.dump(dict(summary=out, wins=wins), open(str(_HERE / 'openset_summary.json'), 'w'), indent=1)
for k, v in out.items(): print(k.ljust(18), ' '.join(f'{m}={v[m]:.4g}' for m in M), 'n', v['n'])
print(wins)
