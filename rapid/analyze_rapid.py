"""Summaries of the frozen-protocol comparison: main table, paired run-by-run wins, Pareto rule, matched operating points."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import json, numpy as np
D = json.load(open(str(_R / 'rapid' / 'protocol_rapid.json'))); R = D['runs']; CV = D['curves']
KEYS = ['TAR', 'eff50', 'pooled', 'worst', 'p99', 'ge10', 'ge50', 'eq100', 'devchosen', 'retry5', 'retry20', 'multi', 'claims', 'EER', 'AUC', 'TAR@FAR0.01', 'TAR@FAR0.001', 'acc']
LOWER = {'pooled', 'worst', 'p99', 'ge10', 'ge50', 'eq100', 'devchosen', 'retry5', 'retry20', 'multi', 'claims', 'EER'}
def rows(sys_, rule): return sorted([r for r in R if r['system'] == sys_ and r['rule'] == rule], key=lambda r: (r['seed'], r['order']))
CFG = [('SMoRFFI-RF128', 'quantile'), ('SMoRFFI-RF128', 'ident'), ('SMoRFFI-RF128', 'floor'), ('RAPID', 'floor'), ('RAPID', 'quantile'), ('RAPID', 'ident'), ('RAPID-Lite', 'floor')]
summary = {}
for s, ru in CFG:
    rr = rows(s, ru); summary[f'{s}|{ru}'] = {k: dict(med=float(np.median([r[k] for r in rr])), lo=float(min(r[k] for r in rr)), hi=float(max(r[k] for r in rr))) for k in KEYS}
    print(f'{s:14s} {ru:8s}', ' '.join(f"{k}={summary[f'{s}|{ru}'][k]['med']:.4g}" for k in KEYS))
# paired comparisons (same seed and order)
def paired(a, b):
    A, B = rows(*a), rows(*b); out = {}
    for k in KEYS:
        better = sum((x[k] < y[k] - 1e-12) if k in LOWER else (x[k] > y[k] + 1e-12) for x, y in zip(A, B))
        equal = sum(abs(x[k] - y[k]) <= 1e-12 for x, y in zip(A, B)); out[k] = (better, equal, len(A) - better - equal)
    # Pareto rule (rule 5): eff50 >=, worst <=, devchosen <= with at least one strict, per order
    par = {}
    for o in ('orig', 'swap'):
        ok = []
        for x, y in zip(A, B):
            if x['order'] != o: continue
            weak = x['eff50'] >= y['eff50'] and x['worst'] <= y['worst'] + 1e-12 and x['devchosen'] <= y['devchosen'] + 1e-12
            strict = x['eff50'] > y['eff50'] or x['worst'] < y['worst'] - 1e-12 or x['devchosen'] < y['devchosen'] - 1e-12
            ok.append(weak and strict)
        par[o] = (sum(ok), len(ok))
    return out, par
comparisons = {}
for a, b in ((('RAPID', 'floor'), ('SMoRFFI-RF128', 'quantile')), (('RAPID', 'ident'), ('SMoRFFI-RF128', 'ident')),
             (('RAPID', 'floor'), ('SMoRFFI-RF128', 'floor')), (('RAPID-Lite', 'floor'), ('SMoRFFI-RF128', 'quantile')), (('RAPID', 'quantile'), ('SMoRFFI-RF128', 'quantile'))):
    o, p = paired(a, b); comparisons[f'{a[0]}|{a[1]} vs {b[0]}|{b[1]}'] = dict(per_metric=o, pareto=p)
    print('\n', a, 'vs', b, 'Pareto', p); print('   ', {k: v for k, v in o.items()})
# operating frontier: for each run, best achievable metric at TAR >= x over all rule families of the system
GRID = np.round(np.arange(0.80, 0.981, 0.01), 3)
def frontier(sys_, key):
    out = []
    for seed in (0, 1, 2):
        for o in ('orig', 'swap'):
            pts = [p for k, c in CV.items() if k.startswith(sys_ + '|') and k.endswith(f'|{seed}|{o}') for p in c]
            out.append([min([p[key] for p in pts if p['TAR'] >= g - 1e-12], default=np.nan) for g in GRID])
    return np.array(out)
front = {}
for s in ('SMoRFFI-RF128', 'RAPID', 'RAPID-Lite'):
    front[s] = {k: frontier(s, k).tolist() for k in ('pooled', 'devchosen', 'ge50', 'p99', 'worst')}
for k in ('pooled', 'devchosen', 'ge50', 'p99', 'worst'):
    b = np.array(front['SMoRFFI-RF128'][k]); r = np.array(front['RAPID'][k]); ok = np.isfinite(b) & np.isfinite(r)
    print(f'frontier {k}: RAPID < BL in {(r < b - 1e-12)[ok].sum()} / {ok.sum()} (run,TAR) cells, equal {(np.abs(r - b) <= 1e-12)[ok].sum()}')
# TAR at matched pooled FAR (max TAR with pooled <= f)
def tar_at(sys_, key, f):
    vals = []
    for seed in (0, 1, 2):
        for o in ('orig', 'swap'):
            pts = [p for k, c in CV.items() if k.startswith(sys_ + '|') and k.endswith(f'|{seed}|{o}') for p in c]
            vals.append(max([p['TAR'] for p in pts if p[key] <= f + 1e-12], default=0.0))
    return vals
matched = {}
for key, fs in (('pooled', (0.002, 0.005, 0.01)), ('devchosen', (0.10, 0.15, 0.20, 0.25)), ('ge50', (25, 50, 100))):
    for f in fs:
        for s in ('SMoRFFI-RF128', 'RAPID', 'RAPID-Lite'):
            v = tar_at(s, key, f); matched[f'{s}|{key}<={f}'] = v
        print(f'TAR at {key}<={f}: BL {np.median(matched[f"SMoRFFI-RF128|{key}<={f}"]):.3f}  RAPID {np.median(matched[f"RAPID|{key}<={f}"]):.3f}  Lite {np.median(matched[f"RAPID-Lite|{key}<={f}"]):.3f}')
costs = {}
for s in ('SMoRFFI-RF128', 'RAPID', 'RAPID-Lite'):
    cc = [c for c in D['costs'] if c['system'] == s]; costs[s] = {k: float(np.median([c[k] for c in cc])) for k in ('fit_s', 'size_MB', 'ms_per_session', 'params_or_nodes')}
print(costs)
json.dump(dict(summary=summary, comparisons=comparisons, frontier=front, grid=GRID.tolist(), matched=matched, costs=costs, info=D['info']),
          open(str(_R / 'rapid' / 'rapid_summary.json'), 'w'), indent=1)
