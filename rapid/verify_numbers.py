"""Recompute every RAPID/simulation number quoted in the paper from the saved result files and check the text."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import json, re, numpy as np
tex = open(str(PAPER_TEX)).read() if PAPER_TEX.exists() else ''
if not tex: print(f'paper source not found at {PAPER_TEX}: printing numbers only (set PAPER_TEX to check them against the paper)')
S = json.load(open('rapid_summary.json'))['summary']; A = json.load(open('ablate_cost.json')); C = A['cost']; RL = json.load(open('rf_latency.json'))
P = json.load(open('rapid_summary.json')); M = json.load(open('matched_tar.json')); FC = json.load(open('frontier_counts.json'))
def pc(x, d=1): return f'{100 * x:.{d}f}'
checks = []
m = lambda c, k: S[c][k]['med']
BQ, BI, RF, RQ, RI, LF = 'SMoRFFI-RF128|quantile', 'SMoRFFI-RF128|ident', 'RAPID|floor', 'RAPID|quantile', 'RAPID|ident', 'RAPID-Lite|floor'
row = lambda label, vals: checks.append((label, ' & '.join(vals)))
# main table rows (strip bold for matching)
plain = re.sub(r'\\textbf\{([^}]*)\}', r'\1', tex)
def trow(label, key, f):
    vals = [f(m(c, key)) for c in (BQ, RQ, BI, RI, RF, LF)]; s = label + ' & ' + ' & '.join(vals)
    checks.append((label, s, s in plain))
pct1 = lambda v: pc(v) + r'\%'; pct2 = lambda v: pc(v, 2) + r'\%'; cnt = lambda v: (f'{v:.1f}' if abs(v - round(v)) > 1e-9 else f'{v:.0f}')
pct0 = lambda v: (pc(v, 0) if abs(100 * v - round(100 * v)) < 1e-9 else pc(v)) + r'\%'
trow('TAR, all identities', 'TAR', pct1); trow('Effective coverage@50', 'eff50', cnt); trow('Identification accuracy', 'acc', pct1)
trow('Pooled FAR', 'pooled', pct2); trow('Development-chosen attacker', 'devchosen', pct1); trow('Retry success, $k=5$', 'retry5', pct1); trow('Retry success, $k=20$', 'retry20', pct1)
trow('Pairs with FAR $\\geq$10\\%', 'ge10', cnt); trow('Pairs with FAR $\\geq$50\\%', 'ge50', cnt); trow('Pairs with FAR $=$100\\%', 'eq100', cnt)
trow('99th-percentile pair FAR', 'p99', pct1); trow('Equal-error rate', 'EER', pct2)
for lab, key in (('TAR at FAR 1\\% (global threshold)', 'TAR@FAR0.01'), ('TAR at FAR 0.1\\% (global threshold)', 'TAR@FAR0.001')): trow(lab, key, pct1)
trow('AUC', 'AUC', lambda v: f'{v:.4f}')
for lab, s, ok in checks: print(('OK ' if ok else ('MISMATCH ' if tex else '')) + s)
# matched table
for g, key in (('0.6', '60'), ('0.75', '75'), ('0.8', '80')):
    r = M[g]; print('matched', key, 'pooled', pc(r['pooled'][0], 2), pc(r['pooled'][1], 2), r['pooled'][3:], 'devch', pc(r['devchosen'][0]), pc(r['devchosen'][1]), r['devchosen'][3:], 'ge50', r['ge50'][0], r['ge50'][1], r['ge50'][3:])
print('frontier counts', FC)
print('cost', {k: (round(v['bytes'] / 2**20, 2), v.get('fit_s'), v['single'], round(v['batch_ms'], 3)) for k, v in C.items()}, 'rf single', RL, 'protocol fit medians', P['costs'])
for a in ('One CosFace', '+ ensemble', '+ rival margin', '+ floor rule (RAPID)', 'RAPID without the ensemble'): pass
