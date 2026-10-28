"""Fill the simulation section from sim2_results.json / sim2_wins.json / sim2_table.json."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'rapid' / 'v2', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import FIG_DIR
import json, numpy as np
R = json.load(open(str(_HERE / 'sim2_results.json'))); W = json.load(open(str(_HERE / 'sim2_wins.json')))
T = json.load(open(str(_HERE / 'sim2_table.json')))
DEF = ['n_dev', 'snr_db', 'spread', 'twins', 'twin_delta', 'drift_ppm']
key = lambda r: (json.dumps({k: r[k] for k in DEF}, sort_keys=True), r.get('twin_channel', False), r['data_seed'], r['order'], r['L'])
A = {key(r): r for r in R if r['system'] == 'RAPID' and r['rule'] == 'ident' and not r.get('twin_channel')}
def cmp(bs, br, m, lower):
    B = {key(r): r for r in R if r['system'] == bs and r['rule'] == br and not r.get('twin_channel')}; c = [k for k in A if k in B]
    bt = sum((A[k][m] < B[k][m] - 1e-12) if lower else (A[k][m] > B[k][m] + 1e-12) for k in c); eq = sum(abs(A[k][m] - B[k][m]) <= 1e-12 for k in c)
    worse = [k for k in c if ((A[k][m] > B[k][m] + 1e-12) if lower else (A[k][m] < B[k][m] - 1e-12))]
    return bt, eq, len(c), worse
out = {}
for m, lower in (('TAR', False), ('pooled', True), ('worst', True), ('devchosen', True), ('EER', True), ('AUC', False), ('acc', False), ('retry20', True)):
    worst_case = None
    for bs, br in (('RF-eq', 'ident'), ('RF-eq', 'quantile'), ('RF', 'quantile'), ('RF', 'ident')):
        bt, eq, n, w = cmp(bs, br, m, lower); out[(m, bs, br)] = (bt, eq, n, w)
N = out[('TAR', 'RF-eq', 'ident')][2]
def be(m):   # better-or-tied against the equal-data forest's identification rule
    bt, eq, n, w = out[(m, 'RF-eq', 'ident')]; return f'{bt + eq}'
exc = sorted({json.loads(k[0])['snr_db'] for m in ('TAR', 'pooled', 'worst', 'devchosen') for k in out[(m, 'RF-eq', 'ident')][3]})
allw = {m: sum(len(out[(m, b, r)][3]) for b, r in (('RF-eq', 'ident'), ('RF-eq', 'quantile'), ('RF', 'quantile'), ('RF', 'ident'))) for m in ('TAR', 'pooled', 'worst', 'devchosen', 'EER', 'AUC')}
print('N', N, {m: (out[(m, 'RF-eq', 'ident')][:3], len(out[(m, 'RF-eq', 'ident')][3])) for m in ('TAR', 'pooled', 'worst', 'devchosen', 'EER', 'AUC', 'acc', 'retry20')})
print('losses vs any forest rule', allw, 'loss configs snr', exc)
for m in ('TAR', 'pooled', 'worst', 'devchosen'):
    for b, r in (('RF-eq', 'ident'), ('RF-eq', 'quantile'), ('RF', 'quantile')):
        w = out[(m, b, r)][3]
        if w: print(m, b, r, [(json.loads(k[0]), k[2], k[3]) for k in w][:6])
it = T['Identical twins']
print('identical', {k: v for k, v in it.items()})
