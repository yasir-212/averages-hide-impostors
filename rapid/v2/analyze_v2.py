"""Summaries of the frozen-protocol results: medians over the 6 runs and run-by-run comparison of RAPID with every baseline."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import json, numpy as np
R = json.load(open(str(_HERE / 'final_results.json')))
HI = ['TAR', 'eff50', 'acc', 'TAR@FAR0.01', 'TAR@FAR0.001', 'AUC']
LO = ['pooled', 'worst', 'p99', 'ge10', 'ge50', 'eq100', 'devchosen', 'retry5', 'retry20', 'multi', 'claims', 'EER']
METRICS = HI + LO
SYS = [('RF', 'quantile'), ('RF', 'ident'), ('MLP', 'quantile'), ('MLP', 'ident'), ('RF-eq', 'quantile'), ('RF-eq', 'ident'),
       ('MLP-eq', 'quantile'), ('MLP-eq', 'ident'), ('RAPID', 'ident')]
def runs(s, r): return {(x['seed'], x['order']): x for x in R['runs'] if x['system'] == s and x['rule'] == r}
summary, wins = {}, {}
for s, r in SYS:
    rr = runs(s, r)
    if not rr: continue
    summary[f'{s}|{r}'] = {m: dict(med=float(np.median([v[m] for v in rr.values()])), min=float(min(v[m] for v in rr.values())),
                                    max=float(max(v[m] for v in rr.values()))) for m in METRICS}
RP = runs('RAPID', 'ident')
for s, r in SYS[:-1]:
    B = runs(s, r)
    if not B: continue
    w = {}
    for m in METRICS:
        better = sum((RP[k][m] > B[k][m] + 1e-12) if m in HI else (RP[k][m] < B[k][m] - 1e-12) for k in RP)
        tie = sum(abs(RP[k][m] - B[k][m]) <= 1e-12 for k in RP); w[m] = [int(better), int(tie), int(len(RP) - better - tie)]
    wins[f'{s}|{r}'] = w
costs = {}
for c in R['costs']:
    costs.setdefault(c['system'], []).append(c)
cost_sum = {s: {k: float(np.median([c[k] for c in v])) for k in v[0] if isinstance(v[0][k], (int, float)) and k not in ('seed',)} for s, v in costs.items()}
json.dump(dict(summary=summary, wins=wins, costs=cost_sum), open(str(_HERE / 'final_summary.json'), 'w'), indent=1)
fmt = lambda m, v: f'{100*v:.2f}' if m in ('pooled', 'EER') else (f'{v:.4f}' if m == 'AUC' else (f'{v:.1f}' if m in ('eff50', 'ge10', 'ge50', 'eq100', 'claims') else f'{100*v:.1f}'))
print('metric'.ljust(14) + ''.join(k.replace('quantile', 'q').replace('ident', 'id').rjust(13) for k in summary))
for m in METRICS:
    print(m.ljust(14) + ''.join(fmt(m, summary[k][m]['med']).rjust(13) for k in summary))
print('\nRAPID better/tie/worse per run vs each baseline')
for k, w in wins.items():
    bad = {m: v for m, v in w.items() if v[2] > 0}
    print(k.ljust(16), 'all metrics better or tied in every run' if not bad else f'losses: {bad}')
print('\ncosts', json.dumps(cost_sum, indent=0)[:1500])
