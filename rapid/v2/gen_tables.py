"""Generate the LaTeX tables and fill the text placeholders of the RAPID section from the result files."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import json, re, sys, numpy as np
S = json.load(open(str(_HERE / 'final_summary.json'))); SUM, COST = S['summary'], S['costs']
R = json.load(open(str(_HERE / 'final_results.json')))['runs']
HI = ['TAR', 'eff50', 'acc', 'TAR@FAR0.01', 'TAR@FAR0.001', 'AUC']
BASES = ['RF|quantile', 'RF|ident', 'RF-eq|quantile', 'RF-eq|ident', 'MLP-eq|quantile', 'MLP-eq|ident']
def fmt(m, v, pct=True):
    if m in ('pooled', 'EER'): s = f'{100 * v:.2f}'
    elif m == 'AUC': return f'{v:.4f}'
    elif m in ('eff50', 'ge10', 'ge50', 'eq100'): return f'{v:.1f}' if abs(v - round(v)) > 1e-9 else f'{v:.0f}'
    elif m == 'claims': return f'{v:.2f}'
    else: s = f'{100 * v:.1f}'
    if s.endswith('.0') and m in ('worst', 'multi') and v in (0.0, 1.0): s = s[:-2]
    return s + (r'\%' if pct else '')
def best(m):
    vals = [SUM[b][m]['med'] for b in BASES]; return max(vals) if m in HI else min(vals)
def run_wins(m, as_tuple=False):
    """per run, RAPID against the best of the six baseline configurations in that run: better / tied / worse"""
    rp = {(r['seed'], r['order']): r[m] for r in R if r['system'] == 'RAPID'}; b_ = t_ = w_ = 0
    for k, v in rp.items():
        bs = [r[m] for r in R if (r['seed'], r['order']) == k and f"{r['system']}|{r['rule']}" in BASES]
        best_b = max(bs) if m in HI else min(bs)
        if abs(v - best_b) <= 1e-12: t_ += 1
        elif (v > best_b) if m in HI else (v < best_b): b_ += 1
        else: w_ += 1
    return (b_, t_, w_) if as_tuple else f'{b_}/{t_}/{w_}'

TABLE_METRICS = ['TAR', 'eff50', 'acc', 'pooled', 'worst', 'p99', 'ge10', 'ge50', 'eq100', 'devchosen', 'retry5', 'retry20', 'multi', 'TAR@FAR0.01', 'TAR@FAR0.001', 'EER', 'AUC']
def runs_total():
    t = [run_wins(m, True) for m in TABLE_METRICS]; return sum(a + b for a, b, c in t), sum(a + b + c for a, b, c in t)
SPECIAL = {
    'RUNS_BT': lambda: str(runs_total()[0]), 'RUNS_ALL': lambda: str(runs_total()[1]),
    'RATIO': lambda: f"{COST['RF-eq']['size_MB'] / COST['RAPID']['size_MB']:.0f}",
    'PARAMS': lambda: f"{int(COST['RAPID']['params']):,}".replace(',', '{,}'),
    'RAPIDMB': lambda: f"{COST['RAPID']['size_MB']:.1f}", 'RAPIDMS': lambda: f"{COST['RAPID']['ms_single']:.1f}",
}
def fill(text):
    def rep(mo):
        if mo.group(1) in SPECIAL: return SPECIAL[mo.group(1)]()
        parts = mo.group(1).split('|')
        if parts[0] == 'BEST': return fmt(parts[1], best(parts[1]))
        if parts[0] == 'WINS': return str(run_wins(parts[1]))
        return fmt(parts[2], SUM[f'{parts[0]}|{parts[1]}'][parts[2]]['med'])
    return re.sub(r'\{\{([^}]+)\}\}', rep, text)
ROWS = [('TAR, all identities', 'TAR'), ('Effective coverage@50', 'eff50'), ('Identification accuracy', 'acc'), None,
        ('Pooled FAR', 'pooled'), ('Worst-pair FAR', 'worst'), ('99th-percentile pair FAR', 'p99'), ('Pairs with FAR $\\geq$10\\%', 'ge10'),
        ('Pairs with FAR $\\geq$50\\%', 'ge50'), ('Pairs with FAR $=$100\\%', 'eq100'), ('Development-chosen attacker', 'devchosen'),
        ('Retry success, $k_{\\mathrm r}=5$', 'retry5'), ('Retry success, $k_{\\mathrm r}=20$', 'retry20'), ('Sessions accepted for $\\geq$2 claims', 'multi'), None,
        ('TAR at FAR 1\\% (global threshold)', 'TAR@FAR0.01'), ('TAR at FAR 0.1\\% (global threshold)', 'TAR@FAR0.001'), ('Equal-error rate', 'EER'), ('AUC', 'AUC')]
def main_table():
    cols = BASES + ['RAPID|ident']; out = []
    for row in ROWS:
        if row is None: out.append('\\midrule'); continue
        lab, m = row; vals = [SUM[c][m]['med'] for c in cols]; b = max(vals) if m in HI else min(vals)
        cells = [('\\textbf{%s}' % fmt(m, v)) if abs(v - b) < 1e-12 else fmt(m, v) for v in vals]
        out.append(f'{lab} & ' + ' & '.join(cells) + f' & {run_wins(m)} \\\\')
    return '\n'.join(out)
if __name__ == '__main__':
    if sys.argv[1] == 'table': print(main_table())
    else:
        src = open(sys.argv[2]).read(); open(sys.argv[3], 'w').write(fill(src)); print('filled', sys.argv[3])
