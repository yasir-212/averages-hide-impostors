"""LaTeX table and text values for the unenrolled-attacker experiment."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import json
S = json.load(open(str(_HERE / 'openset_summary.json'))); Su, Wn = S['summary'], S['wins']
HI = ('TAR', 'eff50')
cols = [('TAR', 'p1'), ('eff50', 'c'), ('unenr_pooled', 'p2'), ('unenr_ge50', 'c'), ('enr_pooled', 'p2'), ('enr_ge50', 'c')]
def f(v, k):
    if k == 'p1': return f'{100 * v:.1f}\\%'
    if k == 'p2': return f'{100 * v:.2f}\\%'
    return f'{v:.1f}' if abs(v - round(v)) > 1e-9 else f'{v:.0f}'
rows = []
for rule, rlab in (('ident', '\\emph{Without absolute check (identification rule; frozen margin for \\rapid)}'), ('ident+abs', '\\emph{With absolute check}')):
    rows.append('\\multicolumn{7}{@{}l}{%s} \\\\' % rlab)
    keys = [f'{s}|{rule}' for s in ('RF-eq', 'MLP-eq', 'RAPID')]
    for s, k in zip(('Forest, equal data', 'MLP, equal data', 'RAPID'), keys):
        cells = []
        for m, kind in cols:
            vals = [Su[kk][m] for kk in keys]; b = max(vals) if m in HI else min(vals); v = Su[k][m]
            cells.append(('\\textbf{%s}' % f(v, kind)) if abs(v - b) < 1e-12 else f(v, kind))
        rows.append(f'{s} & ' + ' & '.join(cells) + ' \\\\')
    rows.append('\\midrule')
rows = rows[:-1]
tab = r'''\begin{table*}[tp]
\caption{Unenrolled attackers on \smorffi: 30 of the 122 devices are never enrolled, and their sessions claim each of the 92 enrolled identities (medians over 3 random device partitions $\times$ 2 role orders). The absolute check uses the 5th percentile of each identity's cross-fitted genuine scores. Pairs: source--target pairs with FAR at or above 50\%. The best value per rule group is in bold.}
\label{tab:openset}
\footnotesize
\setlength{\tabcolsep}{2.2pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{Enrolled genuine} & \multicolumn{2}{c}{Unenrolled attackers} & \multicolumn{2}{c}{Enrolled impostors} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
System & TAR & Eff.@50 & Pooled FAR & Pairs $\geq$50\% & Pooled FAR & Pairs $\geq$50\% \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}'''
open(str(_HERE / 'tab_openset.tex'), 'w').write(tab); print(tab); print(Wn)
