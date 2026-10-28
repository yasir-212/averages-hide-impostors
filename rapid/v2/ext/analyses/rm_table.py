"""Table 12 of the manuscript: the rival margin applied to the equal-data baselines (medians from rm_test_medians.json,
margins selected on the development split by rm_dev.py). Writes tab_rivalbase.tex next to this script."""
import json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
M = json.load(open(HERE / 'rm_test_medians.json'))
rows = [('RF-eq|T', '\\smorffi forest, equal data'), ('MLP-eq|T', 'MLP, equal data'), ('CF-eq|T', 'CosFace prototypes, equal data'),
        ('RAPID|frozen', '\\textbf{\\rapid} (frozen)')]
mets = [('TAR', 'p1', True), ('pooled', 'p2', False), ('worst', 'p1', False), ('ge10', 'n', False), ('ge50', 'n', False),
        ('devchosen', 'p1', False), ('retry20', 'p1', False), ('oracle', 'p1', False)]
def f(k, v): return f'{100 * v:.1f}\\%' if k == 'p1' else (f'{100 * v:.2f}\\%' if k == 'p2' else f'{v:g}')
best = {m: (max if hi else min)(M[r][m] for r, _ in rows) for m, _, hi in mets}
lines = []
for r, lab in rows:
    cells = [('\\textbf{%s}' % f(k, M[r][m])) if abs(M[r][m] - best[m]) < 1e-12 else f(k, M[r][m]) for m, k, _ in mets]
    lines.append(f"{lab} & {M[r]['beta']:g} & " + ' & '.join(cells) + ' \\\\')
tab = r'''\begin{table*}[tp]
\caption{The rival margin applied to the equal-data baselines on \smorffi (medians over 6 runs). Each baseline accepts a claim for $t$ only if its own score of $t$ exceeds that of every other identity by $\beta$, with $\beta$ selected on the development split of \rapid's design so that its development TAR falls by no more than \rapid's does at $\beta=0.02$ (3.0 points). Forest and MLP: posterior; CosFace prototypes: cosine to one mean prototype per identity. Oracle: test-oracle attacker. Bold: best value.}
\label{tab:rivalbase}
\footnotesize
\setlength{\tabcolsep}{2.6pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.9cm}rrrrrrrrr@{}}
\toprule
System with rival margin & $\beta$ & TAR & Pooled & Worst & Pairs & Pairs & Dev.- & Retry & Oracle \\
 & & & FAR & pair & $\geq$10\% & $\geq$50\% & chosen & $k_r$=20 & \\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}
'''
(HERE / 'tab_rivalbase.tex').write_text(tab); print(tab)
