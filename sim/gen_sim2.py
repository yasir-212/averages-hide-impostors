"""LaTeX table and text values for the simulation (RAPID v2 vs the forest)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'rapid' / 'v2', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import FIG_DIR
import json, numpy as np
T = json.load(open(str(_HERE / 'sim2_table.json'))); W = json.load(open(str(_HERE / 'sim2_wins.json')))
order = ['Default', '$N=120$', 'Spread 0.5', 'Drift 1.5\\,ppm', 'Twins 20\\%']   # configurations shown in Table S8
S3 = ['RF|quantile', 'RF-eq|ident', 'RAPID|ident']
def cell(row, key, d=1, scale=100):
    vals = [row[s][key] * scale for s in S3]; hi = key in ('TAR', 'acc'); b = max(vals) if hi else min(vals)
    return ' / '.join(('\\textbf{%.*f}' % (d, v)) if abs(v - b) < 1e-9 else '%.*f' % (d, v) for v in vals)
lines = []
for name in order:
    r = T[name]; lab = name
    lines.append(f"{lab} & {cell(r, 'TAR')} & {cell(r, 'worst', 0)} & {cell(r, 'devchosen')} & {cell(r, 'acc')} & {cell(r, 'EER', 2)} \\\\")
tab = r'''\begin{table}[t]
\caption{Simulated configurations (mean over 2 populations $\times$ 2 role orders). Each cell lists the published forest (quantile rule) / the equal-data forest (identification rule) / RAPID. $^\dagger$10\% of devices are exact copies of another device, including its channel.}
\label{tab:sim}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
Configuration & TAR (\%) & Worst pair (\%) & Dev.-chosen (\%) & Accuracy (\%) & EER (\%) \\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}
\end{table}'''
open(str(_HERE / 'tab_sim.tex'), 'w').write(tab); print(tab)
for k, v in W.items(): print(k, v)
