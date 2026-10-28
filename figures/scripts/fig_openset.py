"""Unenrolled attackers: 30 of 122 devices never enrolled (3 partitions x 2 role orders), medians."""
import pathlib as _pl
_ROOT = next(p for p in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents] if (p / 'paths.py').exists())
import json, numpy as np
from style import plt, W, save, SYS, INK2
D = json.load(open(str(_ROOT / 'rapid/v2/openset_summary.json')))['summary']
SYSTEMS = ['RF-eq', 'MLP-eq', 'RAPID']
MET = [('TAR', 'Enrolled genuine TAR (%)', 100), ('unenr_pooled', 'Unenrolled pooled FAR (%)', 100),
       ('unenr_ge50', 'Unenrolled pairs $\\geq$50% FAR', 1), ('enr_pooled', 'Enrolled-impostor FAR (%)', 100)]
fig, ax = plt.subplots(1, 4, figsize=(W, 2.3))
w = 0.26
for i, (m, lab, sc) in enumerate(MET):
    for j, s in enumerate(SYSTEMS):
        vals = [sc * D[f'{s}|{r}'][m] for r in ('ident', 'ident+abs')]
        _, c, ls, mk = SYS[s]
        bars = ax[i].bar(np.arange(2) + (j - 1) * w, vals, w * 0.92, color=c, label=SYS[s][0], hatch=None if s == 'RAPID' else ('//' if s == 'RF-eq' else '..'),
                         edgecolor='white', linewidth=0.4)
    ax[i].set_xticks([0, 1]); ax[i].set_xticklabels(['ident.', '+ abs.\ncheck']); ax[i].set_title(lab, fontsize=7.6)
    ax[i].grid(axis='x', visible=False)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6); save(fig, 'fig_openset.pdf')
