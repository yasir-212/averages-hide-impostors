"""Figures from the extended evaluation: all-baseline metric bars, session-length sweep, enrolment budget and population size,
and the host-cost comparison."""
import pathlib as _pl
_ROOT = next(p for p in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents] if (p / 'paths.py').exists())
import json, os, numpy as np
from style import plt, W, style, save, SYS, INK2, GRID, BLUE
EXT = str(_ROOT / 'rapid' / 'v2' / 'ext')
E = json.load(open(f'{EXT}/ext_results.json'))['rows']
MAIN = json.load(open(str(_ROOT / 'rapid/v2/final_results.json')))['runs']

def med(name, m, L=4):
    if name in ('RF', 'MLP'):
        v = [r[m] for r in MAIN if r['system'] == name and r['rule'] == 'ident']
    else:
        v = [r[m] for r in E if r['system'] == name and r['L'] == L]
    return float(np.median(v)) if v else np.nan

# ---------------- metric bars, all baselines (identification rule; RAPID frozen rule)
ORDER = ['RF', 'MLP', 'RF-eq', 'MLP-eq', 'kNN-eq', 'GBT-eq', 'CFtrain', 'CFeq', 'RAPID']
ORDER = [s for s in ORDER if s in ('RF', 'MLP') or any(r['system'] == s for r in E)]
MET = [('acc', 'Accuracy (%)', 100, True), ('pooled', 'Pooled FAR (%)', 100, False), ('devchosen', 'Dev.-chosen (%)', 100, False),
       ('retry20', 'Retry, $k_r$=20 (%)', 100, False), ('ge50', 'Pairs $\\geq$50% FAR', 1, False),
       ('ge10', 'Pairs $\\geq$10% FAR', 1, False), ('TAR@FAR0.001', 'TAR@0.1% FAR (%)', 100, True), ('EER', 'EER (%)', 100, False)]
fig, ax = plt.subplots(2, 4, figsize=(W, 5.2), sharey=True); ax = ax.ravel()
yy = np.arange(len(ORDER))[::-1]
for i, (m, lab, sc, hi) in enumerate(MET):
    vals = np.array([sc * med(s, m) for s in ORDER])
    for j, s in enumerate(ORDER):
        _, c, ls, mk = SYS[s]
        ax[i].barh(yy[j], vals[j], 0.72, color=c, alpha=1.0 if s == 'RAPID' else (0.55 if ls == ':' else 0.85),
                   hatch='//' if ls == ':' else None, edgecolor='white', linewidth=0.4)
        ax[i].text(vals[j], yy[j], (f' {vals[j]:.2f}' if m in ('pooled', 'EER') else f' {vals[j]:.1f}' if sc == 100 else f' {vals[j]:g}'),
                   va='center', fontsize=5.8, color=INK2)
    ax[i].set_title(lab + (' $\\uparrow$' if hi else ' $\\downarrow$'), fontsize=7.6); ax[i].grid(axis='y', visible=False)
    ax[i].set_xlim(0, vals[np.isfinite(vals)].max() * 1.32); ax[i].tick_params(axis='x', labelsize=6.5)
ax[0].set_yticks(yy); ax[0].set_yticklabels([SYS[s][0].replace(', ', ',\n') for s in ORDER], fontsize=6.4)
ax[4].set_yticks(yy); ax[4].set_yticklabels([SYS[s][0].replace(', ', ',\n') for s in ORDER], fontsize=6.4)
fig.tight_layout(w_pad=0.3, h_pad=0.6); save(fig, 'fig_bars.pdf')

# ---------------- session length
Ls = [1, 2, 4, 8, 16]; SL = [s for s in ['RF-eq', 'MLP-eq', 'kNN-eq', 'GBT-eq', 'CFtrain', 'CFeq', 'RAPID'] if any(r['system'] == s for r in E)]
KL = [('TAR@FAR0.001', 'TAR at 0.1% FAR (%)'), ('devchosen', 'Dev.-chosen attacker (%)'), ('pooled', 'Pooled FAR (%)'), ('EER', 'EER (%)')]
fig, ax = plt.subplots(1, 4, figsize=(W, 2.3))
for i, (m, lab) in enumerate(KL):
    for s in SL:
        ax[i].plot(Ls, [100 * med(s, m, L) for L in Ls], marker=SYS[s][3], ms=3, lw=1.4, **style(s))
    ax[i].set_xscale('log', base=2); ax[i].set_xticks(Ls); ax[i].set_xticklabels([str(L) for L in Ls]); ax[i].minorticks_off()
    ax[i].set_xlabel('Frames per session $L$'); ax[i].set_title(lab, fontsize=7.8)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, fontsize=6.6, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.4); save(fig, 'fig_length.pdf')

# ---------------- enrolment budget and population size
if os.path.exists(f'{EXT}/sweeps.json'):
    SW = json.load(open(f'{EXT}/sweeps.json'))
    main0 = {s: [r for r in MAIN if r['system'] == s and r['seed'] == 0 and r['rule'] == 'ident'] for s in ('RAPID', 'RF-eq', 'MLP-eq')}
    def val(tag, s, m):
        if tag in ('budget125', 'pop122'): rr = main0[s]
        else: rr = [r for r in SW if r['tag'] == tag and r['system'] == s]
        return 100 * np.mean([r[m] for r in rr]) if rr else np.nan
    fig, ax = plt.subplots(2, 3, figsize=(W, 3.9))
    for row, (tags, xs, xl) in enumerate(((['budget25', 'budget50', 'budget75', 'budget100', 'budget125'], [25, 50, 75, 100, 125], 'Recent enrolment frames per identity'),
                                         (['pop30', 'pop60', 'pop90', 'pop122'], [30, 60, 90, 122], 'Enrolled identities $N$'))):
        for i, (m, lab) in enumerate((('TAR', 'TAR (%)'), ('devchosen', 'Dev.-chosen attacker (%)'), ('pooled', 'Pooled FAR (%)'))):
            for s in ('RF-eq', 'MLP-eq', 'RAPID'):
                ax[row, i].plot(xs, [val(t, s, m) for t in tags], marker=SYS[s][3], ms=3.5, **style(s))
            ax[row, i].set_xlabel(xl, fontsize=7.6); ax[row, i].set_title(lab, fontsize=7.8); ax[row, i].set_xticks(xs)
    h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.8); save(fig, 'fig_budget_pop.pdf')

# ---------------- cost
if os.path.exists(f'{EXT}/cost_bench.json'):
    CB = {r['system']: r for r in json.load(open(f'{EXT}/cost_bench.json'))['rows']}
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5))
    for s, r in CB.items():
        dc = med(s, 'devchosen')
        _, c, ls, mk = SYS[s]
        ax[0].scatter(r['size_MB'], 100 * dc, s=34, color=c, marker=mk, zorder=3, label=SYS[s][0], edgecolor='white', linewidth=0.5)
        ax[1].scatter(r['ms_single'], 100 * dc, s=34, color=c, marker=mk, zorder=3, edgecolor='white', linewidth=0.5)
    for a, xl in zip(ax, ('Model size (MB, log scale)', 'Single-session latency (ms, log scale)')):
        a.set_xscale('log'); a.set_xlabel(xl); a.set_ylabel('Dev.-chosen attacker (%)')
    ax[0].set_title('(a) Security vs. memory'); ax[1].set_title('(b) Security vs. latency')
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, fontsize=6.4, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.1, 1, 1)); save(fig, 'fig_cost.pdf')
