"""Figures from the post-freeze analyses of saved scores (ext/curves.json): retry curves and per-target attacker success,
margin sensitivity, impostor tail."""
import pathlib as _pl
_ROOT = next(p for p in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents] if (p / 'paths.py').exists())
import json, numpy as np
from style import plt, W, style, save, GRID, INK2, BLUE, SYS
C = json.load(open(str(_ROOT / 'rapid/v2/ext/curves.json')))
S = json.load(open(str(_ROOT / 'rapid/v2/final_summary.json')))['summary']
ORDER = ['RF', 'MLP', 'RF-eq', 'MLP-eq', 'RAPID']

# ---------------- retry curves + per-target development-chosen success
fig, ax = plt.subplots(1, 2, figsize=(W, 2.45), gridspec_kw=dict(width_ratios=[1.15, 1]))
k = np.arange(1, 21)
for s in ORDER:
    ax[0].plot(k, 100 * np.array(C['retry'][s]), marker=SYS[s][3], ms=3, markevery=[0, 4, 9, 19], **style(s))
ax[0].set_xlabel('Attempts per session, $k_r$'); ax[0].set_ylabel('Attacker success (%)'); ax[0].set_xticks([1, 5, 10, 15, 20])
ax[0].set_title('(a) Development-chosen attacker with retries')
for s in ORDER:
    v = np.sort(100 * np.array(C['target'][s]).ravel()); ax[1].plot(v, 1 - np.arange(len(v)) / len(v), **style(s))
ax[1].set_xlabel('Per-target attacker success, one attempt (%)'); ax[1].set_ylabel('Share of targets $\\geq x$')
ax[1].set_title('(b) Distribution over targets'); ax[1].set_ylim(0, 0.45); ax[1].set_xlim(2, 100)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.13))
fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, 'fig_retry.pdf')

# ---------------- margin sensitivity
B = C['betas']; M = C['beta_median']
base = {'TAR': ('max', 'TAR (%)'), 'devchosen': ('min', 'Dev.-chosen attacker (%)'), 'pooled': ('min', 'Pooled FAR (%)'),
        'retry20': ('min', 'Retry success, $k_r=20$ (%)'), 'worst': ('min', 'Worst-pair FAR (%)'), 'ge50': ('min', 'Pairs with FAR $\\geq$50%')}
cfg = ['RF|quantile', 'RF|ident', 'RF-eq|quantile', 'RF-eq|ident', 'MLP-eq|quantile', 'MLP-eq|ident']
fig, ax = plt.subplots(2, 3, figsize=(W, 3.6)); ax = ax.ravel()
for i, (m, (agg, lab)) in enumerate(base.items()):
    sc = 1 if m == 'ge50' else 100
    v = [sc * M[str(b)][m] for b in B]; bv = [S[c][m]['med'] for c in cfg]; best = sc * (max(bv) if agg == 'max' else min(bv))
    ax[i].axvspan(0, 0.03, color=BLUE, alpha=0.07, lw=0)
    ax[i].plot(B, v, marker='o', ms=3, **style('RAPID', label='RAPID'))
    ax[i].axhline(best, color=INK2, ls='--', lw=1.1, label='best baseline (any rule)')
    ax[i].axvline(0.02, color=BLUE, ls=':', lw=1)
    ax[i].set_title(lab, fontsize=8.3); ax[i].set_xlabel('Margin $\\beta$')
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, 'fig_beta.pdf')

# ---------------- impostor tail
g = 100 * np.array(C['tail_grid'])
fig, ax = plt.subplots(figsize=(W * 0.62, 2.4))
for s in ORDER:
    ax.step(g, np.maximum(np.array(C['tail'][s]), 0.5), where='post', **style(s))
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Pair FAR threshold $x$ (%)'); ax.set_ylabel('Pairs with FAR $\\geq x$ (median)')
ax.set_ylim(0.8, 600); ax.set_xlim(2.5, 100)
ax.legend(loc='lower left', fontsize=6.6); save(fig, 'fig_tail.pdf')
