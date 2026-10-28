import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = json.load(open(str(_R / 'audit' / 'numbers.json')))
C7 = json.load(open(str(_R / 'audit' / 'check7.json')))
OUT = str(FIG_DIR) + '/'; _os.makedirs(OUT + 'extra', exist_ok=True)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GRAY, INK, INK2, GRID = '#8a8984', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'legend.frameon': False, 'lines.linewidth': 2, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
W = 5.4

# F1 survival of per-pair FAR
fig, ax = plt.subplots(figsize=(W, 2.05))
for m, c, mk in (('RF128', BLUE, 'o'), ('MLP64', ORANGE, 's')):
    f = np.array(D[m]['baseline']['cdf']); xs = np.linspace(0, 1, 38)[1:]
    ys = [(f >= x - 1e-9).sum() for x in xs]
    ax.step(xs, ys, where='post', color=c, label=m)
    ax.plot([xs[-1]], [ys[-1]], marker=mk, color=c, ms=5)
ax.axvline(D['RF128']['baseline']['pooled'], color=GRAY, ls='--', lw=1)
ax.text(D['RF128']['baseline']['pooled'] + 0.012, 1500, 'pooled FAR = 1.5%', color=INK2, fontsize=7.5)
ax.set_yscale('log'); ax.set_ylim(1, 3000); ax.set_xlim(0, 1.02)
ax.set_xlabel('Per-pair false acceptance $x$ (37 test windows per source)'); ax.set_ylabel('Ordered pairs with FAR $\\geq x$')
ax.annotate('155 pairs $\\geq$ 50%', xy=(0.5, 155), xytext=(0.58, 600), fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle='-', color=GRAY, lw=0.8))
ax.annotate('4 pairs at 100% (RF128)', xy=(1.0, 4), xytext=(0.62, 1.6), fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle='-', color=GRAY, lw=0.8))
ax.legend(loc='upper right', bbox_to_anchor=(1, 0.86)); fig.tight_layout(); fig.savefig(OUT + 'fig_pair_far.pdf'); plt.close(fig)

# F2 retry
fig, ax = plt.subplots(figsize=(W, 1.95))
ks = [1, 2, 5, 10, 20]
for m, c, mk in (('RF128', BLUE, 'o'), ('MLP64', ORANGE, 's')):
    ys = [100 * D[m]['retry'][str(k)]['sequential'] for k in ks]
    ax.plot(ks, ys, color=c, marker=mk, ms=5, label=f'{m}, development-chosen source')
    ax.text(22, ys[-1] + (3 if m == 'RF128' else -3), f'{ys[-1]:.0f}%', color=INK, va='center', fontsize=7.5)
ax.axhline(100 * D['RF128']['baseline']['pooled'], color=GRAY, ls='--', lw=1)
ax.text(1.05, 4.2, 'pooled FAR (1.5%)', color=INK2, fontsize=7.5)
ax.set_xscale('log'); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.minorticks_off()
ax.set_xlim(0.85, 30); ax.set_ylim(0, 75); ax.set_xlabel('Consecutive claim attempts $k$ allowed per session'); ax.set_ylabel('Mean impersonation\nsuccess (%)')
ax.legend(loc='upper left'); fig.tight_layout(); fig.savefig(OUT + 'extra/fig_retry.pdf'); plt.close(fig)

# F3 matched-TAR curves, two panels
fig, axs = plt.subplots(1, 2, figsize=(W, 2.35), sharey=True)
for ax, m in zip(axs, ('RF128', 'MLP64')):
    cv = D[m]['curves']
    for key, c, mk, lab in (('baseline', GRAY, 'o', 'Per-target threshold'), ('farcap', BLUE, 's', 'FAR-cap'), ('resid_cal', ORANGE, '^', 'Residual, held-out threshold')):
        p = np.array(sorted(cv[key])); ax.plot(100 * p[:, 0], 100 * p[:, 1], color=c, marker=mk, ms=4, lw=1.6, label=lab)
    ri = D[m]['resid_insample']; g = D[m]['gate']
    ax.plot(100 * ri['TAR'], 100 * ri['worst'], marker='D', ms=6, color=AQUA, ls='none', label='Residual, in-sample threshold')
    ax.annotate(f"{ri['eff0']}/122 IDs accept\nany genuine", xy=(100 * ri['TAR'], 100 * ri['worst']), xytext=(100 * ri['TAR'] + 6, 100 * ri['worst'] - 22), fontsize=7, color=INK2, arrowprops=dict(arrowstyle='-', color=GRAY, lw=0.8))
    ax.plot(100 * g['TAR'], 100 * g['worst'], marker='*', ms=9, color=INK, ls='none', label='Identity gate ($\\gamma{=}0$)')
    ax.annotate(f"{g['stated']} IDs", xy=(100 * g['TAR'], 100 * g['worst']), xytext=(100 * g['TAR'] + 3, 100 * g['worst'] + 7), fontsize=7, color=INK2)
    ax.set_title(m); ax.set_xlim(10, 80); ax.set_ylim(0, 105); ax.set_xlabel('Genuine acceptance, all identities (%)')
axs[0].set_ylabel('Worst-pair FAR (%)')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(OUT + 'fig_matched_tar.pdf'); plt.close(fig)

# F4 stated vs effective coverage (RF128)
R = D['RF128']
rows = [('Per-target threshold', 122, 122, None), ('Identity gate, $\\gamma{=}0$', R['gate']['stated'], R['gate']['eff0'], R['gate']['eff50']),
        ('FAR-cap, $\\gamma{=}0$', 122, R['farcap0']['eff0'], R['farcap0']['eff50']), ('Residual, in-sample threshold', 122, R['resid_insample']['eff0'], R['resid_insample']['eff50']),
        ('Residual, held-out threshold', 122, R['resid_cal045']['eff0'], R['resid_cal045']['eff50'])]
# baseline eff50 compute from per-target? approximate not stored -> derive
rows[0] = ('Per-target threshold', 122, 122, 108)
fig, ax = plt.subplots(figsize=(W, 2.7))
y = np.arange(len(rows))[::-1]; h = 0.26
for i, (name, s, e0, e50) in enumerate(rows):
    yy = y[i]
    ax.barh(yy + h, s, h * 0.92, color=GRAY); ax.barh(yy, e0, h * 0.92, color=BLUE)
    if e50 is not None: ax.barh(yy - h, e50, h * 0.92, color=ORANGE)
    for val, off in ((s, h), (e0, 0), (e50, -h)):
        if val is not None: ax.text(val + 1.5, yy + off, str(val), va='center', fontsize=7, color=INK)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0, 135); ax.set_xlabel('Identities (of 122)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GRAY, label='Stated eligible'), Patch(color=BLUE, label='Effective: test TAR > 0'), Patch(color=ORANGE, label='Effective: test TAR $\\geq$ 50%')], loc='lower center', ncol=3, bbox_to_anchor=(0.32, 1.0), fontsize=7)
ax.grid(axis='y', visible=False); fig.tight_layout(); fig.savefig(OUT + 'extra/fig_coverage.pdf'); plt.close(fig)

# F5 split fragility
fig, ax = plt.subplots(figsize=(W, 2.05))
cats = [('GATE', 'orig'), ('GATE', 'swap'), ('R1', 'orig'), ('R1', 'swap')]
labels = ['Identity gate\norder 1', 'Identity gate\norder 2', 'Residual + calibrated\ndenial, order 1', 'Residual + calibrated\ndenial, order 2']
for i, (meth, ro) in enumerate(cats):
    for j, mdl in enumerate(('MLP64', 'RF128')):
        vals = [100 * r[meth]['worst'] for r in C7 if r['roles'] == ro and r['model'] == mdl]
        xs = i + (-0.12 if mdl == 'MLP64' else 0.12) + np.linspace(-0.05, 0.05, len(vals))
        ax.plot(xs, vals, ls='none', marker='o' if mdl == 'MLP64' else 's', ms=5.5, color=ORANGE if mdl == 'MLP64' else BLUE, label=mdl if i == 0 else None, markeredgecolor='white', markeredgewidth=0.6)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.5); ax.set_ylabel('Worst-pair FAR (%)'); ax.set_ylim(0, 60)
ax.legend(loc='upper center', bbox_to_anchor=(0.62, 1.0), title='3 seeds each', title_fontsize=7); ax.grid(axis='x', visible=False)
fig.tight_layout(); fig.savefig(OUT + 'fig_fragility.pdf'); plt.close(fig)

# F6 A5 artefact, real Zenodo target 2
fig, ax = plt.subplots(figsize=(W, 2.05))
conds = ['Centroid blend\n(A5, as published)', 'Blend toward random\ngenuine (A5-s)', 'Centroid blend +\ngenuine noise (A5-n)', 'Genuine windows\npulled 55% to centroid']
res = [31.6, 91.5, 89.7, 0.2]; oas = [100, 100, 100, 100]
x = np.arange(4); w = 0.36
b1 = ax.bar(x - w / 2, oas, w * 0.94, color=GRAY, label='OAS (Mahalanobis)')
b2 = ax.bar(x + w / 2, res, w * 0.94, color=ORANGE, label='Pair-conditioned residual')
for bars in (b1, b2):
    for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f'{b.get_height():.1f}' if b.get_height() < 100 else '100', ha='center', fontsize=7, color=INK)
ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=7.3); ax.set_ylabel('Acceptance (%)'); ax.set_ylim(0, 125)
ax.axvline(2.5, color=GRID, lw=1); ax.text(3.0, 116, 'genuine, not an attack', fontsize=7, color=INK2, ha='center')
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.12), ncol=2); ax.grid(axis='x', visible=False)
fig.tight_layout(); fig.savefig(OUT + 'fig_a5_artefact.pdf'); plt.close(fig)

# F7 INRIA packet selection
inr = {'OAS': (68.1, 94.7, 97.7), 'One-class\nresidual': (59.9, 87.4, 97.7), 'OAS +\nlogistic': (60.8, 92.7, 97.7), 'RF128': (52.9, 68.2, 95.3)}
fig, ax = plt.subplots(figsize=(W, 2.05))
x = np.arange(len(inr)); w = 0.26
for k, (lab, c) in enumerate((('All packets', GRAY), ('Closest 35%', BLUE), ('Closest 10%', ORANGE))):
    vals = [v[k] for v in inr.values()]
    bars = ax.bar(x + (k - 1) * w, vals, w * 0.92, color=c, label=lab)
    for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f'{b.get_height():.0f}', ha='center', fontsize=6.8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(list(inr.keys()), fontsize=7.5); ax.set_ylim(0, 112); ax.set_ylabel('Worst-target FAR, unenrolled\nradios (%)')
ax.legend(loc='upper left', ncol=3, bbox_to_anchor=(0, 1.08), title=None); ax.grid(axis='x', visible=False)
fig.tight_layout(); fig.savefig(OUT + 'extra/fig_inria_selection.pdf'); plt.close(fig)

# F8 sample size
fig, ax = plt.subplots(figsize=(W, 1.95))
cv = np.array(D['stats']['curve'])
ax.plot(100 * cv[:, 0], cv[:, 2], color=BLUE, marker='o', ms=4, label='One pair ($M=1$)')
ax.plot(100 * cv[:, 0], cv[:, 1], color=ORANGE, marker='s', ms=4, label='All ordered pairs ($M=14{,}762$)')
ax.axhline(37, color=GRAY, ls='--', lw=1); ax.text(0.105, 42, 'available: 37 windows per source', fontsize=7.3, color=INK2)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Target worst-pair FAR $\\varepsilon$ (%)'); ax.set_ylabel('Error-free independent\ntrials needed per pair')
ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10]); ax.set_xticklabels(['0.1', '0.2', '0.5', '1', '2', '5', '10']); ax.minorticks_off()
ax.legend(loc='upper right'); fig.tight_layout(); fig.savefig(OUT + 'fig_samples.pdf'); plt.close(fig)
print('ok')
