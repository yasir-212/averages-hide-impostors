"""Figures for RAPID v2 (frozen protocol on SMoRFFI): ROC + run-by-run panels, matched-operating-point frontiers, impostor tail."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import FIG_DIR, PAPER_TEX
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import load_frames
from evalkit import windows, accept
from core2 import margin
OUT = str(FIG_DIR) + '/'; _os.makedirs(OUT + 'extra', exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.3, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'legend.frameon': False, 'lines.linewidth': 1.8, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
W5 = 5.4; n = 122
STY = {'RAPID': dict(color=BLUE, ls='-', label='RAPID'), 'RF-eq': dict(color=GRAY, ls='-', label='SMoRFFI forest, equal data'),
       'MLP-eq': dict(color=ORANGE, ls='-', label='MLP, equal data'), 'RF': dict(color=GRAY, ls=':', label='SMoRFFI forest, published'),
       'MLP': dict(color=ORANGE, ls=':', label='MLP, training data only')}
x, y, row, _ = load_frames(); Y = {k: windows(x, y, row, n, *v)[1] for k, v in {'A': (600, 725), 'B': (725, 850), 'test': (850, 999)}.items()}
R = json.load(open('final_results.json'))['runs']

def load(name, seed, order):
    if name == 'RF': d = np.load(str(_R / 'rapid' / f'scores_SMoRFFI-RF128_s{seed}.npz')); return {k: d[k] for k in ('A', 'B', 'test')}
    if name == 'MLP': d = np.load(f'scores/MLP_s{seed}.npz'); return {k: d[k] for k in d.files}
    d = np.load(f'scores/{name}_s{seed}_{order}.npz'); return {k: d[k] for k in d.files}

def native(name, S):
    return margin(S) if name == 'RAPID' else S

def roc(S, y):
    G = S[np.arange(len(y)), y]; M = np.ones_like(S, bool); M[np.arange(len(y)), y] = False; I = np.sort(S[M])[::-1]
    fars = np.logspace(-4, -1, 60); tars = [(G > I[max(int(np.ceil(f * len(I))) - 1, 0)]).mean() for f in fars]; return fars, np.array(tars)

def runs(s, r, key):
    d = {(v['seed'], v['order']): v[key] for v in R if v['system'] == s and v['rule'] == r}; return [d[k] for k in sorted(d)]

# ---------------- Figure: ROC (seed 0, orig) + run-by-run TAR (own rule) + TAR at FAR 0.1%
fig, ax = plt.subplots(1, 3, figsize=(W5, 2.35), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
for name in ('RF', 'RF-eq', 'MLP-eq', 'RAPID'):
    S = load(name, 0, 'orig'); f, t = roc(native(name, S['test']), Y['test']); ax[0].plot(100 * f, 100 * t, **STY[name])
ax[0].set_xscale('log'); ax[0].set_xlabel('Pooled FAR, one global threshold (%)'); ax[0].set_ylabel('TAR (%)'); ax[0].set_title('(a) ROC, seed 0', loc='left')
h0, l0 = ax[0].get_legend_handles_labels()
for j, (key, title) in enumerate((('TAR', '(b) TAR, own rule'), ('TAR@FAR0.001', '(c) TAR at FAR 0.1%'))):
    a = ax[j + 1]; cols = [('RF-eq', 'ident'), ('MLP-eq', 'ident'), ('RAPID', 'ident')]
    V = np.array([runs(s, r, key) for s, r in cols]) * 100
    for i in range(V.shape[1]): a.plot(range(3), V[:, i], color=GRID, lw=0.9, zorder=1)
    for c, (s, r) in enumerate(cols): a.scatter(np.full(V.shape[1], c), V[c], s=12, color=STY[s]['color'], zorder=2)
    a.set_xticks(range(3)); a.set_xticklabels(['Forest', 'MLP', 'RAPID'], fontsize=7.2); a.set_title(title, loc='left')
fig.legend(h0, l0, loc='lower center', ncol=4, fontsize=6.6, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.6); fig.savefig(OUT + 'fig_rapid_scores.pdf'); plt.close(fig)

# ---------------- Figure: matched operating points (frontier2.json)
F = json.load(open('frontier2.json')); G = sorted({float(g) for v in F.values() for g in v})
fig, ax = plt.subplots(1, 3, figsize=(W5, 2.15))
for name in ('RF', 'RF-eq', 'MLP-eq', 'RAPID'):
    keys = [k for k in F if k.split('|')[0] == name]
    for j, (idx, title, scale) in enumerate(((0, '(a) Pooled FAR', 100), (4, '(b) Dev.-chosen attacker', 100), (1, '(c) Worst pair', 100))):
        M = np.array([[F[k][str(g)][idx] if F[k].get(str(g)) is not None else np.nan for g in G] for k in keys]) * scale
        med = np.nanmedian(M, 0); ax[j].plot(100 * np.array(G), med, **{**STY[name], 'label': STY[name]['label']})
        if name == 'RAPID': ax[j].fill_between(100 * np.array(G), np.nanmin(M, 0), np.nanmax(M, 0), color=BLUE, alpha=0.12, lw=0)
        ax[j].set_title(title, loc='left'); ax[j].set_xlabel('TAR (%)')
ax[0].set_yscale('log'); ax[0].set_ylabel('FAR (%)')
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, fontsize=6.6, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.6); fig.savefig(OUT + 'fig_rapid_frontier.pdf'); plt.close(fig)

# ---------------- Figure: impostor tail and retries (seed 0, orig: calibration/enrolment block B, attacker development block A)
fig, ax = plt.subplots(1, 2, figsize=(W5, 2.15))
xs = np.linspace(0, 1, 101)
for name, rule in (('RF', 'quantile'), ('RF-eq', 'ident'), ('MLP-eq', 'ident'), ('RAPID', 'ident')):
    S = load(name, 0, 'orig'); th = np.full(n, 1e-12 + (0.02 if name == 'RAPID' else 0))
    if rule == 'quantile':
        from evalkit import thr; Sd = S; th = thr(S['B'], Y['B'], n, 0.9)
    else:
        Sd = {k: margin(v) for k, v in S.items()} if name != 'RAPID' else {k: margin(v) for k, v in S.items()}
    tar, P, acc = accept(Sd['test'], Y['test'], n, th); f = P[np.isfinite(P)]
    lab = f"{STY[name]['label']}{' (quantile)' if rule == 'quantile' else ''} ({100*tar.mean():.1f}%)"
    ax[0].step(100 * xs, [(f >= v - 1e-12).sum() for v in xs], where='post', color=STY[name]['color'], ls=STY[name]['ls'], label=lab)
    _, Pd, _ = accept(Sd['A'], Y['A'], n, th); sel = np.nanargmax(np.where(np.isfinite(Pd), Pd, -1), 0); ks = [1, 2, 5, 10, 20]; vals = []
    for k in ks:
        v = []
        for t in range(n):
            a = acc[Y['test'] == sel[t], t]; nb = len(a) // k
            if nb: v.append(np.mean([a[b * k:(b + 1) * k].any() for b in range(nb)]))
        vals.append(100 * np.mean(v))
    ax[1].plot(ks, vals, marker='o', ms=3, color=STY[name]['color'], ls=STY[name]['ls'])
ax[0].set_yscale('symlog', linthresh=1); ax[0].set_xlabel('Per-pair FAR x (%)'); ax[0].set_ylabel('Pairs with FAR ≥ x'); ax[0].set_title('(a) Impostor tail (TAR in brackets)', loc='left')
ax[1].set_xscale('log'); ax[1].set_xticks([1, 2, 5, 10, 20]); ax[1].set_xticklabels(['1', '2', '5', '10', '20']); ax[1].minorticks_off()
ax[1].set_xlabel('Attempts k per session'); ax[1].set_ylabel('Dev.-chosen success (%)'); ax[1].set_title('(b) Retries', loc='left')
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, fontsize=6.6, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.0); fig.savefig(OUT + 'extra/fig_rapid_tail.pdf'); plt.close(fig)
print('figures done')
