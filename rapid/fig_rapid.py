import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import json, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, str(_R / 'rapid'))
from evalkit import rule_thresholds, accept
from rapid_core import rival
OUT = str(FIG_DIR) + '/'; _os.makedirs(OUT + 'extra', exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False, 'lines.linewidth': 1.8,
    'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
W = 5.4
D = json.load(open('protocol_rapid.json')); CV = D['curves']; R = D['runs']; S = json.load(open('rapid_summary.json'))
BL, RP = 'SMoRFFI-RF128', 'RAPID'
LAB = {BL: 'SMoRFFI framework', RP: 'RAPID (ours)'}; COL = {BL: GRAY, RP: BLUE}
def runs(s, rule): return [r for r in R if r['system'] == s and r['rule'] == rule]
def med(s, rule, k): return np.median([r[k] for r in runs(s, rule)])

# ---------- F1 operating frontier
GRIDT = np.round(np.arange(0.40, 0.871, 0.01), 3)
def fr(sys_, key, seed, o):
    pts = [p for k, c in CV.items() if k.startswith(sys_ + '|') and k.endswith(f'|{seed}|{o}') for p in c]
    return np.array([min([p[key] for p in pts if p['TAR'] >= g - 1e-12], default=np.nan) for g in GRIDT])
fig, axs = plt.subplots(1, 3, figsize=(W, 2.05))
for ax, key, lab, sc in zip(axs, ('pooled', 'devchosen', 'worst'), ('(a) Pooled FAR (%)', '(b) Dev.-chosen attacker (%)', '(c) Worst-pair FAR (%)'), (100, 100, 100)):
    for s_ in (BL, RP):
        M = np.array([fr(s_, key, sd, o) for sd in (0, 1, 2) for o in ('orig', 'swap')]) * sc
        m = np.nanmedian(M, 0); ax.plot(100 * GRIDT, m, color=COL[s_], label=LAB[s_]); ax.fill_between(100 * GRIDT, np.nanmin(M, 0), np.nanmax(M, 0), color=COL[s_], alpha=0.18, lw=0)
    for s_, rule, mk in ((BL, 'quantile', 's'), (BL, 'ident', 'D'), (RP, 'floor', 'o'), (RP, 'ident', 'D')):
        ax.plot(100 * med(s_, rule, 'TAR'), sc * med(s_, rule, key), mk, color=COL[s_], ms=5, mec='white', mew=0.6, zorder=5)
    ax.set_xlabel('TAR (%)'); ax.set_title(lab, fontsize=7.6); ax.set_xlim(40, 87)
axs[0].set_ylim(0, 1.0); axs[2].set_ylim(85, 101)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.6); fig.savefig(OUT + 'extra/v1_fig_rapid_frontier.pdf'); plt.close(fig)

# ---------- pair matrices from saved scores (seed 0, calibration = B, attacker development = A)
def load(s):
    z = np.load(f'scores_{s}_s0.npz'); return {k: z[k] for k in ('A', 'B', 'test')}, {k: z['y_' + k] for k in ('A', 'B', 'test')}
SB, YB = load(BL); SR, YR = load(RP); n = SB['test'].shape[1]
MB = {k: rival(v, 1.0) for k, v in SB.items()}
cfg = {'SMoRFFI, quantile rule': (SB, 'quantile', GRAY, '-'), 'SMoRFFI, identification rule': (MB, 'ident', GRAY, '--'),
       'RAPID, quantile rule': (SR, 'quantile', BLUE, '--'), 'RAPID (floor rule)': (SR, 'floor', BLUE, '-')}
mats = {}
for name, (Sx, rule, c, ls) in cfg.items():
    th = rule_thresholds(rule, Sx['B'], YB['B'], n, 0.9); tar, P, acc = accept(Sx['test'], YB['test'], n, th)
    _, Pd, _ = accept(Sx['A'], YB['A'], n, th); sel = np.nanargmax(np.where(np.isfinite(Pd), Pd, -1), 0)
    mats[name] = dict(tar=tar, P=P, acc=acc, sel=sel)
# F2 CCDF of pair FAR
fig, ax = plt.subplots(figsize=(W, 2.25)); xs = np.linspace(0, 1, 38)[1:]
for name, (Sx, rule, c, ls) in cfg.items():
    f = mats[name]['P'][np.isfinite(mats[name]['P'])]; ys = [(f >= x - 1e-9).sum() for x in xs]
    ax.step(xs * 100, ys, where='post', color=c, ls=ls, label=f"{name} (TAR {100 * mats[name]['tar'].mean():.1f}%)")
ax.set_yscale('log'); ax.set_ylim(0.8, 3000); ax.set_xlim(0, 101)
ax.set_xlabel('Per-pair false acceptance $x$ (%)'); ax.set_ylabel('Ordered pairs with FAR $\\geq x$'); ax.legend(loc='upper right', fontsize=6.6)
fig.tight_layout(); fig.savefig(OUT + 'extra/fig_rapid_pairs.pdf'); plt.close(fig)
# F3 retry curves
ks = [1, 2, 3, 5, 8, 10, 15, 20]; fig, ax = plt.subplots(figsize=(W, 2.15)); retry_out = {}
for name, (Sx, rule, c, ls) in cfg.items():
    acc, sel = mats[name]['acc'], mats[name]['sel']; ys = []
    for k in ks:
        v = []
        for t in range(n):
            a = acc[YB['test'] == sel[t], t]; nb = len(a) // k; v.append(np.mean([a[b * k:(b + 1) * k].any() for b in range(nb)]))
        ys.append(100 * np.mean(v))
    retry_out[name] = ys; ax.plot(ks, ys, color=c, ls=ls, marker='o', ms=3, label=name)
ax.set_xscale('log'); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.minorticks_off()
ax.set_xlabel('Consecutive claim attempts $k$ per session'); ax.set_ylabel('Development-chosen\nattacker success (%)'); ax.set_ylim(0, 70); ax.legend(fontsize=6.6, loc='upper left')
fig.tight_layout(); fig.savefig(OUT + 'extra/fig_rapid_retry.pdf'); plt.close(fig)
# F4 multi-claim acceptance of impostor windows
fig, ax = plt.subplots(figsize=(W, 2.0)); cats = ['0', '1', '2', '3', '4+']; wbar = 0.2; multi_out = {}
for i, (name, (Sx, rule, c, ls)) in enumerate(cfg.items()):
    acc = mats[name]['acc'].copy(); y = YB['test']; acc[np.arange(len(y)), y] = False; k = acc.sum(1)
    h = [np.mean(k == 0), np.mean(k == 1), np.mean(k == 2), np.mean(k == 3), np.mean(k >= 4)]; multi_out[name] = h
    ax.bar(np.arange(5) + (i - 1.5) * wbar, 100 * np.array(h), wbar, color=c, alpha=1 if ls == '-' else 0.45, label=name, edgecolor='white', lw=0.4)
ax.set_xticks(np.arange(5)); ax.set_xticklabels(cats); ax.set_xlabel('Other identities that accept the same test session'); ax.set_ylabel('Sessions (%)'); ax.legend(fontsize=6.4, ncol=2)
fig.tight_layout(); fig.savefig(OUT + 'extra/fig_rapid_multiclaim.pdf'); plt.close(fig)
# F5 per-target development-chosen attacker (quantile rule both = matched rule; and RAPID floor)
fig, axs = plt.subplots(1, 2, figsize=(W, 2.15), sharey=True)
for ax, rn in zip(axs, ('RAPID, quantile rule', 'RAPID (floor rule)')):
    b = mats['SMoRFFI, quantile rule']; r = mats[rn]
    xb = 100 * b['P'][b['sel'], np.arange(n)]; xr = 100 * r['P'][r['sel'], np.arange(n)]
    ax.scatter(xb, xr, s=9, color=BLUE, alpha=0.65, lw=0); ax.plot([0, 100], [0, 100], color=GRAY, lw=0.8, ls='--')
    lo, hi = (xr < xb - 1e-9).sum(), (xr > xb + 1e-9).sum()
    ax.set_title(f'{rn}\nlower on {lo} targets, higher on {hi}', fontsize=7.0); ax.set_xlabel('SMoRFFI framework (%)'); ax.set_xlim(-3, 103); ax.set_ylim(-3, 103)
axs[0].set_ylabel('RAPID (%)'); fig.tight_layout(); fig.savefig(OUT + 'extra/fig_rapid_target.pdf'); plt.close(fig)
# F6 global ROC
def roc(Sx, y):
    G = Sx[np.arange(len(y)), y]; M = np.ones_like(Sx, bool); M[np.arange(len(y)), y] = False; I = np.sort(Sx[M])
    fars = np.logspace(-4, -0.5, 60); tars = [(G > I[int(np.ceil((1 - f) * len(I))) - 1]).mean() for f in fars]; return fars, np.array(tars)
fig, ax = plt.subplots(figsize=(W, 2.1))
for Sx, c, lab in ((SB, GRAY, LAB[BL]), (SR, BLUE, LAB[RP])):
    f, t = roc(Sx['test'], YB['test']); ax.plot(100 * f, 100 * t, color=c, label=lab)
ax.set_xscale('log'); ax.set_xlabel('Pooled false acceptance with one global threshold (%, log scale)'); ax.set_ylabel('TAR (%)'); ax.set_ylim(30, 100); ax.legend(loc='lower right')
fig.tight_layout(); fig.savefig(OUT + 'extra/fig_rapid_roc.pdf'); plt.close(fig)
# F7 cost
C = json.load(open('ablate_cost.json'))['cost']; rl = json.load(open('rf_latency.json'))
names = ['SMoRFFI\nframework', 'RAPID', 'RAPID-Lite']; keys = ['SMoRFFI-RF128', 'RAPID', 'RAPID-Lite']
size = [C[k]['bytes'] / 2**20 for k in keys]; lat = [min(rl['1'][0], rl['2'][0]), C['RAPID']['single'][0], C['RAPID-Lite']['single'][0]]
fit = [S['costs'][k]['fit_s'] for k in keys]
fig, axs = plt.subplots(1, 3, figsize=(W, 1.95))
for ax, v, t, lg in zip(axs, (size, lat, fit), ('Model size (MB)', 'Latency per session (ms)', 'Training time (s)'), (True, True, False)):
    ax.bar(range(3), v, color=[GRAY, BLUE, AQUA], width=0.62)
    for i, x in enumerate(v): ax.text(i, x * (1.25 if lg else 1.03), (f'{x:,.0f}' if x >= 100 else f'{x:.3g}'), ha='center', fontsize=6.8, color=INK)
    if lg: ax.set_yscale('log')
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=6.6); ax.set_title(t, fontsize=7.8); ax.grid(axis='x', visible=False)
axs[0].set_ylim(0.1, 2e4); axs[1].set_ylim(0.1, 60); axs[2].set_ylim(0, 70)
fig.tight_layout(w_pad=0.4); fig.savefig(OUT + 'extra/fig_rapid_cost.pdf'); plt.close(fig)
# F8 paired runs for performance metrics
fig, axs = plt.subplots(1, 4, figsize=(W, 1.95))
for ax, (k, lab) in zip(axs, (('TAR', 'TAR (%)'), ('acc', 'Identification\naccuracy (%)'), ('TAR@FAR0.01', 'TAR at\nFAR = 1% (%)'), ('TAR@FAR0.001', 'TAR at\nFAR = 0.1% (%)'))):
    for sd in (0, 1, 2):
        for o in ('orig', 'swap'):
            b = [r for r in runs(BL, 'quantile') if r['seed'] == sd and r['order'] == o][0][k] * 100
            r_ = [r for r in runs(RP, 'floor') if r['seed'] == sd and r['order'] == o][0][k] * 100
            ax.plot([0, 1], [b, r_], color=GRAY, lw=0.8, alpha=0.7); ax.plot([0], [b], 'o', color=GRAY, ms=3.5); ax.plot([1], [r_], 'o', color=BLUE, ms=3.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['SMoRFFI', 'RAPID'], fontsize=6.8); ax.set_xlim(-0.35, 1.35); ax.set_title(lab, fontsize=7.4); ax.grid(axis='x', visible=False)
fig.tight_layout(w_pad=0.3); fig.savefig(OUT + 'extra/fig_rapid_runs.pdf'); plt.close(fig)
json.dump(dict(retry=retry_out, multi=multi_out, ks=ks, tar={k: float(v['tar'].mean()) for k, v in mats.items()},
               devch_per_target={k: [float(x) for x in v['P'][v['sel'], np.arange(n)]] for k, v in mats.items()}), open('fig_rapid_numbers.json', 'w'), indent=1)
print('figures done')

# ---------- combined compact figures for the page budget
fig, axs = plt.subplots(1, 2, figsize=(W, 2.05), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axs[0]
for name, (Sx, rule, c, ls) in cfg.items():
    f = mats[name]['P'][np.isfinite(mats[name]['P'])]; ys = [(f >= x - 1e-9).sum() for x in xs]
    ax.step(xs * 100, ys, where='post', color=c, ls=ls, label=f"{name} ({100 * mats[name]['tar'].mean():.1f}%)")
ax.set_yscale('log'); ax.set_ylim(0.8, 3000); ax.set_xlim(0, 101); ax.set_xlabel('Per-pair FAR $x$ (%)'); ax.set_ylabel('Pairs with FAR $\\geq x$')
ax.set_title('(a) Impostor tail (TAR in brackets)', fontsize=7.4)
ax = axs[1]
for name, (Sx, rule, c, ls) in cfg.items():
    ax.plot(ks, retry_out[name], color=c, ls=ls, marker='o', ms=2.5)
ax.set_xscale('log'); ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(['1', '2', '5', '10', '20']); ax.minorticks_off()
ax.set_xlabel('Attempts $k$ per session'); ax.set_ylabel('Dev.-chosen success (%)'); ax.set_ylim(0, 60); ax.set_title('(b) Retries', fontsize=7.4)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, fontsize=6.6, handlelength=2.2, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.11, 1, 1), w_pad=0.8); fig.savefig(OUT + 'extra/v1_fig_rapid_tail.pdf'); plt.close(fig)
fig, axs = plt.subplots(1, 3, figsize=(W, 1.75), gridspec_kw=dict(width_ratios=[1.6, 0.8, 0.8]))
ax = axs[0]
for Sx, c, lab in ((SB, GRAY, 'SMoRFFI framework'), (SR, BLUE, 'RAPID')):
    f, t = roc(Sx['test'], YB['test']); ax.plot(100 * f, 100 * t, color=c, label=lab)
ax.set_xscale('log'); ax.set_xlabel('Pooled FAR, one global threshold (%)'); ax.set_ylabel('TAR (%)'); ax.set_ylim(30, 100); ax.legend(loc='lower right', fontsize=6.4)
ax.set_title('(a) ROC, seed 0', fontsize=7.4)
for ax, (k, lab) in zip(axs[1:], (('TAR', '(b) TAR, own rule'), ('TAR@FAR0.001', '(c) TAR at FAR 0.1%'))):
    for sd in (0, 1, 2):
        for o in ('orig', 'swap'):
            b = [r for r in runs(BL, 'quantile') if r['seed'] == sd and r['order'] == o][0][k] * 100
            r_ = [r for r in runs(RP, 'floor') if r['seed'] == sd and r['order'] == o][0][k] * 100
            ax.plot([0, 1], [b, r_], color=GRAY, lw=0.8, alpha=0.7); ax.plot([0], [b], 'o', color=GRAY, ms=3); ax.plot([1], [r_], 'o', color=BLUE, ms=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['SMoRFFI', 'RAPID'], fontsize=6.5); ax.set_xlim(-0.35, 1.35); ax.set_title(lab, fontsize=7.2); ax.grid(axis='x', visible=False)
fig.tight_layout(w_pad=0.5); fig.savefig(OUT + 'extra/v1_fig_rapid_scores.pdf'); plt.close(fig)
print('combined figures done')
