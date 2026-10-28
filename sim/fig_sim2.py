import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'rapid' / 'v2', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import FIG_DIR
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = str(FIG_DIR) + '/'
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 7.8, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False, 'lines.linewidth': 1.6,
    'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
W = 5.4
R = [r for r in json.load(open(str(_HERE / 'sim2_results.json'))) if not r.get('twin_channel')]
RI = [r for r in json.load(open(str(_HERE / 'sim2_results.json'))) if r.get('twin_channel')]
DEF = dict(n_dev=60, snr_db=20.0, spread=1.0, twins=0.05, twin_delta=0.05, drift_ppm=0.6)
SYS = [('RF', 'quantile', GRAY, ':', 's', 'Forest, published, quantile'), ('RF-eq', 'quantile', GRAY, '--', 'v', 'Forest, equal data, quantile'),
       ('RF-eq', 'ident', GRAY, '-', 'D', 'Forest, equal data, identification'), ('RAPID', 'ident', BLUE, '-', 'o', 'RAPID')]
def sel(var, val, L=4, base=DEF):
    want = dict(base); want[var] = val
    if var == 'twin_delta': want['twins'] = 0.1
    if var == 'twin_delta' and abs(val - 0.05) < 1e-12: want['twins'] = 0.1
    return [r for r in R if all(abs(r[k] - v) < 1e-9 for k, v in want.items()) and r['L'] == L]
def series(var, vals, s, rule, key, L=4):
    m = []
    for v in vals:
        rr = [r for r in sel(var, v, L) if r['system'] == s and r['rule'] == rule]
        m.append(np.mean([r[key] for r in rr]) if rr else np.nan)
    return np.array(m)
SW = [('n_dev', [20, 40, 60, 80, 120], 'Enrolled devices $N$'), ('snr_db', [5, 10, 15, 20, 25, 30], 'SNR (dB)'),
      ('spread', [0.5, 0.75, 1.0, 1.5, 2.0], 'Impairment spread (×)'), ('drift_ppm', [0.2, 0.6, 1.0, 1.5], 'Oscillator drift (ppm)')]
# Figure S3 of the paper (simulation sweeps) is drawn by figures/scripts/fig_sim.py.
# twins / separability and session length
fig, axs = plt.subplots(1, 3, figsize=(W, 2.2), gridspec_kw=dict(width_ratios=[1.45, 1, 1]))
dl = [0.0, 0.005, 0.02, 0.05, 0.2, 0.5]
for s, rule, c, ls, mk, lab in SYS:
    axs[0].plot(range(len(dl)), 100 * series('twin_delta', dl, s, rule, 'worst'), color=c, ls=ls, marker=mk, ms=3)
for s_, rule, c, ls, mk, lab in SYS:
    v = [r['worst'] for r in RI if r['system'] == s_ and r['rule'] == rule]
    axs[0].plot([-1.3], [100 * np.mean(v)], mk, color=c, ms=3.5, alpha=1 if ls == '-' else 0.6)
axs[0].axvline(-0.65, color=GRID, lw=1)
axs[0].set_xticks([-1.3] + list(range(len(dl)))); axs[0].set_xticklabels(['ch.', '0', '.005', '.02', '.05', '.2', '.5'], fontsize=6.2)
axs[0].set_xlabel('Twin separation (× spread)'); axs[0].set_ylabel('Worst-pair FAR (%)'); axs[0].set_title('10% twin devices', fontsize=7.4)
tw = [0.0, 0.05, 0.1, 0.2]
for s, rule, c, ls, mk, lab in SYS:
    axs[1].plot([100 * t for t in tw], 100 * series('twins', tw, s, rule, 'pooled'), color=c, ls=ls, marker=mk, ms=3)
axs[1].set_xlabel('Twin devices (%)'); axs[1].set_ylabel('Pooled FAR (%)'); axs[1].set_title('Twin fraction', fontsize=7.4)
Ls = [1, 2, 4, 8, 16]
for s, rule, c, ls, mk, lab in SYS:
    axs[2].plot(Ls, [100 * np.mean([r['devchosen'] for r in R if r['L'] == L and r['system'] == s and r['rule'] == rule and all(abs(r[k] - v) < 1e-9 for k, v in DEF.items())]) for L in Ls], color=c, ls=ls, marker=mk, ms=3, label=lab)
axs[2].set_xscale('log', base=2); axs[2].set_xticks(Ls); axs[2].set_xticklabels([str(L) for L in Ls]); axs[2].minorticks_off()
axs[2].set_xlabel('Frames per session $L$'); axs[2].set_ylabel('Dev.-chosen FAR (%)'); axs[2].set_title('Session length', fontsize=7.4)
h, l = axs[2].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, fontsize=6.4, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.6); fig.savefig(OUT + 'fig_sim_twins.pdf'); plt.close(fig)
# summary table numbers
rows = {}
cfgs = {'Default': ('n_dev', 60), '$N=120$': ('n_dev', 120), 'SNR 5\\,dB': ('snr_db', 5.0), 'Spread 0.5': ('spread', 0.5),
        'Drift 1.5\\,ppm': ('drift_ppm', 1.5), 'Twins 20\\%': ('twins', 0.2), 'Twin sep.\\ 0.02': ('twin_delta', 0.02)}
for name, (var, val) in cfgs.items():
    rr = sel(var, val); rows[name] = {}
    for s, rule, *_ in SYS:
        x = [r for r in rr if r['system'] == s and r['rule'] == rule]
        rows[name][f'{s}|{rule}'] = {k: float(np.mean([r[k] for r in x])) for k in ('TAR', 'pooled', 'worst', 'devchosen', 'p99', 'ge50', 'EER', 'acc', 'retry5', 'TAR@FAR0.001', 'fit_s', 'ms')} if x else None
json.dump(rows, open(str(_HERE / 'sim2_table.json'), 'w'), indent=1)
# win counts across all configurations (RAPID floor vs SMoRFFI quantile; RAPID ident vs SMoRFFI ident)
LOWER = {'pooled', 'worst', 'devchosen', 'p99', 'ge50', 'EER', 'retry5', 'retry20', 'ge10', 'eq100', 'multi', 'fit_s', 'ms'}
wins = {}
for a, b in ((('RAPID', 'ident'), ('RF-eq', 'ident')), (('RAPID', 'ident'), ('RF-eq', 'quantile')), (('RAPID', 'ident'), ('RF', 'quantile')), (('RAPID', 'ident'), ('RF', 'ident'))):
    keyset = {}
    A = {(json.dumps({k: r[k] for k in DEF}, sort_keys=True), r['data_seed'], r['order'], r['L']): r for r in R if r['system'] == a[0] and r['rule'] == a[1]}
    B = {(json.dumps({k: r[k] for k in DEF}, sort_keys=True), r['data_seed'], r['order'], r['L']): r for r in R if r['system'] == b[0] and r['rule'] == b[1]}
    common = [k for k in A if k in B]
    for k_ in ('TAR', 'eff50', 'acc', 'pooled', 'worst', 'p99', 'ge10', 'ge50', 'eq100', 'devchosen', 'retry5', 'retry20', 'multi', 'EER', 'AUC', 'TAR@FAR0.01', 'TAR@FAR0.001', 'fit_s', 'ms'):
        bt = sum((A[c][k_] < B[c][k_] - 1e-12) if k_ in LOWER else (A[c][k_] > B[c][k_] + 1e-12) for c in common)
        eq = sum(abs(A[c][k_] - B[c][k_]) <= 1e-12 for c in common); keyset[k_] = (bt, eq, len(common) - bt - eq)
    wins[f'{a} vs {b}'] = keyset; print(a, b, len(common), keyset)
json.dump(wins, open(str(_HERE / 'sim2_wins.json'), 'w'), indent=1)
print('ok')

# identical twins sharing the channel
rows2 = {}
for s_, rule, *_ in SYS:
    x = [r for r in RI if r['system'] == s_ and r['rule'] == rule]
    rows2[f'{s_}|{rule}'] = {k: float(np.mean([r[k] for r in x])) for k in ('TAR', 'pooled', 'worst', 'devchosen', 'EER', 'acc')} if x else None
tab = json.load(open(str(_HERE / 'sim2_table.json'))); tab['Identical twins'] = rows2; json.dump(tab, open(str(_HERE / 'sim2_table.json'), 'w'), indent=1)
