"""Simulation sweeps (SYNTHETIC 802.11g populations): mean over 2 populations x 2 role orders per configuration."""
import pathlib as _pl
_ROOT = next(p for p in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents] if (p / 'paths.py').exists())
import json, numpy as np
from style import plt, W, save, GRAY, BLUE, INK2
R = [r for r in json.load(open(str(_ROOT / 'sim/sim2_results.json'))) if not r.get('twin_channel')]
DEF = dict(n_dev=60, snr_db=20.0, spread=1.0, twins=0.05, twin_delta=0.05, drift_ppm=0.6)
SYS = [('RF', 'quantile', GRAY, ':', 's', 'SMoRFFI forest, published (quantile)'), ('RF-eq', 'ident', GRAY, '-', 'D', 'SMoRFFI forest, equal data (ident.)'),
       ('RAPID', 'ident', BLUE, '-', 'o', 'RAPID (ours)')]
def series(var, vals, s, rule, key):
    out = []
    for v in vals:
        want = dict(DEF); want[var] = v
        rr = [r for r in R if all(abs(r[k] - w) < 1e-9 for k, w in want.items()) and r['L'] == 4 and r['system'] == s and r['rule'] == rule]
        out.append(np.mean([r[key] for r in rr]) if rr else np.nan)
    return np.array(out)
SW = [('n_dev', [20, 40, 60, 80, 120], 'Enrolled devices $N$'), ('snr_db', [5, 10, 15, 20, 25, 30], 'SNR (dB)'),
      ('spread', [0.5, 0.75, 1.0, 1.5, 2.0], 'Impairment spread (x)'), ('drift_ppm', [0.2, 0.6, 1.0, 1.5], 'Oscillator drift (ppm)')]
KEYS = [('acc', 'Identification\naccuracy (%)'), ('devchosen', 'Dev.-chosen\nattacker (%)'), ('EER', 'EER (%)'), ('worst', 'Worst-pair\nFAR (%)')]
fig, axs = plt.subplots(len(KEYS), 4, figsize=(W, 5.3), sharex='col')
for j, (var, vals, xl) in enumerate(SW):
    for i, (key, yl) in enumerate(KEYS):
        ax = axs[i, j]
        for s, rule, c, ls, mk, lab in SYS:
            ax.plot(vals, 100 * series(var, vals, s, rule, key), color=c, ls=ls, marker=mk, ms=3, label=lab, lw=1.5)
        if key in ('devchosen', 'EER'): ax.set_yscale('symlog', linthresh=0.1); ax.set_ylim(bottom=0)
        if j == 0: ax.set_ylabel(yl, fontsize=7.8)
        if i == len(KEYS) - 1: ax.set_xlabel(xl)
        if var == 'n_dev': ax.set_xticks([20, 60, 120])
h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.4, w_pad=0.4); save(fig, 'fig_sim_sweeps.pdf')
