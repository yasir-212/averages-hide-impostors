"""Generates the result tables of the paper:
tables/tab_ext.tex (all baselines, medians over 6 runs), tables/tab_ek.tex (ensemble size and exemplar count),
tables/tab_cost.tex (host-cost benchmark)."""
import pathlib as _pl
_ROOT = next(p for p in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents] if (p / 'paths.py').exists())
import json, os, numpy as np
EXT = str(_ROOT / 'rapid' / 'v2' / 'ext'); V2 = str(_ROOT / 'rapid' / 'v2'); HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
E = json.load(open(f'{EXT}/ext_results.json'))['rows']; MAIN = json.load(open(f'{V2}/final_results.json'))['runs']
C = json.load(open(f'{EXT}/curves.json'))
LOW = {'pooled', 'worst', 'ge10', 'ge50', 'devchosen', 'retry20', 'EER', 'multi'}

def med(name, m, L=4):
    if name in ('RF', 'MLP'): v = [r[m] for r in MAIN if r['system'] == name and r['rule'] == 'ident']
    elif name == 'RAPID-b0': return C['beta_median']['0.0'][m] if m in C['beta_median']['0.0'] else med('RAPID', m)
    else: v = [r[m] for r in E if r['system'] == name and r['L'] == L]
    return float(np.median(v))

def fmt(m, v):
    if m in ('pooled', 'EER'): return f'{100 * v:.2f}\\%'
    if m == 'AUC': return f'{v:.4f}'
    if m in ('ge50', 'ge10', 'eff50'): return f'{v:g}'
    return f'{100 * v:.1f}\\%'

def table(rows, mets, heads, label, caption, bold_rows=None, size='\\footnotesize', sep='2.6pt', first=3.6):
    vals = {(k, m): med(k, m) for k, _ in rows for m in mets}
    cand = bold_rows or [k for k, _ in rows]
    best = {m: (min if m in LOW else max)(vals[(k, m)] for k in cand) for m in mets}
    L = ['\\begin{table*}[tp]', f'\\caption{{{caption}}}', f'\\label{{{label}}}', size, f'\\setlength{{\\tabcolsep}}{{{sep}}}',
         f'\\begin{{tabular}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{{first}cm}}' + 'r' * len(mets) + '@{}}', '\\toprule',
         'System & ' + ' & '.join(h[0] for h in heads) + ' \\\\', ' & ' + ' & '.join(h[1] for h in heads) + ' \\\\', '\\midrule']
    for k, lab in rows:
        if k == '---': L.append('\\midrule'); continue
        cells = []
        for m in mets:
            s = fmt(m, vals[(k, m)])
            if k in cand and abs(vals[(k, m)] - best[m]) < 1e-12: s = f'\\textbf{{{s}}}'
            cells.append(s)
        L.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
    return '\n'.join(L) + '\n'

def main():
    have = {r['system'] for r in E}
    rows = [('RF', '\\smorffi forest, published'), ('MLP', 'MLP, published'), ('RF-eq', '\\smorffi forest, equal data'), ('MLP-eq', 'MLP, equal data'),
            ('kNN-eq', 'kNN, equal data'), ('GBT-eq', 'GBT, equal data'), ('CFtrain', 'CosFace prototypes, training data'),
            ('CFeq', 'CosFace prototypes, equal data'), ('RAPID', '\\textbf{\\rapid} ($\\beta=0.02$, frozen)'), ('RAPID-b0', '\\rapid, $\\beta=0$')]
    rows = [r for r in rows if r[0] in have or r[0] in ('RF', 'MLP', 'RAPID-b0')]
    mets = ['acc', 'pooled', 'ge10', 'ge50', 'devchosen', 'retry20', 'TAR@FAR0.001', 'EER', 'AUC']
    heads = [('Accu-', 'racy'), ('Pooled', 'FAR'), ('Pairs', '$\\geq$10\\%'), ('Pairs', '$\\geq$50\\%'), ('Dev.-', 'chosen'),
             ('Retry', '$k_r$=20'), ('TAR@', '0.1\\%'), ('EER', ''), ('AUC', '')]
    cap = ('All baselines on \\smorffi under the frozen protocol (medians over 6 runs; identification rule for every baseline, the rule with '
           'the lower single-attempt attacker success for the forest and the MLP in \\cref{tab:main}). TAR@0.1\\%, EER and AUC use one global threshold on each system\'s native score. '
           'The last row is \\rapid with $\\beta=0$, its identification variant, shown for comparison at higher genuine acceptance. '
           'Bold: best value among all rows.')
    open(f'{HERE}/tables/tab_ext.tex', 'w').write(table(rows, mets, heads, 'tab:ext', cap, sep='2.2pt', first=3.7))
    # ensemble size and exemplars
    rk = [('RAPID-E1', '$E=1$, $k=3$'), ('RAPID-E2', '$E=2$, $k=3$'), ('RAPID', '$E=3$, $k=3$ (frozen)'), ('RAPID-k1', '$E=3$, $k=1$'),
          ('RAPID-k5', '$E=3$, $k=5$'), ('RAPID-k10', '$E=3$, $k=10$')]
    mk = ['TAR', 'pooled', 'worst', 'devchosen', 'retry20', 'TAR@FAR0.001', 'EER', 'AUC']
    hk = [('TAR', ''), ('Pooled', 'FAR'), ('Worst', 'pair'), ('Dev.-', 'chosen'), ('Retry', '$k_r$=20'), ('TAR@', '0.1\\%'), ('EER', ''), ('AUC', '')]
    ck = ('Sensitivity of \\rapid to the number of ensemble members $E$ and of nearest exemplars $k$ on \\smorffi (medians over 6 runs, '
          '$\\beta=0.02$; post-freeze analysis on the same trained members). Every variant outperforms every baseline of \\cref{tab:ext} '
          'on the development-chosen attacker, retry success, TAR@0.1\\% and EER.')
    open(f'{HERE}/tables/tab_ek.tex', 'w').write(table(rk, mk, hk, 'tab:ek', ck, sep='3pt', first=3.0))
    # cost
    if os.path.exists(f'{EXT}/cost_bench.json'):
        CB = json.load(open(f'{EXT}/cost_bench.json')); R = {r['system']: r for r in CB['rows']}
        R['RF-eq']['size_MB'] = json.load(open(f'{V2}/final_summary.json'))['costs']['RF-eq']['size_MB']   # median over the six runs
        host = CB['host'].strip().lstrip(':').strip().replace('(R)', '')
        names = [('RF-eq', '\\smorffi forest, equal data'), ('MLP-eq', 'MLP, equal data'), ('kNN-eq', 'kNN, equal data'), ('GBT-eq', 'GBT, equal data'),
                 ('CFtrain', 'CosFace prototypes, training data'), ('RAPID', '\\rapid')]
        L = ['\\begin{table*}[tp]', '\\caption{Cost of every system measured on one two-core host (%s; seed 0): model size (1\\,MB $=2^{20}$ bytes) and the median latency '
             'of a single verification with one thread (400 calls; 200 for the forest). '
             'Model size of the forest: median over the six runs; the equal-data CosFace prototype verifier has the same cost as the one shown. Security: development-chosen attacker (median over 6 runs).}' % host,
             '\\label{tab:cost}', '\\footnotesize', '\\setlength{\\tabcolsep}{4pt}', '\\begin{tabular}{@{}lrrr@{}}', '\\toprule',
             'System & Model size & Latency & Dev.-chosen \\\\', '\\midrule']
        for k, lab in names:
            if k not in R: continue
            r = R[k]; sz = r['size_MB']
            szs = (f'{sz:,.0f}'.replace(',', '{,}') + '\\,MB') if sz >= 100 else (f'{sz:.1f}\\,MB' if sz >= 1 else f'{sz:.2f}\\,MB')
            L.append(f"{lab} & {szs} & {r['ms_single']:.2f}\\,ms & {fmt('devchosen', med(k, 'devchosen'))} \\\\")
        L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
        open(f'{HERE}/tables/tab_cost.tex', 'w').write('\n'.join(L) + '\n')

    # significance
    ST = json.load(open(f'{EXT}/stats.json'))
    def pv(p):
        if p < 1e-3:
            e = int(np.floor(np.log10(p))); return f'$<10^{{{e + 1}}}$'
        return f'{p:.3f}'
    L = ['\\begin{table*}[tp]', '\\caption{Paired per-target comparison of \\rapid with the equal-data baselines on \\smorffi (122 targets, 6 runs each). '
         'Difference: mean over targets of the per-target metric of \\rapid minus that of the baseline, averaged over runs, with a 95\\% cluster-bootstrap '
         'interval over targets (10{,}000 resamples). Better/worse: targets for which \\rapid is better/worse on average over the runs; $p$: exact two-sided sign test.}',
         '\\label{tab:stats}', '\\footnotesize', '\\setlength{\\tabcolsep}{4pt}', '\\begin{tabular}{@{}llrrrr@{}}', '\\toprule',
         'Baseline & Per-target metric & Difference [95\\% CI] & Better & Worse & $p$ \\\\', '\\midrule']
    labs = {'TAR': 'genuine acceptance (pp)', 'devchosen': 'dev.-chosen attacker (pp)', 'worst_per_target': 'worst impostor FAR (pp)'}
    for b, bl in (('RF-eq', '\\smorffi forest, equal data'), ('MLP-eq', 'MLP, equal data')):
        for i, m in enumerate(('TAR', 'devchosen', 'worst_per_target')):
            r = ST[f'{b}.{m}']
            sg = lambda v: (f'+{v:.1f}' if v >= 0 else f'$-${-v:.1f}')
            L.append(f"{bl if i == 0 else ''} & {labs[m]} & {sg(100 * r['mean'])} [{sg(100 * r['lo'])}, {sg(100 * r['hi'])}] & {r['better']} & {r['worse']} & {pv(r['sign_p'])} \\\\")
        if b == 'RF-eq': L.append('\\midrule')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
    open(f'{HERE}/tables/tab_stats.tex', 'w').write('\n'.join(L) + '\n')
    print('tables written')

if __name__ == '__main__':
    main()
