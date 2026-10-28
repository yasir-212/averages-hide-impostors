"""Additional values for numbers.tex: run-level wins, retry curves, margin sweep, open-set, simulation counts,
extended baselines, sensitivity sweeps and cost benchmark (each block only if its result file exists)."""
import json, os, numpy as np
import pathlib as _pl
_ROOT = next(p for p in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents] if (p / 'paths.py').exists())
V2 = str(_ROOT / 'rapid' / 'v2'); EXT = f'{V2}/ext'; SIM = str(_ROOT / 'sim')
def pct(x, d=1): return f'{100 * x:.{d}f}\\%'
def rng(vals, f):
    lo, hi = min(vals), max(vals); a, b = f(lo), f(hi)
    return a if a == b else f"{a.replace(chr(92) + '%', '')}--{b}"

def add(V):
    # ---- run-level comparison of the main table
    W = json.load(open(f'{V2}/final_summary.json'))['wins']
    import sys; sys.path.insert(0, V2)
    S = json.load(open(f'{V2}/final_summary.json'))['summary']
    # counts as in gen_tables.run_wins: per metric, RAPID vs best of six baseline configurations, per run
    R = json.load(open(f'{V2}/final_results.json'))['runs']
    mets = ['TAR', 'eff50', 'acc', 'pooled', 'worst', 'p99', 'ge10', 'ge50', 'eq100', 'devchosen', 'retry5', 'retry20', 'multi',
            'TAR@FAR0.01', 'TAR@FAR0.001', 'EER', 'AUC']
    low = {'pooled', 'worst', 'p99', 'ge10', 'ge50', 'eq100', 'devchosen', 'retry5', 'retry20', 'multi', 'EER'}
    cfgs = [('RF', 'quantile'), ('RF', 'ident'), ('RF-eq', 'quantile'), ('RF-eq', 'ident'), ('MLP-eq', 'quantile'), ('MLP-eq', 'ident')]
    bt = tot = 0
    for seed in (0, 1, 2):
        for order in ('orig', 'swap'):
            rap = [r for r in R if r['system'] == 'RAPID' and r['seed'] == seed and r['order'] == order][0]
            bs = [r for r in R for (s, ru) in cfgs if r['system'] == s and r['rule'] == ru and r['seed'] == seed and r['order'] == order]
            for m in mets:
                vals = [b[m] for b in bs]; best = min(vals) if m in low else max(vals); a = rap[m]
                tot += 1; bt += (a <= best + 1e-12) if m in low else (a >= best - 1e-12)
    V['RUNS_BT'] = str(bt); V['RUNS_ALL'] = str(tot)
    # ---- retry curves, per-target success, margin sweep
    C = json.load(open(f'{EXT}/curves.json'))
    for s, r in C['retry'].items():
        for k in (1, 5, 10, 20): V[f'retry.{s}.k{k}'] = pct(r[k - 1])
    for s, g in C['target'].items():
        a = np.array(g).ravel(); V[f'target.{s}.ge25'] = pct((a >= 0.25 - 1e-12).mean()); V[f'target.{s}.ge50'] = pct((a >= 0.5 - 1e-12).mean())
    for b, r in C['beta_median'].items():
        for m in ('TAR', 'devchosen', 'retry20', 'worst', 'ge50'): V[f'beta.{float(b):g}.{m}'] = pct(r[m]) if m != 'ge50' else f"{r[m]:g}"
        V[f'beta.{float(b):g}.pooled'] = pct(r['pooled'], 2)
    # ---- open set
    O = json.load(open(f'{V2}/openset_summary.json'))['summary']
    for k, r in O.items():
        for m, v in r.items():
            if m == 'n': continue
            V[f'os.{k}.{m}'] = (f'{v:g}' if m in ('eff50', 'unenr_ge50', 'enr_ge50') else pct(v, 2 if 'pooled' in m else 1))
    for rule in ('ident', 'ident+abs'):
        for m in ('TAR', 'unenr_pooled', 'unenr_devchosen', 'unenr_retry20', 'unenr_worst'):
            vals = [O[f'{s}|{rule}'][m] for s in ('RF-eq', 'MLP-eq')]
            V[f'os.base|{rule}.{m}'] = rng(vals, (lambda x: pct(x, 2)) if 'pooled' in m else pct)
    # ---- simulation: RAPID vs equal-data forest (identification) per paired comparison
    R2 = [r for r in json.load(open(f'{SIM}/sim2_results.json')) if not r.get('twin_channel')]; keyf = lambda r: (json.dumps({k: r[k] for k in ('n_dev', 'snr_db', 'spread', 'twins', 'twin_delta', 'drift_ppm')}, sort_keys=True), bool(r.get('twin_channel')), r['data_seed'], r['order'], r['L'])
    A = {keyf(r): r for r in R2 if r['system'] == 'RAPID'}; B = {keyf(r): r for r in R2 if r['system'] == 'RF-eq' and r['rule'] == 'ident'}
    ks = [k for k in A if k in B]; V['SIM_N'] = str(len(ks))
    for m in ('acc', 'TAR', 'pooled', 'worst', 'devchosen', 'EER', 'AUC'):
        hi = m in ('acc', 'TAR', 'AUC'); V[f'SIM.{m}'] = str(sum((A[k][m] >= B[k][m] - 1e-12) if hi else (A[k][m] <= B[k][m] + 1e-12) for k in ks))

def add2(V):
    d = json.load(open(f'{EXT}/dev_choice.json'))
    V['KNN_K'] = str(d['knn_k'])
    g = d['gbt']
    V['GBT_CFG'] = (f"histogram gradient boosting, learning rate {g['learning_rate']:g}, {g['max_iter']} iterations, at most {g['max_leaf_nodes']} leaves, "
                    f"at least {g['min_samples_leaf']} samples per leaf, $\\ell_2={g['l2_regularization']:g}$ (dev.\\ from 3 configurations)")

def add3(V):
    if not os.path.exists(f'{EXT}/stats.json'): return
    ST = json.load(open(f'{EXT}/stats.json'))
    sg = lambda v: (f'+{v:.1f}' if v >= 0 else f'$-${-v:.1f}')
    for k, r in ST.items():
        V[f'stat.{k}'] = f"{sg(100 * r['mean'])} points (95\\% CI {sg(100 * r['lo'])} to {sg(100 * r['hi'])})"
        V[f'stat.{k}.better'] = str(r['better']); V[f'stat.{k}.worse'] = str(r['worse'])

def add4(V):
    p = f'{EXT}/sweeps.json'
    if os.path.exists(p):
        SW = json.load(open(p)); tags = sorted({r['tag'] for r in SW})
        for t in tags:
            for s in ('RAPID', 'RF-eq', 'MLP-eq'):
                rr = [r for r in SW if r['tag'] == t and r['system'] == s]
                if len(rr) < 2: continue
                for m in ('TAR', 'devchosen', 'pooled', 'retry20', 'worst', 'acc'):
                    v = float(np.mean([r[m] for r in rr])); V[f'sw.{t}.{s}.{m}'] = pct(v, 2 if m in ('pooled', 'EER') else 1)
    p = f'{EXT}/ext_results.json'
    if os.path.exists(p):
        E = json.load(open(p))['rows']
        for s in sorted({r['system'] for r in E}):
            for L in (1, 2, 4, 8, 16):
                rr = [r for r in E if r['system'] == s and r['L'] == L]
                if not rr: continue
                for m in ('TAR', 'acc', 'pooled', 'worst', 'ge50', 'devchosen', 'retry20', 'TAR@FAR0.001', 'EER', 'AUC', 'p99', 'ge10', 'eff50'):
                    v = float(np.median([r[m] for r in rr]))
                    V[f'ext.{s}.L{L}.{m}'] = (f'{v:.4f}' if m == 'AUC' else f'{v:g}' if m in ('ge50', 'ge10', 'eff50') else pct(v, 2 if m in ('pooled', 'EER') else 1))
                V[f'ext.{s}.L{L}.n'] = str(len(rr))
    p = f'{EXT}/cost_bench.json'
    if os.path.exists(p):
        CB = json.load(open(p)); V['cost.host'] = CB['host'].strip().lstrip(':').strip().replace('(R)', '')
        for r in CB['rows']:
            if r['system'] == 'RF-eq': r['size_MB'] = json.load(open(f'{V2}/final_summary.json'))['costs']['RF-eq']['size_MB']
        for r in CB['rows']:
            s = r['system']; V[f'cost.{s}.size'] = ((f"{r['size_MB']:,.0f}".replace(',', '{,}') + '\\,MB') if r['size_MB'] >= 100 else f"{r['size_MB']:.1f}\\,MB" if r['size_MB'] >= 1 else f"{r['size_MB']:.2f}\\,MB")
            V[f'cost.{s}.fit'] = f"{r['fit_s']:.0f}\\,s"; V[f'cost.{s}.lat'] = f"{r['ms_single']:.2f}\\,ms"; V[f'cost.{s}.bat'] = f"{r['ms_batched']:.3f}\\,ms"
        R = {r['system']: r for r in CB['rows']}
        if 'RAPID' in R and 'RF-eq' in R:
            V['cost.ratio.size'] = f"{R['RF-eq']['size_MB'] / R['RAPID']['size_MB']:.0f}"; V['cost.ratio.lat'] = f"{R['RF-eq']['ms_single'] / R['RAPID']['ms_single']:.1f}"
            V['cost.ratio.bat'] = f"{R['RAPID']['ms_batched'] / R['RF-eq']['ms_batched']:.1f}"

def add5(V):
    p = f'{EXT}/sweeps.json'
    if not os.path.exists(p): return
    SW = json.load(open(p))
    for t in ('pop30', 'pop60', 'pop90'):
        g = {}
        for s in ('RAPID', 'RF-eq', 'MLP-eq'):
            rr = [r for r in SW if r['tag'] == t and r['system'] == s]
            g[s] = {m: float(np.mean([r[m] for r in rr])) for m in ('TAR', 'devchosen', 'pooled')}
        V[f'gap.{t}.TAR'] = f"{100 * (g['RAPID']['TAR'] - max(g['RF-eq']['TAR'], g['MLP-eq']['TAR'])):.1f}"
        V[f'gap.{t}.devchosen'] = f"{100 * (min(g['RF-eq']['devchosen'], g['MLP-eq']['devchosen']) - g['RAPID']['devchosen']):.1f}"
    # full population (122 identities, frozen budget): the frozen-comparison runs of seed 0, mean of both role orders, as in the sweeps
    R = json.load(open(f'{V2}/final_results.json'))['runs']
    g = {s: {m: float(np.mean([r[m] for r in R if r['system'] == s and r['rule'] == 'ident' and r['seed'] == 0])) for m in ('TAR', 'devchosen')} for s in ('RAPID', 'RF-eq', 'MLP-eq')}
    V['gap.pop122.TAR'] = f"{100 * (g['RAPID']['TAR'] - max(g['RF-eq']['TAR'], g['MLP-eq']['TAR'])):.1f}"
    V['gap.pop122.devchosen'] = f"{100 * (min(g['RF-eq']['devchosen'], g['MLP-eq']['devchosen']) - g['RAPID']['devchosen']):.1f}"

def add6(V):
    p = f'{EXT}/ext_results.json'
    if os.path.exists(p):
        ch = json.load(open(p))['checks']
        mx = max(c['max_abs_diff'] for c in ch if c['system'] == 'RAPID'); e = int(np.floor(np.log10(mx)))
        V['REPRO.maxdiff'] = f'${mx / 10**e:.1f}\\times10^{{{e}}}$'
        V['REPRO.nchecks'] = str(len(ch)); V['REPRO.baseexact'] = 'yes' if all(c['max_abs_diff'] == 0 for c in ch if c['system'] != 'RAPID') else 'no'

def add7(V):
    R = json.load(open(f'{V2}/final_results.json'))['runs']
    for o in ('orig', 'swap'):
        for s in ('RAPID', 'RF-eq', 'MLP-eq'):
            rr = [r for r in R if r['system'] == s and r['rule'] == 'ident' and r['order'] == o]
            for m in ('TAR', 'pooled', 'devchosen', 'retry20', 'EER'):
                V[f'ord.{o}.{s}.{m}'] = pct(float(np.median([r[m] for r in rr])), 2 if m in ('pooled', 'EER') else 1)
    p = f'{EXT}/attacker2.json'
    if os.path.exists(p):
        A = json.load(open(p))
        for s, d in A.items():
            V[f'att2.{s}.devchosen'] = pct(d['median']['devchosen_score']); V[f'att2.{s}.retry20'] = pct(d['median']['retry20_score'])
            V[f'att2.{s}.zero'] = f"{d['median']['zero_dev_targets']:g}"

def add8(V):
    # trained models, counting every ensemble member (see the reproducibility appendix for the breakdown)
    main = 6 * 9 + 3 + 3          # frozen comparison: RAPID (3 members), RF-eq and MLP-eq (1 + 2 cross-fits each); published MLP and forest per seed
    ext = 6 * 8                   # replication with RAPID (3), CosFace, forest, MLP, kNN, GBT
    opn = 6 * 9                   # unenrolled-attacker study
    sweep = 14 * 5 + 9            # budget and population sweeps; component ablation
    sim = 52 * 13 + 8 * 3         # simulation (published forest + 2 orders x (RAPID 3 + forest 3)); low-SNR margin study
    V['WORK.main'] = str(main + ext); V['WORK.open'] = str(opn); V['WORK.sweep'] = str(sweep); V['WORK.sim'] = str(sim)
    V['WORK.models'] = f'{main + ext + opn + sweep + sim:,}'.replace(',', '{,}')
    V['WORK.hours'] = 'eight'
