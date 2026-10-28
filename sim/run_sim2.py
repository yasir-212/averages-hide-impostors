"""SYNTHETIC sensitivity study with RAPID v2 (simulated 802.11g preambles; not measured data).
Every configuration: draw a population, simulate captures, extract SMoRFFI-style features, then evaluate RAPID v2 and the
baselines under the same protocol as on SMoRFFI (2 role orders; the calibration block is the recent enrolment data).
Systems: RAPID (equal data, local prototypes, identification rule), RF-eq (retrained on the same frames; identification and
cross-fitted quantile rules), RF (published framework, training frames only; quantile and identification rules)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'rapid' / 'v2', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import FIG_DIR
import sys, os, json, time, gc, numpy as np
import wifi_sim as WS
from evalkit import windows, blocks_for, full_metrics, global_roc, thr
from core2 import margin
from final_runner import RAPID2, make_base, post, halves, sess_fold, CFG, latency
from run_sim import DEFAULT, SWEEPS, NF
from sklearn.preprocessing import StandardScaler

def run_config(cfg, data_seed, model_seed=0, Ls=(4,), systems=('RAPID', 'RF-eq', 'RF')):
    t0 = time.time(); kw = dict(cfg); n = kw.pop('n_dev')
    x, y, row = WS.dataset(n, NF, seed=data_seed, **kw); bl = blocks_for(NF); res = []
    tr = (row >= bl['train'][0]) & (row < bl['train'][1]); ftr, ytr = x[tr], y[tr]
    for L in Ls:
        W = {k: windows(x, y, row, n, *bl[k], L=L) for k in ('A', 'B', 'test')}; Y = {k: W[k][1] for k in W}
        def ev(name, rule, th, S, cal, att, native, order, extra):
            r = full_metrics(th, S[att], Y[att], S['test'], Y['test'], n); r.update(global_roc(native, Y['test']))
            r.update(system=name, rule=rule, order=order, L=L, data_seed=data_seed, **extra, **cfg); res.append(r)
        if 'RF' in systems:      # published framework: training frames only
            sc = StandardScaler().fit(ftr); t1 = time.time(); m = make_base('RF', model_seed).fit(sc.transform(ftr), ytr); ft = time.time() - t1
            P = {k: post(m, sc, W[k][0], n) for k in W}; lat = latency(lambda w: post(m, sc, w, n), W['test'][0][:300]); del m; gc.collect()
            for order, (cal, att) in (('orig', ('B', 'A')), ('swap', ('A', 'B'))):
                ev('RF', 'quantile', thr(P[cal], Y[cal], n, .9), P, cal, att, P['test'], order, dict(fit_s=ft, ms=lat))
                ev('RF', 'ident', np.full(n, 1e-12), {k: margin(v) for k, v in P.items()}, cal, att, P['test'], order, dict(fit_s=ft, ms=lat))
        for order, (cal, att) in (('orig', ('B', 'A')), ('swap', ('A', 'B'))):
            cm = (row >= bl[cal][0]) & (row < bl[cal][1]); fc, yc, rc = x[cm], y[cm], row[cm]; F1, Y1 = np.r_[ftr, fc], np.r_[ytr, yc]
            if 'RAPID' in systems:
                m = RAPID2(n, model_seed, CFG).fit(F1, Y1, fc, yc); S = {k: m.scores(W[k][0]) for k in (att, 'test')}
                lat = latency(m.scores, W['test'][0][:300])
                ev('RAPID', 'ident', np.full(n, 1e-12 + CFG['delta']), S, cal, att, S['test'], order, dict(fit_s=m.fit_s, ms=lat)); del m; gc.collect()
            for b in ('RF', 'MLP'):
                if f'{b}-eq' not in systems: continue
                sc = StandardScaler().fit(F1); t1 = time.time(); m = make_base(b, model_seed).fit(sc.transform(F1), Y1); ft = time.time() - t1
                P = {k: post(m, sc, W[k][0], n) for k in (att, 'test')}; lat = latency(lambda w: post(m, sc, w, n), W['test'][0][:300]); del m; gc.collect()
                h = halves(yc, rc, n); fo = sess_fold(Y[cal]); Pc = np.zeros((len(Y[cal]), n), np.float32)
                for kf in range(2):
                    keep = h != kf; sk = StandardScaler().fit(np.r_[ftr, fc[keep]]); mk = make_base(b, model_seed).fit(sk.transform(np.r_[ftr, fc[keep]]), np.r_[ytr, yc[keep]])
                    Pc[fo == kf] = post(mk, sk, W[cal][0][fo == kf], n); del mk; gc.collect()
                P[cal] = Pc
                ev(f'{b}-eq', 'ident', np.full(n, 1e-12), {k: margin(v) for k, v in P.items()}, cal, att, P['test'], order, dict(fit_s=ft, ms=lat))
                ev(f'{b}-eq', 'quantile', thr(P[cal], Y[cal], n, .9), P, cal, att, P['test'], order, dict(fit_s=ft, ms=lat))
    print(cfg, 'seed', data_seed, 'total %.1fs' % (time.time() - t0), flush=True)
    return res

def plan():
    p = [(dict(DEFAULT), s, (1, 2, 4, 8, 16)) for s in (11, 12)]
    for key, vals in SWEEPS.items():
        for v in vals:
            c = dict(DEFAULT); c[key] = v
            if key == 'twin_delta': c['twins'] = 0.1
            for s in (11, 12): p.append((c, s, (4,)))
    for s in (11, 12):
        c = dict(DEFAULT); c.update(twins=0.1, twin_delta=0.0, twin_channel=True); p.append((c, s, (4,)))
    return p

def main(out=str(_HERE / 'sim2_results.json')):
    rows = json.load(open(out)) if os.path.exists(out) else []
    done = {(json.dumps({k: r[k] for k in r if k in DEFAULT or k == 'twin_channel'}, sort_keys=True), r['data_seed']) for r in rows}
    for c, s, Ls in plan():
        key = (json.dumps(c, sort_keys=True), s)
        if key in done: continue
        rows += run_config(c, s, Ls=Ls); done.add(key); json.dump(rows, open(out, 'w'))

if __name__ == '__main__':
    main()
