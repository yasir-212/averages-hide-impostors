"""Run the frozen RAPID protocol (RAPID, RAPID-Lite, SMoRFFI-framework baseline) on YOUR data.

Input formats (choose one):
  --features file.npz       per-frame features: arrays x (frames, d), y (device id), row (capture index within device, chronological)
  --preambles file.npz      complex 802.11 OFDM preambles: arrays iq (frames, 320) complex, y, row  -> SMoRFFI-style 15 features
  --iq-cache DIR            INRIA-style cache: meta.json {n, win}, iq_f16.dat float16 (n, 2, win), y.npy; windows assumed stored in
                            chronological order per device -> generic I/Q features (iq_features.py)
Blocks per device follow the SMoRFFI layout scaled to the shortest capture: train 60%, A 12.5%, B 12.5%, test 14.9%.
Outputs (in --out): protocol.json (every run and metric), summary.json, and the same figures as the paper.
Example: python run_on_your_data.py --features /data/my_features.npz --out results_mydata"""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
import argparse, json, os, sys, time, gc, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim'))
from evalkit import windows, blocks_for, SMoRFFIBaseline, RAPIDSystem, rule_thresholds, full_metrics, curve, latency_ms, global_roc
from rapid_core import rival
from sklearn.preprocessing import StandardScaler

def load(args):
    if args.features:
        d = np.load(args.features, allow_pickle=True); return d['x'].astype(float), d['y'], d['row']
    if args.preambles:
        import wifi_sim as WS; d = np.load(args.preambles); return WS.features(d['iq']), d['y'], d['row']
    if args.iq_cache:
        from iq_features import iq_features
        meta = json.load(open(os.path.join(args.iq_cache, 'meta.json'))); n, win = int(meta['n']), int(meta['win'])
        iq = np.memmap(os.path.join(args.iq_cache, 'iq_f16.dat'), dtype=np.float16, mode='r', shape=(n, 2, win))
        y = np.load(os.path.join(args.iq_cache, 'y.npy')); xs = []
        for i in range(0, n, 4096):
            b = np.asarray(iq[i:i + 4096], dtype=np.float32); xs.append(iq_features(b[:, 0] + 1j * b[:, 1]))
        x = np.concatenate(xs); row = np.zeros(n, int)
        for c in np.unique(y): ix = np.flatnonzero(y == c); row[ix] = np.arange(len(ix))
        return x, y, row
    raise SystemExit('give --features, --preambles or --iq-cache')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--features'); ap.add_argument('--preambles'); ap.add_argument('--iq-cache')
    ap.add_argument('--out', default='results_userdata'); ap.add_argument('--L', type=int, default=4); ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--min-frames', type=int, default=200); args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    x, y, row = load(args); ok = np.all(np.isfinite(x), 1); x, y, row = x[ok], y[ok], row[ok]
    labs, cnt = np.unique(y, return_counts=True); keep = labs[cnt >= args.min_frames]; g = np.isin(y, keep)
    x, y, row = x[g], np.searchsorted(keep, y[g]), row[g]; n = len(keep)
    nf = int(min(cnt[cnt >= args.min_frames])); bl = blocks_for(nf)
    # rank frames within each device so blocks are chronological even if row has gaps
    r2 = np.zeros_like(row)
    for j in range(n): ix = np.flatnonzero(y == j); r2[ix[np.argsort(row[ix])]] = np.arange(len(ix))
    row = r2
    W = {k: windows(x, y, row, n, *bl[k], L=args.L) for k in ('A', 'B', 'test')}
    tr = row < bl['train'][1]; ftr, yf = x[tr], y[tr]; sc = StandardScaler().fit(ftr); d = x.shape[1]
    info = dict(devices=int(n), features=int(d), frames_per_device=nf, blocks=bl, sessions={k: int(len(v[1])) for k, v in W.items()}); print(info, flush=True)
    runs, curves, costs = [], {}, []
    for seed in range(args.seeds):
        for sname, make in (('SMoRFFI-RF128', lambda s: SMoRFFIBaseline(n, seed=s)), ('RAPID', lambda s: RAPIDSystem(n, seed=s, d_in=d, E=3)),
                            ('RAPID-Lite', lambda s: RAPIDSystem(n, seed=s, d_in=d, E=2))):
            m = make(seed).fit(ftr, yf, sc); S = {k: m.scores(W[k][0]) for k in W}
            costs.append(dict(system=sname, seed=seed, fit_s=m.fit_s, size_MB=m.size_bytes()[0] / 2**20, ms_per_session=latency_ms(m, W['test'][0][:1000])))
            m.free(); del m; gc.collect()
            M = {k: rival(v, 1.0) for k, v in S.items()} if sname == 'SMoRFFI-RF128' else S
            rules = [('quantile', S), ('ident', M), ('floor', M)] if sname == 'SMoRFFI-RF128' else [('floor', S), ('quantile', S), ('ident', S)]
            gr = global_roc(S['test'], W['test'][1])
            for order in ('orig', 'swap'):
                C, D = ('B', 'A') if order == 'orig' else ('A', 'B')
                for rule, SS in rules:
                    th = rule_thresholds(rule, SS[C], W[C][1], n, 0.9); r = full_metrics(th, SS[D], W[D][1], SS['test'], W['test'][1], n)
                    r.update(gr); r.update(system=sname, rule=rule, seed=seed, order=order); runs.append(r)
                    curves[f'{sname}|{rule}|{seed}|{order}'] = curve(SS[C], W[C][1], SS[D], W[D][1], SS['test'], W['test'][1], n, rule=rule)
            print(sname, seed, 'done', flush=True)
    json.dump(dict(info=info, runs=runs, costs=costs, curves=curves), open(os.path.join(args.out, 'protocol.json'), 'w'))
    keys = ['TAR', 'eff50', 'pooled', 'worst', 'p99', 'ge50', 'devchosen', 'retry5', 'EER', 'AUC', 'TAR@FAR0.01', 'TAR@FAR0.001', 'acc']
    summ = {f'{s}|{ru}': {k: float(np.median([r[k] for r in runs if r['system'] == s and r['rule'] == ru])) for k in keys}
            for s, ru in (('SMoRFFI-RF128', 'quantile'), ('SMoRFFI-RF128', 'ident'), ('RAPID', 'floor'), ('RAPID', 'quantile'), ('RAPID', 'ident'), ('RAPID-Lite', 'floor'))}
    summ['costs'] = {s: {k: float(np.median([c[k] for c in costs if c['system'] == s])) for k in ('fit_s', 'size_MB', 'ms_per_session')} for s in ('SMoRFFI-RF128', 'RAPID', 'RAPID-Lite')}
    json.dump(summ, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)
    for k, v in summ.items(): print(k, {a: round(b, 4) for a, b in v.items()} if k != 'costs' else v)

if __name__ == '__main__':
    main()
