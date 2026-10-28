"""Frozen-protocol evaluation of RAPID v2 against equal-data and published-data baselines on SMoRFFI.
3 seeds x 2 role orders. In each order the calibration block (B or A) is the verifier's recent enrolment data and the other block
is the attacker's development data; the locked test block (frames 850-998) is scored once per fitted model."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import os, sys, json, time, gc, hashlib, argparse, numpy as np
from common import load_frames, FEAT
from evalkit import windows, full_metrics, thr, global_roc, accept
from core2 import train_members, class_means, session_cos, margin, pad_refs, local_cos, unit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

CFG = json.load(open(str(_HERE / 'FROZEN.json')))
BLK = {'A': (600, 725), 'B': (725, 850), 'test': (850, 999)}
ORDERS = {'orig': ('B', 'A'), 'swap': ('A', 'B')}

def frames_in(x, y, row, lo, hi):
    m = (row >= lo) & (row < hi); return x[m], y[m], row[m]

def halves(y_in, r_in, n):
    h = np.zeros(len(r_in), int)
    for j in range(n):
        ix = np.flatnonzero(y_in == j); h[ix] = (np.argsort(np.argsort(r_in[ix])) * 2) // max(len(ix), 1)
    return h

def sess_fold(yc, K=2):
    f = np.zeros(len(yc), int)
    for j in np.unique(yc):
        ix = np.flatnonzero(yc == j); f[ix] = (np.arange(len(ix)) * K) // len(ix)
    return f

class RAPID2:
    def __init__(self, n, seed, cfg):
        self.n, self.seed, self.cfg = n, seed, cfg
    def fit(self, F, y, Fref, yref):
        t0 = time.time(); self.sc = StandardScaler().fit(F); Z = self.sc.transform(F).astype(np.float32)
        kw = dict(h=tuple(self.cfg['h']), d_emb=self.cfg['d_emb'], epochs=self.cfg['epochs'], m=self.cfg['m'], s=self.cfg['s'])
        self.members = train_members(Z, y, self.n, [1000 * self.seed + e for e in range(self.cfg['E'])], kw, workers=2)
        R = np.stack([m.embed(self.sc.transform(Fref).astype(np.float32)) for m in self.members])
        if self.cfg.get('dtype') == 'float16': R = R.astype(np.float16).astype(np.float32)
        self.Rp, self.mask = pad_refs(R, yref, self.n); self.k = int(self.cfg['k']); self.fit_s = time.time() - t0; return self
    def cos(self, W):
        S, L, d = W.shape; Z = self.sc.transform(W.reshape(-1, d)).astype(np.float32)
        U = np.stack([m.embed(Z).reshape(S, L, -1) for m in self.members]); return local_cos(U, self.Rp, self.mask, self.k)
    def scores(self, W):
        return margin(self.cos(W))
    def size(self):
        p = sum(v.size for m in self.members for k, v in m.P.items() if k != 'C'); r = int(self.mask.sum()) * self.Rp.shape[0] * self.Rp.shape[-1]
        bpr = 2 if self.cfg.get('dtype') == 'float16' else 4
        return 4 * p + bpr * r, p, r

def make_base(name, seed):
    if name == 'RF': return RandomForestClassifier(n_estimators=128, random_state=31 + 100 * seed, n_jobs=2)
    return MLPClassifier(hidden_layer_sizes=(256, 256), alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=60,
                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, random_state=7 + seed)

def post(m, sc, W, n):
    P = m.predict_proba(sc.transform(W.reshape(-1, W.shape[2]))); Q = np.zeros((len(P), n), np.float32); Q[:, m.classes_] = P
    return Q.reshape(len(W), W.shape[1], -1).mean(1)

def base_size(m):
    if hasattr(m, 'estimators_'):
        nodes = sum(e.tree_.node_count for e in m.estimators_); return int(sum(e.tree_.value.nbytes for e in m.estimators_) + 64 * nodes), int(nodes)
    p = sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_); return 4 * p, int(p)

def latency(fn, W, reps=3):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); fn(W); best = min(best, time.perf_counter() - t0)
    return 1e3 * best / len(W)

def single_latency(fn, W, k=400):
    ts = []
    for i in range(k):
        t0 = time.perf_counter(); fn(W[i:i + 1]); ts.append(time.perf_counter() - t0)
    return float(1e3 * np.median(ts)), float(1e3 * np.quantile(ts, .95))

def evaluate(rows, sysname, rule, th, S, Y, cal, att, native, seed, order, extra=None):
    r = full_metrics(th, S[att], Y[att], S['test'], Y['test'], len(th)); r.update(global_roc(native, Y['test']))
    r.update(system=sysname, rule=rule, seed=seed, order=order, **(extra or {})); rows.append(r); return r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--out', default='final_results.json')
    ap.add_argument('--systems', default='RAPID,RF-eq,MLP-eq,MLP,RF'); a = ap.parse_args()
    systems = a.systems.split(',')
    x, y, row, n = load_frames(); sha = hashlib.sha256(open(FEAT, 'rb').read()).hexdigest()
    W = {k: windows(x, y, row, n, *v) for k, v in BLK.items()}; Y = {k: W[k][1] for k in W}
    ftr, ytr, _ = frames_in(x, y, row, 0, 600)
    out_path = str(_HERE) + f'/{a.out}'
    res = json.load(open(out_path)) if os.path.exists(out_path) else dict(info=dict(sha=sha, cfg=CFG), runs=[], costs=[])
    done = {(r['system'], r['seed'], r['order']) for r in res['runs']}
    os.makedirs(str(_HERE / 'scores'), exist_ok=True)
    for seed in [int(s) for s in a.seeds.split(',')]:
        for order, (cal, att) in ORDERS.items():
            fc, yc_f, rc = frames_in(x, y, row, *BLK[cal]); F1, Y1 = np.r_[ftr, fc], np.r_[ytr, yc_f]
            # ---------------- RAPID v2 (equal data: training + recent enrolment block; local prototypes = enrolment-block frames)
            if 'RAPID' in systems and ('RAPID', seed, order) not in done:
                m = RAPID2(n, seed, CFG).fit(F1, Y1, fc, yc_f)
                C = {k: m.cos(W[k][0]) for k in (att, 'test')}; S = {k: margin(v) for k, v in C.items()}
                evaluate(res['runs'], 'RAPID', 'ident', np.full(n, 1e-12 + CFG['delta']), S, Y, cal, att, S['test'], seed, order)
                size, npar, nref = m.size(); lat = latency(m.scores, W['test'][0][:1000]); sl = single_latency(m.scores, W['test'][0])
                res['costs'].append(dict(system='RAPID', seed=seed, order=order, fit_s=m.fit_s, size_MB=size / 2**20, params=int(npar), ref_values=int(nref),
                                         ms_batched=lat, ms_single=sl[0], ms_single_p95=sl[1]))
                np.savez_compressed(str(_HERE) + f'/scores/RAPID_s{seed}_{order}.npz', **{k: v for k, v in C.items()})
                print('RAPID', seed, order, 'fit %.1fs' % m.fit_s, 'size %.2f MB' % (size / 2**20), 'lat %.3f ms' % lat, flush=True)
                del m; gc.collect()
            # ---------------- equal-data baselines (same frames as RAPID); quantile rule uses cross-fitted calibration scores
            for bname in ('RF', 'MLP'):
                sname = f'{bname}-eq'
                if sname not in systems or (sname, seed, order) in done: continue
                sc = StandardScaler().fit(F1); t0 = time.time(); m = make_base(bname, seed).fit(sc.transform(F1), Y1); fit_s = time.time() - t0
                P = {k: post(m, sc, W[k][0], n) for k in (att, 'test')}
                size, npar = base_size(m); lat = latency(lambda w: post(m, sc, w, n), W['test'][0][:1000]); sl = single_latency(lambda w: post(m, sc, w, n), W['test'][0], 200 if bname == 'RF' else 400)
                del m; gc.collect()
                h = halves(yc_f, rc, n); fo = sess_fold(Y[cal]); Pc = np.zeros((len(Y[cal]), n), np.float32)
                for kf in range(2):
                    keep = h != kf; sk = StandardScaler().fit(np.r_[ftr, fc[keep]])
                    mk = make_base(bname, seed).fit(sk.transform(np.r_[ftr, fc[keep]]), np.r_[ytr, yc_f[keep]]); Pc[fo == kf] = post(mk, sk, W[cal][0][fo == kf], n); del mk; gc.collect()
                P[cal] = Pc; M = {k: margin(v) for k, v in P.items()}
                evaluate(res['runs'], sname, 'ident', np.full(n, 1e-12), M, Y, cal, att, P['test'], seed, order)
                evaluate(res['runs'], sname, 'quantile', thr(P[cal], Y[cal], n, 0.9), P, Y, cal, att, P['test'], seed, order)
                res['costs'].append(dict(system=sname, seed=seed, order=order, fit_s=fit_s, size_MB=size / 2**20, params=npar, ms_batched=lat, ms_single=sl[0], ms_single_p95=sl[1]))
                np.savez_compressed(str(_HERE) + f'/scores/{sname}_s{seed}_{order}.npz', **P)
                print(sname, seed, order, 'fit %.1fs' % fit_s, 'size %.1f MB' % (size / 2**20), flush=True)
            json.dump(res, open(out_path, 'w'))
        # ---------------- published-data baselines (training frames only; calibration block sets the quantile threshold)
        for bname in ('RF', 'MLP'):
            if bname not in systems or (bname, seed, 'orig') in done: continue
            if bname == 'RF':      # identical forests to the original protocol run: reuse their saved session posteriors
                fn = _R / 'rapid' / f'scores_SMoRFFI-RF128_s{seed}.npz'
                if fn.exists():
                    z = np.load(fn); P = {k: z[k] for k in ('A', 'B', 'test')}
                else:               # retrain the published forest (training frames only) with the original seed
                    sc = StandardScaler().fit(ftr); m = make_base('RF', seed).fit(sc.transform(ftr), ytr); P = {k: post(m, sc, W[k][0], n) for k in ('A', 'B', 'test')}; del m
            else:
                sc = StandardScaler().fit(ftr); t0 = time.time(); m = make_base('MLP', seed).fit(sc.transform(ftr), ytr); fit_s = time.time() - t0
                P = {k: post(m, sc, W[k][0], n) for k in ('A', 'B', 'test')}; size, npar = base_size(m)
                lat = latency(lambda w: post(m, sc, w, n), W['test'][0][:1000]); sl = single_latency(lambda w: post(m, sc, w, n), W['test'][0])
                res['costs'].append(dict(system='MLP', seed=seed, order='both', fit_s=fit_s, size_MB=size / 2**20, params=npar, ms_batched=lat, ms_single=sl[0], ms_single_p95=sl[1]))
                np.savez_compressed(str(_HERE) + f'/scores/MLP_s{seed}.npz', **P); del m
            M = {k: margin(v) for k, v in P.items()}
            for order, (cal, att) in ORDERS.items():
                evaluate(res['runs'], bname, 'quantile', thr(P[cal], Y[cal], n, 0.9), P, Y, cal, att, P['test'], seed, order)
                evaluate(res['runs'], bname, 'ident', np.full(n, 1e-12), M, Y, cal, att, P['test'], seed, order)
            print(bname, seed, 'done', flush=True)
        json.dump(res, open(out_path, 'w'))

if __name__ == '__main__':
    main()
