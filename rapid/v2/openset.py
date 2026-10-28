"""Unenrolled-attacker (open-set) evaluation: a random subset of devices is never enrolled; their sessions claim every enrolled
identity. Systems are trained/enrolled on the enrolled devices only (training frames + the recent enrolment block).
Rules: identification among enrolled identities, with and without an absolute-similarity check calibrated (cross-fitted) at the
5th percentile of each identity's genuine scores on the enrolment block. --split dev uses the inner split (never reads 850-998)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import os, sys, json, time, gc, argparse, numpy as np
from common import load_frames
from evalkit import windows, global_roc
from core2 import margin, pad_refs, local_cos
from final_runner import RAPID2, make_base, post, halves, sess_fold, CFG
from sklearn.preprocessing import StandardScaler

SPLITS = {'dev': dict(train=(0, 350), A=(350, 475), B=(475, 600), test=(600, 850)),
          'test': dict(train=(0, 600), A=(600, 725), B=(725, 850), test=(850, 999))}

def metrics(acc_g, ok_imp, S_att_ok, ens_t, uns, yt_all, ya_all, tgt_map):
    pass

def evaluate(dec_att, dec_test, ya, yt, enrolled, unenrolled):
    """dec_*: (S, n_enrolled) boolean acceptance for every claim; ya/yt: original device labels of the sessions."""
    e_idx = {d: i for i, d in enumerate(enrolled)}; ne = len(enrolled)
    g = np.array([dec_test[(yt == d), e_idx[d]].mean() for d in enrolled])
    U = np.array([[dec_test[yt == u, t].mean() for t in range(ne)] for u in unenrolled])          # (n_unenrolled, ne)
    Ua = np.array([[dec_att[ya == u, t].mean() for t in range(ne)] for u in unenrolled])
    ch = Ua.argmax(0); dc = U[ch, np.arange(ne)]
    r20 = []
    for t in range(ne):
        a = dec_test[yt == unenrolled[ch[t]], t]; nb = len(a) // 20
        if nb: r20.append(np.mean([a[b * 20:(b + 1) * 20].any() for b in range(nb)]))
    Ie = np.array([[dec_test[yt == s, e_idx[t]].mean() if s != t else np.nan for t in enrolled] for s in enrolled])
    f = Ie[np.isfinite(Ie)]
    return dict(TAR=float(g.mean()), eff50=int((g >= .5).sum()), unenr_pooled=float(U.mean()), unenr_worst=float(U.max()),
                unenr_ge50=int((U >= .5).sum()), unenr_devchosen=float(dc.mean()), unenr_retry20=float(np.mean(r20)),
                enr_pooled=float(f.mean()), enr_worst=float(f.max()), enr_ge50=int((f >= .5).sum()))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='dev'); ap.add_argument('--parts', default='0,1,2')
    ap.add_argument('--n_unenrolled', type=int, default=30); ap.add_argument('--systems', default='RAPID,RF-eq,MLP-eq'); ap.add_argument('--out', default=None)
    ap.add_argument('--alpha', type=float, default=0.05); a = ap.parse_args(); SP = SPLITS[a.split]
    x, y, row, n = load_frames()
    if a.split == 'dev':
        keep = row < 850; x, y, row = x[keep], y[keep], row[keep]
    out = a.out or str(_HERE) + f'/openset_{a.split}.json'; rows = json.load(open(out)) if os.path.exists(out) else []
    done = {(r['system'], r['part'], r['order'], r['rule']) for r in rows}
    Wall = {k: windows(x, y, row, n, *SP[k]) for k in ('A', 'B', 'test')}
    for part in [int(p) for p in a.parts.split(',')]:
        rng = np.random.default_rng(500 + part); unen = np.sort(rng.choice(n, a.n_unenrolled, replace=False)); enr = np.setdiff1d(np.arange(n), unen)
        remap = -np.ones(n, int); remap[enr] = np.arange(len(enr)); ne = len(enr)
        trm = (row >= SP['train'][0]) & (row < SP['train'][1]) & np.isin(y, enr); ftr, ytr = x[trm], remap[y[trm]]
        for order, (cal, att) in (('orig', ('B', 'A')), ('swap', ('A', 'B'))):
            cm = (row >= SP[cal][0]) & (row < SP[cal][1]) & np.isin(y, enr); fc, yc, rc = x[cm], remap[y[cm]], row[cm]
            F1, Y1 = np.r_[ftr, fc], np.r_[ytr, yc]
            Wc, ycs = Wall[cal]; ke = np.isin(ycs, enr); Wc, ycs = Wc[ke], remap[ycs[ke]]; fo = sess_fold(ycs)
            Wa, ya = Wall[att]; Wt, yt = Wall['test']
            for sname in a.systems.split(','):
                if (sname, part, order, 'ident') in done: continue
                t0 = time.time()
                if sname == 'RAPID':
                    m = RAPID2(ne, part, CFG).fit(F1, Y1, fc, yc); Ca, Ct = m.cos(Wa), m.cos(Wt)
                    # cross-fitted genuine absolute similarity: calibration sessions of fold k scored against the other fold's frames
                    h = halves(yc, rc, ne); Cc = np.zeros((len(ycs), ne), np.float32)
                    Zc = m.sc.transform(Wc.reshape(-1, 15)).astype(np.float32); Uc = np.stack([mm.embed(Zc).reshape(len(Wc), Wc.shape[1], -1) for mm in m.members])
                    Rall = np.stack([mm.embed(m.sc.transform(fc).astype(np.float32)) for mm in m.members])
                    for k in range(2):
                        Rp, mask = pad_refs(Rall[:, h != k], yc[h != k], ne); Cc[fo == k] = local_cos(Uc[:, fo == k], Rp, mask, min(m.k, mask.sum(1).min()))
                    abs_a, abs_t, abs_c = Ca, Ct, Cc; Ma, Mt = margin(Ca), margin(Ct); del m
                else:
                    bname = sname.split('-')[0]; sc = StandardScaler().fit(F1); mdl = make_base(bname, part).fit(sc.transform(F1), Y1)
                    Pa, Pt = post(mdl, sc, Wa, ne), post(mdl, sc, Wt, ne); del mdl; gc.collect()
                    h = halves(yc, rc, ne); Pc = np.zeros((len(ycs), ne), np.float32)
                    for k in range(2):
                        keep = h != k; sk = StandardScaler().fit(np.r_[ftr, fc[keep]]); mk = make_base(bname, part).fit(sk.transform(np.r_[ftr, fc[keep]]), np.r_[ytr, yc[keep]])
                        Pc[fo == k] = post(mk, sk, Wc[fo == k], ne); del mk; gc.collect()
                    abs_a, abs_t, abs_c = Pa, Pt, Pc; Ma, Mt = margin(Pa), margin(Pt)
                kappa = np.array([np.quantile(abs_c[ycs == t, t], a.alpha) for t in range(ne)])
                tau = CFG['delta'] if sname == 'RAPID' else 0.0
                for rule in ('ident', 'ident+abs'):
                    da, dt = Ma >= 1e-12 + tau, Mt >= 1e-12 + tau
                    if rule == 'ident+abs': da, dt = da & (abs_a >= kappa), dt & (abs_t >= kappa)
                    r = evaluate(da, dt, ya, yt, enr, unen); r.update(system=sname, part=part, order=order, rule=rule, secs=time.time() - t0); rows.append(r)
                    print(sname, part, order, rule, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ('system', 'part', 'order', 'rule')}, flush=True)
                json.dump(rows, open(out, 'w')); gc.collect()

if __name__ == '__main__':
    main()
