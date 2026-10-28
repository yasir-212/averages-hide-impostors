"""Post-freeze sensitivity sweeps on SMoRFFI (seed 0, both role orders; RAPID's frozen configuration throughout):
(1) enrolment budget: only the m most recent frames of the enrolment block per identity are used, by RAPID (training and
    exemplars) and by the equal-data baselines (training);
(2) population size: N identities drawn at random (fixed seed) from the 122; every system is trained, enrolled and attacked
    within the subset.
The identification rule is used for the baselines (their strongest rule on the risk metrics)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R / 'rapid' / 'v2', _R, _HERE):
    _sys.path.insert(0, str(_p))
import os, sys, json, time, gc, numpy as np
from common import load_frames
from evalkit import windows, full_metrics
from core2 import margin
from final_runner import RAPID2, CFG, BLK, ORDERS, frames_in, make_base, post
from sklearn.preprocessing import StandardScaler

OUT = str(_HERE / 'sweeps.json')

def last_m(fc, yc, rc, n, m):
    keep = np.zeros(len(yc), bool)
    for j in range(n):
        ix = np.flatnonzero(yc == j); keep[ix[np.argsort(rc[ix])][-m:]] = True
    return fc[keep], yc[keep]

def run(x, y, row, n, seed, order, tag, m_recent=None):
    cal, att = ORDERS[order]; W = {k: windows(x, y, row, n, *BLK[k]) for k in (att, 'test')}; Y = {k: W[k][1] for k in W}
    ftr, ytr, _ = frames_in(x, y, row, 0, 600); fc, yc, rc = frames_in(x, y, row, *BLK[cal])
    if m_recent is not None: fc, yc = last_m(fc, yc, rc, n, m_recent)
    F1, Y1 = np.r_[ftr, fc], np.r_[ytr, yc]; rows = []
    mdl = RAPID2(n, seed, CFG).fit(F1, Y1, fc, yc); S = {k: mdl.scores(W[k][0]) for k in W}
    r = full_metrics(np.full(n, 1e-12 + CFG['delta']), S[att], Y[att], S['test'], Y['test'], n); r.update(system='RAPID'); rows.append(r)
    del mdl; gc.collect()
    for b in ('RF', 'MLP'):
        sc = StandardScaler().fit(F1); mdl = make_base(b, seed).fit(sc.transform(F1), Y1)
        S = {k: margin(post(mdl, sc, W[k][0], n)) for k in W}; del mdl; gc.collect()
        r = full_metrics(np.full(n, 1e-12), S[att], Y[att], S['test'], Y['test'], n); r.update(system=f'{b}-eq'); rows.append(r)
    for r in rows: r.update(tag=tag, seed=seed, order=order, n=n, m_recent=m_recent)
    return rows

def main():
    x, y, row, n = load_frames(); res = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r['tag'], r['order']) for r in res}
    jobs = [(f'budget{m}', m, None) for m in (25, 50, 75, 100)] + [(f'pop{N}', None, N) for N in (30, 60, 90)]
    for tag, m, N in jobs:
        for order in ('orig', 'swap'):
            if (tag, order) in done: continue
            t0 = time.time()
            if N is None: rows = run(x, y, row, n, 0, order, tag, m)
            else:
                dev = np.sort(np.random.default_rng(2026).choice(n, N, replace=False)); g = np.isin(y, dev)
                rows = run(x[g], np.searchsorted(dev, y[g]), row[g], N, 0, order, tag)
            res += rows; json.dump(res, open(OUT, 'w'))
            print(tag, order, ' '.join('%s TAR %.1f devch %.1f pooled %.3f' % (r['system'], 100 * r['TAR'], 100 * r['devchosen'], 100 * r['pooled']) for r in rows),
                  '%.0fs' % (time.time() - t0), flush=True)

if __name__ == '__main__':
    main()
