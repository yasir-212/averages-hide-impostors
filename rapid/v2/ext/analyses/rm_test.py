"""Frozen-protocol evaluation (3 seeds x 2 role orders) of the baselines with the rival-margin rule at the margins selected on
development data by rm_dev.py (criteria T and A); RAPID's stored scores are reused. CF-eq: RAPID's first member retrained exactly
as in the frozen evaluation (seed 1000*s) with one mean prototype per identity from the recent enrolment block."""
import sys, os, json, numpy as np
os.environ.setdefault('SMORFFI_FEATURES', 'features.npz')   # set to the SMoRFFI feature file
import pathlib as _pl
R = str(next(p for p in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents] if (p / 'paths.py').exists()))
for p in (R + '/rapid/v2', R + '/rapid', R): sys.path.insert(0, p)
from common import load_frames
from evalkit import windows, full_metrics
from core2 import CosFace32, class_means, margin
from final_runner import BLK, ORDERS, CFG, frames_in
from sklearn.preprocessing import StandardScaler
SC = R + '/rapid/v2/scores'; OUT = str(_pl.Path(__file__).resolve().parent)

def main():
    sel = json.load(open(f'{OUT}/rm_dev.json'))['selection']
    x, y, row, n = load_frames(); W = {k: windows(x, y, row, n, *v) for k, v in BLK.items()}; Y = {k: W[k][1] for k in W}
    ftr, ytr, _ = frames_in(x, y, row, 0, 600); runs = []
    for seed in (0, 1, 2):
        for order, (cal, att) in ORDERS.items():
            S = {}
            for m in ('RF-eq', 'MLP-eq'):
                d = np.load(f'{SC}/{m}_s{seed}_{order}.npz'); S[m] = {k: margin(d[k]) for k in (att, 'test')}
            cf_path = f'{OUT}/cfeq_scores_s{seed}_{order}.npz'
            if not os.path.exists(cf_path):
                fc, yc, _ = frames_in(x, y, row, *BLK[cal]); F1, Y1 = np.r_[ftr, fc], np.r_[ytr, yc]; sc = StandardScaler().fit(F1)
                kw = dict(h=tuple(CFG['h']), d_emb=CFG['d_emb'], epochs=CFG['epochs'], m=CFG['m'], s=CFG['s'])
                mem = CosFace32(F1.shape[1], n, seed=1000 * seed, **kw).fit(sc.transform(F1).astype(np.float32), Y1)
                P = class_means(np.stack([mem.embed(sc.transform(fc).astype(np.float32))]), yc, n)[0]
                def cf(Wk):
                    Z = sc.transform(Wk.reshape(-1, Wk.shape[2])).astype(np.float32)
                    return (mem.embed(Z) @ P.T).reshape(len(Wk), Wk.shape[1], -1).mean(1)
                np.savez(cf_path, **{k: cf(W[k][0]) for k in (att, 'test')})
            d = np.load(cf_path); S['CF-eq'] = {k: margin(d[k]) for k in (att, 'test')}
            d = np.load(f'{SC}/RAPID_s{seed}_{order}.npz'); S['RAPID'] = {k: margin(d[k]) for k in (att, 'test')}
            cfgs = [('RAPID', 'frozen', CFG['delta'])]
            for m in ('RF-eq', 'MLP-eq', 'CF-eq'):
                cfgs += [(m, 'ident', 0.0), (m, 'T', sel[m]['beta_T']), (m, 'A', sel[m]['beta_A'])]
            for m, rule, b in cfgs:
                r = full_metrics(np.full(n, 1e-12 + b), S[m][att], Y[att], S[m]['test'], Y['test'], n)
                r.update(system=m, rule=rule, beta=b, seed=seed, order=order); runs.append(r)
            print(seed, order, {f"{r['system']}|{r['rule']}": (round(100 * r['TAR'], 1), round(100 * r['devchosen'], 1)) for r in runs[-10:]}, flush=True)
    json.dump(runs, open(f'{OUT}/rm_test.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
