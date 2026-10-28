"""Equal-data baselines on the inner split: RF128 / MLP retrained on inner-train + the calibration block (the same data RAPID's
prototype refresh uses). Calibration scores are cross-fitted (K=2 contiguous folds) so no threshold is set in-sample."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, time, gc, numpy as np
from devh import load_inner, SPLIT, fold_ids
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import argparse

def make(model, seed):
    if model == 'rf': return RandomForestClassifier(n_estimators=128, random_state=31 + 100 * seed, n_jobs=2)
    return MLPClassifier(hidden_layer_sizes=(256, 256), alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=60,
                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, random_state=7 + seed)

def post(m, sc, W, n):
    P = m.predict_proba(sc.transform(W.reshape(-1, 15))); Q = np.zeros((len(P), n)); Q[:, m.classes_] = P
    return Q.reshape(len(W), W.shape[1], -1).mean(1).astype(np.float32)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='rf'); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
    x, y, row, n, ftr, ytr, W = load_inner(); sc = StandardScaler().fit(ftr); out = {}
    for cal, att in (('B', 'A'), ('A', 'B')):
        lo, hi = SPLIT[cal]; inb = (row >= lo) & (row < hi)
        X1, Y1 = np.r_[ftr, x[inb]], np.r_[ytr, y[inb]]
        m = make(a.model, a.seed).fit(sc.transform(X1), Y1)
        out[f'{cal}_att'] = post(m, sc, W[att][0], n); out[f'{cal}_test'] = post(m, sc, W['test'][0], n); del m; gc.collect()
        # cross-fitted calibration scores: fold k of the calibration block is scored by a model that did not see it
        Wc, yc = W[cal]; fo = fold_ids(yc, 2); Sc = np.zeros((len(yc), n), np.float32)
        # map frames of the block to folds by time: per device, frames in the first/second half of the block range
        r_in, y_in = row[inb], y[inb]; half = np.zeros(len(r_in), int)
        for j in range(n):
            ix = np.flatnonzero(y_in == j); half[ix] = (np.argsort(np.argsort(r_in[ix])) * 2) // len(ix)
        for k in range(2):
            keep = half != k; mk = make(a.model, a.seed).fit(sc.transform(np.r_[ftr, x[inb][keep]]), np.r_[ytr, y_in[keep]])
            Sc[fo == k] = post(mk, sc, Wc[fo == k], n); del mk; gc.collect()
        out[f'{cal}_cal'] = Sc
        print(a.model, a.seed, cal, 'done', flush=True)
    for k in ('A', 'B', 'test'): out[f'y_{k}'] = W[k][1]
    np.savez(str(_HERE) + f'/cache/baseeq_{a.model}_s{a.seed}.npz', **out)

if __name__ == '__main__':
    main()
