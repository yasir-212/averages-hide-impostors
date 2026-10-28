"""Baselines on the inner split: SMoRFFI RF128 (published framework) and a learned per-frame MLP classifier (dev only)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, time, gc, numpy as np
from devh import load_inner
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import argparse
ap = argparse.ArgumentParser(); ap.add_argument('--model', default='rf'); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
x, y, row, n, ftr, ytr, W = load_inner(); sc = StandardScaler().fit(ftr); Ftr = sc.transform(ftr)
t0 = time.time()
if a.model == 'rf':
    m = RandomForestClassifier(n_estimators=128, random_state=31 + 100 * a.seed, n_jobs=2).fit(Ftr, ytr)
else:
    m = MLPClassifier(hidden_layer_sizes=(256, 256), alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=60,
                      early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, random_state=7 + a.seed).fit(Ftr, ytr)
ft = time.time() - t0; out = dict(fit_s=ft)
for k, (Wk, yk) in W.items():
    P = m.predict_proba(sc.transform(Wk.reshape(-1, 15))); Q = np.zeros((len(P), n)); Q[:, m.classes_] = P
    out[f'P_{k}'] = Q.reshape(len(Wk), Wk.shape[1], -1).mean(1).astype(np.float32); out[f'y_{k}'] = yk
np.savez(str(_HERE) + f'/cache/base_{a.model}_s{a.seed}.npz', **out); print(a.model, a.seed, 'fit %.1fs' % ft, flush=True)
