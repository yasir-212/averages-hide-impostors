"""Per-frame scores of the gradient-boosted-trees baseline (configuration chosen on the development split, dev_choice.json),
retrained on equal data for every run; companion of final_ext.py. Writes ext/gbt_s{seed}_{order}.npz."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R / 'rapid' / 'v2', _R, _HERE):
    _sys.path.insert(0, str(_p))
import os, sys, json, time, gc, numpy as np
from common import load_frames
from final_runner import BLK, ORDERS, frames_in
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
EXT = str(_HERE)

def main():
    cfg = json.load(open(f'{EXT}/dev_choice.json'))['gbt']; x, y, row, n = load_frames(); ftr, ytr, _ = frames_in(x, y, row, 0, 600)
    for seed in (0, 1, 2):
        for order, (cal, att) in ORDERS.items():
            path = f'{EXT}/gbt_s{seed}_{order}.npz'
            if os.path.exists(path) or not os.path.exists(f'{EXT}/frames_s{seed}_{order}.npz'): continue
            ref = np.load(f'{EXT}/frames_s{seed}_{order}.npz'); fc, yc, _ = frames_in(x, y, row, *BLK[cal]); F1, Y1 = np.r_[ftr, fc], np.r_[ytr, yc]
            sc = StandardScaler().fit(F1); t0 = time.time()
            m = HistGradientBoostingClassifier(random_state=11 + seed, early_stopping=False, **cfg).fit(sc.transform(F1), Y1); ft = time.time() - t0
            out = {}
            for b in (att, 'test'):
                P = m.predict_proba(sc.transform(x[ref[f'idx_{b}']])); Q = np.zeros((len(P), n), np.float32); Q[:, m.classes_] = P; out[f'GBT-eq_{b}'] = Q
            np.savez_compressed(path, **out); del m; gc.collect(); print(seed, order, 'GBT fit %.0fs' % ft, flush=True)

if __name__ == '__main__':
    main()
