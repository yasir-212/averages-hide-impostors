"""Build features_v1.npz from the SMoRFFI per-device CSV files (<MAC>_pre.csv).

Usage: python data/build_smorffi_features.py /path/to/csv_folder data/features_v1.npz

Keeps the 15 physical-layer feature columns in a fixed order, takes the device identity from the file name
(MAC address and device-number columns are never used as features), records each frame's position in its file
as `row` (chronological capture order) and rejects frames with non-finite values."""
import csv, json, sys, hashlib
from pathlib import Path
import numpy as np

FEATURES = ['CFO', 'short_freq', 'long_freq', 'frac_dimension_1', 'frac_dimension_2', 'iqi_1', 'iqi_2',
            'mag_error_mean_1', 'mag_error_mean_2', 'mag_error_var_1', 'mag_error_var_2',
            'phase_error_mean_1', 'phase_error_mean_2', 'phase_error_var_1', 'phase_error_var_2']

def main(src, dst):
    files = sorted(Path(src).glob('*_pre.csv'))
    if not files: raise SystemExit(f'no *_pre.csv files in {src}')
    labels = [f.name[:-len('_pre.csv')].replace('_', ':') for f in files]
    xs, ys, rows, audit = [], [], [], []
    for j, f in enumerate(files):
        with open(f, newline='') as fh:
            rd = csv.DictReader(fh); missing = [c for c in FEATURES if c not in rd.fieldnames]
            if missing: raise SystemExit(f'{f.name}: missing columns {missing}')
            X = np.array([[float(r[c]) for c in FEATURES] for r in rd], dtype=float)
        ok = np.all(np.isfinite(X), 1); idx = np.flatnonzero(ok)
        xs.append(X[ok]); ys.append(np.full(ok.sum(), j)); rows.append(idx); audit.append(dict(file=f.name, valid_frames=int(ok.sum()), rejected=int((~ok).sum())))
    x = np.concatenate(xs); y = np.concatenate(ys); row = np.concatenate(rows)
    src_hash = hashlib.sha256(b''.join(f.read_bytes() for f in files)).hexdigest()
    np.savez(dst, x=x, y=y, row=row, labels=np.array(labels), audit=json.dumps(dict(files=audit, devices=len(files), frames=int(len(x)), features=FEATURES)), source_hash=src_hash)
    print(f'wrote {dst}: x {x.shape}, devices {len(files)}, frames per device {min(a["valid_frames"] for a in audit)}-{max(a["valid_frames"] for a in audit)}')
    print('expected for the paper: x (122511, 15), 123 devices')

if __name__ == '__main__':
    if len(sys.argv) != 3: raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
