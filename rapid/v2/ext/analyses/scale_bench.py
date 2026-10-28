"""Single-thread latency of RAPID's exemplar comparison (E=3 members, L=4 frames, k=3, 125 exemplars of dimension 64 per
identity) for synthetic unit-norm exemplar banks of N identities; one session per call, median of repeated calls."""
import os
os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
import sys, time, json, numpy as np
import pathlib as _pl
sys.path.insert(0, str(next(p for p in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents] if (p / 'paths.py').exists()) / 'rapid' / 'v2'))
from core2 import local_cos, margin
rng = np.random.default_rng(0); E, L, d, m, k = 3, 4, 64, 125, 3; out = {}
for N in (122, 500, 1000, 2000, 5000):
    Rp = rng.standard_normal((E, N, m, d)).astype(np.float32); Rp /= np.linalg.norm(Rp, axis=-1, keepdims=True)
    mask = np.ones((N, m), bool)
    Uq = rng.standard_normal((E, 1, L, d)).astype(np.float32); Uq /= np.linalg.norm(Uq, axis=-1, keepdims=True)
    ts = []
    for r in range(25 if N <= 1000 else 9):
        t0 = time.perf_counter(); c = local_cos(Uq, Rp, mask, k); R = margin(c); ts.append(time.perf_counter() - t0)
    out[N] = dict(ms=1e3 * float(np.median(ts[2:])), MB=float(Rp.nbytes / 2**20)); print(N, out[N], flush=True)
    del Rp
json.dump(out, open('scale_bench.json', 'w'), indent=1)
