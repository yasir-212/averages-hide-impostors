"""Train ensemble members in parallel worker processes (one BLAS thread each)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def _train(args):
    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rapid_core import CosFace32
    F, y, n, seed, kw = args; m = CosFace32(F.shape[1], n, seed=seed, **kw).fit(F, y)
    return m.P, m.protos

def train_members(F, y, n, seeds, kw, workers=2):
    from rapid_core import CosFace32
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as ex:
        res = list(ex.map(_train, [(F, y, n, s, kw) for s in seeds]))
    ms = []
    for P, protos in res:
        m = CosFace32(F.shape[1], n, **kw); m.P = P; m.protos = protos; ms.append(m)
    return ms
