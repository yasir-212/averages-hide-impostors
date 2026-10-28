"""Matched-operating-point analysis on saved test/development scores: every system's threshold is a single global value swept over
its own score families (rival margin for all systems; native posterior for RF/MLP), and the lowest risk reached at each TAR level
is kept per run. The attacker chooses its source per target on the development block under the same threshold."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, json, glob, numpy as np
from evalkit import windows, accept
from core2 import margin
from common import load_frames
n = 122; GRID = np.round(np.arange(0.40, 0.951, 0.01), 2)

def points(S, Sd, y, yd, ths):
    out = []
    for t in ths:
        th = np.full(n, t); tar, P, _ = accept(S, y, n, th); f = P[np.isfinite(P)]
        _, Pd, _ = accept(Sd, yd, n, th); sel = np.nanargmax(np.where(np.isfinite(Pd), Pd, -1), 0)
        out.append((tar.mean(), f.mean(), f.max(), np.quantile(f, .99), (f >= .5).sum(), P[sel, np.arange(n)].mean()))
    return np.array(out)

def families(name, d, att):
    if name == 'RAPID': C = {k: d[k] for k in (att, 'test')}; return [{k: margin(v) for k, v in C.items()}]
    P = {k: d[k] for k in (att, 'test')}; return [P, {k: margin(v) for k, v in P.items()}]

def frontier(fams, y, yd, att):
    pts = []
    for S in fams:
        q = np.quantile(S['test'][np.arange(len(y)), y], np.linspace(0.0, 0.7, 120)); pts.append(points(S['test'], S[att], y, yd, q))
    pts = np.concatenate(pts); res = {}
    for g in GRID:
        ok = pts[pts[:, 0] >= g]
        res[float(g)] = ok[:, 1:].min(0).tolist() if len(ok) else None
    return res

if __name__ == '__main__':
    x, y, row, _ = load_frames(); Y = {k: windows(x, y, row, n, *v)[1] for k, v in {'A': (600, 725), 'B': (725, 850), 'test': (850, 999)}.items()}
    out = {}
    for name in sys.argv[1].split(','):
        for seed in (0, 1, 2):
            for order, (cal, att) in (('orig', ('B', 'A')), ('swap', ('A', 'B'))):
                if name in ('RF', 'MLP'):
                    d = np.load(str(_R / 'rapid' / f'scores_SMoRFFI-RF128_s{seed}.npz')) if name == 'RF' else np.load(f'scores/MLP_s{seed}.npz')
                else:
                    d = np.load(f'scores/{name}_s{seed}_{order}.npz')
                out[f'{name}|{seed}|{order}'] = frontier(families(name, d, att), Y['test'], Y[att], att)
                print(name, seed, order, flush=True)
    json.dump(out, open('frontier2.json', 'w'))
