import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import numpy as np, sys, json
sys.path.insert(0, '.')
from evalkit import thr, accept
from rapid_core import rival
d = np.load('dev_scores.npz'); PA, PB, yA, yB = d['PA'], d['PB'], d['yA'], d['yB']; n = PA.shape[1]
e = np.load('ens_E3_e30.npz'); RA, RB = rival(e['cA'], 1.0), rival(e['cB'], 1.0)
marg = lambda S: rival(S, 1.0)
def point(SA, SB, th):
    tar, P, acc = accept(SA, yA, n, th); f = P[np.isfinite(P)]
    _, Pd, _ = accept(SB, yB, n, th); sel = np.nanargmax(np.where(np.isfinite(Pd), Pd, -1), 0)
    return dict(TAR=tar.mean(), eff50=int((tar >= .5).sum()), pooled=f.mean(), worst=f.max(), p99=np.quantile(f, .99), ge50=int((f >= .5).sum()),
                eq100=int((f >= 1 - 1e-9).sum()), devch=P[sel, np.arange(n)].mean())
qs = np.r_[0.02, 0.05, np.linspace(0.1, 0.95, 18), 0.97, 0.98, 0.99]
fam = {}
fam['BL quantile'] = [point(PA, PB, thr(PB, yB, n, q)) for q in qs]
fam['BL floor'] = [point(marg(PA), marg(PB), np.minimum(thr(marg(PB), yB, n, q), 0.0)) for q in qs]
fam['BL margin-quantile'] = [point(marg(PA), marg(PB), thr(marg(PB), yB, n, q)) for q in qs]
fam['RAPID quantile'] = [point(RA, RB, thr(RB, yB, n, q)) for q in qs]
fam['RAPID floor'] = [point(RA, RB, np.minimum(thr(RB, yB, n, q), 0.0)) for q in qs]
def frontier(pts, key):
    pts = sorted(pts, key=lambda p: -p['TAR']); out = []; best = np.inf
    for p in pts:
        if p[key] < best - 1e-12: out.append((p['TAR'], p[key])); best = p[key]
    return out
for key in ('pooled', 'devch', 'ge50', 'p99', 'worst', 'eq100'):
    BL = [p for k in fam if k.startswith('BL') for p in fam[k]]; RP = [p for k in fam if k.startswith('RAPID') for p in fam[k]]
    fb = frontier(BL, key); fr = frontier(RP, key)
    # dominance: for every baseline frontier point, exists RAPID point with TAR >= and metric <= ; count strict
    dom = sum(any(r[0] >= b[0] - 1e-9 and r[1] <= b[1] + 1e-12 for r in fr) for b in fb)
    strict = sum(any(r[0] >= b[0] - 1e-9 and r[1] < b[1] - 1e-12 for r in fr) for b in fb)
    print(f'{key:7s} baseline frontier pts {len(fb)}: RAPID weakly dominates {dom}, strictly {strict}')
    print('   BL', [(round(a, 3), round(b, 4)) for a, b in fb][:12]); print('   RP', [(round(a, 3), round(b, 4)) for a, b in fr][:12])
for k, pts in fam.items():
    p = [x for x, q in zip(pts, qs) if abs(q - .9) < 1e-9][0]; print(k, {a: round(float(b), 4) for a, b in p.items()})
    if 'floor' in k: print('   q=0.02:', {a: round(float(b), 4) for a, b in pts[0].items()})
