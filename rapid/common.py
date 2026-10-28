"""Shared data handling and identity-claim metrics (development and protocol runs)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import numpy as np
FEAT = str(DATA_FILE)
def load_frames():
    if not _pl.Path(FEAT).exists():
        raise FileNotFoundError(f'SMoRFFI features not found at {FEAT}. Set SMORFFI_FEATURES or see data/README.md.')
    d = np.load(FEAT); x, y, row = d['x'], d['y'], d['row']
    valid = np.array([j for j in range(len(d['labels'])) if (row[y == j] >= 998).any()])
    g = np.isin(y, valid); return x[g], np.searchsorted(valid, y[g]), row[g], len(valid)

def kalman_seq(Z, r):
    """1-D random-walk Kalman filter applied independently to every column; r = Q/R."""
    out = np.empty_like(Z); s = Z[0].copy(); P = np.ones(Z.shape[1]); out[0] = s
    for k in range(1, len(Z)):
        P = P + r; K = P / (P + 1.0); s = s + K * (Z[k] - s); P = (1 - K) * P; out[k] = s
    return out

def windows(x, y, row, n, lo, hi, L=4):
    """Consecutive non-overlapping windows of L frames per device inside rows [lo,hi). Returns list of (L,15) arrays."""
    W, Y = [], []
    for j in range(n):
        ix = np.flatnonzero((y == j) & (row >= lo) & (row < hi)); ix = ix[np.argsort(row[ix])]
        for p in range(0, len(ix) - L + 1, L):
            ids = ix[p:p + L]
            if np.all(np.diff(row[ids]) == 1): W.append(x[ids]); Y.append(j)
    return np.asarray(W), np.asarray(Y)

def ms(W):
    return np.concatenate([W.mean(1), W.std(1)], 1)

def rival_ratio(S):
    srt = np.sort(S, 1); top, sec = srt[:, -1:], srt[:, -2:-1]
    best_other = np.where(S >= top, sec, top)
    return S - best_other

def thresholds(S, y, n, q):
    return np.array([np.quantile(S[y == t, t], 1 - q, method='higher') if (y == t).any() else np.inf for t in range(n)])

def pair_far(S, y, n, th):
    acc = S >= th[None, :]
    g = acc[np.arange(len(y)), y]; tar = np.array([g[y == t].mean() for t in range(n)])
    a = acc.copy(); a[np.arange(len(y)), y] = False
    P = np.stack([a[y == s].mean(0) for s in range(n)]); np.fill_diagonal(P, np.nan)
    return tar, P, acc

def metrics(S_thr, y_thr, S_ev, y_ev, n, q=0.9, S_dev=None, y_dev=None, retry=20):
    """Thresholds from (S_thr,y_thr). Attacker chooses source per target on (S_dev,y_dev) under the same thresholds."""
    th = thresholds(S_thr, y_thr, n, q)
    tar, P, acc = pair_far(S_ev, y_ev, n, th); f = P[np.isfinite(P)]
    out = dict(TAR=tar.mean(), eff0=int((tar > 0).sum()), eff50=int((tar >= .5).sum()), pooled=f.mean(), worst=f.max(),
               p99=np.quantile(f, .99), p999=np.quantile(f, .999), ge50=int((f >= .5).sum()), eq100=int((f >= 1 - 1e-9).sum()),
               oracle=np.nanmean(np.nanmax(P, 0)))
    if S_dev is not None:
        _, Pd, _ = pair_far(S_dev, y_dev, n, th); Pd = np.where(np.isfinite(Pd), Pd, -1)
        ch = Pd.argmax(0); dc = P[ch, np.arange(n)]; out['devchosen'] = np.nanmean(dc)
        out['retry'] = np.nanmean(1 - (1 - dc) ** retry)   # i.i.d. approximation; exact sequential version in protocol
    # pooled EER (genuine vs all impostor claims)
    G = S_ev[np.arange(len(y_ev)), y_ev]; M = np.ones_like(S_ev, bool); M[np.arange(len(y_ev)), y_ev] = False; I = S_ev[M]
    ts = np.quantile(np.r_[G, I], np.linspace(0, 1, 400)); frr = np.array([(G < t).mean() for t in ts]); far = np.array([(I >= t).mean() for t in ts])
    k = np.argmin(np.abs(frr - far)); out['EER'] = (frr[k] + far[k]) / 2
    out['acc'] = (S_ev.argmax(1) == y_ev).mean()
    return {k: float(v) for k, v in out.items()}
