"""Shared evaluation kit: the 2026 baseline (SMoRFFI framework, Guo et al., Computer Networks 2026),
RAPID, identity-claim metrics, operating curves and cost measurement. Used for SMoRFFI, the synthetic
simulator, and any user dataset given as per-frame features (x, y, row)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import gc, time, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from rapid_core import RAPID, rival

# ---------------------------------------------------------------- data
def blocks_for(n_frames_per_dev):
    """SMoRFFI layout (train 0-599, A 600-724, B 725-849, test 850-998 of 1000) scaled to any capture length."""
    f = n_frames_per_dev / 1000.0
    return {'train': (0, int(600 * f)), 'A': (int(600 * f), int(725 * f)), 'B': (int(725 * f), int(850 * f)), 'test': (int(850 * f), int(999 * f))}

def windows(x, y, row, n, lo, hi, L=4):
    W, Y, S = [], [], []
    for j in range(n):
        ix = np.flatnonzero((y == j) & (row >= lo) & (row < hi)); ix = ix[np.argsort(row[ix])]
        for p in range(0, len(ix) - L + 1, L):
            ids = ix[p:p + L]
            if np.all(np.diff(row[ids]) == 1) and np.all(np.isfinite(x[ids])): W.append(x[ids]); Y.append(j)
    return np.asarray(W), np.asarray(Y)

# ---------------------------------------------------------------- systems
class SMoRFFIBaseline:
    """Published framework: per-frame 15 features -> Random Forest with 128 trees (other hyper-parameters at library
    defaults, as the paper reports none). Verification adaptation: session score for claim t = mean posterior P(t|frame)
    over the session's frames; per-identity threshold on calibration data (same rule as RAPID)."""
    name = 'SMoRFFI-RF128'
    def __init__(self, n_cls, seed=0, n_jobs=2):
        self.n, self.seed, self.n_jobs = n_cls, seed, n_jobs
    def fit(self, F, y, scaler):
        self.sc = scaler; t0 = time.time()
        self.m = RandomForestClassifier(n_estimators=128, random_state=31 + 100 * self.seed, n_jobs=self.n_jobs).fit(scaler.transform(F), y)
        self.fit_s = time.time() - t0; return self
    def scores(self, W):
        P = self.m.predict_proba(self.sc.transform(W.reshape(-1, W.shape[2])))
        out = np.zeros((len(P), self.n)); out[:, self.m.classes_] = P
        return out.reshape(len(W), W.shape[1], -1).mean(1)
    def size_bytes(self):
        nodes = sum(e.tree_.node_count for e in self.m.estimators_)
        return int(sum(e.tree_.value.nbytes for e in self.m.estimators_) + 64 * nodes), int(nodes)
    def free(self):
        del self.m; gc.collect()

class RAPIDSystem:
    name = 'RAPID'
    def __init__(self, n_cls, seed=0, **kw):
        self.r = RAPID(n_cls, seed=seed, **kw); self.n = n_cls
    def fit(self, F, y, scaler):
        self.r.fit(F, y, scaler); self.fit_s = self.r.fit_s; return self
    def scores(self, W):
        return self.r.scores(W)
    def size_bytes(self):
        p = self.r.n_params(); return 4 * p, p
    def free(self):
        pass

# ---------------------------------------------------------------- metrics
def thr(S, y, n, q):
    a = round(1 - q, 10)
    return np.array([np.quantile(S[y == t, t], a, method='higher') if (y == t).any() else np.inf for t in range(n)])

def accept(S, y, n, th):
    acc = S >= th[None, :]; g = acc[np.arange(len(y)), y]
    tar = np.array([g[y == t].mean() for t in range(n)]); a = acc.copy(); a[np.arange(len(y)), y] = False
    P = np.stack([a[y == s].mean(0) for s in range(n)]); np.fill_diagonal(P, np.nan); return tar, P, acc

def global_roc(S, y):
    G = S[np.arange(len(y)), y]; M = np.ones_like(S, bool); M[np.arange(len(y)), y] = False; I = np.sort(S[M])
    out = {}
    for f in (1e-2, 1e-3):
        t = I[int(np.ceil((1 - f) * len(I))) - 1]; out[f'TAR@FAR{f:g}'] = float((G > t).mean())
    ts = np.quantile(np.r_[G, I], np.linspace(0, 1, 801)); frr = np.searchsorted(np.sort(G), ts, 'left') / len(G)
    far = 1 - np.searchsorted(I, ts, 'left') / len(I); k = np.argmin(np.abs(frr - far)); out['EER'] = float((frr[k] + far[k]) / 2)
    # AUC via Mann-Whitney on a subsample of impostor scores
    rng = np.random.default_rng(0); Is = rng.choice(I, min(len(I), 200000), replace=False)
    r = np.argsort(np.argsort(np.r_[G, Is])); out['AUC'] = float((r[:len(G)].sum() - len(G) * (len(G) - 1) / 2) / (len(G) * len(Is)))
    return out

def rule_thresholds(rule, Sc, yc, n, q=0.9):
    """'quantile': per-identity genuine quantile; 'floor': min(quantile, 0) on a margin score (accept every top-1
    identification of t); 'ident': 0 on a margin score (accept iff the claimed identity is the top-1 identity)."""
    if rule == 'quantile': return thr(Sc, yc, n, q)
    if rule == 'floor': return np.minimum(thr(Sc, yc, n, q), 0.0)
    if rule == 'ident': return np.full(n, 1e-12)
    raise ValueError(rule)

def full_metrics(th, Sd, yd, Se, ye, n):
    """th: thresholds fixed on the calibration role; attacker chooses sources on development data (Sd); evaluation on Se."""
    tar, P, acc = accept(Se, ye, n, th); f = P[np.isfinite(P)]
    _, Pd, _ = accept(Sd, yd, n, th); sel = np.nanargmax(np.where(np.isfinite(Pd), Pd, -1), 0); dc = P[sel, np.arange(n)]
    out = dict(TAR=tar.mean(), eff0=(tar > 0).sum(), eff50=(tar >= .5).sum(), pooled=f.mean(), worst=f.max(), p99=np.quantile(f, .99),
               p999=np.quantile(f, .999), ge50=(f >= .5).sum(), ge10=(f >= .1).sum(), eq100=(f >= 1 - 1e-9).sum(), devchosen=dc.mean(),
               oracle=np.nanmean(np.nanmax(P, 0)), acc=(Se.argmax(1) == ye).mean(), multi=np.mean(acc.sum(1) >= 2), claims=acc.sum(1).mean())
    for k in (5, 20):                                   # sequential retry: blocks of k consecutive windows of the chosen source
        vals = []
        for t in range(n):
            a = acc[ye == sel[t], t]; nb = len(a) // k
            if nb: vals.append(np.mean([a[b * k:(b + 1) * k].any() for b in range(nb)]))
        out[f'retry{k}'] = np.mean(vals) if vals else np.nan
    out.update(global_roc(Se, ye))
    return {k: float(v) for k, v in out.items()}

def curve(Sc, yc, Sd, yd, Se, ye, n, rule='quantile', qs=None):
    qs = np.r_[0.02, 0.05, np.linspace(0.10, 0.95, 35), 0.97, 0.98, 0.99] if qs is None else qs; pts = []
    for q in qs:
        th = rule_thresholds(rule, Sc, yc, n, q); tar, P, _ = accept(Se, ye, n, th); f = P[np.isfinite(P)]
        _, Pd, _ = accept(Sd, yd, n, th); sel = np.nanargmax(np.where(np.isfinite(Pd), Pd, -1), 0)
        pts.append(dict(q=float(q), TAR=float(tar.mean()), eff50=int((tar >= .5).sum()), pooled=float(f.mean()), worst=float(f.max()),
                        p99=float(np.quantile(f, .99)), ge50=int((f >= .5).sum()), devchosen=float(P[sel, np.arange(n)].mean())))
    return pts

def latency_ms(system, W, reps=3):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); system.scores(W); best = min(best, time.perf_counter() - t0)
    return 1e3 * best / len(W)
