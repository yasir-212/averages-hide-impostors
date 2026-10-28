"""RAPID: Rival-Aware Prototype IDentity verification (NumPy).
Components: (1) order-invariant window statistics; (2) CosFace embedding trained with an additional
rival margin that pushes each identity away from its most confusable identities; (3) rival-ratio score
S_t(x) = cos(x, p_t) - max_{j != t} cos(x, p_j); (4) per-identity threshold on calibration data."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import numpy as np, sys
sys.path.insert(0, str(_R / 'protocol'))
from cosface import CosFace

def window_stats(Z, kind):
    """Z: (frames, 15). Order-invariant statistics of one window."""
    if kind == 'ms':
        return np.r_[Z.mean(0), Z.std(0)]
    if kind == 'rich':
        return np.r_[Z.mean(0), Z.std(0), Z.min(0), Z.max(0), np.median(Z, 0)]
    if kind == 'rich+':
        d = np.diff(Z, axis=0)
        return np.r_[Z.mean(0), Z.std(0), Z.min(0), Z.max(0), np.median(Z, 0), np.abs(d).mean(0)]
    raise ValueError(kind)

class RivalCosFace(CosFace):
    """CosFace whose logits add a rival margin mr to the k most confusable identities of the true class."""
    def __init__(self, *a, rivals=None, mr=0.15, **k):
        super().__init__(*a, **k); self.R = rivals; self.mr = mr
    def fit(self, X, y):
        n = len(X); ncls = self.P['C'].shape[1]; Y = np.eye(ncls)[y]
        RM = np.zeros((ncls, ncls)) if self.R is None else self.R
        for ep in range(self.epochs):
            perm = self.rng.permutation(n)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]; xb, Yb = X[idx], Y[idx]; B = len(idx)
                u, ne, cache = self._fwd(xb)
                C = self.P['C']; nc = np.linalg.norm(C, axis=0, keepdims=True) + 1e-12; Cn = C / nc
                cos = u @ Cn; z = self.s * (cos - self.m * Yb + self.mr * RM[y[idx]])
                z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
                dz = (p - Yb) / B; dcos = self.s * dz
                du = dcos @ Cn.T; dCn = u.T @ dcos
                dC = (dCn - Cn * (Cn * dCn).sum(0, keepdims=True)) / nc
                de = (du - u * (u * du).sum(1, keepdims=True)) / ne
                G = {'C': dC}; da = de
                for li in range(self.nl - 1, -1, -1):
                    z_, a_ = cache[li + 1]
                    dz_ = da if li == self.nl - 1 else da * (z_ > 0)
                    a_prev = cache[li] if li == 0 else cache[li][1]
                    G[f'W{li}'] = a_prev.T @ dz_ + self.wd * self.P[f'W{li}']; G[f'b{li}'] = dz_.sum(0)
                    da = dz_ @ self.P[f'W{li}'].T
                self._adam(G)
        u = self.embed(X)
        protos = np.stack([u[y == c].mean(0) for c in range(ncls)])
        self.protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
        return self

def rival_matrix(protos, k):
    S = protos @ protos.T; np.fill_diagonal(S, -np.inf)
    R = np.zeros_like(S); nn = np.argsort(-S, 1)[:, :k]
    for i in range(len(S)): R[i, nn[i]] = 1.0
    return R

def rival_ratio(cosines):
    n = cosines.shape[1]; out = np.empty_like(cosines)
    srt = np.sort(cosines, 1); top, second = srt[:, -1], srt[:, -2]
    for t in range(n):
        best_other = np.where(cosines[:, t] >= top, second, top)
        out[:, t] = cosines[:, t] - best_other
    return out

class RAPID:
    def __init__(self, d_in, n_cls, k_rivals=5, mr=0.15, ensemble=1, seed=0, epochs=60, score='ratio', warm_epochs=15):
        self.k, self.mr, self.E, self.seed, self.epochs, self.score_kind, self.warm = k_rivals, mr, ensemble, seed, epochs, score, warm_epochs
        self.d_in, self.n = d_in, n_cls
    def fit(self, X, y):
        warm = CosFace(self.d_in, self.n, seed=self.seed, epochs=self.warm).fit(X, y)
        R = rival_matrix(warm.protos, self.k) if self.k > 0 else None
        self.models = [RivalCosFace(self.d_in, self.n, rivals=R, mr=self.mr, seed=self.seed * 10 + e, epochs=self.epochs).fit(X, y) for e in range(self.E)]
        return self
    def cosines(self, X):
        return np.mean([m.scores(X) for m in self.models], 0)
    def scores(self, X):
        c = self.cosines(X)
        return rival_ratio(c) if self.score_kind == 'ratio' else c
    def n_params(self):
        return sum(v.size for m in self.models for v in m.P.values()) + sum(m.protos.size for m in self.models)
