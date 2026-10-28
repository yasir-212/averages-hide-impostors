"""RAPID — Rival-Aware Prototype IDentity verification (NumPy, float32).

Per-frame metric embedding (CosFace, additive cosine margin) -> cosine to every enrolled prototype ->
session score = mean over the L frames of a session and over E ensemble members ->
rival score  R_t(x) = c_t(x) - lam * max_{j != t} c_j(x)  -> accept claim t iff R_t >= tau_t,
tau_t = per-identity genuine quantile on a held-out calibration block."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import numpy as np, time

class CosFace32:
    def __init__(self, d_in, n_cls, h=(128, 128), d_emb=64, s=16.0, m=0.25, lr=2e-3, epochs=30, batch=256, seed=0, wd=1e-4):
        self.s, self.m, self.lr, self.epochs, self.batch, self.wd = np.float32(s), np.float32(m), lr, epochs, batch, np.float32(wd)
        rng = np.random.default_rng(seed); self.rng = rng; dims = [d_in, *h, d_emb]; self.P = {}
        for i in range(len(dims) - 1):
            self.P[f'W{i}'] = rng.normal(0, np.sqrt(2 / dims[i]), (dims[i], dims[i + 1])).astype(np.float32)
            self.P[f'b{i}'] = np.zeros(dims[i + 1], np.float32)
        self.P['C'] = rng.normal(0, 1, (d_emb, n_cls)).astype(np.float32); self.nl = len(dims) - 1
        self.Mo = {k: np.zeros_like(v) for k, v in self.P.items()}; self.Vo = {k: np.zeros_like(v) for k, v in self.P.items()}; self.t = 0
    def _fwd(self, X):
        cache = [X]; a = X
        for i in range(self.nl):
            z = a @ self.P[f'W{i}'] + self.P[f'b{i}']; a = np.maximum(z, 0) if i < self.nl - 1 else z; cache.append((z, a))
        ne = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12; return a / ne, ne, cache
    def embed(self, X):
        return self._fwd(X.astype(np.float32))[0]
    def fit(self, X, y):
        X = X.astype(np.float32); n = len(X); ncls = self.P['C'].shape[1]; Y = np.eye(ncls, dtype=np.float32)[y]
        for ep in range(self.epochs):
            perm = self.rng.permutation(n)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]; xb, Yb = X[idx], Y[idx]; B = np.float32(len(idx))
                u, ne, cache = self._fwd(xb); C = self.P['C']; nc = np.linalg.norm(C, axis=0, keepdims=True) + 1e-12; Cn = C / nc
                z = self.s * (u @ Cn - self.m * Yb); z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
                dcos = self.s * (p - Yb) / B; du = dcos @ Cn.T; dCn = u.T @ dcos
                G = {'C': (dCn - Cn * (Cn * dCn).sum(0, keepdims=True)) / nc}; da = (du - u * (u * du).sum(1, keepdims=True)) / ne
                for li in range(self.nl - 1, -1, -1):
                    z_, _ = cache[li + 1]; dz_ = da if li == self.nl - 1 else da * (z_ > 0)
                    a_prev = cache[li] if li == 0 else cache[li][1]
                    G[f'W{li}'] = a_prev.T @ dz_ + self.wd * self.P[f'W{li}']; G[f'b{li}'] = dz_.sum(0); da = dz_ @ self.P[f'W{li}'].T
                self._adam(G)
        u = self.embed(X); pr = np.stack([u[y == c].mean(0) for c in range(ncls)])
        self.protos = (pr / (np.linalg.norm(pr, axis=1, keepdims=True) + 1e-12)).astype(np.float32); return self
    def _adam(self, G, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1; c1 = 1 - b1 ** self.t; c2 = 1 - b2 ** self.t
        for k, g in G.items():
            self.Mo[k] *= b1; self.Mo[k] += (1 - b1) * g; self.Vo[k] *= b2; self.Vo[k] += (1 - b2) * g * g
            self.P[k] -= (self.lr * (self.Mo[k] / c1) / (np.sqrt(self.Vo[k] / c2) + eps)).astype(np.float32)
    def cos(self, X):
        return self.embed(X) @ self.protos.T

def rival(C, lam):
    srt = np.sort(C, 1); top, sec = srt[:, -1:], srt[:, -2:-1]; return C - lam * np.where(C >= top, sec, top)

class RAPID:
    def __init__(self, n_cls, d_in=15, E=3, lam=1.0, h=(128, 128), epochs=30, batch=256, seed=0):
        self.n, self.d, self.E, self.lam, self.kw, self.seed = n_cls, d_in, E, lam, dict(h=h, epochs=epochs, batch=batch), seed
    def fit(self, F, y, scaler, workers=2):
        """Members are trained in parallel worker processes (one BLAS thread each); workers=1 trains serially."""
        self.sc = scaler; t0 = time.time(); Fz = scaler.transform(F).astype(np.float32); seeds = [1000 * self.seed + e for e in range(self.E)]
        if workers > 1:
            from partrain import train_members
            self.members = train_members(Fz, y, self.n, seeds, self.kw, workers)
        else:
            self.members = [CosFace32(self.d, self.n, seed=s, **self.kw).fit(Fz, y) for s in seeds]
        self.fit_s = time.time() - t0; return self
    def session_cos(self, W):
        Fz = self.sc.transform(W.reshape(-1, self.d)).astype(np.float32)
        return np.mean([m.cos(Fz) for m in self.members], 0).reshape(len(W), W.shape[1], -1).mean(1)
    def scores(self, W):
        return rival(self.session_cos(W), self.lam)
    def n_params(self):
        return int(sum(v.size for m in self.members for v in m.P.values()) - sum(m.P['C'].size for m in self.members) + sum(m.protos.size for m in self.members))
