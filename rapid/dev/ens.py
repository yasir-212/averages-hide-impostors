"""Batched ensemble of E CosFace members trained simultaneously (float32). Mathematically identical to E
independent members (each has its own initialisation and its own mini-batch order); batching only removes
per-member Python overhead."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import numpy as np

class CosFaceEnsemble:
    def __init__(self, d_in, n_cls, E=3, h=(128, 128), d_emb=64, s=16.0, m=0.25, lr=2e-3, epochs=30, batch=256, seed=0, wd=1e-4):
        self.E, self.s, self.m, self.lr, self.epochs, self.batch, self.wd = E, np.float32(s), np.float32(m), lr, epochs, batch, np.float32(wd)
        self.rngs = [np.random.default_rng(1000 * seed + e) for e in range(E)]; dims = [d_in, *h, d_emb]; self.nl = len(dims) - 1; self.P = {}
        for i in range(self.nl):
            self.P[f'W{i}'] = np.stack([r.normal(0, np.sqrt(2 / dims[i]), (dims[i], dims[i + 1])) for r in self.rngs]).astype(np.float32)
            self.P[f'b{i}'] = np.zeros((E, 1, dims[i + 1]), np.float32)
        self.P['C'] = np.stack([r.normal(0, 1, (d_emb, n_cls)) for r in self.rngs]).astype(np.float32)
        self.Mo = {k: np.zeros_like(v) for k, v in self.P.items()}; self.Vo = {k: np.zeros_like(v) for k, v in self.P.items()}; self.t = 0
    def _fwd(self, X):                               # X: (E,B,d) or (B,d)
        a = X if X.ndim == 3 else np.broadcast_to(X, (self.E, *X.shape)); cache = [a]
        for i in range(self.nl):
            z = a @ self.P[f'W{i}'] + self.P[f'b{i}']; a = np.maximum(z, 0) if i < self.nl - 1 else z; cache.append((z, a))
        ne = np.linalg.norm(a, axis=2, keepdims=True) + 1e-12; return a / ne, ne, cache
    def fit(self, X, y):
        X = X.astype(np.float32); n = len(X); ncls = self.P['C'].shape[2]; Y = np.eye(ncls, dtype=np.float32)
        for ep in range(self.epochs):
            perms = np.stack([r.permutation(n) for r in self.rngs])
            for i in range(0, n, self.batch):
                idx = perms[:, i:i + self.batch]; xb = X[idx]; Yb = Y[y[idx]]; B = np.float32(idx.shape[1])
                u, ne, cache = self._fwd(xb); C = self.P['C']; nc = np.linalg.norm(C, axis=1, keepdims=True) + 1e-12; Cn = C / nc
                z = self.s * (u @ Cn - self.m * Yb); z -= z.max(2, keepdims=True); p = np.exp(z); p /= p.sum(2, keepdims=True)
                dcos = self.s * (p - Yb) / B; du = dcos @ Cn.transpose(0, 2, 1); dCn = u.transpose(0, 2, 1) @ dcos
                G = {'C': (dCn - Cn * (Cn * dCn).sum(1, keepdims=True)) / nc}; da = (du - u * (u * du).sum(2, keepdims=True)) / ne
                for li in range(self.nl - 1, -1, -1):
                    z_, _ = cache[li + 1]; dz_ = da if li == self.nl - 1 else da * (z_ > 0)
                    a_prev = cache[li] if li == 0 else cache[li][1]
                    G[f'W{li}'] = a_prev.transpose(0, 2, 1) @ dz_ + self.wd * self.P[f'W{li}']; G[f'b{li}'] = dz_.sum(1, keepdims=True)
                    da = dz_ @ self.P[f'W{li}'].transpose(0, 2, 1)
                self._adam(G)
        u = self._fwd(X)[0]; pr = np.stack([u[:, y == c].mean(1) for c in range(ncls)], 1)   # (E, ncls, d_emb)
        self.protos = (pr / (np.linalg.norm(pr, axis=2, keepdims=True) + 1e-12)).astype(np.float32); return self
    def _adam(self, G, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1; c1 = 1 - b1 ** self.t; c2 = 1 - b2 ** self.t
        for k, g in G.items():
            self.Mo[k] *= b1; self.Mo[k] += (1 - b1) * g; self.Vo[k] *= b2; self.Vo[k] += (1 - b2) * g * g
            self.P[k] -= (self.lr * (self.Mo[k] / c1) / (np.sqrt(self.Vo[k] / c2) + eps)).astype(np.float32)
    def cos(self, X):
        u = self._fwd(X.astype(np.float32))[0]; return (u @ self.protos.transpose(0, 2, 1)).mean(0)   # ensemble-mean cosine
    def n_params(self):                              # deployed parameters: embedding networks + prototypes (class weights C not needed)
        return int(sum(v.size for k, v in self.P.items() if k != 'C') + self.protos.size)
