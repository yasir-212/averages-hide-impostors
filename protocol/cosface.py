"""Minimal NumPy CosFace (additive-margin softmax) metric-learning verifier.
Embedding: x -> ReLU(W1) -> ReLU(W2) -> W3 -> L2-normalise. Class weights L2-normalised.
Loss: CE over s*(cos - m*onehot). Adam. Verification score: cosine to the class prototype
(normalised mean of training embeddings of that class)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import numpy as np

class CosFace:
    def __init__(self, d_in, n_cls, h=(128, 128), d_emb=64, s=16.0, m=0.25, lr=2e-3, epochs=60, batch=256, seed=0, wd=1e-4):
        self.s, self.m, self.lr, self.epochs, self.batch, self.wd = s, m, lr, epochs, batch, wd
        rng = np.random.default_rng(seed); self.rng = rng
        dims = [d_in, *h, d_emb]
        self.P = {}
        for i in range(len(dims) - 1):
            self.P[f'W{i}'] = rng.normal(0, np.sqrt(2 / dims[i]), (dims[i], dims[i + 1]))
            self.P[f'b{i}'] = np.zeros(dims[i + 1])
        self.P['C'] = rng.normal(0, 1, (d_emb, n_cls))
        self.nl = len(dims) - 1
        self.M = {k: np.zeros_like(v) for k, v in self.P.items()}; self.V = {k: np.zeros_like(v) for k, v in self.P.items()}; self.t = 0

    def _fwd(self, X):
        cache = [X]; a = X
        for i in range(self.nl):
            z = a @ self.P[f'W{i}'] + self.P[f'b{i}']
            a = np.maximum(z, 0) if i < self.nl - 1 else z
            cache.append((z, a))
        e = a; ne = np.linalg.norm(e, axis=1, keepdims=True) + 1e-12; u = e / ne
        return u, ne, cache

    def embed(self, X):
        return self._fwd(X)[0]

    def fit(self, X, y):
        n = len(X); Y = np.eye(self.P['C'].shape[1])[y]
        for ep in range(self.epochs):
            perm = self.rng.permutation(n)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]; xb, Yb = X[idx], Y[idx]; B = len(idx)
                u, ne, cache = self._fwd(xb)
                C = self.P['C']; nc = np.linalg.norm(C, axis=0, keepdims=True) + 1e-12; Cn = C / nc
                cos = u @ Cn; z = self.s * (cos - self.m * Yb)
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
        protos = np.stack([u[y == c].mean(0) for c in range(self.P['C'].shape[1])])
        self.protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
        return self

    def _adam(self, G, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k, g in G.items():
            self.M[k] = b1 * self.M[k] + (1 - b1) * g; self.V[k] = b2 * self.V[k] + (1 - b2) * g * g
            mh = self.M[k] / (1 - b1 ** self.t); vh = self.V[k] / (1 - b2 ** self.t)
            self.P[k] -= self.lr * mh / (np.sqrt(vh) + eps)

    def scores(self, X):
        return self.embed(X) @ self.protos.T   # cosine to each identity prototype

def gradcheck():
    rng = np.random.default_rng(1); X = rng.normal(size=(20, 5)); y = rng.integers(0, 3, 20)
    m = CosFace(5, 3, h=(7,), d_emb=4, epochs=0)
    def loss():
        u, ne, cache = m._fwd(X); C = m.P['C']; Cn = C / np.linalg.norm(C, axis=0, keepdims=True)
        Y = np.eye(3)[y]; z = m.s * (u @ Cn - m.m * Y); z -= z.max(1, keepdims=True)
        return float(-(np.log(np.exp(z) / np.exp(z).sum(1, keepdims=True)) * Y).sum() / len(X))
    # analytic via one step with lr=0 capture
    Y = np.eye(3)[y]; u, ne, cache = m._fwd(X); C = m.P['C']; nc = np.linalg.norm(C, axis=0, keepdims=True); Cn = C / nc
    z = m.s * (u @ Cn - m.m * Y); z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    dz = (p - Y) / len(X); dcos = m.s * dz; du = dcos @ Cn.T; dCn = u.T @ dcos
    dC = (dCn - Cn * (Cn * dCn).sum(0, keepdims=True)) / nc; de = (du - u * (u * du).sum(1, keepdims=True)) / ne
    z1, a1 = cache[2]; dW1 = cache[1][1].T @ de
    for name, ana in (('C', dC), ('W1', dW1)):
        P = m.P[name]; num = np.zeros_like(P)
        for idx in np.ndindex(P.shape):
            o = P[idx]; P[idx] = o + 1e-6; lp = loss(); P[idx] = o - 1e-6; lm = loss(); P[idx] = o; num[idx] = (lp - lm) / 2e-6
        print(name, 'max rel err', float(np.abs(num - ana).max() / (np.abs(num).max() + 1e-12)))

if __name__ == '__main__':
    gradcheck()
