"""RAPID v2 building blocks (NumPy float32): CosFace embedding with optional drift augmentation,
per-member frame embeddings, prototype refresh, cross-fitted calibration, decision rules."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import os, numpy as np, time

class CosFace32:
    def __init__(self, d_in, n_cls, h=(128, 128), d_emb=64, s=16.0, m=0.25, lr=2e-3, epochs=30, batch=256, seed=0, wd=1e-4,
                 aug=None, cosine_lr=False):
        self.s, self.m, self.lr0, self.epochs, self.batch, self.wd = np.float32(s), np.float32(m), lr, epochs, batch, np.float32(wd)
        self.aug = None if aug is None else np.asarray(aug, np.float32); self.cosine_lr = cosine_lr
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
    def embed(self, X, bs=65536):
        X = X.astype(np.float32); return np.concatenate([self._fwd(X[i:i + bs])[0] for i in range(0, len(X), bs)])
    def fit(self, X, y):
        X = X.astype(np.float32); n = len(X); ncls = self.P['C'].shape[1]; Y = np.eye(ncls, dtype=np.float32)[y]
        steps = self.epochs * int(np.ceil(n / self.batch)); st = 0
        for ep in range(self.epochs):
            perm = self.rng.permutation(n)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]; xb, Yb = X[idx], Y[idx]; B = np.float32(len(idx))
                if self.aug is not None:
                    xb = xb + (self.rng.standard_normal(xb.shape).astype(np.float32) * self.aug)
                self.lr = self.lr0 * (0.5 * (1 + np.cos(np.pi * st / steps)) if self.cosine_lr else 1.0); st += 1
                u, ne, cache = self._fwd(xb); C = self.P['C']; nc = np.linalg.norm(C, axis=0, keepdims=True) + 1e-12; Cn = C / nc
                z = self.s * (u @ Cn - self.m * Yb); z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
                dcos = self.s * (p - Yb) / B; du = dcos @ Cn.T; dCn = u.T @ dcos
                G = {'C': (dCn - Cn * (Cn * dCn).sum(0, keepdims=True)) / nc}; da = (du - u * (u * du).sum(1, keepdims=True)) / ne
                for li in range(self.nl - 1, -1, -1):
                    z_, _ = cache[li + 1]; dz_ = da if li == self.nl - 1 else da * (z_ > 0)
                    a_prev = cache[li] if li == 0 else cache[li][1]
                    G[f'W{li}'] = a_prev.T @ dz_ + self.wd * self.P[f'W{li}']; G[f'b{li}'] = dz_.sum(0); da = dz_ @ self.P[f'W{li}'].T
                self._adam(G)
        return self
    def _adam(self, G, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1; c1 = 1 - b1 ** self.t; c2 = 1 - b2 ** self.t
        for k, g in G.items():
            self.Mo[k] *= b1; self.Mo[k] += (1 - b1) * g; self.Vo[k] *= b2; self.Vo[k] += (1 - b2) * g * g
            self.P[k] -= (self.lr * (self.Mo[k] / c1) / (np.sqrt(self.Vo[k] / c2) + eps)).astype(np.float32)

def _train(args):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core2 import CosFace32 as CF
    F, y, n, seed, kw = args; m = CF(F.shape[1], n, seed=seed, **kw).fit(F, y); return m.P

def train_members(F, y, n, seeds, kw, workers=2):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as ex:
        Ps = list(ex.map(_train, [(F, y, n, s, kw) for s in seeds]))
    ms = []
    for P in Ps:
        m = CosFace32(F.shape[1], n, **{k: v for k, v in kw.items()}); m.P = P; ms.append(m)
    return ms

# ------------------------------------------------------------------ prototypes, scores, calibration
def unit(a, ax=-1):
    return a / (np.linalg.norm(a, axis=ax, keepdims=True) + 1e-12)

def class_means(U, y, n):
    """U: (E, N, d) member embeddings of frames; returns (E, n, d) unit mean embedding per class."""
    out = np.zeros((U.shape[0], n, U.shape[2]), np.float32)
    for j in range(n):
        m = y == j
        if m.any(): out[:, j] = U[:, m].mean(1)
    return unit(out)

def session_cos(Us, protos):
    """Us: (E, S, L, d) session-frame embeddings; protos: (E, n, d) -> (S, n) mean cosine over members and frames."""
    return np.einsum('esld,end->sn', Us, protos, optimize=True) / (Us.shape[0] * Us.shape[2])

def margin(C):
    srt = np.sort(C, 1); top, sec = srt[:, -1:], srt[:, -2:-1]; return C - np.where(C >= top, sec, top)

# ------------------------------------------------------------------ local (nearest-exemplar) prototypes
def pad_refs(R, yr, n):
    """R (E,M,d) reference embeddings with labels yr -> (E,n,m,d) padded per class + mask (n,m)."""
    cnt = np.bincount(yr, minlength=n); m = int(cnt.max()); E, _, d = R.shape
    out = np.zeros((E, n, m, d), np.float32); mask = np.zeros((n, m), bool)
    for j in range(n):
        ix = np.flatnonzero(yr == j); out[:, j, :len(ix)] = R[:, ix]; mask[j, :len(ix)] = True
    return out, mask

def local_cos(Uq, Rp, mask, k, chunk=2048):
    """Uq (E,S,L,d); Rp (E,n,m,d) padded references -> (S,n): for every frame and member, mean of the k largest cosines to the
    references of each identity; averaged over the session's frames and the members."""
    E, S, L, d = Uq.shape; n, m = mask.shape; out = np.zeros((S, n), np.float32); neg = np.float32(-4.0)
    for e in range(E):
        Q = Uq[e].reshape(-1, d); Re = Rp[e].reshape(n * m, d)
        for i in range(0, len(Q), chunk):
            G = (Q[i:i + chunk] @ Re.T).reshape(-1, n, m); G = np.where(mask[None], G, neg)
            top = -np.partition(-G, k - 1, axis=2)[:, :, :k] if k < m else G
            v = top.mean(2)                                   # (b, n)
            np.add.at(out, np.arange(i, i + len(v)) // L, v / (E * L))
    return out

def centroids(R, yr, n, K, iters=15, seed=0):
    """Per-identity spherical k-means on reference embeddings (E,M,d) -> (E, n*K, d), labels (n*K,)."""
    rng = np.random.default_rng(seed); E, _, d = R.shape; C = np.zeros((E, n * K, d), np.float32)
    for e in range(E):
        for j in range(n):
            X = R[e, yr == j]; c = X[rng.choice(len(X), K, replace=len(X) < K)].copy()
            for _ in range(iters):
                a = (X @ c.T).argmax(1)
                for q in range(K):
                    if (a == q).any(): c[q] = unit(X[a == q].mean(0))
            C[e, j * K:(j + 1) * K] = c
    return C, np.repeat(np.arange(n), K)
