"""SYNTHETIC 802.11g (OFDM, 20 MS/s) preamble simulator with per-device hardware impairments.
This is a controlled simulation for method development and sensitivity analysis. It is NOT measured data.

Per device d (drawn once):   crystal CFO offset (ppm), I/Q gain and phase imbalance, DC offset,
                             Rapp PA (smoothness, saturation), Wiener phase-noise level, TX-filter tap.
Within a capture (per frame): CFO random walk with mean reversion (temperature), small gain drift,
                             random carrier phase, AWGN at the given SNR, static per-device multipath.
Population knob `spread` scales every between-device impairment spread (small = more similar radios).
`twins` = fraction of devices that are near-copies of another device (same production batch)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import numpy as np

FS = 20e6; FC = 2.412e9
_S = np.sqrt(13 / 6) * np.array([0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, 0,
                                0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0])
_L = np.array([1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0,
               1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1], complex)
SC = np.r_[np.arange(-26, 0), np.arange(1, 27)]           # used subcarriers (52)
def _to_time(F53):
    X = np.zeros(64, complex); X[np.arange(-26, 27) % 64] = F53; return np.fft.ifft(X) * np.sqrt(64)
STF = np.tile(_to_time(_S)[:16], 10)                         # 160 samples
LT = _to_time(_L); LTF = np.r_[LT[-32:], LT, LT]             # 160 samples
PRE = np.r_[STF, LTF]; PRE = PRE / np.sqrt(np.mean(np.abs(PRE) ** 2))
LREF = _L[SC + 26]                                           # reference LTF on used subcarriers

def draw_devices(n, rng, spread=1.0, twins=0.0, twin_delta=0.05, twin_channel=False):
    P = dict(ppm=rng.normal(0, 8 * spread, n), g=rng.normal(0, .03 * spread, n), phi=np.deg2rad(rng.normal(0, 2.0 * spread, n)),
             dc=(rng.normal(0, .01 * spread, n) + 1j * rng.normal(0, .01 * spread, n)), p=np.clip(rng.normal(3, .25 * spread, n), 1.2, 8),
             asat=np.clip(rng.normal(1.6, .05 * spread, n), 1.05, 3), pn=np.abs(rng.normal(1.2e-3, .2e-3 * spread, n)),
             tap=(rng.normal(0, .008 * spread, n) + 1j * rng.normal(0, .008 * spread, n)))
    P['chan'] = np.arange(n, dtype=float); k = int(round(twins * n))
    for i in range(k):                                     # device n-1-i becomes a near-copy of device i
        j = n - 1 - i
        for key, v in P.items():
            if key == 'chan': continue
            scale = {'ppm': 8, 'g': .03, 'phi': np.deg2rad(2), 'dc': .01, 'p': .25, 'asat': .05, 'pn': .2e-3, 'tap': .008}[key] * spread
            noise = rng.normal(0, twin_delta * scale) + (1j * rng.normal(0, twin_delta * scale) if np.iscomplexobj(v) else 0)
            v[j] = v[i] + noise
        if twin_channel: P['chan'][n - 1 - i] = P['chan'][i]     # twin also placed where its original is (same multipath)
    return P

def frames(P, d, n_frames, rng, snr_db=20.0, drift_ppm=0.6, drift_tau=150.0, multipath=0.15):
    """Generate n_frames received preambles (n_frames, 320) for device d, in capture order."""
    ppm0 = P['ppm'][d]; ppm = np.empty(n_frames); u = 0.0; a = np.exp(-1 / drift_tau); sd = drift_ppm * np.sqrt(1 - a * a)
    for k in range(n_frames):
        u = a * u + rng.normal(0, sd); ppm[k] = ppm0 + u
    x = np.tile(PRE, (n_frames, 1))
    x = x + P['tap'][d] * np.roll(x, 1, axis=1)                                      # TX filter mismatch
    x = x * (1 + rng.normal(0, .01, (n_frames, 1)))                                    # small gain drift
    r = np.abs(x); x = x * (1 / (1 + (r / P['asat'][d]) ** (2 * P['p'][d])) ** (1 / (2 * P['p'][d])))   # Rapp PA
    g, phi = P['g'][d], P['phi'][d]                                                    # I/Q imbalance
    I, Q = x.real, x.imag; x = (1 + g) * I + 1j * (1 - g) * (Q * np.cos(phi) + I * np.sin(phi))
    x = x + P['dc'][d]
    t = np.arange(x.shape[1]) / FS
    pn = np.cumsum(rng.normal(0, P['pn'][d], x.shape), 1)
    x = x * np.exp(1j * (2 * np.pi * (ppm[:, None] * 1e-6 * FC) * t[None] + pn + rng.uniform(0, 2 * np.pi, (n_frames, 1))))
    cd = P['chan'][d]; h = np.r_[1.0, multipath * np.exp(1j * 2 * np.pi * ((cd * 0.37) % 1)), 0.5 * multipath * np.exp(1j * 2 * np.pi * ((cd * 0.71) % 1))]
    x = np.stack([np.convolve(f, h)[:x.shape[1]] for f in x])
    p = np.mean(np.abs(x) ** 2, 1, keepdims=True); nv = p / 10 ** (snr_db / 10)
    return x + np.sqrt(nv / 2) * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape))

def _katz(z):
    L = np.abs(np.diff(z, axis=1)).sum(1); dmax = np.abs(z - z[:, :1]).max(1); n = z.shape[1] - 1
    return np.log10(n) / (np.log10(n) + np.log10(dmax / (L + 1e-12) + 1e-12))

def features(X):
    """SMoRFFI-style 15 per-frame features from received preambles X (F, 320)."""
    stf, ltf = X[:, :160], X[:, 160:]
    c16 = np.sum(stf[:, 16:144] * np.conj(stf[:, :128]), 1); f_c = np.angle(c16) * FS / (2 * np.pi * 16)
    n = np.arange(320) / FS; Xc = X * np.exp(-1j * 2 * np.pi * f_c[:, None] * n[None])
    l1, l2 = Xc[:, 192:256], Xc[:, 256:320]
    c64 = np.sum(l2 * np.conj(l1), 1); f_f = np.angle(c64) * FS / (2 * np.pi * 64)
    cfo = f_c + f_f; Xc = Xc * np.exp(-1j * 2 * np.pi * f_f[:, None] * n[None]); l1, l2 = Xc[:, 192:256], Xc[:, 256:320]
    fd1, fd2 = _katz(Xc.real), _katz(Xc.imag)
    I, Q = Xc.real, Xc.imag; iq1 = np.sqrt((I ** 2).mean(1) / (Q ** 2).mean(1)) - 1; iq2 = (I * Q).mean(1) / np.sqrt((I ** 2).mean(1) * (Q ** 2).mean(1))
    out = [cfo, f_c, f_f, fd1, fd2, iq1, iq2]
    mm, mv, pm, pv = [], [], [], []
    for s in (l1, l2):
        Y = np.fft.fft(s, axis=1)[:, SC % 64] / np.sqrt(64)
        hgain = (Y * np.conj(LREF)).mean(1, keepdims=True); e = Y / (hgain * LREF)
        mm.append(np.abs(e).mean(1) - 1); mv.append(np.abs(e).var(1)); ph = np.angle(e); pm.append(ph.mean(1)); pv.append(ph.var(1))
    return np.stack(out + [mm[0], mm[1], mv[0], mv[1], pm[0], pm[1], pv[0], pv[1]], 1)

def dataset(n_dev=60, n_frames=500, seed=0, **kw):
    """Returns per-frame features x (n_dev*n_frames, 15), labels y, capture index row."""
    rng = np.random.default_rng(seed); dkw = {k: kw.pop(k) for k in ('spread', 'twins', 'twin_delta', 'twin_channel') if k in kw}
    P = draw_devices(n_dev, rng, **dkw); xs, ys, rs = [], [], []
    for d in range(n_dev):
        xs.append(features(frames(P, d, n_frames, rng, **kw))); ys.append(np.full(n_frames, d)); rs.append(np.arange(n_frames))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(rs)

if __name__ == '__main__':
    import time; t0 = time.time(); x, y, r = dataset(10, 200, seed=1); print(x.shape, time.time() - t0)
    names = ['CFO', 'short_freq', 'long_freq', 'fd1', 'fd2', 'iqi_1', 'iqi_2', 'mm1', 'mm2', 'mv1', 'mv2', 'pm1', 'pm2', 'pv1', 'pv2']
    for i, nm in enumerate(names):
        between = np.std([x[y == d, i].mean() for d in range(10)]); within = np.mean([x[y == d, i].std() for d in range(10)])
        print(f'{nm:10s} between/within = {between / (within + 1e-12):6.2f}   mean {x[:, i].mean():.3g}')
