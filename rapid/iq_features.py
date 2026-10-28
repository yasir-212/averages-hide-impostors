"""Generic per-window features for raw complex baseband I/Q of any waveform (for corpora that are not 802.11 preambles,
e.g. the INRIA BladeRF windows). 22 features: CFO proxy, I/Q imbalance, DC offset, amplitude/phase-derivative statistics,
PAPR and 8 normalised spectral band powers. For 802.11 OFDM preambles use sim/wifi_sim.features (SMoRFFI-style 15 features)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import numpy as np

def iq_features(X, fs=20e6, bands=8):
    X = np.asarray(X); X = X / (np.sqrt(np.mean(np.abs(X) ** 2, 1, keepdims=True)) + 1e-12)
    I, Q = X.real, X.imag; a = np.abs(X)
    r1 = np.sum(X[:, 1:] * np.conj(X[:, :-1]), 1); cfo = np.angle(r1) * fs / (2 * np.pi)
    dph = np.angle(X[:, 1:] * np.conj(X[:, :-1])) - np.angle(r1)[:, None]
    gain = np.sqrt((I ** 2).mean(1) / ((Q ** 2).mean(1) + 1e-12)) - 1; phase = (I * Q).mean(1) / np.sqrt((I ** 2).mean(1) * (Q ** 2).mean(1) + 1e-12)
    def kurt(v): m = v.mean(1, keepdims=True); s = v.std(1) + 1e-12; return ((v - m) ** 4).mean(1) / s ** 4
    F = np.abs(np.fft.fftshift(np.fft.fft(X, axis=1), axes=1)) ** 2; edges = np.linspace(0, F.shape[1], bands + 1).astype(int)
    bp = np.stack([F[:, edges[b]:edges[b + 1]].sum(1) for b in range(bands)], 1); bp = np.log(bp / bp.sum(1, keepdims=True) + 1e-12)
    feats = [cfo, gain, phase, I.mean(1), Q.mean(1), a.mean(1), a.std(1), kurt(a), dph.std(1), kurt(dph), 10 * np.log10(a.max(1) ** 2 / (a ** 2).mean(1)),
             np.abs(np.fft.fft(a - a.mean(1, keepdims=True), axis=1))[:, 1:4].mean(1), np.mean(np.abs(np.diff(a, axis=1)), 1), np.mean(np.abs(dph), 1)]
    return np.concatenate([np.stack(feats, 1), bp], 1)
