"""SYNTHETIC: margin sensitivity of RAPID at low SNR. Same populations, seeds and protocol as run_sim2.py;
RAPID is evaluated with the frozen margin (0.02, reproduces sim2_results.json) and with beta = 0 (identification rule)."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'rapid' / 'v2', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
import sys, json, gc, numpy as np
import wifi_sim as WS
from evalkit import windows, blocks_for, full_metrics, global_roc
from final_runner import RAPID2, CFG
from run_sim import DEFAULT, NF

def run(cfg, data_seed, model_seed=0, L=4):
    kw = dict(cfg); n = kw.pop('n_dev'); x, y, row = WS.dataset(n, NF, seed=data_seed, **kw); bl = blocks_for(NF); res = []
    tr = (row >= bl['train'][0]) & (row < bl['train'][1]); ftr, ytr = x[tr], y[tr]
    W = {k: windows(x, y, row, n, *bl[k], L=L) for k in ('A', 'B', 'test')}; Y = {k: W[k][1] for k in W}
    for order, (cal, att) in (('orig', ('B', 'A')), ('swap', ('A', 'B'))):
        cm = (row >= bl[cal][0]) & (row < bl[cal][1]); fc, yc = x[cm], y[cm]
        m = RAPID2(n, model_seed, CFG).fit(np.r_[ftr, fc], np.r_[ytr, yc], fc, yc); S = {k: m.scores(W[k][0]) for k in (att, 'test')}
        for beta in (CFG['delta'], 0.0):
            r = full_metrics(np.full(n, 1e-12 + beta), S[att], Y[att], S['test'], Y['test'], n); r.update(global_roc(S['test'], Y['test']))
            r.update(system='RAPID', rule='ident', beta=beta, order=order, L=L, data_seed=data_seed, **cfg); res.append(r)
        del m; gc.collect()
    print(cfg['snr_db'], data_seed, flush=True); return res

if __name__ == '__main__':
    rows = []
    for snr in (5.0, 10.0):
        for s in (11, 12):
            c = dict(DEFAULT); c['snr_db'] = snr; rows += run(c, s)
            json.dump(rows, open(str(_HERE / 'sim_beta_results.json'), 'w'))
