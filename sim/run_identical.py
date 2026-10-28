"""Theory check (Theorem 3.1): 10% of devices are exact copies of another device, including its multipath channel,
so P_t = P_s for those pairs; every verifier must then accept the twin as often as the genuine device."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, json
sys.path.insert(0, str(_R / 'rapid' / 'sim'))
from run_sim import run_config, DEFAULT
if __name__ == '__main__':
    rows = []
    for seed in (11, 12):
        c = dict(DEFAULT); c.update(twins=0.1, twin_delta=0.0, twin_channel=True); rows += run_config(c, seed)
    json.dump(rows, open('sim_identical.json', 'w'))
    import numpy as np
    for s, ru in (('SMoRFFI-RF128', 'quantile'), ('SMoRFFI-RF128', 'ident'), ('RAPID', 'floor'), ('RAPID', 'ident')):
        rr = [r for r in rows if r['system'] == s and r['rule'] == ru]
        print(s, ru, {k: round(float(np.mean([r[k] for r in rr])), 4) for k in ('TAR', 'worst', 'pooled', 'devchosen', 'acc', 'eq100', 'ge50')})
