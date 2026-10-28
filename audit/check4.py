import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, numpy as np
sys.path.insert(0,'.')
from prc import *
sets,n,sha=data_windows(str(DATA_FILE))
sc=StandardScaler().fit(sets['train'][0]); sets={k:(sc.transform(x),y) for k,(x,y) in sets.items()}
(xdev,ydev),(xcal,ycal),(xte,yte)=[sets[k] for k in ('dev','cal','test')]
for m in ('MLP64','RF128'):
    d=np.load(f'probs_{m}.npz'); pdev,pcal,pte=d['pdev'],d['pcal'],d['pte']
    thr=far_capped_thr(pdev,ydev,pcal,ycal,n,0.0)
    g,p=accept_matrix(pte,yte,thr)
    tt=np.array([g[yte==t].mean() for t in range(n)])
    print(m,'FAR-cap g0: TAR %.3f worst %.3f | thr==1.0: %d | TAR=0: %d | TAR<=25%%: %d'%(g.mean(),np.nanmax(p),(thr>=1-1e-12).sum(),(tt==0).sum(),(tt<=.25).sum()))
    # worst-pair FAR among targets that still accept something
    live=tt>0; print('   worst FAR over targets with TAR>0: %.3f (n=%d)'%(np.nanmax(p[:,live]),live.sum()))
    # baseline at gate-like TAR
    for q in (0.25,0.2,0.15):
        th=genuine_thr(pcal,ycal,n,q); g2,p2=accept_matrix(pte,yte,th); print('   baseline q=%.2f TAR %.3f worst %.3f'%(q,g2.mean(),np.nanmax(p2)))
