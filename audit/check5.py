import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, numpy as np, pickle
sys.path.insert(0,'.')
from prc import *
sets,n,sha=data_windows(str(DATA_FILE))
sc=StandardScaler().fit(sets['train'][0]); sets={k:(sc.transform(x),y) for k,(x,y) in sets.items()}
(xdev,ydev),(xcal,ycal),(xte,yte)=[sets[k] for k in ('dev','cal','test')]
for m in ('MLP64','RF128'):
    d=np.load(f'probs_{m}.npz'); pdev,pcal,pte=d['pdev'],d['pcal'],d['pte']
    ver,_=pickle.load(open(f'ver_{m}.pkl','rb')); r=np.load(f'resid_{m}.npz'); rcal,rte=r['rcal'],r['rte']
    thr0=genuine_thr(pcal,ycal,n,0.9); b=np.array([v[1] for v in ver])
    g,p=accept_matrix(pte,yte,thr0,binary_ok=rte>=b[None,:])
    tt=np.array([g[yte==t].mean() for t in range(n)])
    # CAL-only predicted TAR (no test info): backbone & residual on cal genuines
    acal=(pcal>=thr0[None,:])&(rcal>=b[None,:]); ct=np.array([acal[ycal==t,t].mean() for t in range(n)])
    print('==',m,'PAIR-RESID in-sample beta')
    for name,keep in [('test TAR>0 (oracle)',tt>0),('test TAR>=0.5 (oracle)',tt>=.5)]+[(f'CAL TAR>={c}',ct>=c) for c in (0.04,0.2,0.4,0.5,0.6)]:
        w=np.nanmax(p[:,keep]) if keep.any() else np.nan
        print('  %-24s kept %3d  testTAR(all IDs) %.3f  TAR on kept %.3f  worst FAR on kept %.3f'%(name,keep.sum(),(g*keep[yte]).mean(),g[keep[yte]].mean(),w))
    # gate sweep for comparison
    _,dp=accept_matrix(pdev,ydev,thr0); dw=np.nanmax(dp,axis=0)
    g0,p0=accept_matrix(pte,yte,thr0)
    for gm in (0,0.03,0.05,0.1,0.2):
        k=dw<=gm+1e-12; print('  GATE g=%.2f kept %3d  testTAR(all) %.3f  TAR on kept %.3f  worst %.3f'%(gm,k.sum(),(g0*k[yte]).mean(),g0[k[yte]].mean(),np.nanmax(p0[:,k])))
