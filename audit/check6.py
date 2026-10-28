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
(xcal,ycal),(xte,yte)=[sets[k] for k in ('cal','test')]
rng=np.random.default_rng(0)
for m in ('MLP64','RF128'):
    d=np.load(f'probs_{m}.npz'); pcal,pte=d['pcal'],d['pte']
    ver,_=pickle.load(open(f'ver_{m}.pkl','rb')); r=np.load(f'resid_{m}.npz'); rcal,rte=r['rcal'],r['rte']
    thr0=genuine_thr(pcal,ycal,n,0.9); b=np.array([v[1] for v in ver])
    gc,pc=accept_matrix(pcal,ycal,thr0,binary_ok=rcal>=b[None,:])
    g,p=accept_matrix(pte,yte,thr0,binary_ok=rte>=b[None,:])
    ct=np.array([gc[ycal==t].mean() for t in range(n)]); cw=np.nanmax(pc,axis=0)
    print('==',m)
    # rule fixed before test: keep target if CAL genuine pass >= 0.5 AND CAL worst impostor FAR == 0
    for name,keep in [('CAL TAR>=0.5',ct>=.5),('CAL worstFAR==0',cw==0),('CAL TAR>=0.5 & worstFAR==0',(ct>=.5)&(cw==0)),('CAL TAR>=0.3 & worstFAR==0',(ct>=.3)&(cw==0))]:
        A=(pte>=thr0[None,:])&(rte>=b[None,:]); A[np.arange(len(yte)),yte]=False
        idx=[np.flatnonzero(yte==s) for s in range(n)]
        W=[]
        for B in range(200):
            P=np.array([A[rng.choice(i,len(i))].mean(0) for i in idx]); np.fill_diagonal(P,np.nan); W.append(np.nanmax(P[:,keep]))
        print('  %-28s kept %3d  allID TAR %.3f  kept TAR %.3f  worst %.3f  CI %.3f-%.3f'%(name,keep.sum(),(g*keep[yte]).mean(),g[keep[yte]].mean(),np.nanmax(p[:,keep]),*np.quantile(W,[.025,.975])))
