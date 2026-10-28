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
(xtr,ytr),(xdev,ydev),(xcal,ycal),(xte,yte)=[sets[k] for k in ('train','dev','cal','test')]
for mname in sys.argv[1:]:
    d=np.load(f'probs_{mname}.npz'); pdev,pcal,pte=d['pdev'],d['pcal'],d['pte']
    ver,rivals=pickle.load(open(f'ver_{mname}.pkl','rb'))
    # residual scores on cal and test
    rcal=np.zeros((len(ycal),n)); rte=np.zeros((len(yte),n))
    for t,(clf,b) in enumerate(ver):
        rcal[:,t]=clf.predict_proba(score_aug(xcal,pcal,t))[:,1]
        rte[:,t]=clf.predict_proba(score_aug(xte,pte,t))[:,1]
    print('==',mname)
    print('baseline (vary genuine q):')
    for q in (0.9,0.8,0.7,0.6,0.5,0.45,0.4,0.35,0.3):
        thr=genuine_thr(pcal,ycal,n,q); g,p=accept_matrix(pte,yte,thr)
        print('  q=%.2f TAR %.3f worst %.3f pooled %.5f'%(q,g.mean(),np.nanmax(p),np.nanmean(p)))
    print('PAIR-RESID, beta from CAL (out-of-sample), backbone q=0.9:')
    thr0=genuine_thr(pcal,ycal,n,0.9)
    for keep in (0.95,0.9,0.8,0.7,0.6,0.5):
        b=np.array([np.quantile(rcal[ycal==t,t],1-keep,method='higher') for t in range(n)])
        bok=rte>=b[None,:]
        g,p=accept_matrix(pte,yte,thr0,binary_ok=bok)
        print('  keep=%.2f TAR %.3f worst %.3f pooled %.5f'%(keep,g.mean(),np.nanmax(p),np.nanmean(p)))
    print('PAIR-RESID original in-sample beta:')
    b=np.array([v[1] for v in ver]); g,p=accept_matrix(pte,yte,thr0,binary_ok=rte>=b[None,:]); print('  TAR %.3f worst %.3f'%(g.mean(),np.nanmax(p)))
    print('FAR-cap gamma:')
    for gm in (0.0,0.05,0.1):
        thr=far_capped_thr(pdev,ydev,pcal,ycal,n,gm); g,p=accept_matrix(pte,yte,thr)
        print('  g=%.2f TAR %.3f worst %.3f'%(gm,g.mean(),np.nanmax(p)))
    np.savez(f'resid_{mname}.npz',rcal=rcal,rte=rte)
