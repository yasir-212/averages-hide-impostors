import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, json, numpy as np, time
sys.path.insert(0,'.')
from prc import *
sets,n,sha=data_windows(str(DATA_FILE))
sc=StandardScaler().fit(sets['train'][0]); sets={k:(sc.transform(x),y) for k,(x,y) in sets.items()}
(xtr,ytr),(xdev,ydev),(xcal,ycal),(xte,yte)=[sets[k] for k in ('train','dev','cal','test')]
print('sizes',len(ytr),len(ydev),len(ycal),len(yte), 'n',n, 'test per src', np.bincount(yte).min(), np.bincount(yte).max())
res={}
for mname in sys.argv[1:]:
    m=make_model(mname); m.fit(xtr,ytr)
    pdev,pcal,pte=[m.predict_proba(x) for x in (xdev,xcal,xte)]
    np.savez(f'probs_{mname}.npz',pdev=pdev,pcal=pcal,pte=pte)
    thr0=genuine_thr(pcal,ycal,n,0.9)
    rivals=hard_rivals(pdev,ydev,thr0,n,5)
    ver=fit_resid_verifiers(xdev,ydev,pdev,thr0,n,rivals,'hard')
    import pickle; pickle.dump((ver,rivals),open(f'ver_{mname}.pkl','wb'))
    bok=binary_ok_resid(ver,xte,pte,n)
    g,p=accept_matrix(pte,yte,thr0,binary_ok=bok)
    print(mname,'RESID repro TAR %.3f worst %.3f pooled %.5f'%(g.mean(),np.nanmax(p),np.nanmean(p)))
    # in-sample residual threshold stats
    bth=np.array([v[1] for v in ver if v]); print(' beta_t (in-sample 5th pct) median %.3f min %.3f'%(np.median(bth),bth.min()))
    # residual-only TAR on test genuine (with backbone ignored)
    gr=bok[np.arange(len(yte)),yte].mean(); print(' residual-only genuine pass on test %.3f'%gr)
