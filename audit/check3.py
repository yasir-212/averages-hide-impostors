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
for mname in ('MLP64','RF128'):
    d=np.load(f'probs_{mname}.npz'); pdev,pcal,pte=d['pdev'],d['pcal'],d['pte']
    ver,rivals=pickle.load(open(f'ver_{mname}.pkl','rb')); r=np.load(f'resid_{mname}.npz'); rcal,rte=r['rcal'],r['rte']
    thr0=genuine_thr(pcal,ycal,n,0.9)
    b=np.array([v[1] for v in ver]); bok=rte>=b[None,:]
    g,p=accept_matrix(pte,yte,thr0,binary_ok=bok)
    tar_t=np.array([g[yte==t].mean() for t in range(n)])
    print('==',mname,'per-target TAR under original PAIR-RESID')
    for th in (0.0,0.05,0.1,0.25):
        print('  targets with TAR<=%.2f: %d/122'%(th,(tar_t<=th+1e-9).sum()))
    print('  targets with TAR>=0.5: %d'%((tar_t>=0.5).sum()))
    # effective coverage: targets with any genuine accept
    print('  per-target TAR quartiles',np.round(np.quantile(tar_t,[0,.25,.5,.75,1]),3))
    # worst pair + bootstrap CI over test windows per source
    rng=np.random.default_rng(0)
    acc=(pte>=thr0[None,:])&bok
    idx_by_s=[np.flatnonzero(yte==s) for s in range(n)]
    def worst(acc_):
        a=acc_.copy(); a[np.arange(len(yte)),yte]=False
        return a
    W=[]
    A=worst(acc)
    for B in range(300):
        rows=[]
        for s in range(n):
            ii=rng.choice(idx_by_s[s],len(idx_by_s[s]),replace=True); rows.append(A[ii].mean(0))
        P=np.array(rows); np.fill_diagonal(P,np.nan); W.append(np.nanmax(P))
    print('  worst-pair FAR bootstrap 95%% CI: %.3f-%.3f'%tuple(np.quantile(W,[.025,.975])))
    # out-of-sample beta sweep further
    for keep in (0.45,0.4,0.35,0.3):
        bb=np.array([np.quantile(rcal[ycal==t,t],1-keep,method='higher') for t in range(n)])
        g2,p2=accept_matrix(pte,yte,thr0,binary_ok=rte>=bb[None,:])
        print('  cal-beta keep=%.2f TAR %.3f worst %.3f'%(keep,g2.mean(),np.nanmax(p2)))
    # gate-equivalent: deny targets with in-sample-like low TAR -> compare to PAIR-GATE
