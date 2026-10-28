"""Pre-registered robustness test. Rule fixed BEFORE running:
 R1: keep t iff CAL genuine pass (backbone+residual, in-sample beta) >= 0.5 AND CAL worst impostor FAR == 0
 R2: keep t iff CAL genuine pass >= 0.5
 Compare to PAIR-GATE gamma=0 (dev worst FAR == 0) on the same backbone/split.
Variations: backbone seed x residual seed (3), roles: orig (dev=600-750, cal=750-850) and swap."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import sys, numpy as np, json
sys.path.insert(0,'.')
import prc
from prc import *
sets,n,sha=data_windows(str(DATA_FILE))
sc=StandardScaler().fit(sets['train'][0]); sets={k:(sc.transform(x),y) for k,(x,y) in sets.items()}
xtr,ytr=sets['train']; xte,yte=sets['test']
rows=[]
for mname in ('MLP64','RF128'):
  for seed in (0,1,2):
    m=make_model(mname); m.set_params(random_state=[31,67][mname=='MLP64']+100*seed); m.fit(xtr,ytr)
    P={k:m.predict_proba(sets[k][0]) for k in ('dev','cal','test')}
    for roles in ('orig','swap'):
        D,C=('dev','cal') if roles=='orig' else ('cal','dev')
        xd,yd=sets[D]; xc,yc=sets[C]; pd,pc,pt=P[D],P[C],P['test']
        thr0=genuine_thr(pc,yc,n,0.9)
        riv=hard_rivals(pd,yd,thr0,n,5)
        ver=fit_resid_verifiers(xd,yd,pd,thr0,n,riv,'hard',seed=31+1000*seed)
        okc=binary_ok_resid(ver,xc,pc,n); okt=binary_ok_resid(ver,xte,pt,n)
        gc,pcm=accept_matrix(pc,yc,thr0,binary_ok=okc); gt,ptm=accept_matrix(pt,yte,thr0,binary_ok=okt)
        ct=np.array([gc[yc==t].mean() for t in range(n)]); cw=np.nanmax(pcm,axis=0)
        _,dp=accept_matrix(pd,yd,thr0); gate=np.nanmax(dp,axis=0)<=1e-12
        gb,pb=accept_matrix(pt,yte,thr0)
        res={'model':mname,'seed':seed,'roles':roles}
        for name,keep,g,p in [('GATE',gate,gb,pb),('R1',(ct>=.5)&(cw==0),gt,ptm),('R2',ct>=.5,gt,ptm)]:
            res[name]=dict(k=int(keep.sum()),tar_all=float((g*keep[yte]).mean()),tar_kept=float(g[keep[yte]].mean()) if keep.any() else 0,
                           worst=float(np.nanmax(p[:,keep])) if keep.any() else 0)
        rows.append(res)
        print(f"{mname} s{seed} {roles:4s} | "+" | ".join(f"{k}: {res[k]['k']:3d} ids, TARall {res[k]['tar_all']:.3f}, worst {res[k]['worst']:.3f}" for k in ('GATE','R1','R2')),flush=True)
json.dump(rows,open('check7.json','w'),indent=1)
