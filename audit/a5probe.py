import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'audit', _R / 'protocol', _R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import DATA_FILE, FIG_DIR, EMULATED_DIR, PAPER_TEX
if _pl.Path(_sys.argv[0]).resolve() == _pl.Path(__file__).resolve():
    _os.chdir(_HERE)   # relative result files are read and written next to this script
import numpy as np, sys
sys.path.insert(0,'.')
from ras import build_defences, calibrate, accept_defence, centroid_blend, feat_nn_select
def run(d, seed, shift):
    rng=np.random.default_rng(seed)
    A=rng.normal(size=(d,d)); C=A@A.T/d+0.2*np.eye(d)
    L=np.linalg.cholesky(C)
    G=rng.normal(size=(6000,d))@L.T
    mu_a=shift*rng.normal(size=d)/np.sqrt(d)
    Aatk=rng.normal(size=(1500,d))@(L*1.1).T+mu_a
    Rg=np.zeros(len(G)); Ra=np.zeros(len(Aatk))
    tr,ca,te=slice(0,3000),slice(3000,3900),slice(3900,None)
    defs,scaler,_=build_defences(G[tr],Rg[tr],G[ca],seed=seed)
    thr={k:calibrate(v,G[ca],Rg[ca]) for k,v in defs.items()}
    cen=G[tr].mean(0)
    nn,_=feat_nn_select(Aatk,cen,0.10)
    atk={}
    atk['A0 raw']=Aatk
    atk['A5 centroid blend (paper)']=centroid_blend(nn,cen,0.55)
    # defender-aware variants: same 55% budget, but toward random genuine training samples
    pick=G[tr][rng.integers(0,3000,len(nn))]
    atk['A5-s blend to random genuine']=0.45*nn+0.55*pick
    # centroid blend + re-inject genuine-covariance noise to restore typical radius
    noise=(G[tr][rng.integers(0,3000,len(nn))]-cen)
    atk['A5-n centroid blend + genuine noise']=centroid_blend(nn,cen,0.55)+0.55*noise
    out={}
    for k,v in defs.items():
        tm,tb=thr[k]; okg,_=accept_defence(v,G[te],Rg[te],tm,tb)
        out[k]={'TAR':okg.mean()}
        for an,X in atk.items():
            ok,_=accept_defence(v,X,np.zeros(len(X)),tm,tb); out[k][an]=ok.mean()
    return out
for d in (16,32):
  for seed in (1,2):
    o=run(d,seed,6.0)
    print(f'--- d={d} seed={seed}')
    for k in ('OAS','PAIRCONF','PAIRRESID'):
        print(' %-10s '%k+'  '.join(f'{a}: {v:.2f}' for a,v in o[k].items()))

print('--- mechanism check: GENUINE test points pulled 55% toward centroid')
for d in (16,32):
    rng=np.random.default_rng(1)
    A=rng.normal(size=(d,d)); C=A@A.T/d+0.2*np.eye(d); L=np.linalg.cholesky(C)
    G=rng.normal(size=(6000,d))@L.T; R=np.zeros(6000)
    tr,ca,te=slice(0,3000),slice(3000,3900),slice(3900,None)
    defs,_,_=build_defences(G[tr],R[tr],G[ca],seed=1)
    cen=G[tr].mean(0)
    for k in ('OAS','PAIRRESID'):
        tm,tb=calibrate(defs[k],G[ca],R[ca])
        ok,_=accept_defence(defs[k],centroid_blend(G[te],cen,0.55),R[te],tm,tb)
        print(f' d={d} {k}: acceptance of shrunk genuine = {ok.mean():.2f}')
