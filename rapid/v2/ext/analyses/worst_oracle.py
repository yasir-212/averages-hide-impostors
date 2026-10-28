import sys, json, numpy as np
import pathlib as _pl; _ROOT = str(next(p for p in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents] if (p / 'paths.py').exists())); sys.path.insert(0,_ROOT+'/rapid'); sys.path.insert(0,_ROOT)
from evalkit import thr, accept
R=_ROOT+'/rapid/'
n=122; ORD={'orig':('B','A'),'swap':('A','B')}
def lab(S): m=len(S)//n; assert m*n==len(S); return np.repeat(np.arange(n),m)
def margin(C):
    srt=np.sort(C,1); top,sec=srt[:,-1:],srt[:,-2:-1]; return C-np.where(C>=top,sec,top)
out={}
def rec(name, th, S, cal, att):
    yt=lab(S['test']); ya=lab(S[att])
    tar,P,_=accept(S['test'],yt,n,th); _,Pd,_=accept(S[att],ya,n,th)
    Pm=np.where(np.isfinite(P),P,np.nan)
    worst=np.nanmax(Pm); Q=Pm.copy(); Q[85,24]=np.nan; worst_x=np.nanmax(Q)
    Q2=Pm.copy(); Q2[85,:]=np.nan; Q2[:,85]=np.nan; worst_x85=np.nanmax(Q2)
    oracle=np.nanmean(np.nanmax(Pm,0))
    sel=np.nanargmax(np.where(np.isfinite(Pd),Pd,-1),0); dc=P[sel,np.arange(n)].mean()
    out.setdefault(name,[]).append(dict(worst=worst,worst_x=worst_x,worst_x85=worst_x85,oracle=oracle,dc=dc,tar=tar.mean(),ge50=int(np.nansum(Pm>=.5))))
for seed in (0,1,2):
    for order,(cal,att) in ORD.items():
        z=np.load(R+f'v2/scores/RAPID_s{seed}_{order}.npz'); S={k:margin(z[k]) for k in z.files}
        rec('RAPID|id', np.full(n,1e-12+0.02), S, cal, att)
        for b in ('RF-eq','MLP-eq'):
            z=np.load(R+f'v2/scores/{b}_s{seed}_{order}.npz'); P={k:z[k] for k in z.files}
            M={k:margin(v) for k,v in P.items()}
            rec(b+'|id', np.full(n,1e-12), M, cal, att)
            rec(b+'|q', thr(P[cal],lab(P[cal]),n,0.9), P, cal, att)
        z=np.load(R+f'v2/scores/MLP_s{seed}.npz'); P={k:z[k] for k in z.files}; M={k:margin(v) for k,v in P.items()}
        rec('MLP|id', np.full(n,1e-12), M, cal, att); rec('MLP|q', thr(P[cal],lab(P[cal]),n,0.9), P, cal, att)
        z=np.load(R+f'scores_SMoRFFI-RF128_s{seed}.npz'); P={k:z[k] for k in ('A','B','test')}; M={k:margin(v) for k,v in P.items()}
        rec('RF|id', np.full(n,1e-12), M, cal, att); rec('RF|q', thr(P[cal],lab(P[cal]),n,0.9), P, cal, att)
summ={k:{m:float(np.median([r[m] for r in v])) for m in v[0]} for k,v in out.items()}
for k,v in summ.items(): print('%-10s'%k, ' '.join('%s=%.3f'%(m,x) for m,x in v.items()))
json.dump(dict(runs=out,median=summ),open('worst_oracle.json','w'),indent=1,default=float)
