"""Paired per-target significance of RAPID against the equal-data CosFace prototype verifier (RAPID's first ensemble member,
mean prototypes of the recent enrolment block, identification rule), the strongest baseline of the extended comparison.
Member 0 is retrained exactly as in the frozen evaluation (seed 1000*s); RAPID's stored session scores are reused."""
import sys, os, json, numpy as np
os.environ.setdefault('SMORFFI_FEATURES','features.npz')   # set to the SMoRFFI feature file
import pathlib as _pl
_ROOT = str(next(p for p in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents] if (p / 'paths.py').exists()))
R=_ROOT
for p in (R+'/rapid/v2', R+'/rapid', R): sys.path.insert(0,p)
from common import load_frames
from evalkit import windows, accept
from core2 import CosFace32, class_means, margin
from final_runner import BLK, ORDERS, CFG, frames_in
from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest
SC=R+'/rapid/v2/scores'
def main():
    x,y,row,n=load_frames(); W={k:windows(x,y,row,n,*v) for k,v in BLK.items()}; Y={k:W[k][1] for k in W}
    ftr,ytr,_=frames_in(x,y,row,0,600); D={'RAPID':[],'CF':[]}; runs=[]
    for seed in (0,1,2):
        for order,(cal,att) in ORDERS.items():
            fc,yc,_=frames_in(x,y,row,*BLK[cal]); F1,Y1=np.r_[ftr,fc],np.r_[ytr,yc]; sc=StandardScaler().fit(F1)
            kw=dict(h=tuple(CFG['h']),d_emb=CFG['d_emb'],epochs=CFG['epochs'],m=CFG['m'],s=CFG['s'])
            mem=CosFace32(F1.shape[1],n,seed=1000*seed,**kw).fit(sc.transform(F1).astype(np.float32),Y1)
            P=class_means(np.stack([mem.embed(sc.transform(fc).astype(np.float32))]),yc,n)[0]
            def cf(S): Z=sc.transform(S.reshape(-1,S.shape[2])).astype(np.float32); return (mem.embed(Z)@P.T).reshape(len(S),S.shape[1],-1).mean(1)
            for name,S,thv in (('CF',{k:margin(cf(W[k][0])) for k in (att,'test')},0.0),
                               ('RAPID',{k:margin(v) for k,v in np.load(f'{SC}/RAPID_s{seed}_{order}.npz').items() if k in (att,'test')},CFG['delta'])):
                th=np.full(n,1e-12+thv); tar,Pm,_=accept(S['test'],Y['test'],n,th); _,Pd,_=accept(S[att],Y[att],n,th)
                sel=np.nanargmax(np.where(np.isfinite(Pd),Pd,-1),0); D[name].append((tar,Pm[sel,np.arange(n)]))
            runs.append(dict(seed=seed,order=order,CF_TAR=float(D['CF'][-1][0].mean()),CF_dc=float(D['CF'][-1][1].mean())))
            print(runs[-1],flush=True)
    rng=np.random.default_rng(2026); out={}
    for k,name in ((0,'TAR'),(1,'devchosen')):
        d=np.stack([D['RAPID'][r][k]-D['CF'][r][k] for r in range(6)],1); tm=d.mean(1)
        boots=[tm[rng.integers(0,n,n)].mean() for _ in range(10000)]
        better=int((tm>0).sum()) if name=='TAR' else int((tm<0).sum()); worse=int((tm<0).sum()) if name=='TAR' else int((tm>0).sum())
        out[name]=dict(mean=float(tm.mean()),lo=float(np.quantile(boots,.025)),hi=float(np.quantile(boots,.975)),better=better,worse=worse,ties=int(n-better-worse),sign_p=float(binomtest(better,better+worse,.5).pvalue) if better+worse else 1.0)
        print(name,out[name],flush=True)
    json.dump(dict(stats=out,runs=runs),open('cfeq_stats.json','w'),indent=1)
if __name__=='__main__': main()
