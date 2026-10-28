import sys, os, json, time, numpy as np
os.environ.setdefault('SMORFFI_FEATURES','features.npz')   # set to the SMoRFFI feature file
import pathlib as _pl
_ROOT = str(next(p for p in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents] if (p / 'paths.py').exists()))
R=_ROOT
for p in (R+'/rapid/v2', R+'/rapid', R): sys.path.insert(0,p)
import warnings
from common import load_frames
from evalkit import windows, full_metrics, thr
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from final_runner import post, frames_in, BLK, ORDERS, halves, sess_fold
x,y,row,n=load_frames()
W={k:windows(x,y,row,n,*v) for k,v in BLK.items()}; Y={k:W[k][1] for k in W}
ftr,ytr,_=frames_in(x,y,row,0,600)
def mk(seed,mi): return MLPClassifier(hidden_layer_sizes=(256,256),alpha=1e-4,batch_size=256,learning_rate_init=1e-3,max_iter=mi,early_stopping=True,validation_fraction=0.1,n_iter_no_change=8,random_state=7+seed)
def margin(C):
    srt=np.sort(C,1); top,sec=srt[:,-1:],srt[:,-2:-1]; return C-np.where(C>=top,sec,top)
res=[]
seeds=[int(s) for s in sys.argv[1].split(',')] if len(sys.argv)>1 else [0]
orders=sys.argv[2].split(',') if len(sys.argv)>2 else ['orig']
for seed in seeds:
  for order in orders:
    cal,att=ORDERS[order]
    fc,yc,rc=frames_in(x,y,row,*BLK[cal]); F1,Y1=np.r_[ftr,fc],np.r_[ytr,yc]
    sc=StandardScaler().fit(F1)
    for mi in (60,400):
        t0=time.time()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always'); m=mk(seed,mi).fit(sc.transform(F1),Y1)
        conv=not any('ConvergenceWarning' in str(type(ww.message)) or 'converged' in str(ww.message) for ww in w)
        P={k:post(m,sc,W[k][0],n) for k in (att,'test')}
        M={k:margin(v) for k,v in P.items()}
        r=full_metrics(np.full(n,1e-12),M[att],Y[att],M['test'],Y['test'],n)
        res.append(dict(seed=seed,order=order,max_iter=mi,n_iter=int(m.n_iter_),converged=conv,fit_s=time.time()-t0,**{k:r[k] for k in ('TAR','pooled','worst','devchosen','retry20','EER','oracle','acc')}))
        print(res[-1],flush=True)
json.dump(res,open(f'mlp_conv_{"_".join(map(str,seeds))}_{"_".join(orders)}.json','w'),indent=1)
