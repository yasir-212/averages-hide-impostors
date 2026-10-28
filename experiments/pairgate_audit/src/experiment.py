#!/usr/bin/env python3
"""Measured SMoRFFI offline claimed-identity audit.

Frozen chronological splits and model specifications from the 11 October audit.
All target selection / thresholds exclude the test partition. Raw features are
not redistributed in the release. For controlled evaluation only.
"""
import argparse,csv,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA=str(Path(__file__).resolve().parents[1]/'input/features_v1.npz')
MODEL_SPEC={
 'RF128': lambda: RandomForestClassifier(n_estimators=128,min_samples_leaf=2,random_state=31,n_jobs=-1),
 'ET96': lambda:ExtraTreesClassifier(n_estimators=96,min_samples_leaf=2,random_state=31,n_jobs=-1),
 'MLP64': lambda:MLPClassifier(hidden_layer_sizes=(64,),activation='relu',alpha=.1,max_iter=70,learning_rate_init=.001,early_stopping=True,validation_fraction=.12,n_iter_no_change=8,random_state=67,batch_size=256)
}
SPLITS={'train':(0,600),'dev':(600,750),'cal':(750,850),'test':(850,999)}

def data_windows(datafile):
 d=np.load(datafile); X,Y,R=d['x'],d['y'],d['row']
 valid=np.array([j for j in range(len(d['labels'])) if (R[Y==j]>=998).any()]); good=np.isin(Y,valid);X,Y,R=X[good],np.searchsorted(valid,Y[good]),R[good]
 n=len(valid); ret={}
 for segment,(lo,hi) in SPLITS.items():
  xs,ys=[],[]
  for j in range(n):
   ix=np.flatnonzero((Y==j)&(R>=lo)&(R<hi));ix=ix[np.argsort(R[ix])]
   for pos in range(0,len(ix)-3,4):
    ids=ix[pos:pos+4]
    if not np.all(np.diff(R[ids])==1): continue
    z=X[ids];v=np.r_[np.mean(z,axis=0),np.std(z,axis=0)]
    if np.all(np.isfinite(v)):xs.append(v);ys.append(j)
  ret[segment]=(np.asarray(xs),np.asarray(ys))
 return ret,n,valid

def threshold_genuine(p,y,n,q=0.90):
 return np.asarray([np.quantile(p[y==j,j],1-q,method='higher') for j in range(n)])

def matrices(p,y,thr,gate=None):
 n=p.shape[1]; target_gate=np.ones(n,dtype=bool) if gate is None else gate
 accepted=(p>=thr[None,:])&target_gate[None,:]
 genuine=accepted[np.arange(len(y)),y]
 a=accepted.copy();a[np.arange(len(y)),y]=False
 pair=np.stack([a[y==s,:].mean(axis=0) for s in range(n)])
 np.fill_diagonal(pair,np.nan)
 return genuine,pair

def risk_metric(genuine,pair,target_gate=None):
 n=len(genuine) # number of trial windows
 worst=np.nanmax(pair,axis=0)
 return {'TAR_total':float(genuine.mean()),'crossclaim_FAR_pooled':float(np.nanmean(pair)),
         'crossclaim_FAR_maxpair':float(np.nanmax(pair)) if np.isfinite(pair).any() else None,
         'mean_max_source_to_target_FAR':float(np.nanmean(worst)),
         'p99_pair_FAR':float(np.nanquantile(pair,.99)) if np.isfinite(pair).any() else None,
         'eligible_targets':int(np.sum(target_gate)) if target_gate is not None else pair.shape[0],
         'target_coverage':float(np.mean(target_gate)) if target_gate is not None else 1.0}

def worst_source_by_target(p,y,thr):
 g,pair=matrices(p,y,thr)
 n=len(thr)
 return pair, np.asarray([np.nanargmax(pair[:,t]) for t in range(n)],dtype=int)

def group_score(p,y,k):
 # Every source has the same count and is contiguous in chronological order.
 n=p.shape[1];parts=[];labels=[]
 for s in range(n):
  pp=p[y==s]
  for a in range(0,len(pp)-k+1,k):
   parts.append(pp[a:a+k].mean(axis=0));labels.append(s)
 return np.asarray(parts),np.asarray(labels)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--data',default=DEFAULT_DATA);parser.add_argument('--out',default=str(Path(__file__).resolve().parents[1]));parser.add_argument('--model',choices=list(MODEL_SPEC),required=True);args=parser.parse_args()
 out=Path(args.out);(out/'results').mkdir(exist_ok=True,parents=True)
 if not Path(args.data).is_file():raise SystemExit('Missing measured source. Pass --data with the SMoRFFI feature file (see data/README.md)')
 dataset_hash=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(); print('dataset_sha256',dataset_hash,flush=True)
 sets,n,valid=data_windows(args.data)
 scaler=StandardScaler().fit(sets['train'][0]);sets={k:(scaler.transform(x),y) for k,(x,y) in sets.items()}
 print('window shapes', {k:(v[0].shape,len(np.unique(v[1]))) for k,v in sets.items()},flush=True)
 results=[];gating_rows=[];duration=[];profile={}
 for name,model_ctor in MODEL_SPEC.items():
  if name != args.model:continue
  model=model_ctor();start=time.perf_counter();model.fit(*sets['train']);fit_s=time.perf_counter()-start
  probabilities={k:model.predict_proba(v[0]) for k,v in sets.items() if k!='train'}
  for k,v in probabilities.items():
   assert v.shape[1]==n and np.isfinite(v).all(),(name,k)
  ydev=sets['dev'][1];ycal=sets['cal'][1];ytest=sets['test'][1]
  pdev,pcal,ptest=[probabilities[k] for k in ('dev','cal','test')]
  threshold=threshold_genuine(pcal,ycal,n)
  genuine,pair=matrices(ptest,ytest,threshold)
  row={'model':name,'policy':'genuine_calibrated','strategy_param':'0.90','genuine_calibration_count':int(np.bincount(ycal).min()),'test_count':int(np.bincount(ytest).min()),**risk_metric(genuine,pair)}
  results.append(row)
  devpair,_=worst_source_by_target(pdev,ydev,threshold)
  dev_target_worst=np.nanmax(devpair,axis=0)
  sorted_targets=np.argsort(dev_target_worst)
  # Entire gate selection is frozen from dev and calibration; test never consulted.
  for gate_r in [0.,.05,.10,.20,.30,.40,.50,.70,1.0]:
   gate=dev_target_worst<=gate_r+1e-12
   gen,mat=matrices(ptest,ytest,threshold,gate)
   row={'model':name,'policy':'dev_worstsource_target_gate','strategy_param':str(gate_r),
        'genuine_calibration_count':int(np.bincount(ycal).min()),'test_count':int(np.bincount(ytest).min()),
        **risk_metric(gen,mat,gate)}
   results.append(row)
   # evaluate worst only among *eligible* targets. Disabled targets require fallback.
   if gate.any():
    row['eligible_worst_pair_FAR']=float(np.nanmax(mat[:,gate]))
    row['eligible_genuine_TAR']=float(np.mean((ptest[np.arange(len(ytest)),ytest]>=threshold[ytest])[gate[ytest]]))
   else:row['eligible_worst_pair_FAR']=None;row['eligible_genuine_TAR']=None
  # Threshold hardening against the hardest development-source, per claimed identity.
  # This is an empirical development heuristic and has no certified population FAR.
  dev_by_source=np.stack([pdev[ydev==s,:] for s in range(n)])
  for r in [.05,.10,.20]:
   limits=np.quantile(dev_by_source,1-r,axis=1,method='higher')
   np.fill_diagonal(limits,-np.inf)
   th2=np.maximum(threshold, np.max(limits,axis=0))
   gen,mat=matrices(ptest,ytest,th2)
   results.append({'model':name,'policy':'dev_hardest_source_threshold', 'strategy_param':str(r),
                   'genuine_calibration_count':int(np.bincount(ycal).min()),'test_count':int(np.bincount(ytest).min()),
                   **risk_metric(gen,mat)})
  # Holdout selection of the best source for each target on dev (no test oracle).
  selected=np.array([np.nanargmax(devpair[:,t]) for t in range(n)])
  pred_test_pair=np.nanmean(pair[selected,np.arange(n)])
  result={'model':name,'mean_dev_chosen_source_FAR':float(pred_test_pair),
          'mean_oracle_test_source_FAR':float(np.nanmean(np.nanmax(pair,axis=0))),
          'number_target_worst_source_FAR_1':int(np.sum(np.nanmax(pair,axis=0)==1)),
          'number_target_dev_chosen_source_FAR_1':int(np.sum(pair[selected,np.arange(n)]==1)),
          'target_number':n,'source_number':n,'test_trials_per_source':int(np.bincount(ytest).min())}
  gating_rows.append(result)
  # group longer temporal blocks; includes radio/frame count but not time or energy.
  for k in [1,2,4]:
   ca,cy=group_score(pcal,ycal,k);te,ty=group_score(ptest,ytest,k)
   t=threshold_genuine(ca,cy,n)
   gen,mat=matrices(te,ty,t)
   results.append({'model':name,'policy':'nonoverlap_aggregated_k','strategy_param':str(k),
                   'genuine_calibration_count':int(np.bincount(cy).min()),'test_count':int(np.bincount(ty).min()),
                   **risk_metric(gen,mat)})
  duration.append({'model':name,'training_seconds':round(fit_s,3),'pickle_bytes':len(pickle.dumps(model))})
  profile[name]={'dev_pair_max':dev_target_worst.tolist(),'test_pair_max':np.nanmax(pair,axis=0).tolist(),
                 'test_pair_pooled_by_target':np.nanmean(pair,axis=0).tolist(),
                 'test_genuine_TAR_by_target':[float(genuine[ytest==j].mean()) for j in range(n)],
                 'test_pooled_pair_max':float(np.nanmean(np.nanmax(pair,axis=0)))}
  print('trained',name,round(fit_s,2), 'baseline TAR,FAR,worst=',round(float(genuine.mean()),4),round(float(np.nanmean(pair)),4),float(np.nanmax(pair)),flush=True)
 def write_csv(path,rows):
  keys=list(dict.fromkeys(k for v in rows for k in v))
  with open(path,'w',newline='') as f:
   w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
 write_csv(out/('results/risk_policy_'+args.model+'.csv'),results)
 write_csv(out/('results/development_chosen_attacker_'+args.model+'.csv'),gating_rows)
 write_csv(out/('results/model_sizes_'+args.model+'.csv'),duration)
 (out/('results/device_risks_'+args.model+'.json')).write_text(json.dumps(profile,indent=2))
 (out/'results/provenance.json').write_text(json.dumps({'date':'2026-10-11','source_sha256':dataset_hash,'splits':SPLITS,'window_size_frames':4,'n_devices':n,'n_pairs':n*(n-1),'results_type':'offline cross-identity claims','selection_frozen_on':'dev 600-749 + independent genuine cal 750-849','tested_on':'850-998'},indent=2))
 print('Saved to',out,flush=True)
if __name__=='__main__':main()
