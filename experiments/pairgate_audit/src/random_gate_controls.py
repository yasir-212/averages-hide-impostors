"""Descriptive random subset controls, matched target coverage and frozen heldout profiles.
This test-label-based comparison is a *diagnostic*; it never selects the gate.
"""
import json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
output=[]
for name in ['RF128','MLP64']:
 info=json.loads((ROOT/f'results/device_risks_{name}.json').read_text())[name]
 dr=np.asarray(info['dev_pair_max']);worst=np.asarray(info['test_pair_max']);pool=np.asarray(info['test_pair_pooled_by_target']);good=np.asarray(info['test_genuine_TAR_by_target'])
 n=len(dr);rng=np.random.default_rng(20261012)
 for r in [0.,.1]:
  eligible=dr<=r+1e-12;m=int(eligible.sum())
  measured={'worst_pair':float(worst[eligible].max()) if m else None,'FAR_all':float(pool[eligible].sum()/n),'TAR_all':float(good[eligible].sum()/n),'covered_targets':m}
  runs=10000
  perm=np.argsort(rng.random((runs,n)),axis=1)[:,:m]
  rw=worst[perm].max(axis=1);rf=pool[perm].sum(axis=1)/n;rt=good[perm].sum(axis=1)/n
  # not hypothesis testing; these are random-comparator descriptive frequencies.
  output.append({'model':name,'risk_threshold':r,'eligible_targets':m,'development_gate_worst_test':measured['worst_pair'],
                 'development_gate_pooled_test':measured['FAR_all'],'development_gate_total_TAR':measured['TAR_all'],
                 'random_comparator_draws':runs,'random_median_worst_test':float(np.median(rw)),
                 'random_p10_worst_test':float(np.quantile(rw,.1)), 'random_p90_worst_test':float(np.quantile(rw,.9)),
                 'fraction_random_worst_no_worse_than_gate':float(np.mean(rw <= measured['worst_pair']+1e-12)),
                 'random_median_pooled_test':float(np.median(rf)), 'random_median_total_TAR':float(np.median(rt)),
                 'fraction_random_joint_worst_and_tar_better':float(np.mean((rw <= measured['worst_pair']+1e-12)&(rt >= measured['TAR_all']-1e-12)))})
with open(ROOT/'results/random_coverage_control.csv','w',newline='') as f:
 w=csv.DictWriter(f,output[0].keys());w.writeheader();w.writerows(output)
for r in output:print(r)
