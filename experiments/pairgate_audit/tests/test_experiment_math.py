import sys
from pathlib import Path
import numpy as np
from scipy.stats import beta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment import threshold_genuine,matrices,group_score

def test_threshold_only_uses_correct_claim_column():
 p=np.array([[.90,.1],[.80,.2],[.2,.8],[.1,.9]])
 y=np.array([0,0,1,1]);t=threshold_genuine(p,y,2,.5)
 assert abs(t[0]-.9)<1e-10 and abs(t[1]-.9)<1e-10

def test_pair_direction_and_diagonal():
 p=np.array([[.95,.96],[.97,.02],[.20,.99],[.30,.98]])
 y=np.array([0,0,1,1]); g,pair=matrices(p,y,np.array([.90,.90]))
 assert np.isnan(pair[0,0]) and np.isnan(pair[1,1])
 assert pair[0,1]==.5 and pair[1,0]==0

def test_gate_sets_ineligible_target_acceptance_to_zero():
 p=np.array([[.95,.99],[.95,.91],[.8,.95],[.96,.98]])
 y=np.array([0,0,1,1]);g,pair=matrices(p,y,np.array([.9,.9]),np.array([True,False]))
 assert pair[0,1]==0 and np.sum(g[y==1])==0

def test_causally_grouped_windows():
 p=np.array([[.1,.9],[.3,.7],[.8,.2],[.6,.4]])
 y=np.array([0,0,1,1]);g,z=group_score(p,y,2)
 assert g.shape==(2,2) and np.allclose(g,[[.2,.8],[.7,.3]])
 assert np.array_equal(z,[0,1])

def test_binomial_95_upper_for_zero_errors():
 # Standard one-sided CP bound; no attack independence is asserted for current capture.
 n=37;P=26*121;alpha=.05
 bound=1-(alpha/P)**(1/n)
 assert .25<bound<.28

def test_temporal_sample_limit_not_certifiable():
 import math
 p=122*121;alpha=.05;eps=.01
 n=math.ceil(math.log(alpha/p)/math.log(1-eps))
 assert n>1000
