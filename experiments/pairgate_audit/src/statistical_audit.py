from math import log,ceil
from scipy.stats import beta
from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[1]
alpha=.05
output={'confidence_level':.95,'primary_result':'measured offline identity claims','full_pairs':122*121,
 'RF_strict_eligible_pairs':26*121,'per_source_test_windows':37,
 'one_pair_RF_strict_worst_empirical':3/37,'one_pair_RF_strict_worst_95_upper_independent':float(beta.ppf(.95,4,37-3)),
 'simultaneous_upper_even_for_all_zero_errors_full':float(1-(alpha/(122*121))**(1/37)),
 'simultaneous_upper_even_for_all_zero_errors_strict_eligible':float(1-(alpha/(26*121))**(1/37)),
 'sample_need_per_source_full_for_1pct_zero':ceil(log(alpha/(122*121))/log(.99)),
 'sample_need_per_source_strict_for_1pct_zero':ceil(log(alpha/(26*121))/log(.99)),
 'warning':'Independence across time is not established; these are idealized binomial sample requirements, NOT confidence intervals valid for correlated capture windows.'}
(ROOT/'results/statistical_limits.json').write_text(json.dumps(output,indent=2))
print(output)
