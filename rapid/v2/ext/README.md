# Extended evaluation of the frozen RAPID design

Everything in this folder evaluates the **frozen** RAPID design of `../FROZEN.json`; nothing here was used to choose any
parameter of RAPID. Baseline hyper-parameters (kNN, gradient-boosted trees) were chosen on the development split only.

| Step | Script | Output |
|---|---|---|
| kNN and GBT selection on the development split | `dev_ext.py`, `dev_gbt.py`, `dev_gbt2.py` | `dev_ext.json`, `dev_gbt*.json`, `dev_choice.json` |
| Refit RAPID and all baselines, store per-frame scores (6 runs) | `final_ext.py`, `gbt_ext.py` | `frames_s*_*.npz`, `gbt_s*_*.npz` (not in the repository: 150 MB each) |
| Extended baselines, session length, ensemble size and exemplars, reproduction check | `ext_analyze.py` | `ext_results.json` |
| Retry curves, margin sweep, impostor tail (from `../scores/`) | `curves.py` | `curves.json` |
| Enrolment budget and population size | `sweeps.py` | `sweeps.json` |
| Paired per-target significance | `stats.py` | `stats.json` |
| Host-cost benchmark (run alone) | `cost_bench.py` | `cost_bench.json` |
| Stronger (score-ranked) development-chosen attacker | `attacker2.py` | `attacker2.json` |

Note: in `sweeps.json`, the global-threshold fields (EER, AUC, TAR@FAR) of the baselines were computed on the identification margin
rather than on the posterior and are not used in the manuscript; the decision-rule metrics (TAR, FAR, attacker, retry) are correct.
