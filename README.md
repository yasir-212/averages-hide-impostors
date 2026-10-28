# Averages Hide Impostors / RAPID

Code and results for the manuscript **"RAPID: Exposing and Defeating Targeted Impersonation in RF-Fingerprint Authentication through
Provably Exclusive Rival-Aware Verification"** (Yasir Nawaz and Hui Zhang, University of Science and Technology of China; submitted to *ACM Transactions on Privacy and Security*).

An RF-fingerprint verifier that wrongly accepts only 1.5% of impostors on average can still accept its most dangerous impostor on
every attempt. This repository contains the targeted-risk audit that exposes this risk on 122 same-model Wi-Fi transmitters, the
frozen evaluation protocol, and **RAPID**, a rival-aware verifier that provably accepts every transmission for at most one identity.

## Main result

SMoRFFI test block, median of 3 seeds × 2 role orders. Every baseline below is retrained on **exactly the same enrolment frames** as
RAPID and uses the identification rule.

| | SMoRFFI forest | MLP | kNN | GBT | CosFace prototypes | **RAPID** |
|---|---|---|---|---|---|---|
| Pooled FAR | 0.17% | 0.15% | 0.23% | 0.18% | 0.11% | **0.06%** |
| Development-chosen attacker | 8.1% | 7.7% | 8.0% | 8.4% | 5.0% | **2.5%** |
| Retry success, 20 attempts | 36.9% | 35.2% | 48.4% | 38.5% | 26.6% | **13.1%** |
| Pairs with FAR ≥ 50% | 10.5 | 8 | 5 | 10.5 | 4.5 | **2** |
| TAR at FAR 0.1% (one global threshold) | 72.9% | 76.8% | 61.1% | 71.2% | 75.1% | **88.7%** |
| Equal-error rate | 2.76% | 2.43% | 4.49% | 3.02% | 2.93% | **1.97%** |

- **Best on all 17 metrics** against the six configurations of the frozen protocol (SMoRFFI forest as published and retrained,
  retrained MLP, each with two decision rules): genuine acceptance rises from 81.3% to 84.9% and the development-chosen attacker's
  success falls threefold, from 7.7% to 2.5%.
- **The gain comes from the decision rule.** The CosFace prototype verifier uses the embedding of RAPID's first network; RAPID
  halves its attacker success (2.5% against 5.0%) and its 20-attempt retry success (13.1% against 26.6%).
- **The lead holds when the baselines adopt RAPID's rival margin** (Table 12): the retrained forest, MLP and CosFace prototype
  verifier reach TAR 76.7%, 78.3% and 83.3% with attacker success 6.5%, 6.1% and 3.3%, against 84.9% and 2.5% for RAPID.
- **Lightweight:** 1/341 of the forest's memory (11.5 MB against 3,909 MB) and three times faster verification (2.54 ms against 7.74 ms).

Full results: `rapid/v2/final_summary.json`, `rapid/v2/ext/ext_results.json`, `rapid/v2/ext/analyses/rm_test_medians.json`.

## How RAPID works

1. **Enrolment.** Each identity is enrolled with its training frames and a block of recent, authenticated frames.
2. **Embedding.** Three CosFace networks (15→128→128→64, scale 16, margin 0.25) are trained on all enrolment frames.
3. **Local prototypes.** A frame's similarity to identity *j* is the mean cosine to its 3 nearest recent-enrolment exemplars of *j*;
   a 4-frame session's similarity `c_j` is the mean over frames and networks.
4. **Decision.** Accept a claim for *t* iff the rival margin `R_t = c_t − max_{j≠t} c_j ≥ 0.02`. An optional absolute check (`c_t`
   above the 5th percentile of cross-fitted genuine similarities) guards against radios that were never enrolled.
5. **Guarantees.** RAPID accepts every transmission for at most one identity and admits an enrolled impostor only if it resembles
   its target more than its own enrolment, which bounds the confusion of every pair and the success of repeated attempts. The rule
   and its guarantees hold for any representation.

Every design choice was fixed on a development split that never reads the test block; the record is in
[`rapid/v2/FROZEN.json`](rapid/v2/FROZEN.json), `rapid/v2/dev_logs/` and [`rapid/v2/README.md`](rapid/v2/README.md).

## Installation and data

```bash
git clone https://github.com/yasir-212/averages-hide-impostors.git
cd averages-hide-impostors
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # Python 3.10 or newer
export SMORFFI_FEATURES=/path/to/features_v1.npz       # Windows PowerShell: $env:SMORFFI_FEATURES="C:\path\features_v1.npz"
```

The datasets are public and are not redistributed here; [`data/README.md`](data/README.md) explains where to obtain them.

## Reproducing the paper

Table and figure numbers refer to the *ACM TOPS* manuscript; the prefix S refers to its online supplement.

From the saved result files (no retraining):

```bash
python rapid/v2/analyze_v2.py           # medians over the 6 runs and run-by-run comparison with every baseline
python rapid/v2/gen_tables.py table     # rows of the main comparison table (Table 8)
python figures/scripts/fig_ext.py       # Figures 6, 11, 12, 13
python figures/scripts/fig_curves.py    # Figures 7, 8, S2
python sim/gen_sim2.py; python sim/fig_sim2.py                    # Table S8, Figure 14
python numbers/gen_numbers.py; python numbers/gen_numbers_v14.py  # every number quoted in the text
```

From scratch:

| Paper item | Command |
|---|---|
| Tables 4, S1–S2, Figures 2, S1 (targeted-risk audit) | `python audit/check1.py`, `check2.py`, `analyze.py`, `figs.py`; `experiments/` |
| Table 5 (neural verifiers, mitigations) | `python protocol/run_protocol.py` |
| Tables 8, 10, Figures 5, 9 (frozen comparison, ablation) | `python rapid/v2/final_runner.py`, `analyze_v2.py`, `frontier2.py RAPID,RF-eq,MLP-eq,RF,MLP`, `ablation_v2.py`, `fig_rapid2.py` |
| Tables 11, S6, Figures 6, 11 (eight baselines, E and k, session length) | `python rapid/v2/ext/final_ext.py`, `gbt_ext.py`, `ext_analyze.py` |
| Table 9, Figures 7, 8, S2 (significance, retry, score-informed attacker) | `python rapid/v2/ext/stats.py`, `curves.py`, `attacker2.py` |
| Table 12 (rival margin on the baselines) | `python rapid/v2/dev_base_eq.py`, `dev_emb.py`, then `rapid/v2/ext/analyses/rm_dev.py`, `rm_test.py`, `rm_table.py` |
| Sections 5, 6.3, 7.1 (scaling, crossed bootstrap, further analyses) | `python rapid/v2/ext/analyses/scale_bench.py`, `crossed_boot.py`, `mlp_conv.py`, `cfeq_stats.py`, `worst_oracle.py` |
| Figure 12 (enrolment budget, population size) | `python rapid/v2/ext/sweeps.py` |
| Table S7, Figure 13 (cost) | `python rapid/v2/ext/cost_bench.py` |
| Table 13, Figure 10 (unenrolled attackers) | `python rapid/v2/openset.py --split test`, `analyze_openset.py` |
| Table S8, Figures 14, S3 (simulation) | `python sim/run_sim2.py`, `run_sim_beta.py`, `gen_sim2.py`, `fig_sim2.py`; `python figures/scripts/fig_sim.py` |

## Repository layout

| Folder | Content |
|---|---|
| `audit/` | targeted-risk audit of 122 same-model 802.11g transmitters (Section 4, supplement S2) |
| `experiments/` | motivating study: identity gating, residual verifiers, adaptive attacks, INRIA captures (Section 4, supplement S2) |
| `protocol/` | neural verifiers and standard mitigations under the protocol (Section 4.2) |
| `rapid/v2/` | RAPID, equal-data baselines, frozen comparison, unenrolled attackers, design record (Sections 5–7) |
| `rapid/v2/ext/` | eight baselines, rival margin on the baselines, session length, enrolment budget, population size, significance, cost (Section 7) |
| `sim/` | 802.11g preamble simulator and sensitivity study (Section 8, supplement S6) |
| `numbers/` | scripts that compute every number quoted in the text |
| `figures/` | the figures of the paper and their plotting scripts |
| `rapid/` | shared data loading and metrics, and the design history of RAPID |
| `data/`, `SPLITS.md` | how to obtain the datasets; every split, session definition, role order and seed |
