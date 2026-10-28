# RAPID — Rival-Aware Prototype IDentity verification (earlier design)

> **Note.** This folder holds the earlier RAPID design (mean prototypes, floor rule) and its comparison with the published forest. The verifier evaluated in the current paper, with recent enrolment, local prototypes and the identification margin, is in [`v2/`](v2/README.md); `verify_numbers.py` here checks the earlier paper version only.


| File | Purpose |
|---|---|
| `rapid_core.py`, `partrain.py` | RAPID in NumPy (float32 CosFace members, rival margin), parallel member training |
| `evalkit.py` | 2026 SMoRFFI baseline (per-frame RF128), decision rules, all metrics, operating curves, latency |
| `common.py` | SMoRFFI loading (path from `SMORFFI_FEATURES`) |
| `run_rapid_protocol.py` | frozen protocol on SMoRFFI, 3 seeds × 2 role orders → `protocol_rapid.json`, `scores_*.npz` |
| `analyze_rapid.py` | medians, paired wins, Pareto rule, frontiers, matched points → `rapid_summary.json` |
| `ablate_cost.py`, `rf_latency.py` | component ablation and host cost → `ablate_cost.json`, `rf_latency.json` |
| `fig_rapid.py` | Figures 9–11 (and extras) → `../figures/` |
| `verify_numbers.py` | re-derives Table 10 from the result files; with `PAPER_TEX` set, checks each row against the paper |
| `run_on_your_data.py`, `iq_features.py` | run the whole comparison on your own features, 802.11 preambles or raw I/Q windows |
| `FREEZE.md`, `dev/` | design decisions and the development-only experiments behind them (the test block was not read) |

Run every script from the repository root, for example `python rapid/fig_rapid.py`. Only seed-0 score files are kept in the
repository; `run_rapid_protocol.py` regenerates all of them.
