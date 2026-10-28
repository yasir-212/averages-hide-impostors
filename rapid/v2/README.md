# RAPID (current design): recent enrolment, local prototypes, identification margin

This folder holds the verifier evaluated in Sections 5–7 of the paper, its development record and the frozen-protocol results.

## Design (frozen in `FROZEN.json` before the test block was read)

1. **Enrolment.** Every identity is enrolled with its training frames (0–599) and a block of recent frames (the protocol's
   calibration block, 125 frames per device). Recent frames must be authenticated, e.g. by the cryptographic fallback.
2. **Embedding.** Three CosFace members, 15→128→128→64, scale 16, margin 0.25, Adam 2e-3, 30 epochs, batch 256, weight decay 1e-4,
   trained on all enrolment frames (`core2.py`).
3. **Local prototypes.** The similarity of a frame to identity *j* is the mean cosine to its *k* = 3 nearest recent-enrolment
   exemplars of *j*; a session's similarity is the mean over its 4 frames and the 3 members.
4. **Decision.** Accept a claim for *t* iff the rival margin `R_t = c_t − max_{j≠t} c_j` is at least β = 0.02. A session is therefore
   accepted for at most one identity (Proposition 5.1). An optional absolute check `c_t ≥ ω_t` (5th percentile of cross-fitted genuine
   similarities) guards against radios that were never enrolled.

## Development record (test block never loaded)

The development split mirrors the gap between training and test: inner training frames 0–349, inner blocks 350–474 and 475–599,
inner evaluation frames 600–849 (`devh.py` enforces `row < 850`). Scripts: `dev_emb.py` (embedding caches), `dev_base.py` and
`dev_base_eq.py` (forest and MLP baselines, training-only and equal-data), `dev_eval.py`, `dev_eval2.py`, `dev_local.py`
(variants), `openset.py --split dev`. Logs are in `dev_logs/`; the decisions and their evidence are in `FROZEN.json` and supplement Table S3.

## Frozen evaluation

| Step | Command | Output |
|---|---|---|
| RAPID vs forest and MLP (published and equal-data), 3 seeds × 2 role orders | `python final_runner.py` | `final_results.json`, `scores/` |
| Medians and run-by-run comparison | `python analyze_v2.py` | `final_summary.json` |
| Matched operating points | `python frontier2.py RAPID,RF-eq,MLP-eq,RF,MLP` | `frontier2.json` |
| Component ablation (seed 0) | `python ablation_v2.py` | `ablation_v2.json` |
| Unenrolled attackers (3 partitions × 2 orders) | `python openset.py --split test` then `python analyze_openset.py` | `openset_test.json` |
| Worst pair with 95% intervals | `python worstpair.py` | `worstpair.json` |
| Figures 8–9, tables | `python fig_rapid2.py`, `python gen_tables.py table`, `python gen_openset.py` | `figures/`, `tab_*.tex` |
| Check the paper against the results | `PAPER_TEX=path/to/main.tex python verify_v2.py` | prints one line per table row |

The forest needs about 4 GB of RAM; the full evaluation takes about 80 minutes on two CPU cores.
