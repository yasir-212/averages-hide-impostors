# Further analyses of the frozen evaluation

All scripts read the stored session scores of the frozen evaluation (`../../scores/`) or refit models exactly as the frozen
evaluation does; none changes any parameter of RAPID.

| Script | Output | Paper |
|---|---|---|
| `rm_dev.py` | `rm_dev.json`: the rival-margin rule applied to the equal-data forest, MLP and CosFace prototype verifier; each margin selected on the inner development split (frames below 850) so that the development TAR falls by no more than RAPID's at β = 0.02, or so that the development attacker matches RAPID's | Section 7.3, Table 12 |
| `rm_test.py` | `rm_test.json`, `rm_test_medians.json`: frozen-protocol evaluation (3 seeds × 2 role orders) at those margins | Section 7.3, Table 12 |
| `crossed_boot.py` | `crossed_boot.json`: crossed bootstrap over source and target radios for TAR, pooled FAR and the development-chosen attacker | Section 7.1 |
| `cfeq_stats.py` | `cfeq_stats.json`: paired per-target comparison with the equal-data CosFace prototype verifier | Section 7.1 |
| `worst_oracle.py` | `worst_oracle.json`: test-oracle attacker per system | Section 7.4 |
| `mlp_conv.py` | `mlp_conv_*.json`: the equal-data MLP trained with up to 60 (frozen) and 400 epochs (early stopping) | Section 6.3 |
| `scale_bench.py` | `scale_bench.json`: single-core latency and exemplar memory for 122–5,000 identities (synthetic banks) | Section 5 |

`rm_dev.py` reads the inner-split caches in `rapid/v2/cache/`, which `dev_base_eq.py` and `dev_emb.py` regenerate; `rm_test.py`,
`crossed_boot.py`, `cfeq_stats.py` and `mlp_conv.py` need `SMORFFI_FEATURES`.
