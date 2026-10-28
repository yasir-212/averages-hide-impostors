# Numbers quoted in the manuscript

The scripts in this folder compute every result number quoted in the text of the manuscript from the result files of the repository
(no dataset needed). Run from the repository root:

```bash
python numbers/gen_numbers.py        # writes numbers/numbers.tex (main results, extended baselines, sweeps, open set, simulation)
python numbers/gen_numbers_v14.py    # writes numbers/numbers_v14.tex (test-oracle attacker, per-order medians, paired CosFace comparison, converged MLP)
```

The values of Table 12 and of the crossed bootstrap and scaling analyses come from `rapid/v2/ext/analyses/` (`rm_test_medians.json`,
`crossed_boot.json`, `scale_bench.json`). Tables 4 and 5 come from `audit/numbers.json` and `protocol/protocol_results.json`, Tables S1
and S2 from `experiments/` (see `experiments/README.md`), and Table S3 from `rapid/v2/FROZEN.json` and `rapid/v2/dev_logs/`.
