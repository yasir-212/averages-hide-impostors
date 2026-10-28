#!/usr/bin/env bash
# Reproduce every table and figure of the paper that uses SMoRFFI or the simulator.
# Requires: SMORFFI_FEATURES pointing to features_v1.npz (see data/README.md). Run from the repository root.
set -euo pipefail
: "${SMORFFI_FEATURES:?set SMORFFI_FEATURES to your features_v1.npz}"

echo "== Sections 6-7: audit (Tables 2 and 4, Figures 2-5)"
python audit/check1.py
python audit/check2.py
python audit/analyze.py
python audit/check7.py
python audit/figs.py

echo "== Section 10: protocol (Table 8, Figure 7)"
python protocol/run_protocol.py
python protocol/fig_protocol.py

echo "== Section 11: RAPID vs the 2026 SMoRFFI framework (Tables 10-13, Figures 9-11)"
python rapid/run_rapid_protocol.py
python rapid/analyze_rapid.py
python rapid/ablate_cost.py
python rapid/rf_latency.py
python rapid/fig_rapid.py
python rapid/verify_numbers.py

echo "== Section 12: simulation (Table 14, Figures 12-13)"
python sim/run_sim.py
python sim/run_identical.py
python sim/fig_sim.py
echo "done"
