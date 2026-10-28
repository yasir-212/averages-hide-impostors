# Motivating study (Section 4 and supplement S2 of the paper)

Code and raw results of the defences and attacks that Section 4 and supplement S2 evaluate. The paper re-evaluates every
defence below under its protocol (held-out calibration, full coverage, development-chosen and defence-aware attackers); these
results underlie the evaluation shortcuts of supplement Table S1.

| Folder | Content | Paper |
|---|---|---|
| `pairgate_audit/` | identity gating: γ sweep, random-subset control, statistical limits, host and signature cost (`run_all.sh`) | Section 4.3, supplement S2 (Table S1) |
| `pairgate_repro/` | identity-gating operating point reproduced on SMoRFFI (RF128) | supplement S2 |
| `residual_verifier/` | pair-conditioned residual verifier, pair-confidence verifier and FAR cap on SMoRFFI | Section 4.3, supplement S2 (Table S1) |
| `high_coverage_defence/` | full-coverage policies (FAR cap, residual verifier, explicit denial) | Sections 4.2–4.3 |
| `adaptive_attacker/` | attack ladder A0–A5 on the emulated corpus and on SMoRFFI (`macro.csv`, `per_target.csv`) | supplement S2 (Table S1) |
| `inria_captures/` | INRIA BladeRF captures of 12 commercial transmitters: unenrolled and enrolled radios claiming enrolled identities (`results/INRIA_measured/`) | supplement S2 (Table S2) |

Scripts take the data paths as command-line arguments or from `SMORFFI_FEATURES`, `EMULATED_DIR` and `INRIA_CACHE` (see
`../data/README.md`) and write their results into their own folder. `pairgate_audit/` needs `cryptography` and `pytest` (`pairgate_audit/requirements.txt`);
the adaptive suite reads the emulated corpus through `hbd_python.rffi_emulated_attack_benchmark`.
