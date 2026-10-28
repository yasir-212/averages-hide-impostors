# Split definitions

Every split of the manuscript is defined in code; this file states them in one place (frame indices are per device, as stored in the
SMoRFFI feature file).

| Item | Definition | Code |
|---|---|---|
| Devices | every device with frames up to index 998: 122 of the 123 transmitters (one has only 513 usable frames) | `rapid/common.py` (`load_frames`) |
| Training block | frames 0–599 | `rapid/v2/final_runner.py` (`frames_in`) |
| Enrolment / attacker blocks | block A: frames 600–724; block B: frames 725–849 | `rapid/v2/final_runner.py` (`BLK`) |
| Locked test block | frames 850–998 | `rapid/v2/final_runner.py` (`BLK`) |
| Sessions | non-overlapping windows of L = 4 consecutive frames: 37 test sessions and 31 sessions in each of blocks A and B per device | `rapid/common.py` (`windows`) |
| Role orders | orig: recent enrolment = block B, attacker development data = block A; swap: the reverse | `rapid/v2/final_runner.py` (`ORDERS`) |
| Run seeds | 3 seeds × 2 role orders = 6 runs; RAPID member e of run seed r uses seed 1000 r + e | `rapid/v2/final_runner.py`, supplement Table S5 |
| Development split (design only; never reads frames ≥ 850) | inner training 0–349, inner blocks 350–474 and 475–599, inner evaluation 600–849 | `rapid/v2/devh.py` (`SPLIT`) |
| Unenrolled-attacker partitions | 30 of the 122 devices never enrolled, drawn with NumPy `default_rng(500 + p)` for partition p ∈ {0, 1, 2} | `rapid/v2/openset.py` |
| Population-size subsets | identities drawn with NumPy `default_rng(2026)` | `rapid/v2/ext/sweeps.py` |
