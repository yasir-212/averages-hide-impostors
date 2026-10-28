# RAPID design freeze — 2026-10-16 (development data only)

All choices below were made on SMoRFFI frames 0–849 (train 0–599; development blocks A 600–724 and B 725–849).
The locked test block (frames 850–998) was not loaded by any development script (devsearch.py, dev2–dev9.py, devkf.py, frontier.py).

| Decision | Choice | Development evidence (thresholds on B, evaluated on A) |
|---|---|---|
| Representation | per-frame 15 SMoRFFI features, session = 4 consecutive frames (as in the protocol) | window mean/std or richer window statistics were weaker (devsearch.json, dev2) |
| Embedding | CosFace MLP 15-128-128-64, s=16, m=0.25, Adam 2e-3, 30 epochs, batch 256, wd 1e-4 | frame-level CosFace beat window-level CosFace (dev3); 30 epochs ≈ 60 epochs (dev4) |
| Ensemble | E = 3 members, cosine averaged (RAPID); E = 2 (RAPID-Lite) | E = 5 gave no gain (dev7); E = 2 trains in ~19 s (dev8) |
| Score | rival margin R_t = c_t − λ·max_{j≠t} c_j with λ = 1 | λ ∈ {0.75, 0.9, 1.0} similar under the quantile rule; λ = 1 gives the exclusivity bound and the best floor-rule FAR (dev7) |
| Decision rule | τ_t = min(Q_{0.10} of genuine calibration scores of t, 0) ("floor") | TAR 95.0% vs 88.4% for the plain quantile at similar FAR (dev7) |
| Rival-margin training (extra margin to k nearest identities) | rejected | no gain over plain CosFace (devsearch.json) |
| Training-frame subsampling | rejected | stride 2–3 lost 1.5–2.5 points of accuracy (dev9) |

Baseline (Guo et al., Computer Networks 282:112309, 2026 — SMoRFFI framework): per-frame Random Forest, 128 trees, library defaults
(the paper reports no other hyper-parameter). Session posterior = mean of per-frame posteriors. Verification adaptations evaluated:
(i) per-identity posterior quantile (q = 0.9, same rule family as RAPID), (ii) identification decision (accept iff argmax = claim).
Variants checked on development data and rejected as weaker: RF on window mean/std (dev2), Kalman-smoothed features (devkf.log).

Protocol: 3 seeds × 2 role orders (calibration = B or A; the other block is the attacker's development data); test read once per run.
