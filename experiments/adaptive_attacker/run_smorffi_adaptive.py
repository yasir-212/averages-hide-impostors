#!/usr/bin/env python3
"""SMoRFFI adaptive claim attack: hardest source + feature blend toward target.

Complements the emulated-I/Q adaptive suite with offline SMoRFFI claims.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "residual_verifier"))
from pair_resid_compare import (  # noqa: E402
    accept_matrix,
    calibrate_ratio_thr,
    data_windows,
    enroll_prototypes,
    far_capped_thr,
    fit_resid_verifiers,
    genuine_thr,
    hard_rivals,
    proto_ratio_ok,
    binary_ok_resid,
    score_aug,
)

ROOT = Path(__file__).resolve().parent


def blend_toward_target(x_src, proto_t, alpha):
    return (1.0 - alpha) * x_src + alpha * proto_t[None, :]


def main():
    sets, n, sha = data_windows(
        os.environ.get("SMORFFI_FEATURES", "features_v1.npz")
    )
    scaler = StandardScaler().fit(sets["train"][0])
    sets = {k: (scaler.transform(x), y) for k, (x, y) in sets.items()}
    xdev, ydev = sets["dev"]
    xcal, ycal = sets["cal"]
    xtest, ytest = sets["test"]

    model = RandomForestClassifier(
        n_estimators=128, min_samples_leaf=2, random_state=31, n_jobs=-1
    )
    model.fit(*sets["train"])
    pdev = model.predict_proba(xdev)
    pcal = model.predict_proba(xcal)
    ptest = model.predict_proba(xtest)
    thr0 = genuine_thr(pcal, ycal, n, 0.90)
    rivals = hard_rivals(pdev, ydev, thr0, n, top_k=5)
    protos = enroll_prototypes(xcal, ycal, n)
    ver = fit_resid_verifiers(xdev, ydev, pdev, thr0, n, rivals, mode="hard")
    bok = binary_ok_resid(ver, xtest, ptest, n)

    # Baseline accept matrices
    policies = {}
    gen_b, pair_b = accept_matrix(ptest, ytest, thr0)
    policies["baseline"] = (gen_b, pair_b)
    thr_f = far_capped_thr(pdev, ydev, pcal, ycal, n, 0.0)
    policies["farcap0"] = accept_matrix(ptest, ytest, thr_f)
    policies["PAIRRESID"] = accept_matrix(ptest, ytest, thr0, binary_ok=bok)

    # Adaptive: for each target, take DEV-hardest source test samples,
    # blend features toward target prototype, re-score with model+resid
    alphas = [0.0, 0.25, 0.5, 0.75]
    rows = []
    for alpha in alphas:
        # Build synthetic impostor set per target
        far_base = []
        far_resid = []
        far_farcap = []
        for t in range(n):
            s = rivals[t][0]
            xs = xtest[ytest == s]
            if len(xs) == 0:
                continue
            xb = blend_toward_target(xs, protos[t], alpha)
            pb = model.predict_proba(xb)
            # baseline / farcap accept as claim t
            far_base.append(float(np.mean(pb[:, t] >= thr0[t])))
            far_farcap.append(float(np.mean(pb[:, t] >= thr_f[t])))
            # resid
            if ver[t] is None:
                far_resid.append(far_base[-1])
            else:
                clf, bthr = ver[t]
                feats = score_aug(xb, pb, t)
                ok = (pb[:, t] >= thr0[t]) & (clf.predict_proba(feats)[:, 1] >= bthr)
                far_resid.append(float(np.mean(ok)))
        rows.append(
            {
                "alpha_blend_to_target": alpha,
                "baseline_mean_hard_FAR": float(np.mean(far_base)),
                "baseline_worst_hard_FAR": float(np.max(far_base)),
                "farcap0_mean_hard_FAR": float(np.mean(far_farcap)),
                "farcap0_worst_hard_FAR": float(np.max(far_farcap)),
                "PAIRRESID_mean_hard_FAR": float(np.mean(far_resid)),
                "PAIRRESID_worst_hard_FAR": float(np.max(far_resid)),
            }
        )
        print(rows[-1], flush=True)

    # natural (alpha=0) vs policies TAR for reference
    summary = {
        "dataset_sha256": sha,
        "n_devices": n,
        "natural_test": {
            k: {
                "TAR": float(g.mean()),
                "FAR_worst": float(np.nanmax(p)),
                "FAR_pooled": float(np.nanmean(p)),
            }
            for k, (g, p) in policies.items()
        },
        "adaptive_blend_curve": rows,
        "evidence_type": "SMoRFFI offline; feature blend toward target prototype",
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "smorffi_adaptive.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (ROOT / "smorffi_adaptive_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote smorffi adaptive")


if __name__ == "__main__":
    main()
