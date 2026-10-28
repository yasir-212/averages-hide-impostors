#!/usr/bin/env python3
"""Adaptive digital attacker suite on Zenodo emulated IQ.

Threat ladder (attacker has development access to genuine *statistics* or a
surrogate, not the frozen test labels):

  A0 raw          — unmodified attacker packets
  A1 gain_match   — global RMS gain match to genuine train median
  A2 feat_nn      — select attacker packets nearest to genuine enrollment centroid
  A3 surrogate    — black-box style: fit surrogate on train genuines + synthetic
                    negatives; project attacker features toward the accept region
                    with a bounded L2 step (SGGP-inspired low-power attack)
  A4 combo        — gain_match then surrogate projection (strongest digital attack)

Defences evaluated at ~5% FRR conformal thresholds (benign-only calibration):
  EnergyOnly, OAS, IsolationForest, PAIRCONF (OAS+binary), PAIRRESID (OAS+ET residual)

Scope: adaptive digital attacks on emulated traces.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.covariance import OAS
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from hbd_python.rffi_emulated_attack_benchmark import (  # noqa: E402
    extract_features,
    load_packets,
    verify_capture,
)

PAIRS = [
    (1, "legitimate_device_01.npy", "attacker_device_05_impersonates_01.npy"),
    (2, "legitimate_device_02.npy", "attacker_device_06_impersonates_02.npy"),
    (3, "legitimate_device_03.npy", "attacker_device_07_impersonates_03.npy"),
    (4, "legitimate_device_04.npy", "attacker_device_08_impersonates_04.npy"),
]


def split_idx(n: int):
    n_train = int(0.50 * n)
    n_cal = int(0.15 * n)
    return slice(0, n_train), slice(n_train, n_train + n_cal), slice(n_train + n_cal, n)


def conformal_thr(scores: np.ndarray, target_frr: float = 0.05) -> float:
    n = len(scores)
    k = int(np.ceil((1.0 - target_frr) * (n + 1))) - 1
    k = min(max(k, 0), n - 1)
    return float(np.sort(scores)[k])


def gain_match_iq(attacker: np.ndarray, genuine_train: np.ndarray) -> np.ndarray:
    g_rms = np.sqrt(np.mean(np.abs(genuine_train) ** 2, axis=1))
    target = float(np.median(g_rms))
    a_rms = np.sqrt(np.mean(np.abs(attacker) ** 2, axis=1))
    return attacker * (target / np.maximum(a_rms, 1e-12))[:, None]


def feat_nn_select(af: np.ndarray, centroid: np.ndarray, keep_frac: float = 0.35):
    """Keep attacker packets closest to enrollment centroid in feature space."""
    d = np.linalg.norm(af - centroid[None, :], axis=1)
    k = max(1, int(np.ceil(keep_frac * len(af))))
    idx = np.argsort(d)[:k]
    return af[idx], idx


def surrogate_project(
    af: np.ndarray,
    scaler: StandardScaler,
    surrogate: LogisticRegression,
    eps: float = 0.35,
    steps: int = 12,
) -> np.ndarray:
    """Move features toward higher surrogate P(genuine) with bounded L2 budget.

    Uses analytic gradient of logistic P(genuine) in scaled space, then maps
    back. SGGP-inspired low-power attack.
    """
    z0 = scaler.transform(af).astype(np.float64)
    z = z0.copy()
    budget = float(eps) * np.sqrt(z.shape[1])
    w = surrogate.coef_.ravel()
    for _ in range(steps):
        p = surrogate.predict_proba(z)[:, 1]
        # dP/dz = P(1-P) * w
        grad = (p * (1.0 - p))[:, None] * w[None, :]
        z = z + 0.25 * grad
    diff = z - z0
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    scale = np.minimum(1.0, budget / np.maximum(norms, 1e-12))
    z = z0 + diff * scale
    return z * scaler.scale_ + scaler.mean_


def build_defences(Xtr, Rtr, Xca, seed=2026):
    """Return dict name -> (score_fn lower-is-better, cal_scores)."""
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zca = scaler.transform(Xtr), scaler.transform(Xca)

    defences = {}

    # EnergyOnly on log-RMS
    center = float(np.median(Rtr))
    scale_e = max(float(1.4826 * np.median(np.abs(Rtr - center))), 1e-9)
    defences["EnergyOnly"] = ("energy", center, scale_e)

    oas = OAS().fit(Ztr)

    def mahal(Z):
        d = Z - oas.location_
        return np.einsum("ni,ij,nj->n", d, oas.precision_, d)

    defences["OAS"] = ("oas", scaler, mahal)

    forest = IsolationForest(
        n_estimators=200, random_state=seed, n_jobs=2, contamination="auto"
    ).fit(Ztr)
    defences["IsolationForest"] = ("iforest", forest, scaler)

    # PAIRCONF: OAS + logistic vs shuffled/gain negatives
    rng = np.random.default_rng(seed)
    Xneg = Ztr.copy()
    rng.shuffle(Xneg)
    Xneg2 = Ztr * rng.uniform(0.7, 1.3, size=(len(Ztr), 1))
    Xbin = np.vstack([Ztr, Xneg, Xneg2])
    Ybin = np.r_[np.ones(len(Ztr)), np.zeros(len(Xneg) + len(Xneg2))]
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed, solver="lbfgs"
    ).fit(Xbin, Ybin)
    defences["PAIRCONF"] = ("pairconf", scaler, mahal, clf)

    # PAIRRESID: OAS + ExtraTrees residual on score-augmented features
    et = ExtraTreesClassifier(
        n_estimators=96,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=2,
        class_weight="balanced_subsample",
    )
    mtr = mahal(Ztr)
    Zneg = np.vstack([Xneg, Xneg2])
    mneg = mahal(Zneg)
    Feat_pos = np.column_stack([Ztr, mtr, 1.0 / np.maximum(mtr, 1e-9)])
    Feat_neg = np.column_stack([Zneg, mneg, 1.0 / np.maximum(mneg, 1e-9)])
    et.fit(
        np.vstack([Feat_pos, Feat_neg]),
        np.r_[np.ones(len(Feat_pos)), np.zeros(len(Feat_neg))],
    )
    defences["PAIRRESID"] = ("pairresid", scaler, mahal, et)

    return defences, scaler, clf


def score_defence(spec, X, R):
    """Anomaly scores (lower = more genuine-like)."""
    kind = spec[0]
    if kind == "energy":
        _, center, scale = spec
        return np.abs(R - center) / scale
    if kind == "oas":
        _, scaler, mahal = spec
        return mahal(scaler.transform(X))
    if kind == "iforest":
        _, forest, scaler = spec
        return -forest.decision_function(scaler.transform(X))
    if kind == "pairconf":
        _, scaler, mahal, _clf = spec
        return mahal(scaler.transform(X))
    if kind == "pairresid":
        _, scaler, mahal, _et = spec
        return mahal(scaler.transform(X))
    raise ValueError(kind)


def accept_defence(spec, X, R, thr_main, thr_bin=None):
    kind = spec[0]
    if kind in ("energy", "oas", "iforest"):
        s = score_defence(spec, X, R)
        return s <= thr_main, s
    if kind == "pairconf":
        _, scaler, mahal, clf = spec
        Z = scaler.transform(X)
        s = mahal(Z)
        p = clf.predict_proba(Z)[:, 1]
        return (s <= thr_main) & (p >= thr_bin), s
    if kind == "pairresid":
        _, scaler, mahal, et = spec
        Z = scaler.transform(X)
        m = mahal(Z)
        feats = np.column_stack([Z, m, 1.0 / np.maximum(m, 1e-9)])
        p = et.predict_proba(feats)[:, 1]
        return (m <= thr_main) & (p >= thr_bin), m
    raise ValueError(kind)


def calibrate(spec, Xca, Rca, target_frr=0.05):
    kind = spec[0]
    if kind in ("energy", "oas", "iforest"):
        return conformal_thr(score_defence(spec, Xca, Rca), target_frr), None
    if kind == "pairconf":
        _, scaler, mahal, clf = spec
        Z = scaler.transform(Xca)
        thr_m = conformal_thr(mahal(Z), target_frr)
        thr_b = float(np.quantile(clf.predict_proba(Z)[:, 1], target_frr, method="higher"))
        return thr_m, thr_b
    if kind == "pairresid":
        _, scaler, mahal, et = spec
        Z = scaler.transform(Xca)
        m = mahal(Z)
        thr_m = conformal_thr(m, target_frr)
        feats = np.column_stack([Z, m, 1.0 / np.maximum(m, 1e-9)])
        thr_b = float(
            np.quantile(et.predict_proba(feats)[:, 1], target_frr, method="higher")
        )
        return thr_m, thr_b
    raise ValueError(kind)


def centroid_blend(af: np.ndarray, centroid: np.ndarray, alpha: float = 0.55):
    """Convex blend of attacker features toward enrollment centroid (stronger digital)."""
    return (1.0 - alpha) * af + alpha * centroid[None, :]


def make_attacks(g_iq, a_iq, gf, af, g_rms, a_rms, tr, scaler, surrogate, centroid):
    """Build attack feature/rms bundles for each threat level."""
    attacks = {}
    attacks["A0_raw"] = (af, a_rms, "unmodified attacker IQ")
    a1 = gain_match_iq(a_iq, g_iq[tr]).astype(np.complex64)
    a1_rms, a1f = extract_features(a1)
    attacks["A1_gain_match"] = (a1f, a1_rms, "global RMS gain match")
    a2f, idx = feat_nn_select(af, centroid, keep_frac=0.35)
    attacks["A2_feat_nn"] = (a2f, a_rms[idx], "NN select 35% closest to enroll centroid")
    a2t, idxt = feat_nn_select(af, centroid, keep_frac=0.10)
    attacks["A2c_feat_nn_top10"] = (
        a2t,
        a_rms[idxt],
        "NN select top-10% closest to enroll centroid",
    )
    a3f = surrogate_project(af, scaler, surrogate, eps=0.55, steps=16)
    attacks["A3_surrogate"] = (
        a3f,
        a_rms,
        "surrogate-guided L2 feature projection eps=0.55",
    )
    a4f = surrogate_project(a1f, scaler, surrogate, eps=0.55, steps=16)
    attacks["A4_gain+surrogate"] = (
        a4f,
        a1_rms,
        "gain match then surrogate projection",
    )
    a2g, idxg = feat_nn_select(a1f, centroid, keep_frac=0.35)
    attacks["A2b_gain+feat_nn"] = (a2g, a1_rms[idxg], "gain match + NN select")
    # Stronger digital: blend NN-selected packets toward centroid
    blend = centroid_blend(a2t, centroid, alpha=0.55)
    attacks["A5_nn+centroid_blend"] = (
        blend,
        a_rms[idxt],
        "top-10% NN then 55% blend to enroll centroid",
    )
    return attacks


def eval_target(data: Path, tid: int, gname: str, aname: str, seed=2026):
    verify_capture(data / gname)
    verify_capture(data / aname)
    g = np.asarray(load_packets(data / gname))
    a = np.asarray(load_packets(data / aname))
    g_rms, gf = extract_features(g)
    a_rms, af = extract_features(a)
    tr, ca, te = split_idx(len(gf))
    Xtr, Xca, Xte = gf[tr], gf[ca], gf[te]
    Rtr, Rca, Rte = g_rms[tr], g_rms[ca], g_rms[te]

    defences, scaler, surrogate = build_defences(Xtr, Rtr, Xca, seed=seed)
    centroid = Xtr.mean(axis=0)

    # recalibrate each defence
    thrs = {name: calibrate(spec, Xca, Rca) for name, spec in defences.items()}

    attacks = make_attacks(g, a, gf, af, g_rms, a_rms, tr, scaler, surrogate, centroid)

    rows = []
    for dname, spec in defences.items():
        thr_m, thr_b = thrs[dname]
        ok_te, _ = accept_defence(spec, Xte, Rte, thr_m, thr_b)
        tar = float(np.mean(ok_te))
        for aname_atk, (Xf, Rf, desc) in attacks.items():
            ok_a, _ = accept_defence(spec, Xf, Rf, thr_m, thr_b)
            far = float(np.mean(ok_a))
            rows.append(
                {
                    "target": tid,
                    "defence": dname,
                    "attack": aname_atk,
                    "attack_desc": desc,
                    "tar": tar,
                    "far": far,
                    "n_attack": int(len(Xf)),
                    "n_genuine_test": int(len(Xte)),
                }
            )
        print(f"target {tid} {dname} TAR={tar:.3f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("EMULATED_DIR", "RFFI_Emulated"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for tid, gname, aname in PAIRS:
        all_rows.extend(eval_target(data, tid, gname, aname))

    keys = list(dict.fromkeys(k for r in all_rows for k in r))
    with (out / "per_target.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)

    # macro / worst over targets
    defences = sorted({r["defence"] for r in all_rows})
    attacks = sorted({r["attack"] for r in all_rows})
    macro = []
    for d in defences:
        for a in attacks:
            rs = [r for r in all_rows if r["defence"] == d and r["attack"] == a]
            macro.append(
                {
                    "defence": d,
                    "attack": a,
                    "macro_tar": float(np.mean([r["tar"] for r in rs])),
                    "macro_far": float(np.mean([r["far"] for r in rs])),
                    "worst_far": float(np.max([r["far"] for r in rs])),
                    "best_far": float(np.min([r["far"] for r in rs])),
                }
            )
    with (out / "macro.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(macro[0]))
        w.writeheader()
        w.writerows(macro)

    # pivot worst FAR: defence x attack
    pivot = {d: {} for d in defences}
    for row in macro:
        pivot[row["defence"]][row["attack"]] = row["worst_far"]

    # availability = TAR; security under the strongest digital attack
    strongest = "A5_nn+centroid_blend"
    stress = ["A1_gain_match", "A2c_feat_nn_top10", "A5_nn+centroid_blend"]
    system = []
    for d in defences:
        r0 = next(r for r in macro if r["defence"] == d and r["attack"] == "A0_raw")
        row = {
            "defence": d,
            "macro_tar": r0["macro_tar"],
            "worst_far_A0": r0["worst_far"],
        }
        for a in stress:
            ra = next(r for r in macro if r["defence"] == d and r["attack"] == a)
            row[f"worst_far_{a}"] = ra["worst_far"]
        row["worst_far_strongest"] = row[f"worst_far_{strongest}"]
        row["far_inflation_strongest"] = row["worst_far_strongest"] - r0["worst_far"]
        row["residual_security_strongest"] = 1.0 - row["worst_far_strongest"]
        system.append(row)

    summary = {
        "evidence_type": "Zenodo GNU Radio emulated I/Q; adaptive digital attacks",
        "threat_ladder": [
            "A0_raw",
            "A1_gain_match",
            "A2_feat_nn",
            "A2c_feat_nn_top10",
            "A2b_gain+feat_nn",
            "A3_surrogate",
            "A4_gain+surrogate",
            "A5_nn+centroid_blend",
        ],
        "system_under_strongest_attack": system,
        "pivot_worst_far": pivot,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(system, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
