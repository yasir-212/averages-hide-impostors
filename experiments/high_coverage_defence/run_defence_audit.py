#!/usr/bin/env python3
"""High-coverage RF verification audit on SMoRFFI features.

Compares identity-denial PAIR-GATE against policies that keep all identities
RF-eligible. Offline cross-identity claims on SMoRFFI.

Policies
--------
1. baseline_genuine_q90
   Per-target genuine-only calibration quantile; all identities eligible.
2. identity_gate_gamma
   PAIR-GATE: deny RF for targets whose development worst-source FAR > gamma.
3. hardest_source_threshold_r
   Raise each target threshold using the development hardest-source quantile;
   all identities remain eligible.
4. far_capped_threshold_gamma
   Binary-search the lowest threshold such that development worst-source FAR
   <= gamma, subject to a genuine-calibration floor; all identities eligible.
5. score_margin_m
   Accept only if score_t >= genuine tau_t AND (score_t - best_other) >= m;
   all identities eligible.
6. hybrid_farcap_margin
   FAR-capped threshold plus a development-selected margin.

Primary endpoints: all-identity TAR, pooled FAR, worst-pair FAR, identity
coverage, and development-chosen attacker mean FAR.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SPLITS = {"train": (0, 600), "dev": (600, 750), "cal": (750, 850), "test": (850, 999)}


def data_windows(datafile: str):
    d = np.load(datafile)
    x, y, row = d["x"], d["y"], d["row"]
    valid = np.array(
        [j for j in range(len(d["labels"])) if (row[y == j] >= 998).any()]
    )
    good = np.isin(y, valid)
    x, y, row = x[good], np.searchsorted(valid, y[good]), row[good]
    n = len(valid)
    out = {}
    for segment, (lo, hi) in SPLITS.items():
        xs, ys = [], []
        for j in range(n):
            ix = np.flatnonzero((y == j) & (row >= lo) & (row < hi))
            ix = ix[np.argsort(row[ix])]
            for pos in range(0, len(ix) - 3, 4):
                ids = ix[pos : pos + 4]
                if not np.all(np.diff(row[ids]) == 1):
                    continue
                z = x[ids]
                v = np.r_[np.mean(z, axis=0), np.std(z, axis=0)]
                if np.all(np.isfinite(v)):
                    xs.append(v)
                    ys.append(j)
        out[segment] = (np.asarray(xs), np.asarray(ys))
    return out, n, valid, hashlib.sha256(Path(datafile).read_bytes()).hexdigest()


def genuine_thresholds(p: np.ndarray, y: np.ndarray, n: int, q: float = 0.90) -> np.ndarray:
    return np.asarray(
        [np.quantile(p[y == j, j], 1.0 - q, method="higher") for j in range(n)]
    )


def pair_matrix(p: np.ndarray, y: np.ndarray, thr: np.ndarray, gate=None, margin: float = 0.0):
    n = p.shape[1]
    target_gate = np.ones(n, dtype=bool) if gate is None else gate
    # For claimed identity t: best competing score is max_{j != t} p[:, j]
    accept = (p >= thr[None, :]) & target_gate[None, :]
    if margin > 0:
        best_other = np.column_stack(
            [np.max(p[:, [j for j in range(n) if j != t]], axis=1) for t in range(n)]
        )
        accept = accept & ((p - best_other) >= margin)
    genuine = accept[np.arange(len(y)), y]
    a = accept.copy()
    a[np.arange(len(y)), y] = False
    pair = np.stack([a[y == s].mean(axis=0) for s in range(n)])
    np.fill_diagonal(pair, np.nan)
    return genuine, pair


def metrics(genuine, pair, gate=None):
    n_targets = pair.shape[1]
    eligible = np.ones(n_targets, dtype=bool) if gate is None else gate
    out = {
        "TAR_all_identities": float(genuine.mean()),
        "FAR_pooled": float(np.nanmean(pair)),
        "FAR_worst_pair": float(np.nanmax(pair)) if np.isfinite(pair).any() else None,
        "FAR_mean_max_source": float(np.nanmean(np.nanmax(pair, axis=0))),
        "eligible_identities": int(eligible.sum()),
        "identity_coverage": float(eligible.mean()),
    }
    if eligible.any():
        out["FAR_worst_eligible_pair"] = float(np.nanmax(pair[:, eligible]))
        # Genuine TAR among windows whose true identity is eligible.
        # For identity gates this equals RF contribution to availability.
    else:
        out["FAR_worst_eligible_pair"] = None
    return out


def far_capped_thresholds(
    pdev: np.ndarray,
    ydev: np.ndarray,
    pcal: np.ndarray,
    ycal: np.ndarray,
    n: int,
    gamma: float,
    q_floor: float = 0.90,
) -> np.ndarray:
    """Lowest threshold meeting development worst-source FAR <= gamma, above genuine floor."""
    floor = genuine_thresholds(pcal, ycal, n, q=q_floor)
    thr = floor.copy()
    for t in range(n):
        # Candidate scores: genuine cal scores for t and all non-t development scores for class t.
        candidates = np.unique(
            np.r_[
                pcal[ycal == t, t],
                np.concatenate([pdev[ydev == s, t] for s in range(n) if s != t]),
                [floor[t], 1.0],
            ]
        )
        candidates = np.sort(candidates)
        best = 1.0
        for c in candidates:
            if c < floor[t] - 1e-15:
                continue
            worst = 0.0
            for s in range(n):
                if s == t:
                    continue
                scores = pdev[ydev == s, t]
                if len(scores) == 0:
                    continue
                worst = max(worst, float(np.mean(scores >= c)))
            if worst <= gamma + 1e-12:
                best = float(c)
                break
        thr[t] = best
    return thr


def choose_margin(
    pdev: np.ndarray,
    ydev: np.ndarray,
    thr: np.ndarray,
    n: int,
    gamma: float,
    grid: np.ndarray,
) -> float:
    """Smallest development margin with worst-pair FAR <= gamma under fixed thr."""
    for m in grid:
        _, pair = pair_matrix(pdev, ydev, thr, margin=float(m))
        if float(np.nanmax(pair)) <= gamma + 1e-12:
            return float(m)
    return float(grid[-1])


def model_ctor(name: str):
    if name == "RF128":
        return RandomForestClassifier(
            n_estimators=128, min_samples_leaf=2, random_state=31, n_jobs=-1
        )
    if name == "MLP64":
        return MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            alpha=0.1,
            max_iter=70,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=8,
            random_state=67,
            batch_size=256,
        )
    raise ValueError(name)


def evaluate_policy(name, pdev, ydev, ptest, ytest, thr, gate=None, margin=0.0):
    gen, pair = pair_matrix(ptest, ytest, thr, gate=gate, margin=margin)
    row = {"policy": name, **metrics(gen, pair, gate)}
    # Development-chosen attacker: for each target, pick worst development source.
    _, dev_pair = pair_matrix(pdev, ydev, thr, gate=gate, margin=margin)
    selected = np.asarray(
        [np.nanargmax(dev_pair[:, t]) if np.isfinite(dev_pair[:, t]).any() else 0 for t in range(pair.shape[1])]
    )
    # Evaluate those sources on test under the same rule.
    chosen = []
    for t in range(pair.shape[1]):
        if gate is not None and not gate[t]:
            continue
        chosen.append(pair[selected[t], t])
    row["dev_chosen_attacker_mean_FAR"] = float(np.nanmean(chosen)) if chosen else None
    row["margin"] = float(margin)
    if gate is None:
        row["gate_gamma"] = None
    return row, gen, pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.environ.get("SMORFFI_FEATURES", "features_v1.npz"),
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument("--model", choices=["RF128", "MLP64"], default="RF128")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sets, n, valid, sha = data_windows(args.data)
    scaler = StandardScaler().fit(sets["train"][0])
    sets = {k: (scaler.transform(x), y) for k, (x, y) in sets.items()}
    model = model_ctor(args.model)
    model.fit(*sets["train"])
    probs = {k: model.predict_proba(v[0]) for k, v in sets.items() if k != "train"}
    pdev, pcal, ptest = probs["dev"], probs["cal"], probs["test"]
    ydev, ycal, ytest = sets["dev"][1], sets["cal"][1], sets["test"][1]

    base_thr = genuine_thresholds(pcal, ycal, n, q=0.90)
    _, base_dev_pair = pair_matrix(pdev, ydev, base_thr)
    dev_worst = np.nanmax(base_dev_pair, axis=0)

    rows = []
    details = {}

    # 1) Baseline
    row, gen, pair = evaluate_policy(
        "baseline_genuine_q90", pdev, ydev, ptest, ytest, base_thr
    )
    rows.append(row)
    details["baseline_genuine_q90"] = {
        "test_pair_max_by_target": np.nanmax(pair, axis=0).tolist(),
        "TAR": float(gen.mean()),
    }

    # 2) Identity gates (PAIR-GATE style)
    for gamma in [0.0, 0.05, 0.10, 0.20]:
        gate = dev_worst <= gamma + 1e-12
        row, gen, pair = evaluate_policy(
            f"identity_gate_gamma={gamma}",
            pdev,
            ydev,
            ptest,
            ytest,
            base_thr,
            gate=gate,
        )
        row["gate_gamma"] = gamma
        rows.append(row)

    # 3) Hardest-source threshold hardening
    for r in [0.05, 0.10, 0.20]:
        limits = np.quantile(
            np.stack([pdev[ydev == s, :] for s in range(n)]),
            1.0 - r,
            axis=1,
            method="higher",
        )
        np.fill_diagonal(limits, -np.inf)
        thr = np.maximum(base_thr, np.max(limits, axis=0))
        row, gen, pair = evaluate_policy(
            f"hardest_source_threshold_r={r}", pdev, ydev, ptest, ytest, thr
        )
        rows.append(row)

    # 4) FAR-capped thresholds (high coverage alternative)
    for gamma in [0.0, 0.05, 0.10, 0.20]:
        thr = far_capped_thresholds(pdev, ydev, pcal, ycal, n, gamma=gamma)
        row, gen, pair = evaluate_policy(
            f"far_capped_threshold_gamma={gamma}", pdev, ydev, ptest, ytest, thr
        )
        row["gate_gamma"] = gamma
        rows.append(row)
        details[f"far_capped_threshold_gamma={gamma}"] = {
            "mean_threshold": float(np.mean(thr)),
            "mean_threshold_lift": float(np.mean(thr - base_thr)),
            "TAR": float(gen.mean()),
            "FAR_worst_pair": float(np.nanmax(pair)),
        }

    # 5) Score-margin reject option
    for m in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
        row, gen, pair = evaluate_policy(
            f"score_margin_m={m}", pdev, ydev, ptest, ytest, base_thr, margin=m
        )
        rows.append(row)

    # 6) Hybrid: FAR-cap at 0.10 + smallest margin meeting that on development
    thr_cap = far_capped_thresholds(pdev, ydev, pcal, ycal, n, gamma=0.10)
    m_star = choose_margin(
        pdev, ydev, thr_cap, n, gamma=0.10, grid=np.linspace(0.0, 0.30, 31)
    )
    row, gen, pair = evaluate_policy(
        "hybrid_farcap0.10_plus_margin",
        pdev,
        ydev,
        ptest,
        ytest,
        thr_cap,
        margin=m_star,
    )
    row["selected_dev_margin"] = m_star
    rows.append(row)

    # Pareto summary: keep policies with coverage==1 and highlight best worst-pair
    full = [r for r in rows if r["identity_coverage"] >= 0.999]
    if full:
        best = min(full, key=lambda r: (r["FAR_worst_pair"], -r["TAR_all_identities"]))
    else:
        best = None

    summary = {
        "evidence_type": "SMoRFFI offline cross-identity claims",
        "model": args.model,
        "n_devices": n,
        "dataset_sha256": sha,
        "splits": SPLITS,
        "n_policies": len(rows),
        "best_full_coverage_by_worst_pair": best,
        "pair_gate_identity_gate_gamma0": next(
            r for r in rows if r["policy"] == "identity_gate_gamma=0.0"
        ),
        "baseline": next(r for r in rows if r["policy"] == "baseline_genuine_q90"),
    }

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / f"policy_results_{args.model}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out / f"summary_{args.model}.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / f"details_{args.model}.json").write_text(json.dumps(details, indent=2) + "\n")

    print(json.dumps(summary["baseline"], indent=2))
    print("--- identity_gate_gamma=0.0 ---")
    print(json.dumps(summary["pair_gate_identity_gate_gamma0"], indent=2))
    print("--- best full-coverage by worst-pair ---")
    print(json.dumps(best, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
