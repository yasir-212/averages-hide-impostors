#!/usr/bin/env python3
"""Matched comparison: PAIR-GATE vs high-coverage PAIR-CONF and baselines.

PAIR-CONF keeps every identity RF-eligible. For each target it fits a
development-only binary verifier against that target's hardest confusing
sources, then requires both the multiclass score and the binary accept.

Offline identity claims on SMoRFFI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SPLITS = {"train": (0, 600), "dev": (600, 750), "cal": (750, 850), "test": (850, 999)}


def data_windows(datafile: str):
    d = np.load(datafile)
    x, y, row = d["x"], d["y"], d["row"]
    valid = np.array([j for j in range(len(d["labels"])) if (row[y == j] >= 998).any()])
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
    return out, n, hashlib.sha256(Path(datafile).read_bytes()).hexdigest()


def genuine_thr(p, y, n, q=0.90):
    return np.asarray([np.quantile(p[y == j, j], 1 - q, method="higher") for j in range(n)])


def accept_matrix(p, y, thr, gate=None, binary_ok=None):
    n = p.shape[1]
    g = np.ones(n, dtype=bool) if gate is None else gate
    acc = (p >= thr[None, :]) & g[None, :]
    if binary_ok is not None:
        acc = acc & binary_ok
    genuine = acc[np.arange(len(y)), y]
    a = acc.copy()
    a[np.arange(len(y)), y] = False
    pair = np.stack([a[y == s].mean(axis=0) for s in range(n)])
    np.fill_diagonal(pair, np.nan)
    return genuine, pair


def summarize(name, genuine, pair, gate=None, extra=None):
    n = pair.shape[1]
    elig = np.ones(n, dtype=bool) if gate is None else gate
    # development-chosen attacker using test pair rows selected by argmax of pair itself is oracle;
    # caller should pass precomputed dev-chosen FAR if needed.
    row = {
        "method": name,
        "TAR_all": float(genuine.mean()),
        "FAR_pooled": float(np.nanmean(pair)),
        "FAR_worst": float(np.nanmax(pair)),
        "FAR_mean_max_source": float(np.nanmean(np.nanmax(pair, axis=0))),
        "eligible": int(elig.sum()),
        "coverage": float(elig.mean()),
        "FAR_worst_eligible": float(np.nanmax(pair[:, elig])) if elig.any() else None,
    }
    if extra:
        row.update(extra)
    return row


def far_capped_thr(pdev, ydev, pcal, ycal, n, gamma):
    floor = genuine_thr(pcal, ycal, n, 0.90)
    thr = floor.copy()
    for t in range(n):
        cands = np.unique(
            np.r_[
                pcal[ycal == t, t],
                np.concatenate([pdev[ydev == s, t] for s in range(n) if s != t]),
                [floor[t], 1.0],
            ]
        )
        best = 1.0
        for c in np.sort(cands):
            if c < floor[t] - 1e-15:
                continue
            worst = max(
                float(np.mean(pdev[ydev == s, t] >= c))
                for s in range(n)
                if s != t and np.any(ydev == s)
            )
            if worst <= gamma + 1e-12:
                best = float(c)
                break
        thr[t] = best
    return thr


def fit_pair_conf(xdev, ydev, pdev, thr, n, top_k=3, seed=31):
    """Per-target binary verifiers against development hardest sources."""
    verifiers = [None] * n
    rivals = [[] for _ in range(n)]
    for t in range(n):
        far = []
        for s in range(n):
            if s == t:
                continue
            scores = pdev[ydev == s, t]
            far.append((float(np.mean(scores >= thr[t])) if len(scores) else 0.0, s))
        far.sort(reverse=True)
        hard = [s for f, s in far[:top_k] if f > 0]
        if not hard:
            # no confusing source at current thr: verifier always accepts
            rivals[t] = []
            continue
        rivals[t] = hard
        xpos = xdev[ydev == t]
        xneg = np.concatenate([xdev[ydev == s] for s in hard], axis=0)
        X = np.vstack([xpos, xneg])
        Y = np.r_[np.ones(len(xpos)), np.zeros(len(xneg))]
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed, solver="lbfgs"
        )
        clf.fit(X, Y)
        # Calibrate binary threshold on genuine development to keep ~95% of genuines
        pos_scores = clf.predict_proba(xpos)[:, 1]
        bthr = float(np.quantile(pos_scores, 0.05, method="higher"))
        verifiers[t] = (clf, bthr)
    return verifiers, rivals


def binary_ok_matrix(verifiers, x, n):
    ok = np.ones((len(x), n), dtype=bool)
    for t, item in enumerate(verifiers):
        if item is None:
            continue
        clf, bthr = item
        ok[:, t] = clf.predict_proba(x)[:, 1] >= bthr
    return ok


def dev_chosen_far(pdev, ydev, ptest, ytest, thr, gate=None, binary_ok_test=None):
    n = pdev.shape[1]
    _, dev_pair = accept_matrix(pdev, ydev, thr, gate=gate)
    selected = np.array(
        [np.nanargmax(dev_pair[:, t]) if np.isfinite(dev_pair[:, t]).any() else 0 for t in range(n)]
    )
    _, test_pair = accept_matrix(ptest, ytest, thr, gate=gate, binary_ok=binary_ok_test)
    vals = []
    for t in range(n):
        if gate is not None and not gate[t]:
            continue
        vals.append(test_pair[selected[t], t])
    return float(np.nanmean(vals)) if vals else None


def make_model(name):
    if name == "RF128":
        return RandomForestClassifier(
            n_estimators=128, min_samples_leaf=2, random_state=31, n_jobs=-1
        )
    if name == "ET96":
        return ExtraTreesClassifier(
            n_estimators=96, min_samples_leaf=2, random_state=31, n_jobs=-1
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", default=os.environ.get("SMORFFI_FEATURES", "features_v1.npz")
    )
    ap.add_argument(
        "--out", default=str(Path(__file__).resolve().parent)
    )
    ap.add_argument("--models", nargs="+", default=["RF128", "ET96", "MLP64"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sets, n, sha = data_windows(args.data)
    scaler = StandardScaler().fit(sets["train"][0])
    sets = {k: (scaler.transform(x), y) for k, (x, y) in sets.items()}
    xdev, ydev = sets["dev"]
    xcal, ycal = sets["cal"]
    xtest, ytest = sets["test"]

    all_rows = []
    for mname in args.models:
        t0 = time.perf_counter()
        model = make_model(mname)
        model.fit(*sets["train"])
        fit_s = time.perf_counter() - t0
        pdev = model.predict_proba(xdev)
        pcal = model.predict_proba(xcal)
        ptest = model.predict_proba(xtest)
        thr0 = genuine_thr(pcal, ycal, n, 0.90)
        _, dev_pair0 = accept_matrix(pdev, ydev, thr0)
        dev_worst = np.nanmax(dev_pair0, axis=0)

        # 1) Baseline
        gen, pair = accept_matrix(ptest, ytest, thr0)
        all_rows.append(
            summarize(
                f"{mname}|baseline_q90",
                gen,
                pair,
                extra={
                    "fit_seconds": fit_s,
                    "dev_chosen_FAR": dev_chosen_far(pdev, ydev, ptest, ytest, thr0),
                },
            )
        )

        # 2) PAIR-GATE identity gates
        for gamma in [0.0, 0.10]:
            gate = dev_worst <= gamma + 1e-12
            gen, pair = accept_matrix(ptest, ytest, thr0, gate=gate)
            all_rows.append(
                summarize(
                    f"{mname}|identity_gate_g{gamma}",
                    gen,
                    pair,
                    gate=gate,
                    extra={
                        "fit_seconds": fit_s,
                        "dev_chosen_FAR": dev_chosen_far(
                            pdev, ydev, ptest, ytest, thr0, gate=gate
                        ),
                    },
                )
            )

        # 3) FAR-capped full coverage
        for gamma in [0.0, 0.10]:
            thr = far_capped_thr(pdev, ydev, pcal, ycal, n, gamma)
            gen, pair = accept_matrix(ptest, ytest, thr)
            all_rows.append(
                summarize(
                    f"{mname}|farcap_g{gamma}",
                    gen,
                    pair,
                    extra={
                        "fit_seconds": fit_s,
                        "dev_chosen_FAR": dev_chosen_far(
                            pdev, ydev, ptest, ytest, thr
                        ),
                    },
                )
            )

        # 4) PAIR-CONF (new high-coverage defence)
        for top_k in [1, 3, 5]:
            verifiers, rivals = fit_pair_conf(
                xdev, ydev, pdev, thr0, n, top_k=top_k
            )
            bok_test = binary_ok_matrix(verifiers, xtest, n)
            bok_dev = binary_ok_matrix(verifiers, xdev, n)
            gen, pair = accept_matrix(ptest, ytest, thr0, binary_ok=bok_test)
            # optional: also FAR-cap after binary filter for stricter variant
            all_rows.append(
                summarize(
                    f"{mname}|PAIRCONF_k{top_k}",
                    gen,
                    pair,
                    extra={
                        "fit_seconds": fit_s,
                        "dev_chosen_FAR": dev_chosen_far(
                            pdev,
                            ydev,
                            ptest,
                            ytest,
                            thr0,
                            binary_ok_test=bok_test,
                        ),
                        "mean_rivals": float(np.mean([len(r) for r in rivals])),
                        "targets_with_verifier": int(
                            sum(v is not None for v in verifiers)
                        ),
                    },
                )
            )
            # 5) PAIR-CONF + FAR-cap hybrid
            thr = far_capped_thr(pdev, ydev, pcal, ycal, n, 0.10)
            gen, pair = accept_matrix(ptest, ytest, thr, binary_ok=bok_test)
            all_rows.append(
                summarize(
                    f"{mname}|PAIRCONF_k{top_k}+farcap0.1",
                    gen,
                    pair,
                    extra={
                        "fit_seconds": fit_s,
                        "dev_chosen_FAR": dev_chosen_far(
                            pdev,
                            ydev,
                            ptest,
                            ytest,
                            thr,
                            binary_ok_test=bok_test,
                        ),
                    },
                )
            )

        print(f"done {mname} in {fit_s:.1f}s", flush=True)

    keys = list(dict.fromkeys(k for r in all_rows for k in r))
    with (out / "matched_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)

    # Pick best full-coverage by worst FAR, then TAR
    full = [r for r in all_rows if r["coverage"] >= 0.999]
    best_full = min(full, key=lambda r: (r["FAR_worst"], -r["TAR_all"])) if full else None
    gate0 = next(r for r in all_rows if r["method"].endswith("|identity_gate_g0.0"))
    base = next(r for r in all_rows if r["method"].endswith("|baseline_q90"))

    summary = {
        "evidence_type": "SMoRFFI offline identity claims",
        "dataset_sha256": sha,
        "n_devices": n,
        "n_methods": len(all_rows),
        "baseline_RF_style": base,
        "pair_gate_identity_gate0_first_model": gate0,
        "best_full_coverage_by_worst_FAR": best_full,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps({"baseline": base, "gate0": gate0, "best_full": best_full}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
