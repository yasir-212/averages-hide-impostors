#!/usr/bin/env python3
"""PAIR-RESID: full-coverage pair-conditioned residual verifier, compared with PAIR-GATE / FAR-cap / PAIR-CONF.

Keeps every identity RF-eligible. Combines:
  - multiclass score threshold (calibrated genuine quantile),
  - prototype distance-ratio against development hard rivals,
  - score-augmented ExtraTrees binary residual verifiers,
  - optional FAR-capped multiclass thresholds.

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


def hard_rivals(pdev, ydev, thr, n, top_k=5):
    rivals = []
    for t in range(n):
        far = []
        for s in range(n):
            if s == t:
                continue
            scores = pdev[ydev == s, t]
            far.append((float(np.mean(scores >= thr[t])) if len(scores) else 0.0, s))
        far.sort(reverse=True)
        rivals.append([s for f, s in far[:top_k] if f > 0] or [far[0][1]])
    return rivals


def enroll_prototypes(xcal, ycal, n):
    protos = np.zeros((n, xcal.shape[1]), dtype=np.float64)
    for t in range(n):
        xt = xcal[ycal == t]
        protos[t] = xt.mean(axis=0) if len(xt) else 0.0
    return protos


def proto_ratio_ok(x, protos, rivals, ratio_thr):
    # ok[i,t] if dist(x_i, proto_t) / min_rival_dist <= ratio_thr[t]
    n = protos.shape[0]
    ok = np.ones((len(x), n), dtype=bool)
    # distances to all prototypes: (N, n)
    # ||x - mu||^2 = ||x||^2 + ||mu||^2 - 2 x·mu
    x2 = np.sum(x * x, axis=1, keepdims=True)
    p2 = np.sum(protos * protos, axis=1)[None, :]
    d2 = np.maximum(x2 + p2 - 2.0 * (x @ protos.T), 0.0)
    d = np.sqrt(d2)
    for t in range(n):
        rlist = rivals[t]
        rival_d = d[:, rlist].min(axis=1)
        ratio = d[:, t] / np.maximum(rival_d, 1e-12)
        ok[:, t] = ratio <= ratio_thr[t]
    return ok


def calibrate_ratio_thr(xcal, ycal, protos, rivals, n, keep=0.95):
    thr = np.ones(n)
    d_ok = proto_ratio_ok  # noqa: F841 — clarity
    x2 = np.sum(xcal * xcal, axis=1, keepdims=True)
    p2 = np.sum(protos * protos, axis=1)[None, :]
    d = np.sqrt(np.maximum(x2 + p2 - 2.0 * (xcal @ protos.T), 0.0))
    for t in range(n):
        mask = ycal == t
        if not mask.any():
            thr[t] = 1.0
            continue
        rival_d = d[mask][:, rivals[t]].min(axis=1)
        ratio = d[mask, t] / np.maximum(rival_d, 1e-12)
        thr[t] = float(np.quantile(ratio, keep, method="higher"))
    return thr


def score_aug(x, p, t):
    pt = p[:, t]
    # max rival score
    mask = np.ones(p.shape[1], dtype=bool)
    mask[t] = False
    pr = p[:, mask].max(axis=1)
    return np.column_stack([x, pt, pr, pt - pr, pt / np.maximum(pr, 1e-12)])


def fit_resid_verifiers(xdev, ydev, pdev, thr0, n, rivals, mode="hard", seed=31):
    verifiers = [None] * n
    for t in range(n):
        xpos = score_aug(xdev[ydev == t], pdev[ydev == t], t)
        if mode == "all":
            neg_idx = ydev != t
        else:
            hard = rivals[t]
            neg_idx = np.isin(ydev, hard)
        if not neg_idx.any() or len(xpos) < 5:
            continue
        xneg = score_aug(xdev[neg_idx], pdev[neg_idx], t)
        # subsample negatives for speed/balance
        rng = np.random.default_rng(seed + t)
        if len(xneg) > 4 * len(xpos):
            pick = rng.choice(len(xneg), size=4 * len(xpos), replace=False)
            xneg = xneg[pick]
        X = np.vstack([xpos, xneg])
        Y = np.r_[np.ones(len(xpos)), np.zeros(len(xneg))]
        clf = ExtraTreesClassifier(
            n_estimators=64,
            min_samples_leaf=2,
            random_state=seed + t,
            n_jobs=1,
            class_weight="balanced_subsample",
        )
        clf.fit(X, Y)
        pos_scores = clf.predict_proba(xpos)[:, 1]
        bthr = float(np.quantile(pos_scores, 0.05, method="higher"))
        verifiers[t] = (clf, bthr)
    return verifiers


def binary_ok_resid(verifiers, x, p, n):
    ok = np.ones((len(x), n), dtype=bool)
    for t, item in enumerate(verifiers):
        if item is None:
            continue
        clf, bthr = item
        feats = score_aug(x, p, t)
        ok[:, t] = clf.predict_proba(feats)[:, 1] >= bthr
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
            max_iter=120,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=10,
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
    ap.add_argument("--models", nargs="+", default=["RF128", "MLP64"])
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
        rivals = hard_rivals(pdev, ydev, thr0, n, top_k=5)
        protos = enroll_prototypes(xcal, ycal, n)
        rthr = calibrate_ratio_thr(xcal, ycal, protos, rivals, n, keep=0.95)

        # references
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
        gate = dev_worst <= 1e-12
        gen, pair = accept_matrix(ptest, ytest, thr0, gate=gate)
        all_rows.append(
            summarize(
                f"{mname}|identity_gate_g0.0",
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
        for gamma in (0.0, 0.10):
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

        # PROTO ratio alone / with multiclass
        pok_test = proto_ratio_ok(xtest, protos, rivals, rthr)
        gen, pair = accept_matrix(ptest, ytest, thr0, binary_ok=pok_test)
        all_rows.append(
            summarize(
                f"{mname}|PROTO_ratio",
                gen,
                pair,
                extra={
                    "fit_seconds": fit_s,
                    "dev_chosen_FAR": dev_chosen_far(
                        pdev, ydev, ptest, ytest, thr0, binary_ok_test=pok_test
                    ),
                },
            )
        )

        # PAIR-RESID hard / all
        for mode in ("hard", "all"):
            ver = fit_resid_verifiers(
                xdev, ydev, pdev, thr0, n, rivals, mode=mode
            )
            bok = binary_ok_resid(ver, xtest, ptest, n)
            gen, pair = accept_matrix(ptest, ytest, thr0, binary_ok=bok)
            all_rows.append(
                summarize(
                    f"{mname}|PAIRRESID_{mode}",
                    gen,
                    pair,
                    extra={
                        "fit_seconds": fit_s,
                        "dev_chosen_FAR": dev_chosen_far(
                            pdev, ydev, ptest, ytest, thr0, binary_ok_test=bok
                        ),
                        "targets_with_verifier": int(
                            sum(v is not None for v in ver)
                        ),
                    },
                )
            )
            # hybrid: resid + proto
            hybrid = bok & pok_test
            gen, pair = accept_matrix(ptest, ytest, thr0, binary_ok=hybrid)
            all_rows.append(
                summarize(
                    f"{mname}|PAIRRESID_{mode}+PROTO",
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
                            binary_ok_test=hybrid,
                        ),
                    },
                )
            )
            # hybrid + farcap 0.1
            thr = far_capped_thr(pdev, ydev, pcal, ycal, n, 0.10)
            gen, pair = accept_matrix(ptest, ytest, thr, binary_ok=hybrid)
            all_rows.append(
                summarize(
                    f"{mname}|PAIRRESID_{mode}+PROTO+farcap0.1",
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
                            binary_ok_test=hybrid,
                        ),
                    },
                )
            )
            # hybrid + farcap 0.0
            thr0f = far_capped_thr(pdev, ydev, pcal, ycal, n, 0.0)
            gen, pair = accept_matrix(ptest, ytest, thr0f, binary_ok=hybrid)
            all_rows.append(
                summarize(
                    f"{mname}|PAIRRESID_{mode}+PROTO+farcap0.0",
                    gen,
                    pair,
                    extra={
                        "fit_seconds": fit_s,
                        "dev_chosen_FAR": dev_chosen_far(
                            pdev,
                            ydev,
                            ptest,
                            ytest,
                            thr0f,
                            binary_ok_test=hybrid,
                        ),
                    },
                )
            )

        print(f"done {mname} in {fit_s:.1f}s", flush=True)

    keys = list(dict.fromkeys(k for r in all_rows for k in r))
    csv_path = out / "pair_resid_comparison.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)

    full = [r for r in all_rows if r["coverage"] >= 0.999]
    best_full = (
        min(full, key=lambda r: (r["FAR_worst"], -r["TAR_all"])) if full else None
    )
    # also best by (dev_chosen_FAR, FAR_worst) among full coverage with TAR>=0.35
    usable = [r for r in full if r["TAR_all"] >= 0.35]
    best_usable = (
        min(
            usable,
            key=lambda r: (
                r.get("dev_chosen_FAR") or 1.0,
                r["FAR_worst"],
                -r["TAR_all"],
            ),
        )
        if usable
        else None
    )
    gate0 = next(r for r in all_rows if r["method"].endswith("|identity_gate_g0.0"))
    summary = {
        "evidence_type": "SMoRFFI offline identity claims",
        "dataset_sha256": sha,
        "n_devices": n,
        "pair_gate_identity_gate0": gate0,
        "best_full_coverage_by_worst_FAR": best_full,
        "best_full_coverage_TAR_ge_0.35": best_usable,
    }
    (out / "pair_resid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"gate0": gate0, "best_full": best_full, "best_usable": best_usable}, indent=2))
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
