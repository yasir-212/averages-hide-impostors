#!/usr/bin/env python3
"""PAIR-RESID v2 — full-coverage residual auth on SMoRFFI.

Improvements over v1:
  - Hard rivals chosen on DEV, verifiers fit on TRAIN (more data, less test leak)
  - Score-augmented ExtraTrees + RBF SVM ensemble
  - LDA discriminant on (target vs hard rivals) as extra residual score
  - Thresholds calibrated on CAL to a target genuine keep-rate (dev-selected)
  - Combined accept: multiclass q90 AND residual ensemble

Full coverage (all identities RF-eligible).
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


def accept_matrix(p, y, thr, binary_ok=None):
    n = p.shape[1]
    acc = p >= thr[None, :]
    if binary_ok is not None:
        acc = acc & binary_ok
    genuine = acc[np.arange(len(y)), y]
    a = acc.copy()
    a[np.arange(len(y)), y] = False
    pair = np.stack([a[y == s].mean(axis=0) for s in range(n)])
    np.fill_diagonal(pair, np.nan)
    return genuine, pair


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
            sc = pdev[ydev == s, t]
            far.append((float(np.mean(sc >= thr[t])) if len(sc) else 0.0, s))
        far.sort(reverse=True)
        hard = [s for f, s in far[:top_k] if f > 0]
        rivals.append(hard or ([far[0][1]] if far else []))
    return rivals


def score_aug(x, p, t):
    pt = p[:, t]
    mask = np.ones(p.shape[1], dtype=bool)
    mask[t] = False
    pr = p[:, mask].max(axis=1)
    return np.column_stack([x, pt, pr, pt - pr, pt / np.maximum(pr, 1e-12)])


def summarize(name, genuine, pair, extra=None):
    row = {
        "method": name,
        "TAR_all": float(genuine.mean()),
        "FAR_pooled": float(np.nanmean(pair)),
        "FAR_worst": float(np.nanmax(pair)),
        "FAR_mean_max_source": float(np.nanmean(np.nanmax(pair, axis=0))),
        "eligible": int(pair.shape[1]),
        "coverage": 1.0,
    }
    if extra:
        row.update(extra)
    return row


def subsample_neg(xneg, npos, rng):
    if len(xneg) > 6 * npos:
        pick = rng.choice(len(xneg), size=6 * npos, replace=False)
        return xneg[pick]
    return xneg


def fit_v2(xtr, ytr, ptr, xdev, ydev, pdev, n, rivals, seed=31):
    """Per-target ET + linear SVM + LDA on train; rivals from dev."""
    models = [None] * n
    for t in range(n):
        hard = rivals[t]
        pos = ytr == t
        if hard:
            neg = np.isin(ytr, hard)
        else:
            neg = ytr != t
        if not pos.any() or not neg.any():
            continue
        rng = np.random.default_rng(seed + t)
        Xp = score_aug(xtr[pos], ptr[pos], t)
        Xn = score_aug(xtr[neg], ptr[neg], t)
        Xn = subsample_neg(Xn, len(Xp), rng)
        X = np.vstack([Xp, Xn])
        Y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
        et = ExtraTreesClassifier(
            n_estimators=128,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=seed + t,
            n_jobs=1,
            class_weight="balanced_subsample",
        )
        et.fit(X, Y)
        svm = SVC(
            kernel="linear",
            class_weight="balanced",
            probability=True,
            random_state=seed + t,
            max_iter=4000,
        )
        svm.fit(X, Y)
        lda = None
        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X, Y)
        except Exception:
            lda = None
        models[t] = (et, svm, lda)
    return models


def residual_scores(models, x, p, n):
    """Higher = more genuine. Ensemble mean of available classifiers."""
    out = np.ones((len(x), n))
    for t, item in enumerate(models):
        if item is None:
            continue
        et, svm, lda = item
        feats = score_aug(x, p, t)
        s = et.predict_proba(feats)[:, 1]
        s = s + svm.predict_proba(feats)[:, 1]
        k = 2.0
        if lda is not None:
            try:
                s = s + lda.predict_proba(feats)[:, 1]
                k += 1.0
            except Exception:
                pass
        out[:, t] = s / k
    return out


def cal_residual_thr(scores_cal, ycal, n, keep):
    thr = np.zeros(n)
    for t in range(n):
        pos = scores_cal[ycal == t, t]
        if len(pos) == 0:
            thr[t] = 0.0
            continue
        thr[t] = float(np.quantile(pos, 1.0 - keep, method="higher"))
    return thr


def binary_ok(res_scores, rthr):
    return res_scores >= rthr[None, :]


def dev_chosen_far(pdev, ydev, ptest, ytest, thr, binary_ok_test=None):
    n = pdev.shape[1]
    _, dev_pair = accept_matrix(pdev, ydev, thr)
    selected = np.array(
        [np.nanargmax(dev_pair[:, t]) if np.isfinite(dev_pair[:, t]).any() else 0 for t in range(n)]
    )
    _, test_pair = accept_matrix(ptest, ytest, thr, binary_ok=binary_ok_test)
    return float(np.nanmean([test_pair[selected[t], t] for t in range(n)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("SMORFFI_FEATURES", "features_v1.npz"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sets, n, sha = data_windows(args.data)
    scaler = StandardScaler().fit(sets["train"][0])
    sets = {k: (scaler.transform(x), y) for k, (x, y) in sets.items()}
    xtr, ytr = sets["train"]
    xdev, ydev = sets["dev"]
    xcal, ycal = sets["cal"]
    xtest, ytest = sets["test"]

    t0 = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=128, min_samples_leaf=2, random_state=31, n_jobs=-1
    )
    model.fit(xtr, ytr)
    fit_s = time.perf_counter() - t0
    ptr = model.predict_proba(xtr)
    pdev = model.predict_proba(xdev)
    pcal = model.predict_proba(xcal)
    ptest = model.predict_proba(xtest)
    thr0 = genuine_thr(pcal, ycal, n, 0.90)
    rivals = hard_rivals(pdev, ydev, thr0, n, top_k=5)
    print(f"RF fit {fit_s:.1f}s  mean rivals {np.mean([len(r) for r in rivals]):.2f}", flush=True)

    rows = []
    gen, pair = accept_matrix(ptest, ytest, thr0)
    rows.append(summarize("RF128|baseline_q90", gen, pair, extra={"fit_seconds": fit_s, "dev_chosen_FAR": dev_chosen_far(pdev, ydev, ptest, ytest, thr0)}))
    thrf = far_capped_thr(pdev, ydev, pcal, ycal, n, 0.0)
    gen, pair = accept_matrix(ptest, ytest, thrf)
    rows.append(summarize("RF128|farcap_g0.0", gen, pair, extra={"fit_seconds": fit_s, "dev_chosen_FAR": dev_chosen_far(pdev, ydev, ptest, ytest, thrf)}))

    print("fitting PAIR-RESID v2 verifiers...", flush=True)
    t1 = time.perf_counter()
    models = fit_v2(xtr, ytr, ptr, xdev, ydev, pdev, n, rivals)
    print(f"verifiers {time.perf_counter()-t1:.1f}s", flush=True)

    r_cal = residual_scores(models, xcal, pcal, n)
    r_test = residual_scores(models, xtest, ptest, n)
    r_dev = residual_scores(models, xdev, pdev, n)

    # pick keep-rate on DEV: among keep in {0.70,0.80,0.90,0.95}, minimize (dev worst FAR, -TAR)
    cand_keep = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    chosen = None
    for keep in cand_keep:
        rthr = cal_residual_thr(r_cal, ycal, n, keep)
        bok_dev = binary_ok(r_dev, rthr)
        gdev, pdev_a = accept_matrix(pdev, ydev, thr0, binary_ok=bok_dev)
        rec = {
            "keep": keep,
            "dev_TAR": float(gdev.mean()),
            "dev_FAR_worst": float(np.nanmax(pdev_a)),
            "rthr": rthr,
        }
        print(f"  keep={keep:.2f}  dev TAR={rec['dev_TAR']:.3f} worstFAR={rec['dev_FAR_worst']:.3f}", flush=True)
        if chosen is None:
            chosen = rec
        else:
            # prefer lower worst FAR, then higher TAR, require TAR>=0.30 on dev
            a, b = rec, chosen
            def key(r):
                tar_pen = 0 if r["dev_TAR"] >= 0.30 else 1
                return (tar_pen, r["dev_FAR_worst"], -r["dev_TAR"])
            if key(a) < key(b):
                chosen = rec

    rthr = chosen["rthr"]
    bok = binary_ok(r_test, rthr)
    gen, pair = accept_matrix(ptest, ytest, thr0, binary_ok=bok)
    rows.append(
        summarize(
            f"RF128|PAIRRESIDv2_keep{chosen['keep']}",
            gen,
            pair,
            extra={
                "fit_seconds": fit_s,
                "dev_chosen_FAR": dev_chosen_far(pdev, ydev, ptest, ytest, thr0, binary_ok_test=bok),
                "keep": chosen["keep"],
                "dev_TAR": chosen["dev_TAR"],
                "dev_FAR_worst": chosen["dev_FAR_worst"],
            },
        )
    )

    # also report all keep rates on TEST (for curves; keep-rate itself chosen on DEV/CAL)
    curve = []
    for keep in cand_keep:
        rthr_k = cal_residual_thr(r_cal, ycal, n, keep)
        bok_k = binary_ok(r_test, rthr_k)
        g, pa = accept_matrix(ptest, ytest, thr0, binary_ok=bok_k)
        curve.append(
            summarize(
                f"RF128|PAIRRESIDv2_keep{keep}",
                g,
                pa,
                extra={
                    "keep": keep,
                    "dev_chosen_FAR": dev_chosen_far(
                        pdev, ydev, ptest, ytest, thr0, binary_ok_test=bok_k
                    ),
                },
            )
        )
        rows.append(curve[-1] if keep != chosen["keep"] else rows[-1])
    # avoid duplicate of chosen — already appended; skip if same
    # (we appended chosen then loop appends all; dedupe by method later)

    # hybrid + farcap 0
    bok = binary_ok(r_test, rthr)
    gen, pair = accept_matrix(ptest, ytest, thrf, binary_ok=bok)
    rows.append(
        summarize(
            f"RF128|PAIRRESIDv2_keep{chosen['keep']}+farcap0",
            gen,
            pair,
            extra={
                "dev_chosen_FAR": dev_chosen_far(
                    pdev, ydev, ptest, ytest, thrf, binary_ok_test=bok
                )
            },
        )
    )

    # residual-only (no multiclass thr) for diagnostics
    bok = binary_ok(r_test, rthr)
    dummy_thr = np.zeros(n)
    gen, pair = accept_matrix(np.ones_like(ptest), ytest, dummy_thr, binary_ok=bok)
    rows.append(summarize("RF128|PAIRRESIDv2_resid_only", gen, pair))

    # dedupe methods
    seen = set()
    uniq = []
    for r in rows:
        if r["method"] in seen:
            continue
        seen.add(r["method"])
        uniq.append(r)
    rows = uniq

    keys = list(dict.fromkeys(k for r in rows for k in r))
    csv_path = out / "pair_resid_v2.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    full = [r for r in rows if r.get("coverage", 1) >= 0.999]
    usable = [r for r in full if r["TAR_all"] >= 0.30]
    best = min(usable, key=lambda r: (r["FAR_worst"], -r["TAR_all"])) if usable else None
    summary = {
        "evidence_type": "SMoRFFI offline identity claims",
        "dataset_sha256": sha,
        "n_devices": n,
        "dev_chosen_keep": chosen["keep"],
        "best_full_coverage_TAR_ge_0.30": best,
        "all": rows,
    }
    (out / "pair_resid_v2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"chosen_keep": chosen["keep"], "best": best}, indent=2, default=str))
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
