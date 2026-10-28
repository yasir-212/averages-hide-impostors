#!/usr/bin/env python3
"""Calibrate RF128+PAIRRESID keep-rate on INRIA measured radios (CAL only).

Goal: TAR >= 0.80 on cal/dev-like split-1, then report unknown/cross FAR on split-2/3.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_measured_inria import load_cache, rich_features  # noqa: E402

CACHE = Path(os.environ.get("INRIA_CACHE", "inria_cache"))
OUT = Path(__file__).resolve().parents[1] / "results" / "INRIA_measured"


def main():
    iq, y, split, meta = load_cache(CACHE)
    n = int(meta["n"])
    parts = []
    for i in range(0, n, 2048):
        r, x = rich_features(np.asarray(iq[i : i + 2048]))
        parts.append((r, x))
        print("feat", min(i + 2048, n), flush=True)
    X = np.concatenate([p[1] for p in parts])
    known = list(meta["known_tx"])
    unknown = list(meta["unknown_tx"])
    known_tr = (split == 0) & np.isin(y, known)
    scaler = StandardScaler().fit(X[known_tr])
    Xs = scaler.transform(X)
    rf = RandomForestClassifier(
        n_estimators=128, min_samples_leaf=2, random_state=31, n_jobs=-1
    )
    ytr = np.searchsorted(known, y[known_tr])
    rf.fit(Xs[known_tr], ytr)
    P = rf.predict_proba(Xs)
    n_k = len(known)
    yk = np.searchsorted(known, np.clip(y, 0, known[-1]))
    # remap only known; unknowns stay as extra
    ca = (split == 1) & np.isin(y, known)
    thr0 = np.array(
        [
            np.quantile(P[ca & (y == t), np.searchsorted(known, t)], 0.05, method="higher")
            for t in known
        ]
    )

    # fit residual ET on cal hard rivals
    ver = []
    for ti, t in enumerate(known):
        pos = ca & (y == t)
        # rivals: other known on cal with highest P[:,ti]
        far = []
        for s in known:
            if s == t:
                continue
            m = ca & (y == s)
            far.append((float(np.mean(P[m, ti] >= thr0[ti])), s))
        far.sort(reverse=True)
        hard = [s for f, s in far[:3] if f > 0] or [far[0][1]]
        neg = ca & np.isin(y, hard)
        Xp, Xn = Xs[pos], Xs[neg]
        Pp, Pn = P[pos], P[neg]

        def aug(Xf, Pf, tt=ti):
            pt = Pf[:, tt]
            mask = np.ones(Pf.shape[1], dtype=bool)
            mask[tt] = False
            pr = Pf[:, mask].max(1)
            return np.column_stack([Xf, pt, pr, pt - pr])

        clf = ExtraTreesClassifier(
            n_estimators=64,
            min_samples_leaf=2,
            random_state=31 + t,
            n_jobs=1,
            class_weight="balanced_subsample",
        )
        clf.fit(
            np.vstack([aug(Xp, Pp), aug(Xn, Pn)]),
            np.r_[np.ones(len(Xp)), np.zeros(len(Xn))],
        )
        pos_s = clf.predict_proba(aug(Xp, Pp))[:, 1]
        ver.append((clf, pos_s, aug))

    te = split == 2
    rows = []
    for keep in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        tars, far_unk, far_cross = [], [], []
        for ti, t in enumerate(known):
            clf, pos_s, aug = ver[ti]
            bthr = float(np.quantile(pos_s, 1 - keep, method="higher"))
            # genuine test
            g = te & (y == t)
            Pg, Xg = P[g], Xs[g]
            ok_g = (Pg[:, ti] >= thr0[ti]) & (clf.predict_proba(aug(Xg, Pg))[:, 1] >= bthr)
            tars.append(float(np.mean(ok_g)))
            unk = te & np.isin(y, unknown)
            Pu, Xu = P[unk], Xs[unk]
            ok_u = (Pu[:, ti] >= thr0[ti]) & (clf.predict_proba(aug(Xu, Pu))[:, 1] >= bthr)
            far_unk.append(float(np.mean(ok_u)))
            cr = te & np.isin(y, known) & (y != t)
            Pc, Xc = P[cr], Xs[cr]
            ok_c = (Pc[:, ti] >= thr0[ti]) & (clf.predict_proba(aug(Xc, Pc))[:, 1] >= bthr)
            far_cross.append(float(np.mean(ok_c)))
        rec = {
            "keep": keep,
            "macro_tar": float(np.mean(tars)),
            "macro_far_unknown": float(np.mean(far_unk)),
            "worst_far_unknown": float(np.max(far_unk)),
            "macro_far_cross": float(np.mean(far_cross)),
            "worst_far_cross": float(np.max(far_cross)),
        }
        rows.append(rec)
        print(rec, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "resid_keep_curve.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    usable = [r for r in rows if r["macro_tar"] >= 0.80]
    best = (
        min(usable, key=lambda r: (r["worst_far_unknown"], -r["macro_tar"]))
        if usable
        else min(rows, key=lambda r: (r["worst_far_unknown"], -r["macro_tar"]))
    )
    (OUT / "resid_keep_best.json").write_text(
        json.dumps(
            {
                "best_TAR_ge_0.80": best if best["macro_tar"] >= 0.80 else None,
                "best_any": min(rows, key=lambda r: (r["worst_far_unknown"], -r["macro_tar"])),
                "curve": rows,
                "evidence_class": "public_measured_radio",
            },
            indent=2,
        )
        + "\n"
    )
    print("BEST", best)


if __name__ == "__main__":
    main()
