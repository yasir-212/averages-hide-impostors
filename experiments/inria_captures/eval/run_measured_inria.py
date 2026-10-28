#!/usr/bin/env python3
"""Measured radios: INRIA PLA BladeRF I/Q (Zenodo 18268648), public measured
2.4 GHz BladeRF AX4 I/Q from 12 COTS transmitters (Alla et al., ACSAC 2024).

Impersonation = a physically different radio (unknown TX 9–11, or another
known TX) claiming a known identity. Adaptive digital stress = NN-select and
centroid blend in feature space (windows are already unit-power, so A1 gain
match is nearly a no-op and is reported as such).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import math

import numpy as np
from sklearn.covariance import OAS
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CACHE = Path(os.environ.get("INRIA_CACHE", "inria_cache"))
OUT = Path(__file__).resolve().parents[1] / "results" / "INRIA_measured"


def load_cache(cache: Path):
    meta = json.loads((cache / "meta.json").read_text())
    n, win = int(meta["n"]), int(meta["win"])
    iq = np.memmap(cache / "iq_f16.dat", dtype=np.float16, mode="r", shape=(n, 2, win))
    y = np.load(cache / "y.npy")
    split = np.load(cache / "split.npy")
    return iq, y, split, meta


def _moments(values: np.ndarray):
    mean = np.mean(values, axis=1)
    centered = values - mean[:, None]
    std = np.sqrt(np.mean(centered ** 2, axis=1))
    safe = np.maximum(std, 1e-12)
    normalized = centered / safe[:, None]
    skewness = np.mean(normalized ** 3, axis=1)
    excess_kurtosis = np.mean(normalized ** 4, axis=1) - 3.0
    return mean, std, skewness, excess_kurtosis


def rich_features(iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Length-agnostic 23-D IQ descriptors (same family as Zenodo extractor).

    iq: (N, 2, L) real/imag.
    """
    re = np.asarray(iq[:, 0], dtype=np.float64)
    im = np.asarray(iq[:, 1], dtype=np.float64)
    batch = re + 1j * im
    sample_count = batch.shape[1]
    frequency = np.fft.fftshift(np.fft.fftfreq(sample_count))
    frequency_centered = frequency - np.mean(frequency)
    spectral_slope_denominator = np.sum(frequency_centered ** 2)
    epsilon = np.finfo(np.float64).tiny
    rms = np.sqrt(np.mean(np.abs(batch) ** 2, axis=1))
    rms = np.maximum(rms, 1e-12)
    log_rms = np.log(rms)
    normalized = batch / rms[:, None]
    real, imag = normalized.real, normalized.imag
    time_stats = (*_moments(real), *_moments(imag))
    power = np.abs(np.fft.fftshift(np.fft.fft(normalized, axis=1), axes=1)) ** 2
    normalized_power = power / np.maximum(np.sum(power, axis=1, keepdims=True), epsilon)
    spectral_floor = np.maximum(np.max(power, axis=1, keepdims=True) * 1e-12, 1e-15)
    spectral_centroid = np.sum(normalized_power * frequency[None, :], axis=1)
    spectral_spread = np.sqrt(
        np.sum(normalized_power * (frequency[None, :] - spectral_centroid[:, None]) ** 2, axis=1)
    )
    spectral_flatness = np.exp(np.mean(np.log(np.maximum(power, spectral_floor)), axis=1)) / np.maximum(
        np.mean(power, axis=1), epsilon
    )
    spectral_entropy = -np.sum(normalized_power * np.log(normalized_power + epsilon), axis=1) / math.log(
        sample_count
    )
    log_power = np.log(np.maximum(power, spectral_floor))
    spectral_slope = np.sum(log_power * frequency_centered[None, :], axis=1) / spectral_slope_denominator
    real_var = np.var(real, axis=1)
    imag_var = np.var(imag, axis=1)
    iq_variance_ratio_db = 10 * np.log10(np.maximum(real_var, epsilon) / np.maximum(imag_var, epsilon))
    real_centered = real - np.mean(real, axis=1, keepdims=True)
    imag_centered = imag - np.mean(imag, axis=1, keepdims=True)
    iq_correlation = np.mean(real_centered * imag_centered, axis=1) / np.maximum(
        np.sqrt(real_var * imag_var), epsilon
    )
    increments = np.angle(normalized[:, 1:] * np.conj(normalized[:, :-1]))
    cfo_mean = np.angle(np.mean(np.exp(1j * increments), axis=1))
    phase_residual = np.angle(np.exp(1j * (increments - cfo_mean[:, None])))
    _, phase_std, phase_skew, phase_kurtosis = _moments(phase_residual)
    phase_present = phase_std > 1e-6
    phase_stats = (
        cfo_mean,
        np.where(phase_present, np.mean(phase_residual ** 2, axis=1), 0.0),
        np.where(phase_present, phase_std, 0.0),
        np.where(phase_present, phase_skew, 0.0),
        np.where(phase_present, phase_kurtosis, 0.0),
    )
    autocorrelation = []
    for component in (real_centered, imag_centered):
        numerator = np.sum(component[:, :-1] * component[:, 1:], axis=1)
        denominator = np.sqrt(
            np.sum(component[:, :-1] ** 2, axis=1) * np.sum(component[:, 1:] ** 2, axis=1)
        )
        autocorrelation.append(numerator / np.maximum(denominator, epsilon))
    quarter = sample_count // 4
    quarter_energy = np.stack(
        [np.mean(np.abs(normalized[:, i * quarter : (i + 1) * quarter]) ** 2, axis=1) for i in range(4)],
        axis=1,
    )
    energy_variation = np.std(quarter_energy, axis=1)
    feats = np.column_stack(
        [
            *time_stats,
            spectral_centroid,
            spectral_spread,
            spectral_flatness,
            spectral_entropy,
            spectral_slope,
            iq_variance_ratio_db,
            iq_correlation,
            *phase_stats,
            *autocorrelation,
            energy_variation,
        ]
    )
    return log_rms, feats


def conformal_thr(scores, target_frr=0.05):
    n = len(scores)
    k = int(np.ceil((1.0 - target_frr) * (n + 1))) - 1
    k = min(max(k, 0), n - 1)
    return float(np.sort(scores)[k])


def mahal_fn(oas):
    loc, prec = oas.location_, oas.precision_

    def mahal(Z):
        d = Z - loc
        return np.einsum("ni,ij,nj->n", d, prec, d)

    return mahal


def nn_select(X, centroid, keep_frac=0.35):
    d = np.linalg.norm(X - centroid[None, :], axis=1)
    k = max(1, int(np.ceil(keep_frac * len(X))))
    idx = np.argsort(d)[:k]
    return X[idx], idx


def blend_to(X, centroid, alpha):
    return (1.0 - alpha) * X + alpha * centroid[None, :]


def eval_target(t, X, R, y, split, known, unknown, seed=2026):
    tr = (y == t) & (split == 0)
    ca = (y == t) & (split == 1)
    te = (y == t) & (split == 2)
    p2 = (y == t) & (split == 3)
    Xtr, Xca, Xte, Xp2 = X[tr], X[ca], X[te], X[p2]
    Rtr, Rca, Rte, Rp2 = R[tr], R[ca], R[te], R[p2]
    if min(len(Xtr), len(Xca), len(Xte)) < 20:
        return None

    scaler = StandardScaler().fit(Xtr)
    Ztr, Zca, Zte, Zp2 = map(scaler.transform, (Xtr, Xca, Xte, Xp2))
    centroid = Xtr.mean(axis=0)

    oas = OAS().fit(Ztr)
    mahal = mahal_fn(oas)
    rng = np.random.default_rng(seed + t)
    Zneg = np.vstack(
        [Ztr[rng.permutation(len(Ztr))], Ztr * rng.uniform(0.7, 1.3, size=(len(Ztr), 1))]
    )
    Ybin = np.r_[np.ones(len(Ztr)), np.zeros(len(Zneg))]
    lr = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed, solver="lbfgs"
    ).fit(np.vstack([Ztr, Zneg]), Ybin)
    mtr, mneg = mahal(Ztr), mahal(Zneg)
    et = ExtraTreesClassifier(
        n_estimators=96,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=2,
        class_weight="balanced_subsample",
    )
    et.fit(
        np.vstack(
            [
                np.column_stack([Ztr, mtr, 1 / np.maximum(mtr, 1e-9)]),
                np.column_stack([Zneg, mneg, 1 / np.maximum(mneg, 1e-9)]),
            ]
        ),
        Ybin,
    )

    center = float(np.median(Rtr))
    scale_e = max(float(1.4826 * np.median(np.abs(Rtr - center))), 1e-9)
    thr_e = conformal_thr(np.abs(Rca - center) / scale_e)
    thr_m = conformal_thr(mahal(Zca))
    thr_lr = float(np.quantile(lr.predict_proba(Zca)[:, 1], 0.05, method="higher"))
    feats_ca = np.column_stack([Zca, mahal(Zca), 1 / np.maximum(mahal(Zca), 1e-9)])
    thr_et = float(np.quantile(et.predict_proba(feats_ca)[:, 1], 0.05, method="higher"))

    def energy_ok(R):
        return np.abs(R - center) / scale_e <= thr_e

    def oas_ok(Z):
        return mahal(Z) <= thr_m

    def pairconf_ok(Z):
        return (mahal(Z) <= thr_m) & (lr.predict_proba(Z)[:, 1] >= thr_lr)

    def pairresid_ok(Z):
        m = mahal(Z)
        f = np.column_stack([Z, m, 1 / np.maximum(m, 1e-9)])
        return (m <= thr_m) & (et.predict_proba(f)[:, 1] >= thr_et)

    defences = {
        "EnergyOnly": energy_ok,
        "OAS": oas_ok,
        "PAIRCONF": pairconf_ok,
        "PAIRRESID": pairresid_ok,
    }

    # attack feature sets (in original X space, then scaled)
    cross_mask = (split == 2) & np.isin(y, [k for k in known if k != t])
    unk_mask = (split == 2) & np.isin(y, unknown)
    Xcross, Rcross = X[cross_mask], R[cross_mask]
    Xunk, Runk = X[unk_mask], R[unk_mask]
    Xnn, inn = nn_select(Xunk, centroid, 0.35) if len(Xunk) else (Xunk, np.array([], int))
    Rnn = Runk[inn] if len(inn) else Runk
    Xnn10, i10 = nn_select(Xunk, centroid, 0.10) if len(Xunk) else (Xunk, np.array([], int))
    Rnn10 = Runk[i10] if len(i10) else Runk
    Xblend = blend_to(Xnn10, centroid, 0.55) if len(Xnn10) else Xnn10

    attacks = {
        "genuine_p1test": (Xte, Rte),
        "genuine_p2": (Xp2, Rp2),
        "cross_known": (Xcross, Rcross),
        "unknown_raw": (Xunk, Runk),
        "unknown_nn35": (Xnn, Rnn),
        "unknown_nn10": (Xnn10, Rnn10),
        "unknown_nn10_blend55": (Xblend, Rnn10),
    }

    rows = []
    for dname, fn in defences.items():
        if dname == "EnergyOnly":
            tar_p1 = float(np.mean(energy_ok(Rte)))
            tar_p2 = float(np.mean(energy_ok(Rp2))) if len(Xp2) else None
        else:
            tar_p1 = float(np.mean(fn(Zte)))
            tar_p2 = float(np.mean(fn(scaler.transform(Xp2)))) if len(Xp2) else None
        for aname, (Xa, Ra) in attacks.items():
            if aname.startswith("genuine"):
                continue
            if len(Xa) == 0:
                far = None
            elif dname == "EnergyOnly":
                far = float(np.mean(energy_ok(Ra)))
            else:
                far = float(np.mean(fn(scaler.transform(Xa))))
            rows.append(
                {
                    "target": int(t),
                    "defence": dname,
                    "attack": aname,
                    "tar_p1": tar_p1,
                    "tar_p2": tar_p2,
                    "far": far,
                    "n_attack": int(len(Xa)),
                    "n_genuine_p1": int(len(Xte)),
                    "n_genuine_p2": int(len(Xp2)),
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    iq, y, split, meta = load_cache(cache)
    print("extracting features from", meta["n"], "windows...", flush=True)
    # memmap slice to numpy in chunks
    R_parts, X_parts = [], []
    bs = 1024
    n = int(meta["n"])
    for i in range(0, n, bs):
        sl = np.asarray(iq[i : i + bs])
        r, x = rich_features(sl)
        R_parts.append(r)
        X_parts.append(x)
        print(f"  {min(i+bs,n)}/{n}", flush=True)
    R = np.concatenate(R_parts)
    X = np.concatenate(X_parts)
    known = list(meta["known_tx"])
    unknown = list(meta["unknown_tx"])

    all_rows = []
    for t in known:
        rows = eval_target(t, X, R, y, split, known, unknown)
        if rows:
            all_rows.extend(rows)
            print("target", t, "one-class done", flush=True)

    # Multiclass RF128 + PAIR-RESID residual on known devices (SMoRFFI-style).
    print("fitting RF128 multiclass on known train...", flush=True)
    known_tr = (split == 0) & np.isin(y, known)
    known_ca = (split == 1) & np.isin(y, known)
    scaler_m = StandardScaler().fit(X[known_tr])
    rf = RandomForestClassifier(
        n_estimators=128, min_samples_leaf=2, random_state=31, n_jobs=-1
    )
    ytr = np.searchsorted(known, y[known_tr])
    rf.fit(scaler_m.transform(X[known_tr]), ytr)
    Pca = rf.predict_proba(scaler_m.transform(X[known_ca]))
    yca = np.searchsorted(known, y[known_ca])
    n_k = len(known)
    thr0 = np.array(
        [np.quantile(Pca[yca == j, j], 0.05, method="higher") for j in range(n_k)]
    )
    Pdev = rf.predict_proba(scaler_m.transform(X[(split == 1) & np.isin(y, known)]))
    ydev = np.searchsorted(known, y[(split == 1) & np.isin(y, known)])
    # hard rival = known source with highest FAR at thr0 on cal
    rivals = []
    for t in range(n_k):
        far = []
        for s in range(n_k):
            if s == t:
                continue
            far.append((float(np.mean(Pdev[ydev == s, t] >= thr0[t])), s))
        far.sort(reverse=True)
        rivals.append([s for f, s in far[:3] if f > 0] or ([far[0][1]] if far else []))
    Xdev = scaler_m.transform(X[(split == 1) & np.isin(y, known)])
    verifiers = [None] * n_k
    for t in range(n_k):
        hard = rivals[t]
        if not hard:
            continue
        pos = Xdev[ydev == t]
        neg = np.concatenate([Xdev[ydev == s] for s in hard], axis=0)
        Ppos = Pdev[ydev == t]
        Pneg = np.concatenate([Pdev[ydev == s] for s in hard], axis=0)

        def aug(Xf, Pf, tt=t):
            pt = Pf[:, tt]
            mask = np.ones(Pf.shape[1], dtype=bool)
            mask[tt] = False
            pr = Pf[:, mask].max(axis=1)
            return np.column_stack([Xf, pt, pr, pt - pr])

        clf = ExtraTreesClassifier(
            n_estimators=64,
            min_samples_leaf=2,
            random_state=31 + t,
            n_jobs=1,
            class_weight="balanced_subsample",
        )
        clf.fit(
            np.vstack([aug(pos, Ppos), aug(neg, Pneg)]),
            np.r_[np.ones(len(pos)), np.zeros(len(neg))],
        )
        bthr = float(np.quantile(clf.predict_proba(aug(pos, Ppos))[:, 1], 0.05, method="higher"))
        verifiers[t] = (clf, bthr)

    Pte = rf.predict_proba(scaler_m.transform(X[split == 2]))
    yte_raw = y[split == 2]
    Xp2m = X[split == 3]
    Pp2 = rf.predict_proba(scaler_m.transform(Xp2m)) if len(Xp2m) else None
    yp2 = y[split == 3]

    def accept_rf(P, Xz, t_idx, use_resid):
        ok = P[:, t_idx] >= thr0[t_idx]
        if use_resid and verifiers[t_idx] is not None:
            clf, bthr = verifiers[t_idx]
            pt = P[:, t_idx]
            mask = np.ones(P.shape[1], dtype=bool)
            mask[t_idx] = False
            pr = P[:, mask].max(axis=1)
            feats = np.column_stack([Xz, pt, pr, pt - pr])
            ok = ok & (clf.predict_proba(feats)[:, 1] >= bthr)
        return ok

    Zte_all = scaler_m.transform(X[split == 2])
    Zp2_all = scaler_m.transform(X[split == 3]) if len(Xp2m) else None
    for t_idx, t in enumerate(known):
        te_m = yte_raw == t
        p2_m = yp2 == t
        for dname, use_resid in (("RF128", False), ("RF128+PAIRRESID", True)):
            tar_p1 = float(np.mean(accept_rf(Pte[te_m], Zte_all[te_m], t_idx, use_resid)))
            tar_p2 = (
                float(np.mean(accept_rf(Pp2[p2_m], Zp2_all[p2_m], t_idx, use_resid)))
                if Pp2 is not None and p2_m.any()
                else None
            )
            attacks_m = {
                "cross_known": yte_raw != t,
                "unknown_raw": np.isin(yte_raw, unknown),
            }
            # NN/blend on unknown features in scaled space
            unk_m = np.isin(yte_raw, unknown)
            centroid = scaler_m.transform(X[(split == 0) & (y == t)]).mean(axis=0)
            Xunk_s = Zte_all[unk_m]
            Punk = Pte[unk_m]
            if len(Xunk_s):
                d = np.linalg.norm(Xunk_s - centroid[None, :], axis=1)
                k10 = max(1, int(np.ceil(0.10 * len(Xunk_s))))
                k35 = max(1, int(np.ceil(0.35 * len(Xunk_s))))
                i10 = np.argsort(d)[:k10]
                i35 = np.argsort(d)[:k35]
                blend = 0.45 * Xunk_s[i10] + 0.55 * centroid
                # For blend, re-score RF on blended features
                Pblend = rf.predict_proba(blend)
            extra = {}
            if len(Xunk_s):
                extra["unknown_nn35"] = (Xunk_s[i35], Punk[i35])
                extra["unknown_nn10"] = (Xunk_s[i10], Punk[i10])
                extra["unknown_nn10_blend55"] = (blend, Pblend)
            for aname, mask in attacks_m.items():
                far = float(np.mean(accept_rf(Pte[mask], Zte_all[mask], t_idx, use_resid)))
                all_rows.append(
                    {
                        "target": int(t),
                        "defence": dname,
                        "attack": aname,
                        "tar_p1": tar_p1,
                        "tar_p2": tar_p2,
                        "far": far,
                        "n_attack": int(mask.sum()),
                        "n_genuine_p1": int(te_m.sum()),
                        "n_genuine_p2": int(p2_m.sum()),
                    }
                )
            for aname, (Xa, Pa) in extra.items():
                far = float(np.mean(accept_rf(Pa, Xa, t_idx, use_resid)))
                all_rows.append(
                    {
                        "target": int(t),
                        "defence": dname,
                        "attack": aname,
                        "tar_p1": tar_p1,
                        "tar_p2": tar_p2,
                        "far": far,
                        "n_attack": int(len(Xa)),
                        "n_genuine_p1": int(te_m.sum()),
                        "n_genuine_p2": int(p2_m.sum()),
                    }
                )
        print("target", t, "RF residual done", flush=True)

    keys = list(dict.fromkeys(k for r in all_rows for k in r))
    with (out / "per_target.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)

    defs = sorted({r["defence"] for r in all_rows})
    atks = sorted({r["attack"] for r in all_rows})
    macro = []
    for d in defs:
        tar = [r["tar_p1"] for r in all_rows if r["defence"] == d and r["attack"] == atks[0]]
        tar2 = [r["tar_p2"] for r in all_rows if r["defence"] == d and r["attack"] == atks[0] and r["tar_p2"] is not None]
        row = {
            "defence": d,
            "macro_tar_p1": float(np.mean(tar)),
            "macro_tar_p2": float(np.mean(tar2)) if tar2 else None,
        }
        for a in atks:
            rs = [r for r in all_rows if r["defence"] == d and r["attack"] == a]
            fars = [r["far"] for r in rs if r["far"] is not None]
            row[f"macro_far_{a}"] = float(np.mean(fars)) if fars else None
            row[f"worst_far_{a}"] = float(np.max(fars)) if fars else None
        macro.append(row)

    with (out / "macro.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(macro[0]))
        w.writeheader()
        w.writerows(macro)

    summary = {
        "evidence_class": "public_measured_radio",
        "dataset": meta,
        "threat": (
            "Unknown COTS transmitters (TX 9-11) and other known TX claim a "
            "known identity. Distinct physical radios, same public BladeRF capture."
        ),
        "macro": macro,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(macro, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
