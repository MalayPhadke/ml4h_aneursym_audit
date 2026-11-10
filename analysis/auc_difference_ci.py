#!/usr/bin/env python3
"""Paired bootstrap CIs for full versus strict-removal test AUC."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


from cdel.paths import PIPELINE_ROOT

PIPELINE = PIPELINE_ROOT


def read_predictions(path: Path) -> dict[str, dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["patch_id"]: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"Duplicate patch IDs in {path}")
    return result


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def percentile(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    return [float(x) for x in np.percentile(finite, [2.5, 50, 97.5])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=PIPELINE / "results" / "full_vs_strict_auc_bootstrap.json")
    args = parser.parse_args()

    full = read_predictions(PIPELINE / "conditions" / "b2_full_n1" / "test_predictions.csv")
    strict = read_predictions(PIPELINE / "conditions" / "b1_pos_mode2_strict" / "test_predictions.csv")
    if set(full) != set(strict):
        raise RuntimeError("Full and strict prediction files do not contain identical patch IDs.")
    ids = sorted(full)
    labels = np.array([int(full[key]["label"]) for key in ids], dtype=np.int8)
    full_scores = np.array([float(full[key]["probability_positive"]) for key in ids])
    strict_scores = np.array([float(strict[key]["probability_positive"]) for key in ids])
    patients = np.array([full[key]["patient_id"] for key in ids])
    if any(int(full[key]["label"]) != int(strict[key]["label"]) for key in ids):
        raise RuntimeError("Label mismatch between paired prediction files.")

    full_auc = auc(labels, full_scores)
    strict_auc = auc(labels, strict_scores)
    observed = strict_auc - full_auc
    rng = np.random.default_rng(args.seed)

    patch_deltas = np.empty(args.resamples)
    for index in range(args.resamples):
        sample = rng.integers(0, len(ids), len(ids))
        patch_deltas[index] = auc(labels[sample], strict_scores[sample]) - auc(labels[sample], full_scores[sample])

    unique_patients = np.unique(patients)
    patient_indices = {patient: np.flatnonzero(patients == patient) for patient in unique_patients}
    cluster_deltas = np.empty(args.resamples)
    for index in range(args.resamples):
        sampled_patients = rng.choice(unique_patients, len(unique_patients), replace=True)
        sample = np.concatenate([patient_indices[patient] for patient in sampled_patients])
        cluster_deltas[index] = auc(labels[sample], strict_scores[sample]) - auc(labels[sample], full_scores[sample])

    report = {
        "comparison": "strict removal minus full-data test AUC",
        "paired_test_patch_count": len(ids),
        "test_patient_count": len(unique_patients),
        "full_auc": full_auc,
        "strict_auc": strict_auc,
        "observed_delta_strict_minus_full": observed,
        "bootstrap_resamples": args.resamples,
        "seed": args.seed,
        "patch_bootstrap_delta_percentiles_2.5_50_97.5": percentile(patch_deltas),
        "patient_cluster_bootstrap_delta_percentiles_2.5_50_97.5": percentile(cluster_deltas),
        "patient_cluster_probability_delta_ge_zero": float(np.mean(cluster_deltas >= 0)),
        "interpretation": "A CI containing zero does not establish equivalence; it means this single-run paired test set does not resolve the AUC difference from zero.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
