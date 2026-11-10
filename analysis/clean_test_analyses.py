#!/usr/bin/env python3
"""Clean-test, bootstrap and attribution-agreement analyses that require no model retraining.

Produces the positive-only clean-test comparison, finite attribution-map
accounting, bootstrap membership audit, and held-out attribution-consistency
audit from frozen predictions and patch-ID-aware attribution records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr


from audit.paths import PIPELINE_ROOT

PIPELINE = PIPELINE_ROOT
ATTRIBUTION = PIPELINE / "phase0_gates" / "attribution_per_patch.csv"
FULL_DIR = PIPELINE / "conditions" / "b2_full_n1"
STRICT_DIR = PIPELINE / "conditions" / "b1_pos_mode2_strict"
DEFAULT_OUTPUT = PIPELINE / "multiseed_clean_test"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value: str | float | None) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def sha256_ids(ids: list[str]) -> str:
    payload = "".join(f"{x}\n" for x in sorted(ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fpr_at(labels: np.ndarray, scores: np.ndarray, target: float) -> tuple[float, float, float]:
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    if not len(positive_scores) or not len(negative_scores):
        return float("nan"), float("nan"), float("nan")
    needed = int(math.ceil(target * len(positive_scores)))
    threshold = float(np.sort(positive_scores)[::-1][needed - 1])
    achieved = float(np.mean(positive_scores >= threshold))
    fpr = float(np.mean(negative_scores >= threshold))
    return fpr, achieved, threshold


def metrics(rows: list[dict], validation_threshold: float) -> dict:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    scores = np.asarray([float(row["probability_positive"]) for row in rows])
    result = {
        "n": len(rows),
        "n_patients": len({row["patient_id"] for row in rows}),
        "n_positive": int(labels.sum()),
        "n_negative": int(np.sum(labels == 0)),
        "auc": auc(labels, scores),
    }
    for target in (0.80, 0.90, 0.95):
        fpr, achieved, threshold = fpr_at(labels, scores, target)
        suffix = int(target * 100)
        result[f"fpr_at_{suffix}"] = fpr
        result[f"achieved_sensitivity_at_{suffix}"] = achieved
        result[f"threshold_at_{suffix}"] = threshold
    predictions = scores >= validation_threshold
    result["validation_selected_threshold"] = validation_threshold
    result["sensitivity_at_validation_threshold"] = float(np.mean(predictions[labels == 1]))
    result["specificity_at_validation_threshold"] = float(np.mean(~predictions[labels == 0]))
    return result


def prediction_bundle(directory: Path) -> tuple[list[dict], float]:
    rows = read_csv(directory / "test_predictions.csv")
    summary = json.loads((directory / "condition_summary.json").read_text())
    threshold = summary["validation"]["metrics"]["all_negatives"]["threshold_at_90_sensitivity"]
    return rows, float(threshold)


def paired_bootstrap(
    full_rows: list[dict],
    strict_rows: list[dict],
    test_set: str,
    resamples: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    full = {row["patch_id"]: row for row in full_rows}
    strict = {row["patch_id"]: row for row in strict_rows}
    if set(full) != set(strict):
        raise RuntimeError(f"Prediction coverage differs for {test_set}")
    ids = sorted(full)
    labels = np.asarray([int(full[x]["label"]) for x in ids], dtype=np.int8)
    full_scores = np.asarray([float(full[x]["probability_positive"]) for x in ids])
    strict_scores = np.asarray([float(strict[x]["probability_positive"]) for x in ids])
    patients = np.asarray([full[x]["patient_id"] for x in ids])
    unique_patients = np.unique(patients)
    patient_indices = {p: np.flatnonzero(patients == p) for p in unique_patients}
    rng = np.random.default_rng(seed)
    # Preserve the RNG stream used by the original manuscript audit: that
    # script generated an independent patch bootstrap before the patient
    # bootstrap. We do not report the patch bootstrap here, but advancing the
    # stream keeps the original-test patient CI exactly reproducible.
    for _ in range(resamples):
        rng.integers(0, len(ids), len(ids))
    samples: list[dict] = []
    for index in range(resamples):
        sampled = rng.choice(unique_patients, len(unique_patients), replace=True)
        take = np.concatenate([patient_indices[p] for p in sampled])
        y = labels[take]
        f = full_scores[take]
        s = strict_scores[take]
        full_fpr90 = fpr_at(y, f, 0.90)[0]
        strict_fpr90 = fpr_at(y, s, 0.90)[0]
        full_auc = auc(y, f)
        strict_auc = auc(y, s)
        samples.append({
            "test_set": test_set,
            "resample": index,
            "full_auc": full_auc,
            "strict_auc": strict_auc,
            "delta_auc_strict_minus_full": strict_auc - full_auc,
            "full_fpr90": full_fpr90,
            "strict_fpr90": strict_fpr90,
            "delta_fpr90_strict_minus_full": strict_fpr90 - full_fpr90,
        })
    summaries: list[dict] = []
    observed_labels = labels
    for metric_name, full_values, strict_values in (
        ("auc", full_scores, strict_scores),
        ("fpr90", full_scores, strict_scores),
    ):
        if metric_name == "auc":
            full_observed = auc(observed_labels, full_values)
            strict_observed = auc(observed_labels, strict_values)
        else:
            full_observed = fpr_at(observed_labels, full_values, 0.90)[0]
            strict_observed = fpr_at(observed_labels, strict_values, 0.90)[0]
        key = f"delta_{metric_name}_strict_minus_full"
        values = np.asarray([row[key] for row in samples], dtype=float)
        summaries.append({
            "test_set": test_set,
            "metric": metric_name,
            "n_patients": len(unique_patients),
            "n_patches": len(ids),
            "full_observed": full_observed,
            "strict_observed": strict_observed,
            "observed_delta_strict_minus_full": strict_observed - full_observed,
            "bootstrap_median_delta": float(np.nanmedian(values)),
            "ci_2_5": float(np.nanpercentile(values, 2.5)),
            "ci_97_5": float(np.nanpercentile(values, 97.5)),
            "probability_delta_ge_zero": float(np.nanmean(values >= 0)),
            "resamples": resamples,
            "seed": seed,
        })
    return samples, summaries


def patient_cluster_median_ci(rows: list[dict], field: str, resamples: int, seed: int) -> tuple[float, float, float]:
    usable = [row for row in rows if finite(row[field])]
    values = np.asarray([float(row[field]) for row in usable])
    patients = sorted({row["patient_id"] for row in usable})
    by_patient = {
        patient: np.asarray([float(row[field]) for row in usable if row["patient_id"] == patient])
        for patient in patients
    }
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    for index in range(resamples):
        sampled = rng.choice(patients, len(patients), replace=True)
        boot[index] = np.median(np.concatenate([by_patient[p] for p in sampled]))
    return float(np.median(values)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def patient_cluster_rho_ci(rows: list[dict], resamples: int, seed: int) -> tuple[float | None, float | None, float | None, float | None, int, int]:
    usable = [
        row for row in rows
        if finite(row["gradcam_bounded_fraction"]) and finite(row["ig_bounded_fraction"])
    ]
    if len(usable) < 3:
        return None, None, None, None, len(usable), len({row["patient_id"] for row in usable})
    x = np.asarray([float(row["gradcam_bounded_fraction"]) for row in usable])
    y = np.asarray([float(row["ig_bounded_fraction"]) for row in usable])
    observed = spearmanr(x, y)
    patients = sorted({row["patient_id"] for row in usable})
    by_patient = {
        patient: np.asarray([i for i, row in enumerate(usable) if row["patient_id"] == patient])
        for patient in patients
    }
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    for index in range(resamples):
        sampled = rng.choice(patients, len(patients), replace=True)
        take = np.concatenate([by_patient[p] for p in sampled])
        boot[index] = spearmanr(x[take], y[take]).statistic
    finite_boot = boot[np.isfinite(boot)]
    return (
        float(observed.statistic),
        float(observed.pvalue),
        float(np.percentile(finite_boot, 2.5)),
        float(np.percentile(finite_boot, 97.5)),
        len(usable),
        len(patients),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--attribution-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    attribution = read_csv(ATTRIBUTION)
    test_attr = [row for row in attribution if row["split"] == "test"]
    raw_zero_positive_ids = {
        row["patch_id"] for row in test_attr
        if row["class"] == "positive" and int(row["mode2_strict_exact_zero"]) == 1
    }
    if len(raw_zero_positive_ids) != 34:
        raise RuntimeError(f"Expected 34 raw-zero test positives; found {len(raw_zero_positive_ids)}")

    full_rows, full_val_threshold = prediction_bundle(FULL_DIR)
    strict_rows, strict_val_threshold = prediction_bundle(STRICT_DIR)
    if {row["patch_id"] for row in full_rows} != {row["patch_id"] for row in strict_rows}:
        raise RuntimeError("Full and strict test predictions do not have identical IDs")
    for rows in (full_rows, strict_rows):
        matched = [row for row in rows if row["patch_id"] in raw_zero_positive_ids]
        if len(matched) != 34 or any(int(row["label"]) != 1 for row in matched):
            raise RuntimeError("Raw-zero positive exclusion IDs do not match predictions")

    clean_metrics: list[dict] = []
    model_inputs = (
        ("full", full_rows, full_val_threshold),
        ("strict", strict_rows, strict_val_threshold),
    )
    for model, rows, threshold in model_inputs:
        for test_set, selected in (
            ("original", rows),
            ("clean_exclude_34_raw_zero_positives", [row for row in rows if row["patch_id"] not in raw_zero_positive_ids]),
        ):
            clean_metrics.append({"model": model, "test_set": test_set, **metrics(selected, threshold)})
    write_csv(args.output / "clean_test_metrics.csv", clean_metrics)

    all_bootstrap: list[dict] = []
    bootstrap_summary: list[dict] = []
    for test_set, ids_to_exclude in (
        ("original", set()),
        ("clean_exclude_34_raw_zero_positives", raw_zero_positive_ids),
    ):
        selected_full = [row for row in full_rows if row["patch_id"] not in ids_to_exclude]
        selected_strict = [row for row in strict_rows if row["patch_id"] not in ids_to_exclude]
        samples, summaries = paired_bootstrap(
            selected_full, selected_strict, test_set, args.bootstrap_resamples, args.seed
        )
        all_bootstrap.extend(samples)
        bootstrap_summary.extend(summaries)
    write_csv(args.output / "clean_test_bootstrap.csv", all_bootstrap)
    write_csv(args.output / "clean_test_bootstrap_summary.csv", bootstrap_summary)

    count_rows: list[dict] = []
    subsets = [
        ("positive", "all", [row for row in test_attr if row["class"] == "positive"]),
        ("positive", "raw_zero", [row for row in test_attr if row["patch_id"] in raw_zero_positive_ids]),
        ("positive", "nonzero", [row for row in test_attr if row["class"] == "positive" and row["patch_id"] not in raw_zero_positive_ids]),
    ]
    subsets.extend((cls, "all", [row for row in test_attr if row["class"] == cls]) for cls in ("frangi", "hard", "easy"))
    for cls, subset, rows in subsets:
        gc_finite = sum(finite(row["rar"]) for row in rows)
        ig_finite = sum(finite(row["ig_bounded_fraction"]) for row in rows)
        paired = sum(
            finite(row["gradcam_bounded_fraction"]) and finite(row["ig_bounded_fraction"])
            for row in rows
        )
        count_rows.append({
            "class": cls,
            "subset": subset,
            "total_n": len(rows),
            "finite_gradcam_rar_n": gc_finite,
            "undefined_gradcam_rar_n": len(rows) - gc_finite,
            "finite_ig_summary_n": ig_finite,
            "undefined_ig_summary_n": len(rows) - ig_finite,
            "finite_paired_gradcam_ig_n": paired,
            "undefined_paired_n": len(rows) - paired,
            "ig_exact_zero_n": sum(int(row["ig_exact_zero"]) == 1 for row in rows),
            "gradcam_exact_zero_n": sum(int(row["gradcam_exact_zero"]) == 1 for row in rows),
        })
    write_csv(args.output / "finite_map_counts.csv", count_rows)

    table_rows: list[dict] = []
    for cls in ("positive", "frangi", "hard", "easy"):
        rows = [row for row in test_attr if row["class"] == cls]
        median, low, high = patient_cluster_median_ci(
            rows, "rar", args.attribution_resamples, args.seed
        )
        table_rows.append({
            "class": cls,
            "total_n": len(rows),
            "finite_gradcam_rar_n": sum(finite(row["rar"]) for row in rows),
            "undefined_gradcam_rar_n": sum(not finite(row["rar"]) for row in rows),
            "finite_paired_gradcam_ig_n": sum(
                finite(row["gradcam_bounded_fraction"]) and finite(row["ig_bounded_fraction"])
                for row in rows
            ),
            "median_gradcam_rar": median,
            "patient_cluster_ci_2_5": low,
            "patient_cluster_ci_97_5": high,
        })
    write_csv(args.output / "table1_revised.csv", table_rows)

    agreement_groups = [
        ("positive_all", [row for row in test_attr if row["class"] == "positive"]),
        ("positive_raw_zero", [row for row in test_attr if row["patch_id"] in raw_zero_positive_ids]),
        ("positive_nonzero", [row for row in test_attr if row["class"] == "positive" and row["patch_id"] not in raw_zero_positive_ids]),
        ("pooled_negatives", [row for row in test_attr if row["class"] != "positive"]),
    ]
    agreement_groups.extend((cls, [row for row in test_attr if row["class"] == cls]) for cls in ("frangi", "hard", "easy"))
    agreement_rows: list[dict] = []
    for group, rows in agreement_groups:
        rho, pvalue, low, high, paired_n, patient_n = patient_cluster_rho_ci(
            rows, args.attribution_resamples, args.seed
        )
        agreement_rows.append({
            "group": group,
            "total_n": len(rows),
            "finite_paired_n": paired_n,
            "finite_paired_patients": patient_n,
            "spearman_rho": rho if rho is not None else "",
            "two_sided_p": pvalue if pvalue is not None else "",
            "patient_cluster_ci_2_5": low if low is not None else "",
            "patient_cluster_ci_97_5": high if high is not None else "",
            "bootstrap_resamples": args.attribution_resamples if paired_n >= 3 else 0,
            "undefined_policy": "exclude non-finite pairs before patient resampling",
        })
    write_csv(args.output / "attribution_agreement_test.csv", agreement_rows)

    membership: list[dict] = []
    for name, rows, undefined_policy in (
        ("auc_original_full_vs_strict", full_rows, "not applicable; all frozen-test predictions included"),
        ("auc_clean_full_vs_strict", [row for row in full_rows if row["patch_id"] not in raw_zero_positive_ids], "34 strict raw-zero positives excluded before patient resampling"),
    ):
        membership.append({
            "bootstrap_name": name,
            "n_patients": len({row["patient_id"] for row in rows}),
            "n_patches": len(rows),
            "n_positive": sum(int(row["label"]) == 1 for row in rows),
            "n_raw_zero_positive": sum(row["patch_id"] in raw_zero_positive_ids for row in rows),
            "undefined_policy": undefined_policy,
            "patch_ids_sha256": sha256_ids([row["patch_id"] for row in rows]),
        })
    for cls in ("positive", "frangi", "hard", "easy"):
        rows = [row for row in test_attr if row["class"] == cls and finite(row["rar"])]
        membership.append({
            "bootstrap_name": f"gradcam_rar_test_{cls}",
            "n_patients": len({row["patient_id"] for row in rows}),
            "n_patches": len(rows),
            "n_positive": sum(int(row["label"]) == 1 for row in rows),
            "n_raw_zero_positive": sum(row["patch_id"] in raw_zero_positive_ids for row in rows),
            "undefined_policy": "undefined GradCAM++ RAR excluded before patient resampling",
            "patch_ids_sha256": sha256_ids([row["patch_id"] for row in rows]),
        })
    for group, rows in agreement_groups:
        usable = [row for row in rows if finite(row["gradcam_bounded_fraction"]) and finite(row["ig_bounded_fraction"])]
        membership.append({
            "bootstrap_name": f"spearman_test_{group}",
            "n_patients": len({row["patient_id"] for row in usable}),
            "n_patches": len(usable),
            "n_positive": sum(int(row["label"]) == 1 for row in usable),
            "n_raw_zero_positive": sum(row["patch_id"] in raw_zero_positive_ids for row in usable),
            "undefined_policy": "non-finite GradCAM++/IG pairs excluded before patient resampling",
            "patch_ids_sha256": sha256_ids([row["patch_id"] for row in usable]),
        })
    write_csv(args.output / "bootstrap_membership_audit.csv", membership)

    figure_metrics = {(row["model"], row["test_set"]): row for row in clean_metrics}
    labels = ["Original", "Exclude 34 raw-zero\npositives"]
    test_sets = ["original", "clean_exclude_34_raw_zero_positives"]
    x = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for axis, metric, title in ((axes[0], "auc", "AUC"), (axes[1], "fpr_at_90", "FPR@90")):
        plotted = []
        for model, model_label, color in (("full", "Full", "#4c78a8"), ("strict", "Strict removal", "#f58518")):
            values = [figure_metrics[(model, t)][metric] for t in test_sets]
            plotted.extend(values)
            axis.plot(x, values, marker="o", linewidth=1.6, markersize=6, label=model_label, color=color)
            for position, value in zip(x, values):
                axis.annotate(f"{value:.3f}", (position, value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
        axis.set_xticks(x, labels)
        axis.set_title(title)
        margin = max(0.005, (max(plotted) - min(plotted)) * 0.35)
        axis.set_ylim(min(plotted) - margin, max(plotted) + margin * 1.8)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / "fig_original_vs_clean_test.pdf", bbox_inches="tight")
    plt.close(fig)

    clean_positive = [
        row for row in test_attr
        if row["class"] == "positive" and row["patch_id"] not in raw_zero_positive_ids
        and finite(row["gradcam_bounded_fraction"]) and finite(row["ig_bounded_fraction"])
    ]
    rho = spearmanr(
        [float(row["gradcam_bounded_fraction"]) for row in clean_positive],
        [float(row["ig_bounded_fraction"]) for row in clean_positive],
    )
    fig, axis = plt.subplots(figsize=(4.2, 3.5))
    axis.scatter(
        [float(row["gradcam_bounded_fraction"]) for row in clean_positive],
        [float(row["ig_bounded_fraction"]) for row in clean_positive],
        s=14, alpha=0.55, linewidth=0,
    )
    axis.set_xlabel("GradCAM++ bounded density")
    axis.set_ylabel("IG bounded density")
    axis.set_title(f"Nonzero test positives: rho={rho.statistic:.3f}, n={len(clean_positive)}")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output / "fig_attribution_agreement_clean_positive.pdf", bbox_inches="tight")
    plt.close(fig)

    report = {
        "raw_zero_positive_test_n": len(raw_zero_positive_ids),
        "raw_zero_positive_ids_sha256": sha256_ids(list(raw_zero_positive_ids)),
        "outputs": sorted(path.name for path in args.output.iterdir()),
        "note": "The clean test excludes exactly 34 strict raw-zero positives and retains the raw-zero Frangi negative excluded by the older 35-patch nondegenerate diagnostic.",
    }
    (args.output / "static_analysis_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
