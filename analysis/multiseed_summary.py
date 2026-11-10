#!/usr/bin/env python3
"""Aggregate paired FULL/STRICT/RANDOM training runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


from audit.paths import PIPELINE_ROOT

PIPELINE = PIPELINE_ROOT
CONDITIONS_ROOT = PIPELINE / "conditions"
DEFAULT_OUTPUT = PIPELINE / "multiseed_clean_test"
SEEDS = (42, 43, 44, 45, 46)
CONDITION_DIRS = {
    "full": {
        42: "b2_full_n1",
        **{seed: f"ms_full_train{seed}" for seed in SEEDS if seed != 42},
    },
    "strict": {
        42: "b1_pos_mode2_strict",
        **{seed: f"ms_strict_train{seed}" for seed in SEEDS if seed != 42},
    },
    "random": {
        42: "b2_rand_pos170_seed42",
        **{seed: f"ms_random170_train{seed}_sel42" for seed in SEEDS if seed != 42},
    },
}


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


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = len(labels) - n_pos
    ranks = rankdata(scores, method="average")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fpr_at(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    needed = int(math.ceil(target * len(positive_scores)))
    threshold = float(np.sort(positive_scores)[::-1][needed - 1])
    return float(np.mean(negative_scores >= threshold))


def prediction_metrics(path: Path) -> dict:
    rows = read_csv(path)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    scores = np.asarray([float(row["probability_positive"]) for row in rows])
    return {
        "test_n": len(rows),
        "test_auc": auc(labels, scores),
        "test_fpr80": fpr_at(labels, scores, 0.80),
        "test_fpr90": fpr_at(labels, scores, 0.90),
        "test_fpr95": fpr_at(labels, scores, 0.95),
    }


def load_run(condition: str, seed: int, directory_name: str) -> dict:
    directory = CONDITIONS_ROOT / directory_name
    summary_path = directory / "condition_summary.json"
    predictions_path = directory / "test_predictions.csv"
    history_path = directory / "training_history.csv"
    missing = [path for path in (summary_path, predictions_path, history_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing run artifacts: " + ", ".join(str(path) for path in missing))
    summary = json.loads(summary_path.read_text())
    history = read_csv(history_path)
    best_epoch = int(summary["best_epoch"])
    selected = next(row for row in history if int(row["epoch"]) == best_epoch)
    final = max(history, key=lambda row: int(row["epoch"]))
    validation = summary["validation"]["metrics"]["all_negatives"]
    training_seed = int(summary.get("training_seed", summary["condition"]["seed"]))
    selection_seed = int(summary.get("selection_seed", summary["condition"]["seed"]))
    if training_seed != seed:
        raise RuntimeError(f"Training-seed mismatch in {directory}: {training_seed} != {seed}")
    return {
        "condition": condition,
        "training_seed": training_seed,
        "selection_seed": selection_seed,
        "condition_id": directory_name,
        "selected_epoch": best_epoch,
        "validation_auc": validation["auc"],
        "validation_fpr90": validation["fpr_at_90_sensitivity"],
        **prediction_metrics(predictions_path),
        "selected_checkpoint_train_loss": float(selected["train_loss"]),
        "final_train_loss": float(final["train_loss"]),
        "positive_removed": int(summary["positive_removed"]),
        "negative_removed": int(summary["negative_removed"]),
        "train_selection_sha256": summary["train_selection_sha256"],
        "test_predictions": str(predictions_path),
    }


def paired_rows(by_key: dict[tuple[str, int], dict], metric: str) -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        full = float(by_key[("full", seed)][metric])
        strict = float(by_key[("strict", seed)][metric])
        random = float(by_key[("random", seed)][metric])
        rows.append({
            "training_seed": seed,
            "full": full,
            "strict": strict,
            "random_fixed_subset_seed42": random,
            "strict_minus_full": strict - full,
            "random_minus_full": random - full,
            "strict_minus_random": strict - random,
        })
    return rows


def summarize_differences(metric: str, rows: list[dict], resamples: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    output: list[dict] = []
    for comparison in ("strict_minus_full", "random_minus_full", "strict_minus_random"):
        values = np.asarray([float(row[comparison]) for row in rows])
        boot = np.empty(resamples)
        for index in range(resamples):
            boot[index] = np.mean(rng.choice(values, len(values), replace=True))
        output.append({
            "metric": metric,
            "comparison": comparison,
            "n_paired_seeds": len(values),
            "mean_difference": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)),
            "median_difference": float(np.median(values)),
            "min_difference": float(np.min(values)),
            "max_difference": float(np.max(values)),
            "paired_seed_bootstrap_ci_2_5": float(np.percentile(boot, 2.5)),
            "paired_seed_bootstrap_ci_97_5": float(np.percentile(boot, 97.5)),
            "bootstrap_resamples": resamples,
            "bootstrap_seed": seed,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for condition, paths in CONDITION_DIRS.items():
        for seed in SEEDS:
            results.append(load_run(condition, seed, paths[seed]))
    results.sort(key=lambda row: (int(row["training_seed"]), ("full", "strict", "random").index(row["condition"])))

    # The random arm deliberately holds the removed subset fixed at selection
    # seed 42 while varying only the paired training seed.
    random_hashes = {row["train_selection_sha256"] for row in results if row["condition"] == "random"}
    strict_hashes = {row["train_selection_sha256"] for row in results if row["condition"] == "strict"}
    full_hashes = {row["train_selection_sha256"] for row in results if row["condition"] == "full"}
    if len(random_hashes) != 1 or len(strict_hashes) != 1 or len(full_hashes) != 1:
        raise RuntimeError(
            f"Training selections changed across seeds: full={full_hashes}, strict={strict_hashes}, random={random_hashes}"
        )
    if any(row["selection_seed"] != 42 for row in results if row["condition"] == "random"):
        raise RuntimeError("Random arm does not consistently use selection seed 42")
    write_csv(args.output / "results_multiseed.csv", results)

    by_key = {(row["condition"], int(row["training_seed"])): row for row in results}
    auc_rows = paired_rows(by_key, "test_auc")
    fpr90_rows = paired_rows(by_key, "test_fpr90")
    fpr95_rows = paired_rows(by_key, "test_fpr95")
    write_csv(args.output / "paired_auc_differences.csv", auc_rows)
    write_csv(args.output / "paired_fpr90_differences.csv", fpr90_rows)
    write_csv(args.output / "paired_fpr95_differences.csv", fpr95_rows)
    summaries = []
    for metric, rows in (("auc", auc_rows), ("fpr90", fpr90_rows), ("fpr95", fpr95_rows)):
        summaries.extend(summarize_differences(metric, rows, args.bootstrap_resamples, args.seed))
    write_csv(args.output / "paired_difference_summary.csv", summaries)

    # Sensitivity-only comparison with the previously completed random arm,
    # where subset-selection and training seeds co-varied. This is not a
    # formal crossed variance decomposition, but bounds what the five-by-five
    # experiment would need to resolve.
    variable_random = {
        seed: prediction_metrics(
            CONDITIONS_ROOT / f"b2_rand_pos170_seed{seed}" / "test_predictions.csv"
        )
        for seed in SEEDS
    }
    variance_rows: list[dict] = []
    for metric in ("test_auc", "test_fpr90", "test_fpr95"):
        fixed_values = np.asarray([
            float(by_key[("random", seed)][metric]) for seed in SEEDS
        ])
        combined_values = np.asarray([
            float(variable_random[seed][metric]) for seed in SEEDS
        ])
        fixed_variance = float(np.var(fixed_values, ddof=1))
        combined_variance = float(np.var(combined_values, ddof=1))
        variance_rows.extend([
            {
                "metric": metric,
                "design": "fixed_subset_seed42_vary_training_seed",
                "n": len(fixed_values),
                "mean": float(np.mean(fixed_values)),
                "sample_sd": float(np.std(fixed_values, ddof=1)),
                "sample_variance": fixed_variance,
                "interpretation": "training-seed variation conditional on one random subset",
            },
            {
                "metric": metric,
                "design": "co_varied_subset_and_training_seed",
                "n": len(combined_values),
                "mean": float(np.mean(combined_values)),
                "sample_sd": float(np.std(combined_values, ddof=1)),
                "sample_variance": combined_variance,
                "interpretation": "combined variation; subset and training effects are not identifiable",
            },
        ])
    write_csv(args.output / "random_seed_subset_variance_sensitivity.csv", variance_rows)

    colors = {"full": "#4c78a8", "strict": "#e45756", "random": "#72b7b2"}
    labels = {"full": "Full", "strict": "Strict removal", "random": "Random removal (subset seed 42)"}
    for metric, ylabel, filename in (
        ("test_auc", "Test AUC", "fig_multiseed_auc.pdf"),
        ("test_fpr90", "Test FPR@90", "fig_multiseed_fpr90.pdf"),
    ):
        fig, axis = plt.subplots(figsize=(5.2, 3.5))
        for condition in ("full", "strict", "random"):
            selected = sorted((row for row in results if row["condition"] == condition), key=lambda row: row["training_seed"])
            axis.plot(
                [row["training_seed"] for row in selected],
                [row[metric] for row in selected],
                marker="o", linewidth=1.5, label=labels[condition], color=colors[condition],
            )
        axis.set_xticks(SEEDS)
        axis.set_xlabel("Paired training seed")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output / filename, bbox_inches="tight")
        plt.close(fig)

    aggregate = {
        condition: {
            metric: {
                "mean": statistics.mean(float(row[metric]) for row in results if row["condition"] == condition),
                "sample_sd": statistics.stdev(float(row[metric]) for row in results if row["condition"] == condition),
            }
            for metric in ("test_auc", "test_fpr90", "test_fpr95")
        }
        for condition in ("full", "strict", "random")
    }
    report = {
        "design": "Five paired training seeds (42--46). The random arm uses one fixed 170-positive subset selected with seed 42, separating training-seed variation from subset choice.",
        "selection_hashes": {
            "full": next(iter(full_hashes)),
            "strict": next(iter(strict_hashes)),
            "random": next(iter(random_hashes)),
        },
        "aggregate": aggregate,
        "outputs": sorted(path.name for path in args.output.iterdir()),
    }
    (args.output / "multiseed_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    clean_metrics = read_csv(args.output / "clean_test_metrics.csv")
    clean_bootstrap = read_csv(args.output / "clean_test_bootstrap_summary.csv")
    finite_counts = read_csv(args.output / "finite_map_counts.csv")
    agreement = read_csv(args.output / "attribution_agreement_test.csv")
    clean_by_key = {(row["model"], row["test_set"]): row for row in clean_metrics}
    clean_ci = {
        (row["test_set"], row["metric"]): row for row in clean_bootstrap
    }
    diff_by_key = {(row["metric"], row["comparison"]): row for row in summaries}
    positive_counts = next(
        row for row in finite_counts
        if row["class"] == "positive" and row["subset"] == "all"
    )
    positive_agreement = next(row for row in agreement if row["group"] == "positive_nonzero")

    def pm(mean: float, sample_sd: float) -> str:
        return f"{mean:.4f} +/- {sample_sd:.4f}"

    strict_auc_wins = sum(
        float(by_key[("strict", seed)]["test_auc"]) > float(by_key[("full", seed)]["test_auc"])
        for seed in SEEDS
    )
    strict_fpr_wins = sum(
        float(by_key[("strict", seed)]["test_fpr90"]) < float(by_key[("full", seed)]["test_fpr90"])
        for seed in SEEDS
    )
    lines = [
        "# ML4H 434 Rebuttal Experiment Results",
        "",
        "## Paired multi-seed retraining",
        "",
        "Five paired training seeds (42--46) were used for Full, Strict, and Random removal. "
        "The Random arm held the 170-positive subset fixed with selection seed 42, so variation across rows is training-seed variation rather than subset-selection variation.",
        "",
        "| Condition | Test AUC, mean +/- SD | Test FPR@90, mean +/- SD | Test FPR@95, mean +/- SD |",
        "|---|---:|---:|---:|",
    ]
    for condition, display in (("full", "Full"), ("strict", "Strict removal"), ("random", "Random removal (fixed subset)")):
        lines.append(
            f"| {display} | {pm(**aggregate[condition]['test_auc'])} | "
            f"{pm(**aggregate[condition]['test_fpr90'])} | {pm(**aggregate[condition]['test_fpr95'])} |"
        )
    strict_full_auc = diff_by_key[("auc", "strict_minus_full")]
    strict_full_fpr = diff_by_key[("fpr90", "strict_minus_full")]
    strict_random_auc = diff_by_key[("auc", "strict_minus_random")]
    variance_by_key = {
        (row["metric"], row["design"]): row for row in variance_rows
    }
    lines.extend([
        "",
        f"Strict minus Full AUC averaged {float(strict_full_auc['mean_difference']):.4f} "
        f"(paired-seed bootstrap 95% CI {float(strict_full_auc['paired_seed_bootstrap_ci_2_5']):.4f} to "
        f"{float(strict_full_auc['paired_seed_bootstrap_ci_97_5']):.4f}); Strict had higher AUC in {strict_auc_wins}/5 paired seeds.",
        f"Strict minus Full FPR@90 averaged {float(strict_full_fpr['mean_difference']):.4f} "
        f"(95% CI {float(strict_full_fpr['paired_seed_bootstrap_ci_2_5']):.4f} to "
        f"{float(strict_full_fpr['paired_seed_bootstrap_ci_97_5']):.4f}); Strict had lower FPR@90 in {strict_fpr_wins}/5 seeds.",
        f"Strict minus fixed-subset Random AUC averaged {float(strict_random_auc['mean_difference']):.4f} "
        f"(95% CI {float(strict_random_auc['paired_seed_bootstrap_ci_2_5']):.4f} to "
        f"{float(strict_random_auc['paired_seed_bootstrap_ci_97_5']):.4f}). This comparison is conditional on one random subset.",
        "",
        "## Random-subset sensitivity",
        "",
        f"With the subset fixed, Random-removal AUC was "
        f"{pm(float(variance_by_key[('test_auc','fixed_subset_seed42_vary_training_seed')]['mean']), float(variance_by_key[('test_auc','fixed_subset_seed42_vary_training_seed')]['sample_sd']))}. "
        f"In the earlier five-run grid, where subset and optimization seed co-varied, it was "
        f"{pm(float(variance_by_key[('test_auc','co_varied_subset_and_training_seed')]['mean']), float(variance_by_key[('test_auc','co_varied_subset_and_training_seed')]['sample_sd']))}. "
        "This sensitivity comparison is not a crossed variance decomposition; subset and optimization effects are not separately identifiable in the earlier grid.",
        "",
        "## Clean-test analysis",
        "",
        "The cleaned test removes exactly the 34 strict raw-zero positive patches, retains all 968 negatives (including the one raw-zero Frangi patch), and does not retrain either model.",
        "",
        "| Model | Original AUC | Clean AUC | Original FPR@90 | Clean FPR@90 |",
        "|---|---:|---:|---:|---:|",
        f"| Full | {float(clean_by_key[('full','original')]['auc']):.4f} | "
        f"{float(clean_by_key[('full','clean_exclude_34_raw_zero_positives')]['auc']):.4f} | "
        f"{float(clean_by_key[('full','original')]['fpr_at_90']):.4f} | "
        f"{float(clean_by_key[('full','clean_exclude_34_raw_zero_positives')]['fpr_at_90']):.4f} |",
        f"| Strict | {float(clean_by_key[('strict','original')]['auc']):.4f} | "
        f"{float(clean_by_key[('strict','clean_exclude_34_raw_zero_positives')]['auc']):.4f} | "
        f"{float(clean_by_key[('strict','original')]['fpr_at_90']):.4f} | "
        f"{float(clean_by_key[('strict','clean_exclude_34_raw_zero_positives')]['fpr_at_90']):.4f} |",
        "",
        f"On the clean test, Strict minus Full AUC was "
        f"{float(clean_ci[('clean_exclude_34_raw_zero_positives','auc')]['observed_delta_strict_minus_full']):.4f} "
        f"(patient-cluster bootstrap 95% CI "
        f"{float(clean_ci[('clean_exclude_34_raw_zero_positives','auc')]['ci_2_5']):.4f} to "
        f"{float(clean_ci[('clean_exclude_34_raw_zero_positives','auc')]['ci_97_5']):.4f}).",
        "",
        "## Finite-map and attribution-consistency audit",
        "",
        f"All {positive_counts['total_n']} positive test patches have finite GradCAM++ RAR. "
        f"Only {positive_counts['finite_paired_gradcam_ig_n']} have finite paired GradCAM++/IG bounded-density summaries; "
        f"the {positive_counts['undefined_paired_n']} undefined pairs are exactly the 34 raw-zero positives. "
        "With a zero baseline, IG on these all-zero inputs is expected to be identically zero.",
        "",
        f"Among the 134 nonzero positives, Spearman rho was {float(positive_agreement['spearman_rho']):.3f} "
        f"(nominal p={float(positive_agreement['two_sided_p']):.4f}; patient-cluster bootstrap 95% CI "
        f"{float(positive_agreement['patient_cluster_ci_2_5']):.3f} to "
        f"{float(positive_agreement['patient_cluster_ci_97_5']):.3f}). "
        "This is an attribution-consistency audit, not a causal faithfulness test.",
        "",
        "## Interpretation",
        "",
        "The case study evaluates anomaly identification and intervention utility. "
        "Direct harmfulness remains an unresolved middle link. The paired-seed and clean-test results do not show a reproducible advantage for targeted deletion over Full training. "
        "Targeted deletion exceeds the one fixed Random subset in mean AUC, but random-subset variability was not replicated, so this comparison should remain qualified.",
        "",
        "## Machine-readable outputs",
        "",
    ])
    lines.extend(f"- `{path.name}`" for path in sorted(args.output.iterdir()))
    results_markdown = args.output / "multiseed_results.md"
    results_markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
