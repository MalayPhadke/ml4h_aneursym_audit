#!/usr/bin/env python3
"""Summarize the five exact random-170-positive control runs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


from cdel.paths import PIPELINE_ROOT

PIPELINE = PIPELINE_ROOT
SEEDS = (42, 43, 44, 45, 46)


def mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def main() -> None:
    rows = []
    for seed in SEEDS:
        path = PIPELINE / "conditions" / f"b2_rand_pos170_seed{seed}" / "condition_summary.json"
        payload = json.loads(path.read_text())
        test = payload["test"]["metrics"]["all_negatives"]
        predictions = PIPELINE / "conditions" / f"b2_rand_pos170_seed{seed}" / "test_predictions.csv"
        rows.append({
            "seed": seed,
            "condition_id": payload["condition"]["condition_id"],
            "best_epoch": payload["best_epoch"],
            "positive_removed": payload["positive_removed"],
            "negative_removed": payload["negative_removed"],
            "test_auc": test["auc"],
            "test_fpr_at_90_sensitivity": test["fpr_at_90_sensitivity"],
            "test_predictions": str(predictions),
        })
    report = {
        "design": "five independent seeded selections/training runs; exactly 170 of 868 train positives removed per run; frozen validation and test IDs",
        "runs": rows,
        "aggregate": {
            "test_auc": mean_sd([row["test_auc"] for row in rows]),
            "test_fpr_at_90_sensitivity": mean_sd([row["test_fpr_at_90_sensitivity"] for row in rows]),
        },
    }
    output = PIPELINE / "results" / "random170_positive_multiseed_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
