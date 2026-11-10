"""Frozen-test evaluation helpers shared by all AUDIT training conditions."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from audit.core import binary_metrics


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Prediction table is empty: {path}")
    return rows


def validate_prediction_coverage(
    rows: list[dict], expected_records: list[dict], expected_split: str
) -> dict:
    expected = {
        row["patch_id"]: row for row in expected_records if row["split"] == expected_split
    }
    actual: dict[str, dict] = {}
    for row in rows:
        patch_id = row["patch_id"]
        if patch_id in actual:
            raise RuntimeError(f"Duplicate prediction patch ID: {patch_id}")
        actual[patch_id] = row
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:10]
        extra = sorted(set(actual) - set(expected))[:10]
        raise RuntimeError(
            f"Frozen {expected_split} coverage mismatch: missing={missing}, extra={extra}"
        )
    for patch_id, row in actual.items():
        reference = expected[patch_id]
        for field in ("patient_id", "class", "label", "split"):
            if str(row[field]) != str(reference[field]):
                raise RuntimeError(f"Metadata mismatch for {patch_id}, field={field}")
    return {
        "expected_n": len(expected),
        "actual_n": len(actual),
        "complete": True,
        "duplicates": 0,
    }


def metrics_by_negative_class(rows: list[dict]) -> dict:
    result = {
        "all_negatives": binary_metrics(
            [int(row["label"]) for row in rows],
            [float(row["probability_positive"]) for row in rows],
        )
    }
    positives = [row for row in rows if int(row["label"]) == 1]
    for class_name in ("frangi", "hard", "easy"):
        selected = positives + [row for row in rows if row["class"] == class_name]
        result[f"positive_vs_{class_name}"] = binary_metrics(
            [int(row["label"]) for row in selected],
            [float(row["probability_positive"]) for row in selected],
        )
    return result


def summarize_cas(cas_rows: list[dict], expected_patch_ids: set[str]) -> dict:
    by_id: dict[str, float] = {}
    for row in cas_rows:
        patch_id = row["patch_id"]
        if patch_id in by_id:
            raise RuntimeError(f"Duplicate CAS patch ID: {patch_id}")
        value = float(row["cas"])
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite CAS for {patch_id}")
        by_id[patch_id] = value
    missing = expected_patch_ids - set(by_id)
    extra = set(by_id) - expected_patch_ids
    if missing or extra:
        raise RuntimeError(
            f"CAS test coverage mismatch: missing={sorted(missing)[:10]}, extra={sorted(extra)[:10]}"
        )
    values = np.asarray(list(by_id.values()), dtype=np.float64)
    return {
        "available": True,
        "n": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def evaluate_rows(
    rows: list[dict],
    manifest: list[dict],
    split: str = "test",
    cas_rows: list[dict] | None = None,
) -> dict:
    coverage = validate_prediction_coverage(rows, manifest, split)
    metrics = metrics_by_negative_class(rows)
    cas = {"available": False, "reason": "No valid paired/recentered CAS table supplied."}
    if cas_rows is not None:
        cas = summarize_cas(cas_rows, {row["patch_id"] for row in rows})
    return {
        "split": split,
        "coverage": coverage,
        "metrics": metrics,
        "cas": cas,
    }
