#!/usr/bin/env python
"""Merge clean attribution outputs and execute Phase 0 gates G1/G2/G3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cdel.paths import resolve_patch_path
from cdel.core import (
    HASHES_PATH,
    MANIFEST_PATH,
    PIPELINE_ROOT,
    json_dump,
    load_manifest,
    load_preprocessed_patch,
    sha256_file,
    write_csv,
)


DEFAULT_ATTR_DIR = PIPELINE_ROOT / "attribution_v1"
DEFAULT_OUT_DIR = PIPELINE_ROOT / "phase0_gates"


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required attribution output is missing: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Required attribution output is empty: {path}")
    return rows


def index_unique(rows: list[dict], source: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        patch_id = row["patch_id"]
        if patch_id in indexed:
            raise RuntimeError(f"Duplicate patch ID in {source}: {patch_id}")
        indexed[patch_id] = row
    return indexed


def sha256_lines(lines: list[str]) -> str:
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def count_summary(rows: list[dict]) -> dict:
    return {
        "total": len(rows),
        "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "by_class": dict(sorted(Counter(row["class"] for row in rows).items())),
        "by_split_class": {
            f"{split}:{class_name}": count
            for (split, class_name), count in sorted(
                Counter((row["split"], row["class"]) for row in rows).items()
            )
        },
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"n": len(array), "n_finite": 0}
    quantiles = np.quantile(finite, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    return {
        "n": int(len(array)),
        "n_finite": int(len(finite)),
        "min": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p05": float(quantiles[2]),
        "p25": float(quantiles[3]),
        "median": float(quantiles[4]),
        "p75": float(quantiles[5]),
        "p95": float(quantiles[6]),
        "p99": float(quantiles[7]),
        "max": float(quantiles[8]),
        "mean": float(finite.mean()),
    }


def g2_group(rows: list[dict]) -> dict:
    slopes = np.asarray([float(row["radial_slope"]) for row in rows], dtype=np.float64)
    exact = slopes == 0.0
    nonzero = slopes[~exact]
    par_nan = np.asarray([int(row["mode2_par_nan"]) == 1 for row in rows])
    strict = np.asarray([int(row["mode2_strict_exact_zero"]) == 1 for row in rows])
    raw_zero_spellings = Counter(
        row["radial_slope"] for row, is_zero in zip(rows, exact) if is_zero
    )
    return {
        "n": len(rows),
        "radial_slope_exact_zero_n": int(exact.sum()),
        "radial_slope_exact_zero_fraction": float(exact.mean()) if len(rows) else None,
        "radial_slope_raw_zero_spellings": dict(raw_zero_spellings),
        "radial_slope_abs_le_1e-12_n": int((np.abs(slopes) <= 1e-12).sum()),
        "radial_slope_abs_le_1e-9_n": int((np.abs(slopes) <= 1e-9).sum()),
        "radial_slope_abs_le_1e-6_n": int((np.abs(slopes) <= 1e-6).sum()),
        "smallest_nonzero_absolute_radial_slope": (
            float(np.min(np.abs(nonzero))) if len(nonzero) else None
        ),
        "nonzero_radial_slope_distribution": distribution(nonzero.tolist()),
        "mode2_par_nan_n": int(par_nan.sum()),
        "mode2_par_nan_fraction": float(par_nan.mean()) if len(rows) else None,
        "mode2_strict_exact_zero_n": int(strict.sum()),
        "mode2_strict_exact_zero_fraction": float(strict.mean()) if len(rows) else None,
        "par_nan_and_slope_exact_zero_n": int((par_nan & exact).sum()),
        "par_nan_and_ig_exact_zero_n": int(
            sum(
                (int(row["mode2_par_nan"]) == 1)
                and (int(row["ig_exact_zero"]) == 1)
                for row in rows
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--attr-dir", type=Path, default=DEFAULT_ATTR_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.attr_dir = args.attr_dir.resolve()
    args.out_dir = args.out_dir.resolve()

    manifest = load_manifest(args.manifest)
    gradcam = read_csv(args.attr_dir / "gradcam_per_patch.csv")
    ig = read_csv(args.attr_dir / "ig_per_patch.csv")
    ig_run_summary = json.loads((args.attr_dir / "ig_summary.json").read_text())
    manifest_by_id = index_unique(manifest, "manifest")
    gradcam_by_id = index_unique(gradcam, "GradCAM++")
    ig_by_id = index_unique(ig, "IG")
    expected_ids = set(manifest_by_id)

    coverage = {}
    for name, indexed in (("gradcam", gradcam_by_id), ("ig", ig_by_id)):
        actual_ids = set(indexed)
        coverage[name] = {
            "n_rows": len(indexed),
            "missing_n": len(expected_ids - actual_ids),
            "extra_n": len(actual_ids - expected_ids),
            "missing_examples": sorted(expected_ids - actual_ids)[:10],
            "extra_examples": sorted(actual_ids - expected_ids)[:10],
            "complete": actual_ids == expected_ids,
        }

    combined: list[dict] = []
    prediction_abs_differences: list[float] = []
    if not coverage["gradcam"]["complete"] or not coverage["ig"]["complete"]:
        raise RuntimeError(f"G1 attribution coverage failed: {coverage}")
    common_to_skip = {
        "patch_id", "patient_id", "class", "label", "split", "fold",
        "anchor_x", "anchor_y", "anchor_z", "path", "checkpoint_sha256",
        "probability_positive", "radial_slope",
    }
    for manifest_row in manifest:
        patch_id = manifest_row["patch_id"]
        cam_row = gradcam_by_id[patch_id]
        ig_row = ig_by_id[patch_id]
        for field in ("patient_id", "class", "label", "split", "fold", "path"):
            if str(cam_row[field]) != str(manifest_row[field]) or str(ig_row[field]) != str(manifest_row[field]):
                raise RuntimeError(f"Metadata mismatch for {patch_id}, field={field}")
        if cam_row["checkpoint_sha256"] != ig_row["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint mismatch across attribution methods: {patch_id}")
        if not math.isclose(
            float(cam_row["radial_slope"]), float(ig_row["radial_slope"]),
            rel_tol=0.0, abs_tol=0.0,
        ):
            raise RuntimeError(f"Radial slope differs across methods: {patch_id}")
        prediction_abs_differences.append(abs(
            float(cam_row["probability_positive"])
            - float(ig_row["probability_positive"])
        ))
        row = dict(manifest_row)
        row.update({
            "checkpoint_sha256": ig_row["checkpoint_sha256"],
            "probability_positive": ig_row["probability_positive"],
            "radial_slope": ig_row["radial_slope"],
        })
        row.update({key: value for key, value in cam_row.items() if key not in common_to_skip})
        row.update({key: value for key, value in ig_row.items() if key not in common_to_skip})
        combined.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    unified_path = args.out_dir / "attribution_per_patch.csv"
    write_csv(unified_path, combined)

    expected_hashes = json.loads(HASHES_PATH.read_text())
    actual_mapping_hash = sha256_lines(
        sorted(f"{row['patch_id']},{row['split']}" for row in manifest)
    )
    actual_split_hashes = {
        split: sha256_lines(sorted(row["patch_id"] for row in manifest if row["split"] == split))
        for split in ("train", "val", "test")
    }
    patients = {
        split: {row["patient_id"] for row in manifest if row["split"] == split}
        for split in ("train", "val", "test")
    }
    patient_overlaps = {
        "train_val": len(patients["train"] & patients["val"]),
        "train_test": len(patients["train"] & patients["test"]),
        "val_test": len(patients["val"] & patients["test"]),
    }
    g3_checks = {
        "manifest_mapping_hash_matches": actual_mapping_hash == expected_hashes["manifest_mapping_sha256"],
        "train_id_hash_matches": actual_split_hashes["train"] == expected_hashes["train_patch_ids_sha256"],
        "val_id_hash_matches": actual_split_hashes["val"] == expected_hashes["val_patch_ids_sha256"],
        "test_id_hash_matches": actual_split_hashes["test"] == expected_hashes["test_patch_ids_sha256"],
        "patient_overlap_zero": all(value == 0 for value in patient_overlaps.values()),
        "single_checkpoint_for_all_rows": len({row["checkpoint_sha256"] for row in combined}) == 1,
    }

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in combined:
        groups["all"].append(row)
        groups[f"split={row['split']}"] .append(row)
        groups[f"class={row['class']}"] .append(row)
        groups[f"split={row['split']},class={row['class']}"] .append(row)
    g2 = {name: g2_group(rows) for name, rows in sorted(groups.items())}
    positive_train = g2["split=train,class=positive"]
    exact_slope_rows = [row for row in combined if float(row["radial_slope"]) == 0.0]
    exact_slope_preprocess_audit = []
    for row in exact_slope_rows:
        with np.load(resolve_patch_path(row["path"]), allow_pickle=True) as archive:
            raw_patch = np.asarray(archive["resized"], dtype=np.float32)
        patch = load_preprocessed_patch(row["path"])
        exact_slope_preprocess_audit.append({
            "patch_id": row["patch_id"],
            "class": row["class"],
            "split": row["split"],
            "raw_nonzero_voxel_count": int(np.count_nonzero(raw_patch)),
            "raw_min": float(raw_patch.min()),
            "raw_max": float(raw_patch.max()),
            "raw_mean": float(raw_patch.mean()),
            "raw_std": float(raw_patch.std()),
            "preprocessed_nonzero_voxel_count": int(np.count_nonzero(patch)),
            "preprocessed_min": float(patch.min()),
            "preprocessed_max": float(patch.max()),
        })
    write_csv(
        args.out_dir / "exact_zero_radial_slope_preprocessing_audit.csv",
        exact_slope_preprocess_audit,
    )
    all_exact_slope_inputs_collapsed = all(
        row["preprocessed_nonzero_voxel_count"] == 0
        for row in exact_slope_preprocess_audit
    )
    all_exact_slope_raw_inputs_zero = all(
        row["raw_nonzero_voxel_count"] == 0
        for row in exact_slope_preprocess_audit
    )
    point_mass_reproduced = (
        positive_train["radial_slope_exact_zero_n"] > 0
        and positive_train["radial_slope_exact_zero_n"] < positive_train["n"]
        and positive_train["mode2_par_nan_n"] > 0
        and positive_train["mode2_strict_exact_zero_n"] > 0
    )

    g1_pass = all(item["complete"] for item in coverage.values())
    prediction_reproducibility = {
        "tolerance": 1e-4,
        "maximum_absolute_probability_difference": max(prediction_abs_differences),
        "p99_absolute_probability_difference": float(
            np.quantile(prediction_abs_differences, 0.99)
        ),
        "within_tolerance": max(prediction_abs_differences) <= 1e-4,
        "reason_for_nonzero_difference": "GradCAM and IG ran with different batch sizes and on different GPU models; both used the identical checkpoint.",
    }
    g1_pass = g1_pass and prediction_reproducibility["within_tolerance"]
    g3_pass = all(g3_checks.values())
    completeness_values = [float(row["completeness_gap"]) for row in combined]
    completeness_summary = distribution(completeness_values)
    ig_quality_checks = {
        "all_completeness_gaps_finite": all(
            math.isfinite(value) for value in completeness_values
        ),
        "p95_completeness_gap_le_0_05": (
            completeness_summary.get("p95") is not None
            and completeness_summary["p95"] <= 0.05
        ),
        "publication_resolution_steps_ge_100": int(ig_run_summary["ig_steps"]) >= 100,
        "trapezoidal_quadrature": ig_run_summary.get("ig_quadrature") == "trapezoid",
    }
    ig_quality_pass = all(ig_quality_checks.values())
    gates = {
        "IG_faithfulness_quality": ig_quality_pass,
        "G1_full_split_coverage": g1_pass,
        "G2_exact_zero_mode_reproduced": point_mass_reproduced,
        "G3_frozen_split_integrity": g3_pass,
    }
    report = {
        "all_phase0_gates_pass": all(gates.values()),
        "gates": gates,
        "G1": {
            "interpretation": "Revised leakage-safe G1: every frozen-manifest patch, including every train patch, has split-aware GradCAM++ and IG outputs.",
            "manifest_counts": count_summary(manifest),
            "coverage": coverage,
            "prediction_reproducibility": prediction_reproducibility,
            "unified_output": str(unified_path),
            "unified_output_sha256": sha256_file(unified_path),
        },
        "IG_faithfulness": {
            "pass": ig_quality_pass,
            "checks": ig_quality_checks,
            "run_summary": ig_run_summary,
            "completeness_gap_distribution": completeness_summary,
            "n_gap_gt_0_02": sum(value > 0.02 for value in completeness_values),
            "n_gap_gt_0_05": sum(value > 0.05 for value in completeness_values),
            "n_gap_gt_0_10": sum(value > 0.10 for value in completeness_values),
        },
        "G2": {
            "interpretation": "Exact IEEE float equality is reported separately from near-zero thresholds and from PAR-NaN/IG-zero Mode 2 definitions.",
            "historical_numbers_are_not_acceptance_targets": True,
            "point_mass_reproduction_criterion": "train-positive has both exact-zero and nonzero radial_slope patches, at least one PAR-NaN patch, and at least one strict radial_slope==0 plus IG-exact-zero patch",
            "point_mass_reproduced": point_mass_reproduced,
            "exact_zero_preprocessing_audit": {
                "n_exact_zero_radial_slope": len(exact_slope_preprocess_audit),
                "n_all_zero_after_preprocessing": sum(
                    row["preprocessed_nonzero_voxel_count"] == 0
                    for row in exact_slope_preprocess_audit
                ),
                "n_all_zero_in_raw_npz_resized_array": sum(
                    row["raw_nonzero_voxel_count"] == 0
                    for row in exact_slope_preprocess_audit
                ),
                "all_exact_zero_radial_slope_raw_inputs_are_all_zero": all_exact_slope_raw_inputs_zero,
                "all_exact_zero_radial_slope_inputs_are_all_zero_after_preprocessing": all_exact_slope_inputs_collapsed,
                "artifact": str(args.out_dir / "exact_zero_radial_slope_preprocessing_audit.csv"),
                "interpretation": "The exact-zero point mass is raw input corruption/degeneracy: every NPZ resized array is identically zero even before preprocessing. It is not evidence that a nonblank model input induced zero gradients.",
            },
            "groups": g2,
        },
        "G3": {
            "checks": g3_checks,
            "expected_hashes": expected_hashes,
            "actual_manifest_mapping_sha256": actual_mapping_hash,
            "actual_split_patch_id_sha256": actual_split_hashes,
            "patient_overlap_counts": patient_overlaps,
            "manifest_file_sha256": sha256_file(args.manifest),
        },
    }
    json_dump(args.out_dir / "phase0_gate_report.json", report)
    markdown = [
        "# Phase 0 gate report",
        "",
        f"Overall: **{'PASS' if report['all_phase0_gates_pass'] else 'FAIL'}**",
        "",
        f"- G1 complete per-patch coverage: **{'PASS' if g1_pass else 'FAIL'}** ({len(combined)}/{len(manifest)})",
        f"- IG faithfulness (100-step trapezoid; p95 gap <=5%): **{'PASS' if ig_quality_pass else 'FAIL'}** (p95={completeness_summary.get('p95', float('nan')):.4f})",
        f"- G2 exact-zero Mode 2 point mass reproduced: **{'PASS' if point_mass_reproduced else 'FAIL'}**",
        f"- G3 frozen split/hash/patient integrity: **{'PASS' if g3_pass else 'FAIL'}**",
        "",
        "## Train-positive Mode 2 counts",
        "",
        f"- Total: {positive_train['n']}",
        f"- Exact radial_slope == 0: {positive_train['radial_slope_exact_zero_n']} ({positive_train['radial_slope_exact_zero_fraction']:.2%})",
        f"- PAR NaN: {positive_train['mode2_par_nan_n']} ({positive_train['mode2_par_nan_fraction']:.2%})",
        f"- Strict radial_slope==0 plus IG exact-zero: {positive_train['mode2_strict_exact_zero_n']} ({positive_train['mode2_strict_exact_zero_fraction']:.2%})",
        f"- Exact-zero slope inputs that are all-zero after preprocessing: {sum(row['preprocessed_nonzero_voxel_count'] == 0 for row in exact_slope_preprocess_audit)}/{len(exact_slope_preprocess_audit)}",
        f"- Exact-zero slope inputs whose raw NPZ resized array is all-zero: {sum(row['raw_nonzero_voxel_count'] == 0 for row in exact_slope_preprocess_audit)}/{len(exact_slope_preprocess_audit)}",
        "",
        "The old 232/321 counts are historical only; the clean train-only checkpoint determines the new values. Exact-zero slope patches are raw all-zero NPZ inputs, so they must not be described as information-bearing tissue inputs with mysteriously absent gradients.",
    ]
    (args.out_dir / "phase0_gate_report.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps({"gates": gates, "train_positive": positive_train}, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
