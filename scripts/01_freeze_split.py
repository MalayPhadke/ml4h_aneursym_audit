#!/usr/bin/env python
"""Create the one-time, patient-grouped v1 train/val/test split manifest.

The manifest is the canonical patch-ID -> split-name mapping for all later
CDEL experiments.  Splitting is performed at patient/scan UID level so that
patches from the same scan never appear in different splits.  Twenty seeded,
stratified group folds are combined as 14 train, 3 validation, and 3 test
folds (approximately 70/15/15).

Existing outputs are never overwritten unless --force is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


from cdel.paths import PATCH_ROOT, PIPELINE_ROOT

DEFAULT_OUT_DIR = PIPELINE_ROOT / "manifests"

CLASS_SPECS = {
    "positive": {"directory": "positive", "marker": "_positive_", "label": 1},
    "frangi": {"directory": "fast_frangi", "marker": "_frangi_", "label": 0},
    "hard": {"directory": "negative_hard", "marker": "_negative_hard_", "label": 0},
    "easy": {"directory": "negative_easy", "marker": "_negative_easy_", "label": 0},
}


def sha256_lines(lines: list[str]) -> str:
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_filename(path: Path, marker: str) -> tuple[str, tuple[int, int, int]]:
    try:
        patient_id, coordinate_text = path.stem.split(marker, 1)
        coordinate = tuple(int(value) for value in coordinate_text.rsplit("_", 3)[-3:])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Cannot parse patient ID/anchor coordinate from {path.name}") from exc
    if not patient_id or len(coordinate) != 3:
        raise ValueError(f"Invalid patch filename: {path.name}")
    return patient_id, coordinate


def discover_v1_patches() -> list[dict]:
    records: list[dict] = []
    for class_index, (class_name, spec) in enumerate(CLASS_SPECS.items()):
        directory = PATCH_ROOT / "dataset_v1" / spec["directory"]
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing v1 patch directory: {directory}")
        files = sorted(directory.glob("*.npz"))
        if not files:
            raise RuntimeError(f"No NPZ files found in {directory}")
        for path in files:
            patient_id, coordinate = parse_filename(path, spec["marker"])
            records.append(
                {
                    "patch_id": path.relative_to(PATCH_ROOT).as_posix(),
                    "patient_id": patient_id,
                    "class": class_name,
                    "class_index": class_index,
                    "label": int(spec["label"]),
                    "anchor_x": coordinate[0],
                    "anchor_y": coordinate[1],
                    "anchor_z": coordinate[2],
                    "path": str(path.resolve()),
                }
            )
    patch_ids = [record["patch_id"] for record in records]
    if len(patch_ids) != len(set(patch_ids)):
        duplicates = [key for key, count in Counter(patch_ids).items() if count > 1]
        raise RuntimeError(f"Duplicate patch IDs found: {duplicates[:10]}")
    return records


def assign_splits(records: list[dict], seed: int, n_folds: int) -> None:
    if n_folds != 20:
        raise ValueError("This frozen protocol requires exactly 20 folds (14/3/3).")
    y = np.asarray([record["class_index"] for record in records], dtype=np.int64)
    groups = np.asarray([record["patient_id"] for record in records])
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_by_row = np.full(len(records), -1, dtype=np.int64)
    placeholder = np.zeros((len(records), 1), dtype=np.float32)
    for fold, (_, held_out_indices) in enumerate(splitter.split(placeholder, y, groups)):
        fold_by_row[held_out_indices] = fold
    if np.any(fold_by_row < 0):
        raise RuntimeError("At least one patch was not assigned to a fold.")

    for record, fold in zip(records, fold_by_row.tolist()):
        record["fold"] = fold
        if fold < 14:
            record["split"] = "train"
        elif fold < 17:
            record["split"] = "val"
        else:
            record["split"] = "test"


def validate_split(records: list[dict]) -> dict:
    patients_by_split: dict[str, set[str]] = defaultdict(set)
    for record in records:
        patients_by_split[record["split"]].add(record["patient_id"])
    names = ("train", "val", "test")
    overlaps = {
        f"{left}_{right}": len(patients_by_split[left] & patients_by_split[right])
        for i, left in enumerate(names)
        for right in names[i + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Patient leakage detected: {overlaps}")

    class_counts: dict[str, dict[str, int]] = {}
    for split in names:
        class_counts[split] = dict(
            sorted(Counter(r["class"] for r in records if r["split"] == split).items())
        )
    return {
        "n_patches": len(records),
        "n_patients": len({r["patient_id"] for r in records}),
        "patch_counts_by_split_and_class": class_counts,
        "patient_counts_by_split": {
            split: len(patients_by_split[split]) for split in names
        },
        "patient_overlap_counts": overlaps,
    }


def write_outputs(records: list[dict], out_dir: Path, seed: int, force: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split_manifest.csv"
    protected = [
        manifest_path,
        out_dir / "train_patch_ids.txt",
        out_dir / "val_patch_ids.txt",
        out_dir / "test_patch_ids.txt",
        out_dir / "split_hashes.json",
        out_dir / "split_summary.json",
    ]
    existing = [path for path in protected if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Frozen split outputs already exist; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    records = sorted(records, key=lambda record: record["patch_id"])
    fields = [
        "patch_id",
        "patient_id",
        "class",
        "label",
        "split",
        "fold",
        "anchor_x",
        "anchor_y",
        "anchor_z",
        "path",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record[field] for field in fields} for record in records)

    ids_by_split = {
        split: sorted(record["patch_id"] for record in records if record["split"] == split)
        for split in ("train", "val", "test")
    }
    for split, patch_ids in ids_by_split.items():
        (out_dir / f"{split}_patch_ids.txt").write_text(
            "".join(f"{patch_id}\n" for patch_id in patch_ids)
        )

    manifest_mapping = sorted(f"{r['patch_id']},{r['split']}" for r in records)
    hashes = {
        "algorithm": "sha256",
        "seed": seed,
        "protocol": "StratifiedGroupKFold(20): folds 0-13 train, 14-16 val, 17-19 test",
        "manifest_mapping_sha256": sha256_lines(manifest_mapping),
        **{
            f"{split}_patch_ids_sha256": sha256_lines(ids_by_split[split])
            for split in ("train", "val", "test")
        },
    }
    (out_dir / "split_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")

    summary = validate_split(records)
    summary.update(
        {
            "seed": seed,
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "grouping_unit": "patient_id parsed from patch filename",
            "stratification_target": "four patch classes",
            "hashes": hashes,
        }
    )
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Frozen manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    records = discover_v1_patches()
    assign_splits(records, seed=args.seed, n_folds=args.n_folds)
    validate_split(records)
    write_outputs(records, args.out_dir.resolve(), args.seed, args.force)


if __name__ == "__main__":
    main()
