#!/usr/bin/env python
"""Train one frozen-split AUDIT curation/baseline condition from scratch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from audit.evaluation import evaluate_rows, metrics_by_negative_class
from audit.paths import CONDITION_PLAN_DIR
from audit.core import (
    HASHES_PATH,
    MANIFEST_PATH,
    PIPELINE_ROOT,
    build_model,
    json_dump,
    load_manifest,
    load_model_checkpoint,
    make_loader,
    run_prediction,
    set_global_seed,
    sha256_file,
    write_csv,
)


DEFAULT_ATTRIBUTION = PIPELINE_ROOT / "phase0_gates" / "attribution_per_patch.csv"
DEFAULT_BASE_SELECTION = PIPELINE_ROOT / "baseline_v1" / "train_selection.csv"
DEFAULT_OUT_ROOT = PIPELINE_ROOT / "conditions"
DEFAULT_CONDITIONS = CONDITION_PLAN_DIR / "phase3_5_conditions.csv"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def sha256_ids(patch_ids: list[str]) -> str:
    payload = "".join(f"{patch_id}\n" for patch_id in sorted(patch_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def configure_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(out_dir / "training.log"), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger("audit_condition")


def get_condition(path: Path, index: int) -> dict:
    rows = read_csv(path)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Condition index {index} outside [0,{len(rows)-1}]")
    return rows[index]


def score_for(row: dict, score_name: str) -> float:
    probability = float(row["probability_positive"])
    label = int(row["label"])
    if score_name == "radial_slope":
        return float(row["radial_slope"])
    if score_name == "gradcam_bounded_fraction":
        return float(row["gradcam_bounded_fraction"])
    if score_name == "ig_bounded_fraction":
        return float(row["ig_bounded_fraction"])
    if score_name == "loss":
        correct_probability = probability if label == 1 else 1.0 - probability
        return -math.log(max(correct_probability, 1e-12))
    if score_name == "lowest_margin":
        return -abs(probability - 0.5) * 2.0
    if score_name == "el2n":
        return math.sqrt(2.0) * (1.0 - probability if label == 1 else probability)
    raise ValueError(f"Unsupported score: {score_name}")


def select_records(
    base_records: list[dict],
    attribution_by_id: dict[str, dict],
    mode: str,
    score_name: str,
    fraction: float,
    direction: str,
    seed: int,
    positive_remove_n: int = 0,
) -> tuple[list[dict], list[dict], dict]:
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"Pruning fraction must lie in [0,1): {fraction}")
    if mode == "full":
        return list(base_records), [], {
            "negative_removed": 0,
            "positive_removed": 0,
            "ranking_rows": [],
        }

    remove_ids: set[str] = set()
    ranking_rows: list[dict] = []
    negative_records = [row for row in base_records if int(row["label"]) == 0]
    positive_records = [row for row in base_records if int(row["label"]) == 1]
    if mode in {"negative", "combined"}:
        remove_n = int(round(len(negative_records) * fraction))
        if remove_n < 1:
            raise RuntimeError("Negative condition would remove zero patches.")
        if score_name == "random":
            rng = np.random.default_rng(seed)
            ordered = list(negative_records)
            rng.shuffle(ordered)
            for rank, record in enumerate(ordered, start=1):
                ranking_rows.append({
                    "patch_id": record["patch_id"], "rank": rank,
                    "score": "", "remove": int(rank <= remove_n),
                    "ranking": "seeded_random", "direction": "random",
                    "ineligible_reason": "",
                })
        else:
            scored = []
            unscored = []
            for record in negative_records:
                attr = attribution_by_id[record["patch_id"]]
                value = score_for(attr, score_name)
                if not math.isfinite(value):
                    unscored.append(record)
                else:
                    scored.append((value, record))
            if remove_n > len(scored):
                raise RuntimeError(
                    f"Pruning budget {remove_n} exceeds {len(scored)} finite {score_name} scores."
                )
            reverse = direction == "high"
            ordered_pairs = sorted(scored, key=lambda item: (item[0], item[1]["patch_id"]), reverse=reverse)
            ordered = [record for _, record in ordered_pairs]
            for rank, (value, record) in enumerate(ordered_pairs, start=1):
                ranking_rows.append({
                    "patch_id": record["patch_id"], "rank": rank,
                    "score": value, "remove": int(rank <= remove_n),
                    "ranking": score_name, "direction": direction,
                    "ineligible_reason": "",
                })
            for record in sorted(unscored, key=lambda item: item["patch_id"]):
                ranking_rows.append({
                    "patch_id": record["patch_id"], "rank": "",
                    "score": "", "remove": 0,
                    "ranking": score_name, "direction": direction,
                    "ineligible_reason": "non-finite score",
                })
        remove_ids.update(record["patch_id"] for record in ordered[:remove_n])

    positive_remove_ids: set[str] = set()
    if mode == "positive_random_matched":
        if positive_remove_n < 1:
            raise ValueError("Matched random-positive pruning requires positive_remove_n >= 1.")
        if positive_remove_n >= len(positive_records):
            raise ValueError(
                f"Cannot remove {positive_remove_n} of {len(positive_records)} positives."
            )
        rng = np.random.default_rng(seed)
        ordered = sorted(positive_records, key=lambda row: row["patch_id"])
        rng.shuffle(ordered)
        for rank, record in enumerate(ordered, start=1):
            remove = int(rank <= positive_remove_n)
            ranking_rows.append({
                "patch_id": record["patch_id"], "rank": rank,
                "score": "", "remove": remove,
                "ranking": "seeded_random_positive",
                "direction": "random",
                "ineligible_reason": "",
            })
            if remove:
                positive_remove_ids.add(record["patch_id"])
        remove_ids.update(positive_remove_ids)

    if mode in {"positive", "combined"}:
        if score_name not in {"mode2_strict", "random", "radial_slope", "gradcam_bounded_fraction", "ig_bounded_fraction", "loss", "el2n", "lowest_margin"}:
            raise ValueError(f"Unsupported combined/positive score {score_name}")
        for record in positive_records:
            attr = attribution_by_id[record["patch_id"]]
            if int(attr["mode2_strict_exact_zero"]) == 1:
                positive_remove_ids.add(record["patch_id"])
                ranking_rows.append({
                    "patch_id": record["patch_id"], "rank": "", "score": 1,
                    "remove": 1, "ranking": "mode2_strict_exact_zero",
                    "direction": "fixed",
                    "ineligible_reason": "",
                })
        if not positive_remove_ids:
            raise RuntimeError("Positive Mode 2 pruning selected zero training patches.")
        remove_ids.update(positive_remove_ids)

    retained = [record for record in base_records if record["patch_id"] not in remove_ids]
    removed = [record for record in base_records if record["patch_id"] in remove_ids]
    return retained, removed, {
        "negative_removed": sum(int(row["label"]) == 0 for row in removed),
        "positive_removed": sum(int(row["label"]) == 1 for row in removed),
        "ranking_rows": ranking_rows,
    }


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    condition: dict,
    val_metrics: dict,
    manifest_hash: str,
    manifest_path: Path,
    selection_hash: str,
    training_seed: int,
    selection_seed: int,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "seed": int(condition["seed"]),
        "training_seed": training_seed,
        "selection_seed": selection_seed,
        "condition": condition,
        "architecture": "MONAI resnet10, spatial_dims=3, one input channel, two classes",
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_mapping_sha256": manifest_hash,
        "train_selection_sha256": selection_hash,
        "validation_metrics": val_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS)
    parser.add_argument("--condition-index", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--attribution", type=Path, default=DEFAULT_ATTRIBUTION)
    parser.add_argument("--base-selection", type=Path, default=DEFAULT_BASE_SELECTION)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    condition = get_condition(args.conditions.resolve(), args.condition_index)
    condition_id = condition["condition_id"]
    out_dir = args.out_root.resolve() / condition_id
    if (out_dir / "condition_summary.json").exists() and not args.force:
        print(f"Already complete: {condition_id}")
        return
    log = configure_logging(out_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for condition training.")
    device = torch.device("cuda")
    legacy_seed = int(condition["seed"])
    training_seed = int(condition.get("training_seed") or legacy_seed)
    selection_seed = int(condition.get("selection_seed") or legacy_seed)
    set_global_seed(training_seed)
    manifest = load_manifest(args.manifest.resolve())
    manifest_by_id = {record["patch_id"]: record for record in manifest}
    base_ids = [row["patch_id"] for row in read_csv(args.base_selection.resolve())]
    if len(base_ids) != len(set(base_ids)):
        raise RuntimeError("Baseline train selection contains duplicate IDs.")
    base_records = [manifest_by_id[patch_id] for patch_id in base_ids]
    if any(record["split"] != "train" for record in base_records):
        raise RuntimeError("Baseline selection contains non-train records.")
    attribution_rows = read_csv(args.attribution.resolve())
    attribution_by_id = {row["patch_id"]: row for row in attribution_rows}
    if set(base_ids) - set(attribution_by_id):
        raise RuntimeError("Attribution table does not cover the baseline training selection.")

    mode = condition["mode"]
    score_name = condition["score"]
    fraction = float(condition["negative_fraction"] or 0.0)
    direction = condition["direction"]
    positive_remove_n = int(condition.get("positive_remove_n") or 0)
    train_records, removed_records, details = select_records(
        base_records, attribution_by_id, mode, score_name, fraction, direction,
        selection_seed, positive_remove_n,
    )
    write_csv(out_dir / "train_selection.csv", train_records)
    if removed_records:
        write_csv(out_dir / "removed_selection.csv", removed_records)
    if details["ranking_rows"]:
        write_csv(out_dir / "ranking.csv", details["ranking_rows"])
    selection_hash = sha256_file(out_dir / "train_selection.csv")
    val_records = [record for record in manifest if record["split"] == "val"]
    test_records = [record for record in manifest if record["split"] == "test"]
    log.info("GPU: %s", torch.cuda.get_device_name(0))
    log.info(
        "Condition: %s (training_seed=%d, selection_seed=%d)",
        condition,
        training_seed,
        selection_seed,
    )
    log.info("Retained=%d removed=%d counts=%s", len(train_records), len(removed_records), dict(Counter(row["class"] for row in train_records)))

    model = build_model(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    loader = make_loader(
        train_records, args.batch_size, True, training_seed, args.num_workers
    )
    manifest_hash = json.loads(HASHES_PATH.read_text())["manifest_mapping_sha256"]
    best_key = (float("inf"), float("inf"))
    best_epoch = -1
    history: list[dict] = []
    best_checkpoint = out_dir / "best_checkpoint.pth"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_examples = 0
        for patches, labels, _ in loader:
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(patches)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(labels)
            n_examples += len(labels)
        val_rows, val_metrics_flat = run_prediction(
            model, val_records, device, args.eval_batch_size, args.num_workers
        )
        evaluation = evaluate_rows(val_rows, manifest, split="val")
        val_metrics = evaluation["metrics"]
        primary = val_metrics["all_negatives"]
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / n_examples,
            "val_auc": primary["auc"],
            "val_fpr_at_90_sensitivity": primary["fpr_at_90_sensitivity"],
            "val_achieved_sensitivity": primary["achieved_sensitivity"],
        })
        log.info("Epoch %02d/%02d loss=%.5f val_auc=%.5f val_fpr90=%.5f", epoch, args.epochs, total_loss / n_examples, primary["auc"], primary["fpr_at_90_sensitivity"])
        key = (primary["fpr_at_90_sensitivity"], -primary["auc"])
        if key < best_key:
            best_key = key
            best_epoch = epoch
            torch.save(
                checkpoint_payload(
                    model, optimizer, epoch, condition, val_metrics,
                    manifest_hash, args.manifest.resolve(), selection_hash,
                    training_seed, selection_seed,
                ),
                best_checkpoint,
            )
            write_csv(out_dir / "best_val_predictions.csv", val_rows)
    write_csv(out_dir / "training_history.csv", history)

    best_model, _ = load_model_checkpoint(best_checkpoint, device)
    val_rows, _ = run_prediction(best_model, val_records, device, args.eval_batch_size, args.num_workers)
    test_rows, _ = run_prediction(best_model, test_records, device, args.eval_batch_size, args.num_workers)
    write_csv(out_dir / "val_predictions.csv", val_rows)
    write_csv(out_dir / "test_predictions.csv", test_rows)
    val_evaluation = evaluate_rows(val_rows, manifest, split="val")
    test_evaluation = evaluate_rows(test_rows, manifest, split="test")
    nondegenerate_test_ids = {
        row["patch_id"]
        for row in attribution_rows
        if row["split"] == "test" and float(row["radial_slope"]) != 0.0
    }
    nondegenerate_test_rows = [
        row for row in test_rows if row["patch_id"] in nondegenerate_test_ids
    ]
    if len(nondegenerate_test_rows) != len(nondegenerate_test_ids):
        raise RuntimeError("Nondegenerate fixed-test subset coverage mismatch.")
    summary = {
        "condition": condition,
        "training_seed": training_seed,
        "selection_seed": selection_seed,
        "gpu_model": torch.cuda.get_device_name(0),
        "best_epoch": best_epoch,
        "checkpoint": str(best_checkpoint),
        "manifest_mapping_sha256": manifest_hash,
        "test_patch_ids_sha256": json.loads(HASHES_PATH.read_text())["test_patch_ids_sha256"],
        "train_selection_sha256": selection_hash,
        "base_train_n": len(base_records),
        "retained_train_n": len(train_records),
        "removed_train_n": len(removed_records),
        "retained_fraction": len(train_records) / len(base_records),
        "negative_removed": details["negative_removed"],
        "positive_removed": details["positive_removed"],
        "retained_counts_by_class": dict(Counter(row["class"] for row in train_records)),
        "validation": val_evaluation,
        "test": test_evaluation,
        "secondary_nondegenerate_test": {
            "criterion": "exclude frozen-test patches whose raw resized NPZ is all-zero, identified by exact radial_slope==0 and audited in Phase 0",
            "patch_ids_sha256": sha256_ids(list(nondegenerate_test_ids)),
            "n_retained": len(nondegenerate_test_rows),
            "n_excluded": len(test_rows) - len(nondegenerate_test_rows),
            "metrics": metrics_by_negative_class(nondegenerate_test_rows),
        },
        "cas": {"available": False, "reason": "No valid paired/recentered CAS source data."},
    }
    json_dump(out_dir / "condition_summary.json", summary)
    log.info("COMPLETE %s test=%s", condition_id, test_evaluation["metrics"]["all_negatives"])


if __name__ == "__main__":
    main()
