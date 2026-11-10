#!/usr/bin/env python
"""Train the leakage-safe full-v1 baseline on the frozen train split only."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cdel.core import (
    HASHES_PATH,
    MANIFEST_PATH,
    PIPELINE_ROOT,
    binary_metrics,
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


DEFAULT_OUT_DIR = PIPELINE_ROOT / "baseline_v1"


def configure_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "training.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("cdel_train_v1")


def balanced_training_subset(train_records: list[dict], seed: int) -> list[dict]:
    """Use all positives and an equal seeded budget from each negative subclass."""
    by_class: dict[str, list[dict]] = {
        name: [record for record in train_records if record["class"] == name]
        for name in ("positive", "frangi", "hard", "easy")
    }
    n_positive = len(by_class["positive"])
    if not n_positive:
        raise RuntimeError("Frozen training split contains no positives.")
    rng = np.random.default_rng(seed)
    selected = list(by_class["positive"])
    for class_name in ("frangi", "hard", "easy"):
        candidates = by_class[class_name]
        n_select = min(n_positive, len(candidates))
        indices = rng.choice(len(candidates), n_select, replace=False)
        selected.extend(candidates[int(index)] for index in indices)
    return sorted(selected, key=lambda record: record["patch_id"])


def metrics_by_negative_class(rows: list[dict]) -> dict:
    result = {
        "all_negatives": binary_metrics(
            [row["label"] for row in rows],
            [row["probability_positive"] for row in rows],
        )
    }
    positives = [row for row in rows if row["label"] == 1]
    for class_name in ("frangi", "hard", "easy"):
        selected = positives + [row for row in rows if row["class"] == class_name]
        result[f"positive_vs_{class_name}"] = binary_metrics(
            [row["label"] for row in selected],
            [row["probability_positive"] for row in selected],
        )
    return result


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    seed: int,
    args: argparse.Namespace,
    val_metrics: dict,
    manifest_hash: str,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "seed": seed,
        "architecture": "MONAI resnet10, spatial_dims=3, one input channel, two classes",
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "negative_sampling": "all train positives plus up to n_positive from each negative subclass",
        },
        "manifest_path": str(args.manifest),
        "manifest_file_sha256": sha256_file(args.manifest),
        "manifest_mapping_sha256": manifest_hash,
        "validation_metrics": val_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.out_dir = args.out_dir.resolve()
    best_checkpoint = args.out_dir / "best_checkpoint.pth"
    if best_checkpoint.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {best_checkpoint}")

    log = configure_logging(args.out_dir)
    set_global_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for baseline training.")
    device = torch.device("cuda")
    log.info("GPU: %s", torch.cuda.get_device_name(0))
    log.info("Manifest: %s", args.manifest)

    records = load_manifest(args.manifest)
    train_pool = [record for record in records if record["split"] == "train"]
    val_records = [record for record in records if record["split"] == "val"]
    test_records = [record for record in records if record["split"] == "test"]
    train_records = balanced_training_subset(train_pool, args.seed)
    log.info("Train pool counts: %s", dict(Counter(r["class"] for r in train_pool)))
    log.info("Selected train counts: %s", dict(Counter(r["class"] for r in train_records)))
    log.info("Validation counts: %s", dict(Counter(r["class"] for r in val_records)))
    log.info("Test counts (held until final): %s", dict(Counter(r["class"] for r in test_records)))
    write_csv(args.out_dir / "train_selection.csv", train_records)

    manifest_hashes = json.loads(HASHES_PATH.read_text())
    manifest_mapping_hash = manifest_hashes["manifest_mapping_sha256"]
    model = build_model(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    train_loader = make_loader(
        train_records,
        args.batch_size,
        True,
        args.seed,
        args.num_workers,
    )

    history: list[dict] = []
    best_key = (float("inf"), float("inf"))
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_examples = 0
        for patches, labels, _ in train_loader:
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(patches)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * len(labels)
            n_examples += len(labels)

        val_rows, val_metrics = run_prediction(
            model,
            val_records,
            device,
            args.eval_batch_size,
            args.num_workers,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_examples, 1),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_row)
        log.info(
            "Epoch %02d/%02d loss=%.5f val_auc=%.5f val_fpr90=%.5f",
            epoch,
            args.epochs,
            epoch_row["train_loss"],
            val_metrics["auc"],
            val_metrics["fpr_at_90_sensitivity"],
        )
        selection_key = (
            val_metrics["fpr_at_90_sensitivity"],
            -val_metrics["auc"],
        )
        if selection_key < best_key:
            best_key = selection_key
            best_epoch = epoch
            torch.save(
                checkpoint_payload(
                    model,
                    optimizer,
                    epoch,
                    args.seed,
                    args,
                    val_metrics,
                    manifest_mapping_hash,
                ),
                best_checkpoint,
            )
            write_csv(args.out_dir / "best_val_predictions.csv", val_rows)

    write_csv(args.out_dir / "training_history.csv", history)
    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            args.epochs,
            args.seed,
            args,
            history[-1],
            manifest_mapping_hash,
        ),
        args.out_dir / "last_checkpoint.pth",
    )

    best_model, checkpoint = load_model_checkpoint(best_checkpoint, device)
    val_rows, val_metrics = run_prediction(
        best_model, val_records, device, args.eval_batch_size, args.num_workers
    )
    # The frozen test set is touched only here, after model selection is complete.
    test_rows, test_metrics = run_prediction(
        best_model, test_records, device, args.eval_batch_size, args.num_workers
    )
    write_csv(args.out_dir / "val_predictions.csv", val_rows)
    write_csv(args.out_dir / "test_predictions.csv", test_rows)

    summary = {
        "condition": "B2-Full clean v1 baseline",
        "checkpoint": str(best_checkpoint),
        "best_epoch": best_epoch,
        "seed": args.seed,
        "manifest_mapping_sha256": manifest_mapping_hash,
        "manifest_file_sha256": sha256_file(args.manifest),
        "counts": {
            "train_pool": dict(Counter(r["class"] for r in train_pool)),
            "train_selected": dict(Counter(r["class"] for r in train_records)),
            "val": dict(Counter(r["class"] for r in val_records)),
            "test": dict(Counter(r["class"] for r in test_records)),
        },
        "validation_metrics": metrics_by_negative_class(val_rows),
        "test_metrics": metrics_by_negative_class(test_rows),
        "test_used_for_model_selection": False,
        "checkpoint_validation_metrics": checkpoint.get("validation_metrics", {}),
    }
    json_dump(args.out_dir / "baseline_summary.json", summary)
    log.info("Best epoch: %d", best_epoch)
    log.info("Validation metrics: %s", summary["validation_metrics"])
    log.info("Final held-out test metrics: %s", summary["test_metrics"])
    log.info("COMPLETE")


if __name__ == "__main__":
    main()
