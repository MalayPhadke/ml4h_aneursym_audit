#!/usr/bin/env python
"""Compute split-aware GradCAM++ or IG for every frozen-manifest v1 patch.

Each method writes a resumable partial CSV and promotes it to a final CSV only
after every manifest patch has been processed exactly once.  Attribution always
targets the positive (aneurysm) logit, including for negative patches.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audit.core import (
    HASHES_PATH,
    MANIFEST_PATH,
    PATCH_SIZE,
    PIPELINE_ROOT,
    compute_radial_slope,
    json_dump,
    load_manifest,
    load_model_checkpoint,
    load_preprocessed_patch,
    radial_ratio,
    sha256_file,
)


DEFAULT_CHECKPOINT = PIPELINE_ROOT / "baseline_v1" / "best_checkpoint.pth"
DEFAULT_OUT_DIR = PIPELINE_ROOT / "attribution_v1"
TARGET_CLASS = 1

COMMON_FIELDS = [
    "patch_id", "patient_id", "class", "label", "split", "fold",
    "anchor_x", "anchor_y", "anchor_z", "path", "checkpoint_sha256",
    "probability_positive", "radial_slope",
]
GRADCAM_FIELDS = COMMON_FIELDS + [
    "rar", "gradcam_bounded_fraction", "gradcam_inner_mean",
    "gradcam_outer_mean", "gradcam_exact_zero",
]
IG_FIELDS = COMMON_FIELDS + [
    "par", "ig_bounded_fraction", "ig_inner_mean", "ig_outer_mean",
    "ig_signed_sum", "ig_abs_sum", "ig_abs_max", "ig_output_difference",
    "ig_completeness_absolute_residual", "completeness_gap",
    "ig_exact_zero", "mode2_par_nan", "mode2_strict_exact_zero",
]


def configure_logging(out_dir: Path, method: str) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / f"{method}.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger(f"audit_{method}")


def target_last_conv(model: nn.Module) -> nn.Conv3d:
    layers = [module for module in model.modules() if isinstance(module, nn.Conv3d)]
    if not layers:
        raise RuntimeError("The model has no Conv3d layer for GradCAM++.")
    return layers[-1]


class BatchedGradCAMPlusPlus3D:
    def __init__(self, model: nn.Module):
        self.model = model
        self.features: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        layer = target_last_conv(model)
        self.forward_handle = layer.register_forward_hook(self._save_features)
        self.backward_handle = layer.register_full_backward_hook(self._save_gradients)

    def _save_features(self, _module, _inputs, output) -> None:
        self.features = output.detach()

    def _save_gradients(self, _module, _grad_inputs, grad_outputs) -> None:
        self.gradients = grad_outputs[0].detach()

    def generate(self, patches: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(patches)
        logits[:, TARGET_CLASS].sum().backward()
        if self.features is None or self.gradients is None:
            raise RuntimeError("GradCAM++ hooks did not capture features/gradients.")
        features = self.features
        gradients = self.gradients
        grad2 = gradients.square()
        grad3 = grad2 * gradients
        feature_sum = features.sum(dim=(2, 3, 4), keepdim=True)
        denominator = 2.0 * grad2 + feature_sum * grad3
        alpha = torch.where(
            denominator != 0,
            grad2 / denominator,
            torch.zeros_like(denominator),
        )
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3, 4))
        cams = F.relu((weights[:, :, None, None, None] * features).sum(dim=1))
        cams = F.interpolate(
            cams[:, None],
            size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
            mode="trilinear",
            align_corners=False,
        )[:, 0]
        flat = cams.flatten(1)
        mins = flat.min(dim=1).values[:, None, None, None]
        maxs = flat.max(dim=1).values[:, None, None, None]
        spans = maxs - mins
        cams = torch.where(spans > 0, (cams - mins) / spans, torch.zeros_like(cams))
        probabilities = torch.softmax(logits.detach().float(), dim=1)[:, 1]
        return cams.detach().cpu().numpy().astype(np.float32), probabilities.cpu().numpy()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


class BatchedIntegratedGradients3D:
    def __init__(self, model: nn.Module, steps: int, alpha_chunk: int, quadrature: str):
        if steps < 1 or alpha_chunk < 1:
            raise ValueError("IG steps and alpha chunk must both be positive.")
        self.model = model
        self.steps = steps
        self.alpha_chunk = alpha_chunk
        self.quadrature = quadrature

    def generate(self, patches: torch.Tensor) -> dict[str, np.ndarray]:
        batch_size = patches.shape[0]
        gradient_sum = torch.zeros_like(patches)
        if self.quadrature == "right":
            alphas = torch.linspace(
                1.0 / self.steps, 1.0, self.steps,
                device=patches.device, dtype=patches.dtype,
            )
            weights = torch.full_like(alphas, 1.0 / self.steps)
        elif self.quadrature == "trapezoid":
            alphas = torch.linspace(
                0.0, 1.0, self.steps + 1,
                device=patches.device, dtype=patches.dtype,
            )
            weights = torch.full_like(alphas, 1.0 / self.steps)
            weights[0] *= 0.5
            weights[-1] *= 0.5
        else:
            raise ValueError(f"Unknown IG quadrature: {self.quadrature}")
        for start in range(0, len(alphas), self.alpha_chunk):
            alpha = alphas[start : start + self.alpha_chunk]
            weight = weights[start : start + self.alpha_chunk]
            interpolated = (
                patches[:, None] * alpha[None, :, None, None, None, None]
            ).reshape(-1, 1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
            interpolated = interpolated.detach().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(interpolated)
            gradients = torch.autograd.grad(
                logits[:, TARGET_CLASS].sum(), interpolated, retain_graph=False
            )[0]
            gradients = gradients.reshape(
                batch_size, len(alpha), 1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE
            )
            gradient_sum += (
                gradients * weight[None, :, None, None, None, None]
            ).sum(dim=1)

        signed_ig = patches * gradient_sum
        with torch.no_grad():
            input_logits = self.model(patches)
            baseline_logits = self.model(torch.zeros_like(patches))
            probabilities = torch.softmax(input_logits.float(), dim=1)[:, 1]
            output_difference = (
                input_logits[:, TARGET_CLASS] - baseline_logits[:, TARGET_CLASS]
            ).float()

        signed_sum = signed_ig.flatten(1).sum(dim=1).float()
        residual = (signed_sum - output_difference).abs()
        gap = residual / (output_difference.abs() + 1e-8)
        absolute = signed_ig.abs().float()
        absolute_flat = absolute.flatten(1)
        absolute_sum = absolute_flat.sum(dim=1)
        absolute_max = absolute_flat.max(dim=1).values
        absolute_min = absolute_flat.min(dim=1).values
        span = absolute_max - absolute_min
        normalized = torch.where(
            span[:, None, None, None, None] > 0,
            (absolute - absolute_min[:, None, None, None, None])
            / span[:, None, None, None, None],
            torch.zeros_like(absolute),
        )
        return {
            "map": normalized[:, 0].cpu().numpy().astype(np.float32),
            "probability": probabilities.cpu().numpy(),
            "signed_sum": signed_sum.cpu().numpy(),
            "abs_sum": absolute_sum.cpu().numpy(),
            "abs_max": absolute_max.cpu().numpy(),
            "output_difference": output_difference.cpu().numpy(),
            "residual": residual.cpu().numpy(),
            "gap": gap.cpu().numpy(),
        }


def batches(items: list[dict], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def read_completed(path: Path, fieldnames: list[str]) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and list(rows[0]) != fieldnames:
        raise RuntimeError(f"Partial CSV schema does not match current script: {path}")
    output: dict[str, dict] = {}
    for row in rows:
        patch_id = row["patch_id"]
        if patch_id in output:
            raise RuntimeError(f"Duplicate patch in partial output: {patch_id}")
        output[patch_id] = row
    return output


def append_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def common_row(record: dict, checkpoint_hash: str, probability: float, slope: float) -> dict:
    return {
        **{field: record[field] for field in COMMON_FIELDS if field in record},
        "checkpoint_sha256": checkpoint_hash,
        "probability_positive": float(probability),
        "radial_slope": float(slope),
    }


def validate_final(path: Path, records: list[dict], checkpoint_hash: str) -> None:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_ids = [record["patch_id"] for record in records]
    actual_ids = [row["patch_id"] for row in rows]
    if len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError(f"Duplicate IDs in final attribution file: {path}")
    if set(actual_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(actual_ids))[:10]
        extra = sorted(set(actual_ids) - set(expected_ids))[:10]
        raise RuntimeError(f"Coverage failure: missing={missing}, extra={extra}")
    if any(row["checkpoint_sha256"] != checkpoint_hash for row in rows):
        raise RuntimeError("Attribution rows mix checkpoints.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("gradcam", "ig"), required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--ig-alpha-chunk", type=int, default=10)
    parser.add_argument("--ig-quadrature", choices=("right", "trapezoid"), default="trapezoid")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Smoke-test only; final output is not promoted when set.")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.out_dir = args.out_dir.resolve()
    batch_size = args.batch_size or (16 if args.method == "gradcam" else 4)
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")

    log = configure_logging(args.out_dir, args.method)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for all-patch attribution.")
    device = torch.device("cuda")
    records = load_manifest(args.manifest)
    if args.max_records is not None:
        records = records[: args.max_records]
    checkpoint_hash = sha256_file(args.checkpoint)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    expected_mapping_hash = json.loads(HASHES_PATH.read_text())["manifest_mapping_sha256"]
    if checkpoint.get("manifest_mapping_sha256") != expected_mapping_hash:
        raise RuntimeError("Checkpoint was not trained against the frozen manifest mapping.")
    if checkpoint.get("manifest_file_sha256") != sha256_file(args.manifest):
        raise RuntimeError("Checkpoint manifest file hash differs from the requested manifest.")

    stem = f"{args.method}_per_patch"
    final_path = args.out_dir / f"{stem}.csv"
    partial_path = args.out_dir / f"{stem}.partial.csv"
    fields = GRADCAM_FIELDS if args.method == "gradcam" else IG_FIELDS
    if args.force:
        final_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)
    if final_path.exists():
        validate_final(final_path, records, checkpoint_hash)
        log.info("Validated existing complete output: %s", final_path)
        return

    completed = read_completed(partial_path, fields)
    if completed and any(
        row["checkpoint_sha256"] != checkpoint_hash for row in completed.values()
    ):
        raise RuntimeError("Partial attribution file belongs to a different checkpoint.")
    remaining = [record for record in records if record["patch_id"] not in completed]
    log.info("GPU: %s", torch.cuda.get_device_name(0))
    log.info("Method=%s records=%d resumed=%d remaining=%d", args.method, len(records), len(completed), len(remaining))
    log.info("Checkpoint SHA256: %s", checkpoint_hash)

    gradcam = BatchedGradCAMPlusPlus3D(model) if args.method == "gradcam" else None
    ig = (
        BatchedIntegratedGradients3D(
            model, args.ig_steps, args.ig_alpha_chunk, args.ig_quadrature
        )
        if args.method == "ig" else None
    )
    processed = len(completed)
    for batch_records in batches(remaining, batch_size):
        patch_arrays = [load_preprocessed_patch(record["path"]) for record in batch_records]
        slopes = [compute_radial_slope(patch) for patch in patch_arrays]
        patches = torch.from_numpy(np.stack(patch_arrays)[:, None]).to(device)
        output_rows: list[dict] = []
        if gradcam is not None:
            maps, probabilities = gradcam.generate(patches)
            for record, slope, attribution, probability in zip(
                batch_records, slopes, maps, probabilities
            ):
                ratio, bounded, inner, outer = radial_ratio(attribution)
                output_rows.append({
                    **common_row(record, checkpoint_hash, probability, slope),
                    "rar": ratio,
                    "gradcam_bounded_fraction": bounded,
                    "gradcam_inner_mean": inner,
                    "gradcam_outer_mean": outer,
                    "gradcam_exact_zero": int(np.count_nonzero(attribution) == 0),
                })
        else:
            assert ig is not None
            result = ig.generate(patches)
            for index, (record, slope, attribution) in enumerate(
                zip(batch_records, slopes, result["map"])
            ):
                ratio, bounded, inner, outer = radial_ratio(attribution)
                exact_zero = bool(result["abs_max"][index] == 0.0)
                par_nan = not math.isfinite(ratio)
                output_rows.append({
                    **common_row(
                        record, checkpoint_hash, result["probability"][index], slope
                    ),
                    "par": ratio,
                    "ig_bounded_fraction": bounded,
                    "ig_inner_mean": inner,
                    "ig_outer_mean": outer,
                    "ig_signed_sum": float(result["signed_sum"][index]),
                    "ig_abs_sum": float(result["abs_sum"][index]),
                    "ig_abs_max": float(result["abs_max"][index]),
                    "ig_output_difference": float(result["output_difference"][index]),
                    "ig_completeness_absolute_residual": float(result["residual"][index]),
                    "completeness_gap": float(result["gap"][index]),
                    "ig_exact_zero": int(exact_zero),
                    "mode2_par_nan": int(par_nan),
                    "mode2_strict_exact_zero": int(slope == 0.0 and exact_zero),
                })
        append_rows(partial_path, fields, output_rows)
        processed += len(output_rows)
        if processed % 100 <= len(output_rows) or processed == len(records):
            log.info("Processed %d/%d", processed, len(records))

    if gradcam is not None:
        gradcam.close()
    if args.max_records is not None:
        log.info("Smoke test complete; partial output retained and not promoted.")
        return
    validate_final(partial_path, records, checkpoint_hash)
    os.replace(partial_path, final_path)
    validate_final(final_path, records, checkpoint_hash)
    with final_path.open(newline="") as handle:
        final_rows = list(csv.DictReader(handle))
    summary = {
        "method": args.method,
        "n_manifest": len(records),
        "n_rows": len(final_rows),
        "coverage_complete": True,
        "duplicate_patch_ids": 0,
        "counts_by_split": dict(Counter(row["split"] for row in final_rows)),
        "counts_by_class": dict(Counter(row["class"] for row in final_rows)),
        "manifest_mapping_sha256": expected_mapping_hash,
        "manifest_file_sha256": sha256_file(args.manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "target_class": TARGET_CLASS,
        "ig_steps": args.ig_steps if args.method == "ig" else None,
        "ig_alpha_chunk": args.ig_alpha_chunk if args.method == "ig" else None,
        "ig_quadrature": args.ig_quadrature if args.method == "ig" else None,
    }
    json_dump(args.out_dir / f"{args.method}_summary.json", summary)
    log.info("COMPLETE: %s", final_path)


if __name__ == "__main__":
    main()
