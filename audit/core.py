"""Shared data, model, metrics, and attribution utilities for AUDIT phases."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, roc_curve

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from monai.networks.nets import resnet10


from audit.paths import PATCH_ROOT, PIPELINE_ROOT, resolve_patch_path

MANIFEST_PATH = PIPELINE_ROOT / "manifests" / "split_manifest.csv"
HASHES_PATH = PIPELINE_ROOT / "manifests" / "split_hashes.json"

PATCH_SIZE = 64
CLIP_MAX = 800.0
SIGMA_FIELD = 0.5
N_RADIAL_BINS = 8
LEGACY_INNER_FRACTION_PARAMETER = 0.60
R_MAX = PATCH_SIZE / 2.0
LEGACY_R_INNER = R_MAX * (LEGACY_INNER_FRACTION_PARAMETER ** (1.0 / 3.0))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen split manifest not found: {path}")
    with path.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    required = {
        "patch_id",
        "patient_id",
        "class",
        "label",
        "split",
        "fold",
    }
    if not records or not required.issubset(records[0]):
        raise ValueError(f"Manifest is empty or lacks required columns: {required}")
    for record in records:
        for field in ("label", "fold", "anchor_x", "anchor_y", "anchor_z"):
            record[field] = int(record[field])
        if record["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {record['split']} for {record['patch_id']}")
        record["path"] = str(resolve_patch_path(record.get("path") or record["patch_id"]))
        path_value = Path(record["path"])
        if not path_value.is_file():
            raise FileNotFoundError(f"Manifest patch is missing: {path_value}")
    return records


def preprocess_patch(raw_patch: np.ndarray) -> np.ndarray:
    patch = np.asarray(raw_patch, dtype=np.float32)
    if patch.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"Expected 64^3 patch; received shape {patch.shape}")
    patch = np.clip(patch, 0.0, CLIP_MAX)
    patch = gaussian_filter(patch, sigma=SIGMA_FIELD)
    low = float(patch.min())
    high = float(patch.max())
    return ((patch - low) / (high - low + 1e-6)).astype(np.float32)


def load_preprocessed_patch(path: str | Path) -> np.ndarray:
    path = resolve_patch_path(path)
    with np.load(path, allow_pickle=True) as archive:
        if "resized" not in archive.files:
            raise KeyError(f"NPZ lacks 'resized': {path}")
        raw = archive["resized"]
    return preprocess_patch(raw)


class ManifestPatchDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        patch = load_preprocessed_patch(record["path"])
        return (
            torch.from_numpy(patch[np.newaxis]),
            torch.tensor(record["label"], dtype=torch.long),
            index,
        )


def make_loader(
    records: list[dict],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 4,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ManifestPatchDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def build_model(device: torch.device) -> nn.Module:
    return resnet10(
        pretrained=False,
        spatial_dims=3,
        n_input_channels=1,
        num_classes=2,
    ).to(device)


def checkpoint_state_dict(checkpoint: dict | object) -> dict:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state dictionary.")
    return checkpoint


def load_model_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(device)
    model.load_state_dict(checkpoint_state_dict(checkpoint), strict=True)
    model.eval()
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    return model, metadata


def fpr_at_sensitivity(
    labels: np.ndarray, probabilities: np.ndarray, target_sensitivity: float = 0.90
) -> tuple[float, float, float]:
    false_positive_rates, true_positive_rates, thresholds = roc_curve(labels, probabilities)
    valid = np.flatnonzero(true_positive_rates >= target_sensitivity)
    if not len(valid):
        return float("nan"), float("nan"), float("nan")
    index = valid[np.argmin(false_positive_rates[valid])]
    return (
        float(false_positive_rates[index]),
        float(true_positive_rates[index]),
        float(thresholds[index]),
    )


def binary_metrics(labels: Iterable[int], probabilities: Iterable[float]) -> dict:
    y_true = np.asarray(list(labels), dtype=np.int64)
    y_prob = np.asarray(list(probabilities), dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return {"n": int(len(y_true)), "auc": None, "fpr_at_90_sensitivity": None}
    auc_value = float(roc_auc_score(y_true, y_prob))
    fpr90, achieved_sensitivity, threshold = fpr_at_sensitivity(y_true, y_prob, 0.90)
    return {
        "n": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "auc": auc_value,
        "fpr_at_90_sensitivity": fpr90,
        "achieved_sensitivity": achieved_sensitivity,
        "threshold_at_90_sensitivity": threshold,
    }


def run_prediction(
    model: nn.Module,
    records: list[dict],
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[list[dict], dict]:
    loader = make_loader(records, batch_size, False, 0, num_workers)
    criterion = nn.CrossEntropyLoss(reduction="none")
    output_rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for patches, labels, indices in loader:
            patches = patches.to(device, non_blocking=True)
            labels_gpu = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(patches)
                losses = criterion(logits, labels_gpu)
            probabilities = torch.softmax(logits.float(), dim=1)
            one_hot = torch.nn.functional.one_hot(labels_gpu, num_classes=2).float()
            el2n = torch.linalg.vector_norm(probabilities - one_hot, dim=1)
            for batch_position, manifest_index in enumerate(indices.tolist()):
                record = records[manifest_index]
                probability = float(probabilities[batch_position, 1].cpu())
                output_rows.append(
                    {
                        **record,
                        "probability_positive": probability,
                        "predicted_label": int(probability >= 0.5),
                        "loss": float(losses[batch_position].float().cpu()),
                        "margin": float(abs(probability - 0.5) * 2.0),
                        "el2n": float(el2n[batch_position].cpu()),
                    }
                )
    metrics = binary_metrics(
        [row["label"] for row in output_rows],
        [row["probability_positive"] for row in output_rows],
    )
    return output_rows, metrics


def build_radial_masks() -> dict[str, np.ndarray | float | int]:
    center = PATCH_SIZE / 2.0 - 0.5
    zz, yy, xx = np.mgrid[0:PATCH_SIZE, 0:PATCH_SIZE, 0:PATCH_SIZE]
    distance = np.sqrt((zz - center) ** 2 + (yy - center) ** 2 + (xx - center) ** 2)
    inner = distance <= LEGACY_R_INNER
    outer = ~inner
    return {
        "distance": distance,
        "inner": inner,
        "outer": outer,
        "r_inner": LEGACY_R_INNER,
        "inner_count": int(inner.sum()),
        "outer_count": int(outer.sum()),
        "inner_cube_fraction": float(inner.mean()),
        "outer_cube_fraction": float(outer.mean()),
    }


RADIAL_MASKS = build_radial_masks()


def compute_radial_slope(patch: np.ndarray, n_bins: int = N_RADIAL_BINS) -> float:
    distance = np.asarray(RADIAL_MASKS["distance"]).ravel()
    values = patch.ravel()
    edges = np.linspace(0.0, R_MAX, n_bins + 1)
    means: list[float] = []
    centers: list[float] = []
    for index in range(n_bins):
        selected = (distance >= edges[index]) & (distance < edges[index + 1])
        if np.any(selected):
            means.append(float(values[selected].mean()))
            centers.append(float((edges[index] + edges[index + 1]) / 2.0))
    if len(centers) < 2:
        return 0.0
    x = np.asarray(centers, dtype=np.float64)
    x -= x.mean()
    y = np.asarray(means, dtype=np.float64)
    return float(np.dot(x, y) / (np.dot(x, x) + 1e-12))


def radial_ratio(attribution: np.ndarray) -> tuple[float, float, float, float]:
    inner_values = attribution[np.asarray(RADIAL_MASKS["inner"])]
    outer_values = attribution[np.asarray(RADIAL_MASKS["outer"])]
    inner_mean = float(inner_values.mean())
    outer_mean = float(outer_values.mean())
    ratio = float("nan") if inner_mean <= 1e-12 else outer_mean / inner_mean
    denominator = outer_mean + inner_mean
    bounded_fraction = float("nan") if denominator <= 1e-12 else outer_mean / denominator
    return ratio, bounded_fraction, inner_mean, outer_mean


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value: object) -> None:
    def convert(item):
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        raise TypeError(f"Cannot JSON-encode {type(item).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n")
