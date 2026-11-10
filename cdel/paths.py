"""Central path configuration.

Everything is relative to the repository root unless overridden by env vars:

  CDEL_DATA_ROOT  directory holding ``patches/`` and the size table (default: <repo>/data)
  CDEL_OUT_ROOT   pipeline output root (default: <repo>/outputs).
                  Set to <repo>/artifacts to run analyses on the shipped results.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("CDEL_DATA_ROOT", REPO_ROOT / "data"))
PATCH_ROOT = DATA_ROOT / "patches"
SIZE_CSV = DATA_ROOT / "aneurysm_size_estimates_v3.csv"
PIPELINE_ROOT = Path(os.environ.get("CDEL_OUT_ROOT", REPO_ROOT / "outputs"))
CONDITION_PLAN_DIR = REPO_ROOT / "configs" / "condition_plan"


def resolve_patch_path(path: str | Path) -> Path:
    """Map a manifest/attribution ``path`` onto PATCH_ROOT.

    Shipped CSVs keep the original absolute paths (so their SHA256s match the
    paper); anything after the last ``/patches/`` component is re-rooted here.
    """
    path = Path(path)
    if path.is_absolute() and path.is_file():
        return path
    parts = path.parts
    if "patches" in parts:
        path = Path(*parts[len(parts) - parts[::-1].index("patches"):])
    return PATCH_ROOT / path
