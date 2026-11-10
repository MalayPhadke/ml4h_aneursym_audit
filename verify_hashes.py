#!/usr/bin/env python3
"""Check shipped artifacts/configs against the SHA256 table in the paper (Appendix F, Table A-hashes)."""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
A, C = ROOT / "artifacts", ROOT / "configs" / "condition_plan"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

h = json.loads((A / "manifests" / "split_hashes.json").read_text())
CHECKS = {
    "Manifest mapping (split_hashes.json)": (h["manifest_mapping_sha256"], "dd11fed82c60a86a3bbb3704359147803c0267d498dde952fee64b4ad1a5456c"),
    "Train ID list": (sha(A / "manifests/train_patch_ids.txt"), "26a0bd83e7e9bffd0e859fcc5331f10a4b7f4f951b59564a82f91536d561d1b6"),
    "Validation ID list": (sha(A / "manifests/val_patch_ids.txt"), "553c6d4ca73996af211b9d0d8b8455e24c4f076ec805bbaade1dd71982568784"),
    "Test ID list": (sha(A / "manifests/test_patch_ids.txt"), "f9e8f98a2bcbe4eecdd8d22e0154cbb6ece531fd8b3ee1691d15578859d443c8"),
    "Manifest CSV file": (sha(A / "manifests/split_manifest.csv"), "2a28f92957fd644c1f2e0330bc17e3f7e9f3f18dfc31d0f7796848d9bac996de"),
    "Joined attribution CSV": (sha(A / "phase0_gates/attribution_per_patch.csv"), "4369142346d21e02d0cda4e7c115322aed884c09c95ffbd42f01028a4d8ca9c2"),
    "GradCAM++ CSV": (sha(A / "attribution_v1/gradcam_per_patch.csv"), "476edddaa48db61401f11a29e691c402e2cc1aba72e9d9ff1f78e90bf343482d"),
    "IG CSV": (sha(A / "attribution_v1/ig_per_patch.csv"), "357003cefedca5841c2d999733c5e7fd9b7b1750c16683438e28e60c39392579"),
    "Initial grid configuration": (sha(C / "phase3_5_conditions.csv"), "0b0f3b85aa443c9126651e77b1f88b76ea58428bf12742fed297e30f0c1388e6"),
    "Matched-positive configuration": (sha(C / "random170_positive_condition.csv"), "0bd31394b93950da780521c172cd1fe515201aa985ad27e19256e47715e22ab3"),
    "Multi-seed matched configuration": (sha(C / "random170_positive_multiseed.csv"), "571e666869b7febad8de9d63d313c4d7528523b98ea954bf6f187312932f92e2"),
    "Multi-seed summary": (sha(A / "results/random170_positive_multiseed_summary.json"), "a03fb0555c6bc2615cda39e8f55fe8b6b0b8b964b83c59b5d732c2202c536778"),
    "AUC-difference bootstrap": (sha(A / "results/full_vs_strict_auc_bootstrap.json"), "3ea3b5b8e48bbce0f3139fabae03ad9d07cc126c14cb6de8c7bec895d171d3a8"),
}
bad = 0
for name, (got, want) in CHECKS.items():
    ok = got == want
    bad += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name}: {got[:16]}")
sys.exit(1 if bad else 0)
