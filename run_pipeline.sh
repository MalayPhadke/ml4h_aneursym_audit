#!/bin/bash
# Full pipeline, in order. Needs the patches (CDEL_DATA_ROOT) and a GPU.
# Run a single stage on its own by copying one line.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/01_freeze_split.py                        # patient-grouped split + hashes
python scripts/02_train_baseline.py                      # train-only baseline
python scripts/03_attribution.py --method gradcam --batch-size 16
python scripts/03_attribution.py --method ig --batch-size 4 --ig-steps 100 --ig-alpha-chunk 10 --ig-quadrature trapezoid --force
python scripts/04_audit_gates.py                         # ID coverage, raw-zero audit, split hashes

# Curation conditions (one CSV row = one training run; add INDEX handling or parallelise as you like)
for plan in phase3_5_conditions n1_full_condition random170_positive_condition \
            random170_positive_multiseed paired_multiseed_missing; do
  n=$(( $(wc -l < configs/condition_plan/$plan.csv) - 1 ))
  for i in $(seq 0 $((n - 1))); do
    python scripts/05_train_condition.py --conditions configs/condition_plan/$plan.csv --condition-index $i
  done
done
python scripts/06_summarize_grid.py                      # grid table + follow-up plan

CDEL_OUT_ROOT="${CDEL_OUT_ROOT:-$PWD/outputs}" bash reproduce_tables.sh
