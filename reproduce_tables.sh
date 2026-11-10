#!/bin/bash
# Regenerate the paper's analysis tables from per-run predictions (CPU only).
# By default works on a scratch copy of artifacts/ so shipped files stay untouched;
# set AUDIT_OUT_ROOT=outputs to analyse your own runs.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
if [ -z "${AUDIT_OUT_ROOT:-}" ]; then
  rm -rf outputs_repro && cp -r artifacts outputs_repro
  export AUDIT_OUT_ROOT="$PWD/outputs_repro"
fi
python analysis/size_enrichment.py         # raw-zero vs small lesions (OR 2.28)
python analysis/auc_difference_ci.py       # strict - full AUC bootstrap
python analysis/random170_variance.py      # 5 random 170-positive subsets
python analysis/clean_test_analyses.py     # clean-test metrics, attribution agreement
python analysis/multiseed_summary.py       # Table 2: 5 paired seeds
