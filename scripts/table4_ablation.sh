#!/usr/bin/env bash
# Table IV (ablation of the input keypoints of Uni-Sign (pose) under SI): three variants (one GPU each, Tesla T4 in the
# paper), then the tests (CPU) -> $SIBI_PAPER/table_ablation.tex. SKIP_TRAIN=1 only analyzes existing runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SIBI_DATA=${SIBI_DATA:-data} SIBI_QC=${SIBI_QC:-qc} SIBI_PAPER=${SIBI_PAPER:-paper}
if [ -z "${SKIP_TRAIN:-}" ]; then
  for d in hands thumbs face; do DROP=$d scripts/train_unisign.sh; done
fi
mkdir -p "$SIBI_PAPER"
$PY evaluate.py analyze ablasi "$SIBI_QC/final"
$PY evaluate.py analyze perclass "$SIBI_QC/final"
$PY evaluate.py tables
