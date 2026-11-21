#!/usr/bin/env bash
# SignBart with its official recipe and input (mp48), SI (Table III). One GPU per fold (Tesla T4 in the paper), seed 0.
# The official repository is cloned on first use into $SIGNBART_DIR (default ext/SignBart).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SIBI_DATA=${SIBI_DATA:-data} SIBI_QC=${SIBI_QC:-qc} SIBI_PAPER=${SIBI_PAPER:-paper}
export SLOT2_DATA=${SLOT2_DATA:-$SIBI_DATA/pilot} SLOT2_FOLDS=${SLOT2_FOLDS:-$SIBI_QC} SLOT2_OUT=${SLOT2_OUT:-$SIBI_QC/final/signbart}
mkdir -p "$SLOT2_OUT"
for k in ${FOLDS:-$(seq 0 21)}; do
  $PY train.py --model signbart --protocol si "$k" batch=128 meanloss mp48 cat=combined "$@"
done
