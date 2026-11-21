#!/usr/bin/env bash
# DSTA-SLR with its official recipe and input (27 RTMW points, rtmw27), SI (Table III). One GPU per fold (Tesla T4 in
# the paper; DSTA-SLR runs on CUDA only), seed 0, 250 epochs. For sessions with a time limit, add stop=125 to the first
# run and resume to the second (checkpoints in $SLOT2_CKPT_OUT, read from $SLOT2_CKPT); the result equals one run.
# The official repository is cloned on first use into $DSTA_SLR_DIR (default ext/DSTA-SLR).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SIBI_DATA=${SIBI_DATA:-data} SIBI_QC=${SIBI_QC:-qc} SIBI_PAPER=${SIBI_PAPER:-paper}
export SLOT2_DATA=${SLOT2_DATA:-$SIBI_DATA/pilot} SLOT2_FOLDS=${SLOT2_FOLDS:-$SIBI_QC} SLOT2_OUT=${SLOT2_OUT:-$SIBI_QC/final/dsta_slr}
mkdir -p "$SLOT2_OUT"
for k in ${FOLDS:-$(seq 0 21)}; do
  $PY train.py --model dsta_slr --protocol si "$k" epochs=250 rtmw27 cat=combined "$@"
done
