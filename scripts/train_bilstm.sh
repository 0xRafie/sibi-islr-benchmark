#!/usr/bin/env bash
# BiLSTM baseline (Table III, rows 1-2). One GPU (the paper used a Tesla T4); 5 seeds per fold.
# SI: leave-one-signer-out, folds 0-21. SD: stratified 5-fold, folds 0-4 (PROTOCOL=sd).
# Extra options go to training/bilstm.py, e.g. --shard 0/2 to split the jobs over two GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SIBI_DATA=${SIBI_DATA:-data} SIBI_QC=${SIBI_QC:-qc} SIBI_PAPER=${SIBI_PAPER:-paper}
PROTOCOL=${PROTOCOL:-si}
FOLDS=$([ "$PROTOCOL" = sd ] && echo 0-4 || echo 0-21)
$PY train.py --model bilstm --protocol "$PROTOCOL" --setting combined --folds "$FOLDS" --seeds 0-4 \
  --data "$SIBI_DATA/pilot" --folds-dir "$SIBI_QC" --out "$SIBI_QC/final/bilstm_$PROTOCOL" "$@"
