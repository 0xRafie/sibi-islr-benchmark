#!/usr/bin/env bash
# Uni-Sign (pose) fine-tuning, SI (Table III). One GPU (Tesla T4 in the paper, about 4 hours per fold), seed 0.
# Needs the Uni-Sign clone ($UNISIGN_DIR, default ext/Uni-Sign), its checkpoint ($UNISIGN_CKPT) and the extras
# pip install -e ".[unisign]"; see sibi_islr/models/unisign.py. Fold tables are read from ./qc (or the working directory).
# DROP=hands|thumbs|face trains an ablation variant of Table IV; --save-dir DIR keeps the weights of each fold.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SIBI_DATA=${SIBI_DATA:-data} SIBI_QC=${SIBI_QC:-qc} SIBI_PAPER=${SIBI_PAPER:-paper}
P=$SIBI_DATA/pilot
SRC=$P/rtmw_affix.npz,$P/rtmw_number.npz,$P/rtmw_alphabet.npz,$P/rtmw_kata.npz
OUT=$SIBI_QC/final/unisign${DROP:+_$DROP}
$PY train.py --model unisign --protocol si combined "$SRC" "$OUT" --folds ${FOLDS:-0-21} --seeds 0 --epochs 20 \
  ${DROP:+--drop $DROP} "$@"
