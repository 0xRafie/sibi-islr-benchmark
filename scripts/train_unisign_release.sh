#!/usr/bin/env bash
# The released model (huggingface.co/0xRafie/uni-sign-sibi): Uni-Sign (pose) fine-tuned on all 22 signers, no test
# data. One GPU (Tesla T4, about 4.3 hours). Same requirements as train_unisign.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SIBI_DATA=${SIBI_DATA:-data} SIBI_QC=${SIBI_QC:-qc} SIBI_PAPER=${SIBI_PAPER:-paper}
P=$SIBI_DATA/pilot
SRC=$P/rtmw_affix.npz,$P/rtmw_number.npz,$P/rtmw_alphabet.npz,$P/rtmw_kata.npz
$PY train.py --model unisign --protocol all combined "$SRC" "$SIBI_QC/final/unisign_release" --seeds 0 --epochs 20 \
  --save-dir "${SAVE_DIR:-weights/unisign_release}" "$@"
