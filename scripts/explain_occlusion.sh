#!/usr/bin/env bash
# Test-time occlusion of the fine-tuned Uni-Sign (pose) on one held-out signer (keypoint groups, Shapley values; an
# analysis beyond the paper tables). One GPU. Needs the weights of fold FOLD from train_unisign.sh --save-dir in STATE.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SIBI_DATA=${SIBI_DATA:-data} SIBI_QC=${SIBI_QC:-qc} SIBI_PAPER=${SIBI_PAPER:-paper}
: "${FOLD:?set FOLD}" "${STATE:?set STATE to the saved weights of that fold}"
P=$SIBI_DATA/pilot
SRC=$P/rtmw_affix.npz,$P/rtmw_number.npz,$P/rtmw_alphabet.npz,$P/rtmw_kata.npz
$PY explain.py run combined "$SRC" "$STATE" "$SIBI_QC/explain/fold$FOLD" --fold "$FOLD" "$@"
