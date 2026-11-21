#!/usr/bin/env bash
# SHuBERT + rank-1 LoRA, SI (Table III). One GPU (Tesla T4 in the paper).
# Two environments (see sibi_islr/models/shubert.py): stage a (Python 3.12, this package with ".[shubert]") turns each
# clip into per-frame DINOv2 and pose features, FRAMES/frames_<category>.npz:
#   python prepare_data.py shubert-features fetch
#   python prepare_data.py shubert-features a <clips.json> <frames_<category>.npz> [--shard i/n]   (then merge)
# The LoRA runs need environment b (Python 3.10 + the authors' fairseq); fold tables are read from ./qc.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SIBI_DATA=${SIBI_DATA:-data} SIBI_QC=${SIBI_QC:-qc} SIBI_PAPER=${SIBI_PAPER:-paper}
FRAMES=${FRAMES:-$SIBI_DATA/pilot/shubert_frames}
$PY train.py --model shubert --protocol si run "$FRAMES" "$SIBI_QC/final/shubert" ${FOLDS:-$(seq 0 21)} "$@"
