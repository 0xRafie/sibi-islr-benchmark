#!/usr/bin/env bash
# Fig. 1-5 -> $SIBI_PAPER/figures/ and Table II -> $SIBI_PAPER/table_keypoints.tex. CPU.
# Needs the RTMW keypoint packs ($SIBI_DATA/pilot/rtmw_*.npz) and, for panel (c) of Fig. 2 and Fig. 3, the SIBI
# dictionary clip imbuhan__Awalan-Me in $SIBI_DICT/video and $SIBI_DICT/keypoints.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SIBI_DATA=${SIBI_DATA:-data} SIBI_QC=${SIBI_QC:-qc} SIBI_PAPER=${SIBI_PAPER:-paper}
$PY make_figures.py "$@"
