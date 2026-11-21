#!/usr/bin/env bash
# Fig. 6 (per-signer affix accuracy under SI) and Fig. 7 (affix confusion of Uni-Sign) -> $SIBI_QC/figures/. CPU.
# Needs the runs in $SIBI_QC/final and $SIBI_QC/folds_combined.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SIBI_DATA=${SIBI_DATA:-data} SIBI_QC=${SIBI_QC:-qc} SIBI_PAPER=${SIBI_PAPER:-paper}
$PY make_figures.py final "$@"
