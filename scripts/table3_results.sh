#!/usr/bin/env bash
# Table III (results of the five models under SD and SI): metrics, Wilcoxon signed-rank tests with Holm correction,
# per-class and per-signer scores -> $SIBI_QC/results/, then $SIBI_PAPER/table_results.tex. CPU, a few minutes.
# Needs the runs of the train_*.sh scripts in $SIBI_QC/final (any layout below it) and $SIBI_QC/folds_combined.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SIBI_DATA=${SIBI_DATA:-data} SIBI_QC=${SIBI_QC:-qc} SIBI_PAPER=${SIBI_PAPER:-paper}
mkdir -p "$SIBI_PAPER"
$PY evaluate.py analyze final "$SIBI_QC/final"
$PY evaluate.py analyze perclass "$SIBI_QC/final"
$PY evaluate.py tables
