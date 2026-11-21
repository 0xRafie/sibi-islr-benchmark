#!/usr/bin/env bash
# Table I (dataset statistics): clip counts per category and protocol -> $SIBI_QC/protocol_counts.csv.
# CPU; needs $SIBI_QC/darmawan_files.csv (prepare_data.sh). The table itself was typed from these counts.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SIBI_DATA=${SIBI_DATA:-data} SIBI_QC=${SIBI_QC:-qc} SIBI_PAPER=${SIBI_PAPER:-paper}
$PY evaluate.py protocol-counts "$@"
