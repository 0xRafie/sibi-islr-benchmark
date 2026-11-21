#!/usr/bin/env bash
# Every paper result, in order. Needs the dataset (data use agreement) and one or more GPUs; on one Tesla T4 the
# training takes several weeks of GPU time (110 BiLSTM runs per protocol, 22 folds for each other model, plus the
# Uni-Sign ablations). The evaluation steps run on a CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SIBI_DATA=${SIBI_DATA:-data} SIBI_QC=${SIBI_QC:-qc} SIBI_PAPER=${SIBI_PAPER:-paper}
S=scripts
$S/prepare_data.sh
$S/table1_dataset.sh
$S/train_bilstm.sh; PROTOCOL=sd $S/train_bilstm.sh
$S/train_signbart.sh
$S/train_dsta_slr.sh
$S/train_unisign.sh
$S/train_shubert.sh          # run in environment b (see the script)
$S/table3_results.sh
$S/table4_ablation.sh
$S/figures_1_to_5.sh
$S/figures_6_7.sh
