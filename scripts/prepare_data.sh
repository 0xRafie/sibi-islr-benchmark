#!/usr/bin/env bash
# Data preparation: inventory -> MediaPipe and RTMW keypoints -> trimmed keypoint packs -> word clips -> folds.
# Needs the SIBI dataset (data use agreement) in $SIBI_DATA/darmawan. MediaPipe runs on a CPU (hours); RTMW and the
# word aligner need one GPU. The word clips also need the manual boundaries of the 110 annotated sentence videos
# ($SIBI_DATA/annotations/words_manual.csv), available from the authors once the data owner has granted access.
# Options of every step: the docstring of its module (python prepare_data.py --help lists them).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export SIBI_DATA=${SIBI_DATA:-data} SIBI_QC=${SIBI_QC:-qc} SIBI_PAPER=${SIBI_PAPER:-paper}

$PY prepare_data.py inventory                                   # -> $SIBI_QC/darmawan_files.csv
for c in affix number alphabet; do                               # MediaPipe Holistic -> $SIBI_DATA/keypoints/
  $PY prepare_data.py extract --darmawan "$c/*/*.mp4"
done
$PY prepare_data.py extract --darmawan 'sentence/neutral/*.mp4'
$PY prepare_data.py pack                                        # -> $SIBI_DATA/pilot/mp_{affix,number,alphabet}.npz
for c in affix number alphabet; do                               # RTMW (GPU) -> $SIBI_DATA/pilot/rtmw_<category>.npz
  $PY prepare_data.py extract-rtmw "$SIBI_DATA/darmawan/$c" "$SIBI_DATA/pilot/rtmw_$c.npz"
done
$PY prepare_data.py extract-rtmw "$SIBI_DATA/darmawan/sentence/neutral" "$SIBI_DATA/pilot/rtmw_neutral.npz"

# Word clips from the neutral sentence videos (semi-supervised forced alignment, F+)
mkdir -p "$SIBI_DATA/kata" && ln -sfn ../keypoints "$SIBI_DATA/kata/keypoints"
$PY prepare_data.py align pack                                  # -> $SIBI_DATA/kata/align_data.npz
$PY prepare_data.py fplus train "$SIBI_DATA/kata/align_data.npz" "$SIBI_DATA/kata"   # GPU
$PY prepare_data.py cut F+                                      # -> $SIBI_DATA/pilot/mp_kata.npz, $SIBI_QC/kata_clips.csv
$PY prepare_data.py cut rtmw                                    # -> $SIBI_DATA/pilot/rtmw_kata.npz

$PY prepare_data.py folds                                       # -> $SIBI_QC/folds_combined.csv, folds_cr.csv
