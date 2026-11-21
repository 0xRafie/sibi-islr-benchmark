"""Data preparation: inventory, keypoints, trimming and packing, word clips, folds, model input features.

    python prepare_data.py [--data-dir DIR] [--qc-dir DIR] [--dict-dir DIR] STEP [ARGS ...]

Steps (module in sibi_islr/; ARGS and --demo as documented in its docstring):
    inventory         data/inventory.py          list the dataset videos -> <qc>/darmawan_files.csv
    extract           data/extract_mediapipe.py  MediaPipe Holistic keypoints (--darmawan PATTERN | --jobs ...)
    extract-rtmw      data/extract_rtmw.py       RTMW whole-body keypoints through rtmlib (GPU)
    trim              data/trim.py               trim clips to the frames with a detected hand
    pack              data/pack.py               keypoint packs mp_<category>.npz [--face ...] [--fx ...]
    segment           data/words/segment.py      candidate word boundaries in the sentence videos
    align             data/words/align.py        HMM/Viterbi word alignment: pack | train ... | usulan
    fplus             data/words/fplus.py        semi-supervised forced alignment: train ...
    cut               data/words/cut.py          cut the word clips: <win> | rtmw ... | face ...
    folds             data/folds.py              SD (stratified 5-fold) and SI (leave-one-signer-out) folds
    unisign-features  models/unisign.py          frozen Uni-Sign features: extract ...
    shubert-features  models/shubert.py          SHuBERT inputs: fetch | a ... | merge ... | b ...

Example: python prepare_data.py folds          # -> <qc>/folds_combined.csv, <qc>/folds_cr.csv
         python prepare_data.py folds --demo   # self-check on synthetic data
"""
from sibi_islr.cli import dispatch

STEPS = {
    "inventory": "sibi_islr.data.inventory",
    "extract": "sibi_islr.data.extract_mediapipe",
    "extract-rtmw": "sibi_islr.data.extract_rtmw",
    "trim": "sibi_islr.data.trim",
    "pack": "sibi_islr.data.pack",
    "segment": "sibi_islr.data.words.segment",
    "align": "sibi_islr.data.words.align",
    "fplus": "sibi_islr.data.words.fplus",
    "cut": "sibi_islr.data.words.cut",
    "folds": "sibi_islr.data.folds",
    "unisign-features": "sibi_islr.models.unisign",
    "shubert-features": "sibi_islr.models.shubert",
}

if __name__ == "__main__":
    dispatch(__doc__, "prepare_data.py", STEPS)
