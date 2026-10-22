# SIBI-ISLR Benchmark

Code for a signer-independent benchmark of isolated sign recognition for SIBI (Sistem Isyarat Bahasa Indonesia), the
signed form of Indonesian taught in Indonesian schools for deaf students. The benchmark covers 114 classes in four
categories (18 affixes, 41 numbers, 26 letters, and 29 words) recorded by 22 signers. It compares five models under two
protocols:

- **SD (signer-dependent):** stratified 5-fold cross-validation over clips, the protocol of earlier SIBI work.
- **SI (signer-independent):** leave-one-signer-out cross-validation over 22 folds, so every test signer is unseen.

The paper is under review. A citation will be added here once it is published.

## Results

Accuracy (%) under SI: per instance (P-I), and per class (P-C) for Overall. The BiLSTM under SD is shown for reference.

| Input | Model | Protocol | Affix | Number | Alphabet | Word | Overall P-I | Overall P-C |
|---|---|---|---|---|---|---|---|---|
| MediaPipe | BiLSTM (Ho and Santoso) | SD | 95.92 | 99.44 | 98.73 | 98.89 | 97.29 | 97.34 |
| MediaPipe | BiLSTM (Ho and Santoso) | SI | 83.70 | 95.13 | 90.59 | 95.92 | 85.72 | 86.09 |
| MediaPipe | SignBart | SI | 86.16 | 91.64 | 95.71 | 97.16 | 86.68 | 87.17 |
| RTMW | DSTA-SLR | SI | 83.48 | 97.32 | 90.44 | 99.07 | 88.57 | 89.29 |
| RTMW | Uni-Sign (pose) | SI | **93.56** | 96.98 | **96.36** | 99.25 | **95.16** | **95.40** |
| DINOv2 + MediaPipe | SHuBERT | SI | 84.52 | 80.37 | 86.42 | **99.42** | 83.10 | 84.27 |

Category columns use the within-category score. Overall uses the 114-class score.

## Data

The SIBI dataset of Darmawan et al. is available from its owner under a data use agreement for non-commercial research.
This repository contains **no** videos, images, keypoints, fold files, or predictions. The fold files are regenerated
from the dataset with `scripts/folds.py`. Please cite the dataset articles required by the agreement.

## Setup

Python 3.12 with the pinned packages in `requirements.txt` (MediaPipe must stay at 0.10.14, the version the BiLSTM baseline
was built on):

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
```

Training uses one NVIDIA GPU (all reported runs used a Tesla T4). Uni-Sign and SHuBERT are loaded from their authors'
repositories at the commits used in the paper (see the module docstrings).

## Pipeline

Each script documents its inputs, outputs, and options in its docstring, and most have a `--demo` self-check. Data paths
default to `data/` and `qc/` and can be changed with the environment variables `SIBI_DATA`, `SIBI_QC`, `SIBI_PAPER`, and
`SIBI_DICT`.

1. **Inventory and keypoints:** `inventory_darmawan.py` lists the dataset files. `extract.py` extracts MediaPipe
   Holistic keypoints, and `extract_rtmw.py` extracts RTMW keypoints through rtmlib. Both run once and cache their output.
2. **Clips:** `trim.py` and `pack_mp.py` trim the isolated clips to the frames with a detected hand. `seg_fplus.py` cuts
   word clips from the sentence videos by semi-supervised forced alignment, on top of `align_words.py`, and
   `cut_words.py` writes the clips. The neutral sentence keypoints are expected in `data/kata/keypoints`; a link
   (`ln -s ../keypoints data/kata/keypoints`) is enough.
3. **Folds:** `folds.py` writes the SD (stratified 5-fold) and SI (leave-one-signer-out) fold assignments.
4. **Models:**
   - `train_main.py`: BiLSTM.
   - `slot2.py` (options `mp48` and `rtmw27`): SignBart and DSTA-SLR. SignBart needs a clone of its official repository
     at the commit given in `scripts/models/signbart.py`.
   - `unisign_finetune.py`: Uni-Sign. `--drop` trains the ablation variants, and `--protocol all` trains the released
     model on all 22 signers.
   - `shubert_features.py` and `shubert_lora.py`: SHuBERT.
5. **Evaluation:** `analyze.py` computes the metrics and the Wilcoxon signed-rank tests with Holm correction.
   `paper_tables.py` and `figures.py` produce the tables and figures.
6. **Occlusion analysis:** `explain_unisign.py` and `explain_unisign_run.py`.

Three inputs derived from the dataset are not included and are available from the authors once the data owner has granted
access:
- the manual sign boundaries of the 110 annotated sentence videos used to train the aligner;
- the corrected labels of one signer's sentence files;
- the fold files.

## Released model

A Uni-Sign (pose) model fine-tuned on all 22 signers is released on Hugging Face at
[0xRafie/uni-sign-sibi](https://huggingface.co/0xRafie/uni-sign-sibi) under CC BY-NC 4.0. That is the license of the
Uni-Sign weights it starts from, and it matches the non-commercial terms of the data. Its expected accuracy on a new signer
is the SI result above.

## License

- The code in this repository is released under the MIT License (`LICENSE`).
- `scripts/models/dsta_slr.py` is adapted from DSTA-SLR and stays under CC BY-NC 4.0.
- SignBart, Uni-Sign, and SHuBERT are used from their own repositories under their own terms.

See `NOTICE.md` for details.
