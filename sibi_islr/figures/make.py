"""Figures of the paper and its keypoint table (Table II), drawn from the results and keypoints. Dataset clips appear
only as skeletons: the dataset's data use agreement does not allow publishing its video frames, so the image panels
use a frame of the public online SIBI dictionary instead. Output: PDF + PNG.

    python make_figures.py --demo      # self-check on synthetic data
    python make_figures.py             # Fig. 1-5, Table II and the occlusion figure -> paper/figures/, paper/
    python make_figures.py final       # Fig. 6 and 7 (+ confusion matrices of the other models) -> qc/figures/

Fig. 1  fig_frames            me-, 1, a, saya from signer 1: the 69 RTMW keypoints read by Uni-Sign in five frames per
                              sign; also writes Table II (raw RTMW against Uni-Sign-normalized coordinates)
Fig. 2  fig_inputs            input of each model for me-: MediaPipe Holistic (543 points) and RTMW (133 points, 69 used by
                              Uni-Sign) on one frame of signer 1; SHuBERT's three image crops and seven pose points on a
                              dictionary frame (shubert_crops copies crop_hands and crop_face of the SHuBERT Space)
Fig. 3  fig_unisign           Uni-Sign as fine-tuned here (pose branch, mT5 encoder, linear 114-class head), drawn from
                              ext/Uni-Sign/models.py:206-267 and scripts/unisign_finetune.py
Fig. 4  fig_protocols         one example fold of SD (stratified 5-fold) and SI (leave-one-signer-out)
Fig. 5  fig_ablation_unisign  the 69 Uni-Sign input keypoints in the full model and the three ablation variants
Fig. 6  fig_signer_heat       per-signer SI affix accuracy of the five models (final runs, seed 0)
Fig. 7  fig_confusion_affix   affix confusion matrix of Uni-Sign under SI, row-normalized (final runs, seed 0)
        fig_shapley           test-time occlusion of the full Uni-Sign on signer 5: Shapley values of six keypoint groups
                              (needs qc/results/e22_report.json from explain_unisign_run.py --report)

Inputs: RTMW keypoint packs (data/pilot/rtmw_<category>.npz, rtmw_neutral.npz for words), MediaPipe keypoints
(data/keypoints or data/keypoints_fig; words cut from data/kata/keypoints by qc/kata_clips.csv), the final runs with
qc/folds_combined.csv, and the dictionary clip video/imbuhan__Awalan-Me.webm with keypoints/imbuhan__Awalan-Me.npy.
Paths are relative to the working directory; environment variables move them:
    SIBI_QC    (default qc)     folds, kata_clips.csv, final runs, results/e22_report.json, Fig. 6-7 output (figures/)
    SIBI_DATA  (default data)   keypoint packs and keypoints
    SIBI_PAPER (default paper)  Fig. 1-5 output (figures/) and table_keypoints.tex
    SIBI_DICT  (default: working directory)  the dictionary's video/ and keypoints/ folders

Sizes follow the journal template: one column 86.4 mm, two columns 181.4 mm, Times. Colors come from a sequential blue
ramp and three categorical slots, and each series also has its own marker, so the figures stay readable in black and
white. No hatching.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle

from sibi_islr.figures.common import AFFIX, OUT, PAPER
from sibi_islr.figures.results import FINAL_MODELS, FINAL_OUT, HELD, N_CLS, N_SIG, N_TAKE, fig_confusion_affix, fig_protocols, fig_signer_heat, final_confusion, final_signer, protocol_masks
from sibi_islr.figures.keypoints import ABLATIONS, UNISIGN_PTS, fig_frames, fig_inputs, frame_ids, parts, shubert_crops, unisign_scale
from sibi_islr.figures.unisign import GROUP_LABEL, fig_ablation_unisign, fig_shapley, fig_unisign


def demo():
    import tempfile
    from sibi_islr.explain.occlusion import GROUPS
    m = protocol_masks()
    assert set(m) == {"SD", "SI"}
    assert (m["SI"] == "test").sum() == N_CLS * N_TAKE and (m["SI"][HELD] == "test").all()
    assert all((m["SD"][s, c] == "train").any() for s in range(N_SIG) for c in range(N_CLS))
    assert all((m["SD"][:, c] == "test").sum() == round(N_SIG * N_TAKE / 5) for c in range(N_CLS))
    arr = np.zeros((60, 1662))
    arr[10:50, 1599:] = 0.5                                                     # right hand detected in frames 10-49
    arr[3, 1599:] = 0.5                                                         # one-frame blip before the span: cleaned
    arr[30, 1599:] = 0                                                          # one-frame gap inside the span: filled
    assert frame_ids(arr).tolist() == [10, 20, 30, 39, 49]
    pose, face, lh, rh = parts(arr[0])
    assert pose.shape == (33, 4) and face.shape == (468, 3) and lh.shape == rh.shape == (21, 3)
    k = np.zeros((70, 133, 2))
    k[:, [0, *range(3, 11)]] = np.linspace([0.3, 0.2], [0.7, 0.4], 9)          # body box 0.4 x 0.2 -> scale 0.4
    assert abs(unisign_scale(k, np.ones((70, 133))) - 0.4) < 1e-9
    assert len(set(UNISIGN_PTS)) == len(UNISIGN_PTS) == 69                        # 9 body + 42 hand + 18 face
    keep = lambda drop: len(set(UNISIGN_PTS) - set(drop))
    assert [keep(d_) for _, d_ in ABLATIONS] == [69, 44, 61, 51]                   # hands only = 42 + 2 wrists
    assert set(GROUP_LABEL) == set(GROUPS)                                         # the groups of explain_unisign_run.py --report
    img = np.arange(200 * 300 * 3, dtype=np.uint8).reshape(200, 300, 3)
    fc = np.zeros((468, 3))
    fc[[69, 168, 156, 118, 54], :2] = [[0.40, 0.30], [0.45, 0.30], [0.40, 0.33], [0.42, 0.34], [0.41, 0.31]]
    fc[[299, 347, 336, 301], :2] = [[0.55, 0.30], [0.50, 0.34], [0.52, 0.31], [0.55, 0.33]]
    fc[[164, 212, 432, 18], :2] = [[0.47, 0.45], [0.43, 0.48], [0.52, 0.48], [0.47, 0.52]]
    hand = np.zeros((21, 3))
    hand[:, :2] = np.linspace([0.10, 0.60], [0.20, 0.75], 21)                    # 30..60 px -> side 31 + 2 x 7
    crops, boxes = shubert_crops(img, fc, hand, hand)
    assert all(c.shape == (224, 224, 3) for c in crops.values()) and boxes["right"] == (22, 112, 45)   # center (44, 134)
    assert (crops["face"][-1, 0] == 128).all() and boxes["face"] == (110, 50, 65)  # corner outside eyes and mouth = grey
    with tempfile.TemporaryDirectory() as d:     # synthetic final runs: 18 affixes (labels in reverse AFFIX order) + 1 number
        rows = [("affix", c, 17 - i) for i, c in enumerate(AFFIX)] + [("number", "1", 18)]
        folds = f"{d}/folds_combined.csv"
        with open(folds, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["category", "class", "label", "subject", "cl"])
            w.writerows([(*r, s, s - 1) for s in (1, 2) for r in rows])        # signer s = fold s - 1, 19 clips each
        y = np.array([r[2] for r in rows])
        right = np.eye(19)[y]                                                   # every clip correct
        wrong = right.copy()
        wrong[AFFIX.index("ti")] = np.eye(19)[17 - AFFIX.index("wati")]         # -ti taken for -wati
        wrong[AFFIX.index("an"), 18] = 5                                        # a number scores highest: ignored
        mt = lambda a: dict(metrics=dict(acc=a, macro_f1=a, acc_within_affix=a, f1_within_affix=a, acc_affix=a, f1_affix=a))
        os.makedirs(f"{d}/runs")
        for fold, seed, models in ((0, 0, {"bilstm": (0.8, right), "unisign_ft": (0.95, wrong)}),
                                   (0, 1, {"bilstm": (0.6, right)}),
                                   (1, 0, {"bilstm": (0.9, right), "unisign_ft": (1.0, right)})):
            stem = f"{d}/runs/r_f{fold}_s{seed}"
            json.dump(dict(setting="combined", protocol="cl", fold=fold, seed=seed, test_index=list(range(19 * fold, 19 * fold + 19)),
                           models={m_: mt(a) for m_, (a, _) in models.items()}), open(stem + ".json", "w"))
            np.savez(stem + "_scores.npz", **{m_: sc for m_, (_, sc) in models.items()})
        fa = final_signer(f"{d}/runs", [FINAL_MODELS[0], FINAL_MODELS[3]], folds_csv=folds)
        assert {s: np.round(v, 6).tolist() for s, v in fa["bilstm"].items()} == {1: [80.0], 2: [90.0]}, fa   # seed 0 only
        assert np.round(fa["unisign_ft"][2], 6).tolist() == [100.0]
        assert sorted(np.round(final_signer(f"{d}/runs", FINAL_MODELS[:1], None, folds)["bilstm"][1], 6)) == [60.0, 80.0]
        M, n = final_confusion(f"{d}/runs", "unisign_ft", (0,), folds)
        i, j = AFFIX.index("ti"), AFFIX.index("wati")
        assert n == 2 and np.allclose(M.sum(1), 100) and M[i, j] == M[i, i] == 50 and np.trace(M) == 17 * 100 + 50, M
        assert final_confusion(f"{d}/runs", "bilstm", None, folds)[1] == 3 and final_confusion(f"{d}/runs", "bilstm", (0,), folds)[1] == 2
        os.makedirs(f"{d}/slot2")                                               # slot2.py format: one list per fold
        json.dump([dict(cat="combined", proto="cl", model="dsta", fold=0, seed=0, test_idx=list(range(19)), options=["rtmw27"],
                        pred=y.tolist())], open(f"{d}/slot2/dsta.json", "w"))
        np.savez(f"{d}/slot2/dsta_f00_scores.npz", dsta=wrong)
        M, n = final_confusion(f"{d}/slot2", "dsta_am21", None, folds)
        assert n == 1 and M[i, j] == 100 and M[i, i] == 0 and np.allclose(np.delete(np.diag(M), i), 100)
    print("demo ok: example folds, frame choice, Uni-Sign scale and keypoint counts, SHuBERT crops, per-signer accuracy "
          "and affix confusion matrix from synthetic final runs (both result formats)")


def cli():
    if "--demo" in sys.argv:
        demo()
    elif "final" in sys.argv:                                     # figures.py final -> qc/figures/ (combined model)
        fig_signer_heat(final_signer())                                              # Fig. 6
        fig_confusion_affix(final_confusion(model="unisign_ft", seeds=(0,))[0], name="fig_confusion_affix_final", out=FINAL_OUT)  # Fig. 7
        for m in ("bilstm", "signbart_am21", "dsta_am21", "shubert_lora"):
            fig_confusion_affix(final_confusion(model=m, seeds=(0,))[0], name=f"fig_confusion_affix_final_{m}", out=FINAL_OUT)
        print(f"written to {FINAL_OUT}/ (fig_signer_cl_final, fig_confusion_affix_final + four other models)")
    else:
        fig_protocols()
        fig_frames()
        fig_inputs()
        fig_ablation_unisign()
        fig_shapley()
        fig_unisign()
        print(f"written to {OUT}/ (fig_protocols, fig_frames, fig_inputs, fig_ablation_unisign, fig_shapley, fig_unisign) "
              f"and {PAPER}/table_keypoints.tex")


if __name__ == "__main__":
    cli()
