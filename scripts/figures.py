"""Figures of the paper and its keypoint table (Table II), drawn from the results and keypoints. Dataset clips appear
only as skeletons: the dataset's data use agreement does not allow publishing its video frames, so the image panels
use a frame of the public online SIBI dictionary instead. Output: PDF + PNG.

    python scripts/figures.py --demo      # self-check on synthetic data
    python scripts/figures.py             # Fig. 1-5, Table II and the occlusion figure -> paper/figures/, paper/
    python scripts/figures.py final       # Fig. 6 and 7 (+ confusion matrices of the other models) -> qc/figures/

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

QC = os.environ.get("SIBI_QC", "qc")
DATA = os.environ.get("SIBI_DATA", "data")
PAPER = os.environ.get("SIBI_PAPER", "paper")
DICT = os.environ.get("SIBI_DICT", "")
OUT = f"{PAPER}/figures"
COL, FULL = 86.4 / 25.4, 181.4 / 25.4                 # inches
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]           # slots 1-3, distinguishable in every pair
BLUE_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEST, TRAIN = "#256abf", "#d6d5cf"
# class order: prefixes, suffixes, particles
AFFIX = ["ber", "di", "ke", "me", "pe", "se", "ter", "an", "i", "kan", "man", "nya", "ti", "wan", "wati", "kah", "lah",
         "pun"]
DISPLAY = {c: (f"{c}-" if i < 7 else f"-{c}") for i, c in enumerate(AFFIX)}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "axes.linewidth": 0.6,
})


def save(fig, name, out=OUT):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/{name}.{ext}", bbox_inches="tight", pad_inches=0.02,
                    metadata={"CreationDate": None} if ext == "pdf" else None)    # byte-identical PDF on every run
    plt.close(fig)


# ---------- Fig. 4: evaluation protocols ----------

N_SIG, N_CLS, N_TAKE = 6, 4, 3
HELD = 2                 # signer tested in the SI example


def protocol_masks(seed=0):
    """-> {protocol: array (signer, class, take)} of 'test' or 'train' for one example fold."""
    rng = np.random.default_rng(seed)
    base = np.full((N_SIG, N_CLS, N_TAKE), "train", dtype=object)
    cs = base.copy()                                  # SD: per class, about 1/5 of the clips at random go to test
    cells = [(s, t) for s in range(N_SIG) for t in range(N_TAKE)]
    for c in range(N_CLS):
        while True:                                   # as in nearly every SD fold, each signer-class pair keeps
            pick = rng.choice(len(cells), round(len(cells) / 5), replace=False)   # a training take
            if max(np.bincount([cells[i][0] for i in pick], minlength=N_SIG)) < N_TAKE:
                break
        for i in pick:
            cs[cells[i][0], c, cells[i][1]] = "test"
    cl = base.copy()                                  # SI: one signer
    cl[HELD] = "test"
    return {"SD": cs, "SI": cl}                      # signer-dependent / signer-independent


def fig_protocols():
    """One example fold of SD and SI, with axis labels (signer, class) and, in SD, a dashed box around a test take whose
    other takes of the same signer and class are in training. Panel titles are only SD/SI; the caption explains them."""
    masks = protocol_masks()
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.05))
    cw, gap = 1.0, 0.5
    xc = lambda c, t=0: c * (N_TAKE * cw + gap) + t * cw
    for ax, (name, mk) in zip(axes, masks.items()):
        for s in range(N_SIG):
            for c in range(N_CLS):
                for t in range(N_TAKE):
                    ax.add_patch(Rectangle((xc(c, t) + 0.06, N_SIG - 1 - s + 0.06), cw - 0.12, 0.88,
                                           facecolor=TEST if mk[s, c, t] == "test" else TRAIN, edgecolor="none"))
        if name == "SD":                               # one test take whose other takes of that signer and class train
            s0, c0 = next((s, c) for c in range(N_CLS) for s in range(N_SIG) if list(mk[s, c]).count("test") == 1)
            ax.add_patch(Rectangle((xc(c0) - 0.06, N_SIG - 1 - s0 - 0.08), N_TAKE * cw + 0.12, 1.16, fill=False,
                                   edgecolor=INK, lw=0.8, ls=(0, (2, 1.2)), zorder=3))
        width = N_CLS * (N_TAKE * cw + gap) - gap
        ax.set_xlim(-1.9, width + 0.1)                  # same in both panels: same scale
        ax.set_ylim(-1.55, N_SIG + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name, pad=2)                       # explained in the caption
        for c in range(N_CLS):
            ax.text(xc(c) + N_TAKE * cw / 2, -0.4, str(c + 1), ha="center", va="center", fontsize=6, color=MUTED)
        for s in range(N_SIG):
            ax.text(-0.45, N_SIG - 1 - s + 0.5, str(s + 1), ha="center", va="center", fontsize=6, color=MUTED)
        ax.text(width / 2, -1.15, "class", ha="center", va="center", fontsize=6, color=INK2)
        ax.text(-1.35, N_SIG / 2, "signer", ha="center", va="center", rotation=90, fontsize=6, color=INK2)
    fig.legend(handles=[Patch(facecolor=TEST, label="test"), Patch(facecolor=TRAIN, label="train")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.2)
    fig.subplots_adjust(wspace=0.04, bottom=0.12, top=0.88)
    save(fig, "fig_protocols")


# ---------- Fig. 7: affix confusion matrix ----------

def fig_confusion_affix(M, name="fig_confusion_affix", out=OUT):
    """M: row-normalized confusion matrix (%) in AFFIX order, true x predicted, e.g. from final_confusion()."""
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(COL, COL * 1.02))
    n = len(AFFIX)
    ax.pcolormesh(np.arange(n + 1) - 0.5, np.arange(n + 1) - 0.5, np.sqrt(M), cmap=cmap, vmin=0, vmax=10,
                  edgecolors="white", linewidth=0.4)            # square root: small confusions stay visible; white cell borders
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    for i in range(n):
        for j in range(n):
            if i == j or M[i, j] >= 1:                              # numbers from 1%
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7,   # at least 7 pt
                        color="white" if M[i, j] >= 30 else INK)
    ticks = [DISPLAY[c] for c in AFFIX]
    ax.set_xticks(range(n), ticks, rotation=90)
    ax.set_yticks(range(n), ticks)
    ax.tick_params(length=0, labelsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for b in (6.5, 14.5):                                         # borders prefix | suffix | particle
        ax.axhline(b, color=INK2, linewidth=0.5)
        ax.axvline(b, color=INK2, linewidth=0.5)
    for lo, hi, group in ((0, 6, "prefixes"), (7, 14, "suffixes"), (15, 17, "particles")):   # group names
        ax.text((lo + hi) / 2, -0.9, group, ha="center", va="bottom", fontsize=7, color=INK2, style="italic")
    for side in ax.spines.values():
        side.set_visible(False)
    save(fig, name, out)


# ---------- Fig. 6 and 7: final runs of the combined 114-class model (within-category affix scores) ----------

FINAL_OUT = f"{QC}/figures"                  # final-run figures are written apart from paper/figures
FINAL_MODELS = [("bilstm", "BiLSTM", "o"), ("signbart_am21", "SignBart", "D"), ("dsta_am21", "DSTA-SLR", "s"),
                ("unisign_ft", "Uni-Sign (pose)", "^"), ("shubert_lora", "SHuBERT", "v")]   # order of Table III


def final_signer(root=f"{QC}/final", models=FINAL_MODELS, seeds=(0,), folds_csv=f"{QC}/folds_combined.csv"):
    """-> {model: {signer: [within-category affix accuracy under SI, per seed]}} from the final runs (analyze.load)."""
    sys.path.insert(0, "scripts")
    import analyze
    runs = analyze.load(root, seeds)                                   # the paper uses seed 0 for every model
    f2s = {int(r["cl"]): int(r["subject"]) for r in csv.DictReader(open(folds_csv))}
    acc = defaultdict(lambda: defaultdict(list))
    for m, _, _ in models:
        for (f, _), v in sorted(runs[("combined", "cl", f"{m}|affix")].items()):
            acc[m][f2s[f]].append(100 * v[0])
    return acc


def fig_signer_heat(acc, models=FINAL_MODELS, name="fig_signer_cl_final", out=FINAL_OUT):
    """Fig. 6: SI affix accuracy of each test signer for the five models as an annotated heatmap, columns ordered by the
    mean of the five models, plus the mean and standard deviation over signers."""
    order = sorted(acc["bilstm"], key=lambda s: np.mean([np.mean(acc[m][s]) for m, _, _ in models]))
    A = np.array([[np.mean(acc[m][s]) for s in order] for m, _, _ in models])
    cmap, lo, hi, n = plt.get_cmap("Greens"), 70, 100, len(order)       # the lowest cell in the paper is 71.2
    fig, ax = plt.subplots(figsize=(FULL, 1.3))
    fig.subplots_adjust(left=0.115, right=0.915, top=0.86, bottom=0.27)       # the Mean/Std. columns fit the text width
    ax.imshow(A, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    for i, row in enumerate(A):
        for j, v in enumerate(row):
            r, g, b, _ = cmap((v - lo) / (hi - lo))
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else INK)   # white text on dark cells
        ax.text(n + 0.1, i, f"{row.mean():.1f}", va="center", fontsize=7, fontweight="bold")
        ax.text(n + 1.4, i, f"{row.std(ddof=1):.1f}", va="center", fontsize=7)
    ax.text(n + 0.1, -0.75, "Mean", va="bottom", fontsize=7, fontweight="bold")
    ax.text(n + 1.4, -0.75, "Std.", va="bottom", fontsize=7)                 # not "SD": SD means signer-dependent here
    ax.set_yticks(range(len(models)), [label for _, label, _ in models])
    ax.set_xticks(range(n), [str(s) for s in order])
    ax.set_xlabel("test signer (ordered by five-model mean accuracy)")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    save(fig, name, out)


def final_confusion(root=f"{QC}/final", model="bilstm", seeds=None, folds_csv=f"{QC}/folds_combined.csv"):
    """-> (row-normalized confusion matrix in %, true x predicted, AFFIX order, within-affix argmax; number of runs,
    fold x seed) over all SI runs of the model. Two result formats: one JSON per fold and seed (`test_index`, scores in
    <stem>_scores.npz) and slot2.py (SignBart, DSTA-SLR: one list per fold, `test_idx`, scores in <stem>_fNN_scores.npz)."""
    sys.path.insert(0, "scripts")
    import json, analyze
    rows = list(csv.DictReader(open(folds_csv)))
    y = np.array([int(r["label"]) for r in rows])
    lab = np.array([int(next(r["label"] for r in rows if r["category"] == "affix" and r["class"] == c)) for c in AFFIX])
    srt, order = np.sort(lab), np.argsort(lab)
    C, seen = np.zeros((len(AFFIX), len(AFFIX))), set()
    for p in analyze.result_paths(root):
        r = json.load(open(p))
        if isinstance(r, list):
            items = [(e, f"{p[:-5]}_f{int(e['fold']):02d}_scores.npz", e["model"], e.get("test_idx")) for e in r
                     if isinstance(e, dict) and e.get("cat") == "combined" and e.get("proto") == "cl" and "model" in e
                     and analyze.s4_name(e) == model]   # runs tagged _am7 and _am21 are not mixed
        elif isinstance(r, dict) and r.get("setting") == "combined" and r.get("protocol") == "cl" and model in r.get("models", {}):
            items = [(r, p[:-5] + "_scores.npz", model, r.get("test_index"))]
        else:
            continue
        for e, zp, key, te in items:
            k = (int(e["fold"]), int(e["seed"]))
            if k in seen or te is None or not os.path.exists(zp) or key not in np.load(zp).files or (seeds and k[1] not in seeds):
                continue
            seen.add(k)
            te, sc = np.array(te), np.load(zp)[key]
            sel = np.isin(y[te], lab)
            np.add.at(C, (order[np.searchsorted(srt, y[te][sel])], order[sc[sel][:, srt].argmax(1)]), 1)
    return 100 * C / C.sum(1, keepdims=True), len(seen)


# ---------- Fig. 1: signs frame by frame (the 69 Uni-Sign keypoints) and Table II ----------

KP = f"{DATA}/keypoints" if os.path.isdir(f"{DATA}/keypoints/affix") else f"{DATA}/keypoints_fig"   # _fig: only the clips drawn here
FRAME_SIGNS = [("affix/me/subject-1-affix-me-rep-1", "me- (affix)"), ("number/1/subject-1-number-1-rep-1", "1 (number)"),
               ("alphabet/a/subject-1-alphabet-a-rep-7", "a (alphabet)"),   # take 7: RTMW tracks the resting left hand
               ("kata/saya/subject-1-sentence-1-neutral-rep-1__0", "saya (word)")]   # cleanly (take 1 misses it)
N_FRAMES = 5
FPS = 30                                    # the dataset videos have 30 fps
TABLE_PTS = [(0, "wrist"), (4, "thumb tip"), (8, "index tip"), (12, "middle tip"), (20, "little tip")]   # RTMW order = MediaPipe
TABLE_TEX = r"""% generated by scripts/figures.py (fig_frames) from the RTMW keypoints; do not edit by hand
\begin{table}[!t]
\caption{Raw and Normalized Right-Hand Keypoints of Number 1 (Fig.~\ref{fig:frames}, TIME~s)}
\label{tab:keypoints}
\centering\footnotesize
\begin{tabular}{lcccc}
\hline
 & \multicolumn{2}{c}{RTMW} & \multicolumn{2}{c}{Normalized} \\
Keypoint & $x$ & $y$ & $x$ & $y$ \\
\hline
ROWS
\hline
\end{tabular}
\par\vspace{1mm}\parbox{\linewidth}{\scriptsize RTMW coordinates are divided by the frame width and height. Uni-Sign takes each hand keypoint relative to the wrist and divides it by the longest side of the box around the body keypoints of the clip.}
\end{table}
"""


def load_sign(rel):
    """relpath -> (T, 1662). A word clip is the [awal, akhir) cut (start, end) of its sentence video in kata_clips.csv."""
    if not rel.startswith("kata/"):
        return np.load(f"{KP}/{rel}.npy")
    r = next(r for r in csv.DictReader(open(f"{QC}/kata_clips.csv")) if r["relpath"] == rel)
    return np.load(f"{DATA}/kata/keypoints/{r['video'][:-4]}.npy")[int(r["awal"]):int(r["akhir"])]


def parts(row):
    """One row of 1,662 features -> (pose 33x4, face 468x3, left hand 21x3, right hand 21x3); zero = not detected."""
    return (row[:132].reshape(33, 4), row[132:1536].reshape(468, 3), row[1536:1599].reshape(21, 3),
            row[1599:1662].reshape(21, 3))


def frame_ids(arr, n=N_FRAMES):
    """n frames evenly spaced from the first to the last frame with a hand (blips cleaned as in trim.span, without
    padding, so every panel shows a hand)."""
    from trim import clean, hand_mask
    idx = np.flatnonzero(clean(hand_mask(arr)))
    return np.linspace(idx[0], idx[-1], n).round().astype(int)


def draw_hand(ax, hand, col, mk, lw=0.8, size=3):
    import mediapipe as mp
    for i, j in mp.solutions.holistic.HAND_CONNECTIONS:
        ax.plot(hand[[i, j], 0], hand[[i, j], 1], color=col, lw=lw)
    ax.scatter(hand[:, 0], hand[:, 1], s=size, color=col, marker=mk, zorder=3, linewidths=0)


# RTMW (COCO-WholeBody, 133 keypoints): upper-body and hand connections (21-point hands, same topology as MediaPipe)
RT_BODY = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)]
RT_HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
           (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def frame_axis(ax):
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GRID); side.set_linewidth(0.5)


def rtmw_clip(rel):
    """relpath -> (kp (T,133,2) divided by (W,H), sc (T,133)) from data/pilot/rtmw_<category>.npz. A word clip is the
    [awal, akhir) cut of its sentence video in rtmw_neutral.npz given by kata_clips.csv, as in load_sign."""
    if rel.startswith("kata/"):
        r = next(r for r in csv.DictReader(open(f"{QC}/kata_clips.csv")) if r["relpath"] == rel)
        d, a, b = np.load(f"{DATA}/pilot/rtmw_neutral.npz"), int(r["awal"]), int(r["akhir"])
        i = next(i for i, q in enumerate(d["relpath"]) if q.endswith("/" + os.path.basename(r["video"])[:-4]))
    else:
        d = np.load(f"{DATA}/pilot/rtmw_{rel.split('/')[0]}.npz")
        i, a, b = list(d["relpath"]).index(rel), 0, None
    s = slice(d["off"][i], d["off"][i + 1])
    return d["kp"][s][a:b].astype(float), d["sc"][s][a:b].astype(float)


def unisign_scale(kp, sc):
    """Uni-Sign's normalization scale of one clip (crop_scale in ext/Uni-Sign/datasets.py, checked in unisign_features.demo):
    the longest side of the box around the 9 body keypoints scored > 0.3 in the frames read at test time (all if <= 64,
    else 64 evenly spaced). load_part_kp then shifts each hand to its wrist, divides it by this scale and clips to [-1, 1]."""
    i = np.arange(len(kp)) if len(kp) <= 64 else np.linspace(0, len(kp) - 1, 64).round().astype(int)
    body = [0, *range(3, 11)]
    xy = kp[i][:, body][sc[i][:, body] > 0.3]
    return (xy.max(0) - xy.min(0)).max()


def fig_frames(table=f"{PAPER}/table_keypoints.tex"):
    """Grid of signs x frames across the page; the coordinate table (Table II) is written to `table` (LaTeX)."""
    fig = plt.figure(figsize=(FULL, 4.6))
    grid = fig.add_gridspec(len(FRAME_SIGNS), N_FRAMES, left=0.035, right=0.995, top=0.9, bottom=0.005, wspace=0.03,
                            hspace=0.05)
    for r, (rel, name) in enumerate(FRAME_SIGNS):
        kp, sc = rtmw_clip(rel)
        arr = load_sign(rel)
        assert len(kp) == len(arr), rel                                        # RTMW frames = MediaPipe frames
        ids = frame_ids(arr, N_FRAMES + 2)[1:-1]     # without the two ends of the span: RTMW often swaps resting hands
        use = np.isin(np.arange(133), UNISIGN_PTS)
        xy = np.concatenate([kp[t][use & (sc[t] > 0.3)] for t in ids])           # one window per row: box of all points + 6%
        c, size = (xy.min(0) + xy.max(0)) / 2, (xy.max(0) - xy.min(0)) * 1.12
        w = max(size[0], size[1] * 1.32)
        for col, t in enumerate(ids):
            ax = fig.add_subplot(grid[r, col])
            draw_unisign(ax, kp[t], [], sc[t] > 0.3)
            ax.set_xlim(c[0] - w / 2, c[0] + w / 2)
            ax.set_ylim(c[1] + w / 2.64, c[1] - w / 2.64)                      # width:height 1.32; image y points down
            frame_axis(ax)
            if col == 0:
                ax.set_ylabel(name, fontsize=7)
            ax.text(0.97, 0.04, f"{t / FPS:.2f} s", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6,
                    color=INK2, zorder=6, bbox=dict(fc="#ffffff", ec="none", pad=0.4, alpha=0.85))   # time from clip start
        if r == 1:                         # table: right hand of number 1 in the middle frame; only the index finger is up
            t, scale = ids[N_FRAMES // 2], unisign_scale(kp, sc)
            t_tab = t / FPS                                            # the caption of Table II gives the panel time
            rh = kp[t, 112:133]
            assert (sc[t, [112 + i for i, _ in TABLE_PTS]] > 0.3).all()
            rows = [f"{nm.capitalize()} & {rh[i, 0]:.3f} & {rh[i, 1]:.3f} & "
                    + " & ".join(f"${v:+.2f}$" for v in np.clip((rh[i] - rh[0]) / scale, -1, 1)) + r" \\"
                    for i, nm in TABLE_PTS]
    fig.legend(handles=[plt.Line2D([], [], color=SERIES[0], marker="o", ms=3, lw=0.8, label="right hand"),
                        plt.Line2D([], [], color=SERIES[1], marker="s", ms=3, lw=0.8, label="left hand"),
                        plt.Line2D([], [], color=INK, marker="o", ms=3, lw=0.8, label="body and face")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.6)
    over = fig.add_axes([0, 0, 1, 1], frameon=False)                  # one time arrow above the grid, as in Uni-Sign
    over.set_xlim(0, 1); over.set_ylim(0, 1); over.axis("off")
    over.annotate("", (0.995, 0.912), (0.035, 0.912), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, mutation_scale=7))
    over.text(0.985, 0.922, "time", ha="right", va="bottom", fontsize=6.5, color=INK2)
    save(fig, "fig_frames")
    with open(table, "w") as fh:
        fh.write(TABLE_TEX.replace("ROWS", "\n".join(rows)).replace("TIME", f"{t_tab:.2f}"))


# ---------- Fig. 2: the input of each model on one frame ----------

INPUT_SIGN, INPUT_T = "affix/me/subject-1-affix-me-rep-1", 36          # middle frame with both hands and the face
KAMUS_SIGN, KAMUS_T = "imbuhan__Awalan-Me", 66        # me- from the SIBI dictionary for panel (c): dataset pixels may not
                                                     # be published, dictionary frames may with their source
UNISIGN_PTS = [0, *range(3, 11), *range(91, 133), *range(23, 40, 2), *range(83, 91), 53]   # datasets.py load_part_kp
SHUBERT_POSE = [0, 11, 12, 13, 14, 15, 16]          # body_features: nose, shoulders, elbows, wrists (14 values)
SIGNBART_ARM = [11, 12, 13, 14, 15, 16]              # SignBart (mp48): shoulders, elbows, wrists + the 42 hand points
DSTA_PTS = [0, *range(5, 11), *[b + k for b in (91, 112) for k in (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)]]  # slot2.SIGN27
SHUBERT_FACE = ([69, 168, 156, 118, 54], [168, 299, 347, 336, 301], [164, 212, 432, 18])   # crop_face: both eyes, mouth


def shubert_crops(img, face, lh, rh):
    """Three 224 x 224 SHuBERT crops from an RGB frame and normalized landmarks, copying crop_face.cues_on_grey_background
    (boxes of the left eye, right eye and mouth on grey 128, 10 px padding, made square) and crop_hands.get_bounding_box +
    adjust_bounding_box (square box x 1.5) of the Space ShesterG/TTIC-SHuBERT-ASLVideo-to-EnglishText.
    -> (crops {face, right, left}, pixel boxes {face: (x, y, side), inner: [(x0, y0, x1, y1)] x 3, right/left: (x, y, side)})."""
    import cv2
    H, W = img.shape[:2]
    inner = [(int(face[i, 0].min() * W), int(face[i, 1].min() * H), int(face[i, 0].max() * W), int(face[i, 1].max() * H))
             for i in map(list, SHUBERT_FACE)]
    x0, y0 = max(0, min(b[0] for b in inner) - 10), max(0, min(b[1] for b in inner) - 10)
    x1, y1 = min(W, max(b[2] for b in inner) + 10), min(H, max(b[3] for b in inner) + 10)
    side = max(x1 - x0, y1 - y0)
    x0 = max(0, x0 - (side - (x1 - x0)) // 2) if x1 - x0 < side else x0
    y0 = max(0, y0 - (side - (y1 - y0)) // 2) if y1 - y0 < side else y0
    g = np.full((side, side, 3), 128, np.uint8)
    for a, b, c, d in inner:
        crop = img[b:d, a:c]
        g[b - y0:b - y0 + crop.shape[0], a - x0:a - x0 + crop.shape[1]] = crop
    crops, boxes = {"face": cv2.resize(g, (224, 224), interpolation=cv2.INTER_AREA)}, {"face": (x0, y0, side), "inner": inner}
    for name, hand in (("right", rh), ("left", lh)):
        px = np.array([(int(x * W), int(y * H)) for x, y in hand[:, :2]], dtype=np.int32)
        cx, cy = np.mean(px, axis=0, dtype=int)
        _, _, wb, hb = cv2.boundingRect(px)
        n = max(wb, hb)
        pad = int(0.5 * n / 2)
        x, y, n = cx - n // 2 - pad, cy - n // 2 - pad, n + 2 * pad
        x, y = max(W - n if x + n > W else x, 0), max(H - n if y + n > H else y, 0)
        crops[name] = cv2.resize(img[y:y + n, x:x + n], (224, 224), interpolation=cv2.INTER_AREA)
        boxes[name] = (x, y, n)
    return crops, boxes


def kamus_frame():
    """-> (RGB frame, pose, face, lh, rh) of frame KAMUS_T of the dictionary video; landmarks from keypoints/ (same MediaPipe)."""
    import cv2
    arr = np.load(os.path.join(DICT, "keypoints", f"{KAMUS_SIGN}.npy"))
    cap = cv2.VideoCapture(os.path.join(DICT, "video", f"{KAMUS_SIGN}.webm"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, KAMUS_T)
    ok, bgr = cap.read()
    assert ok and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) in (len(arr), 0), KAMUS_SIGN
    return (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), *parts(arr[KAMUS_T]))


def fig_inputs():
    import mediapipe as mp
    hol = mp.solutions.holistic
    arr = load_sign(INPUT_SIGN)
    pose, face, lh, rh = parts(arr[INPUT_T])
    d = np.load(f"{DATA}/pilot/rtmw_{INPUT_SIGN.split('/')[0]}.npz")
    i = list(d["relpath"]).index(INPUT_SIGN)
    assert d["off"][i + 1] - d["off"][i] == len(arr), INPUT_SIGN                   # RTMW frames = MediaPipe frames
    kp, sc = d["kp"][d["off"][i] + INPUT_T].astype(float), d["sc"][d["off"][i] + INPUT_T].astype(float)
    # one tight window for (a) and (b): box around every drawn point except the hips and below, + 6%
    up = np.vstack([face[face[:, :2].any(1), :2], pose[:17][pose[:17, :2].any(1), :2], lh[lh.any(1), :2], rh[rh.any(1), :2],
                    kp[(sc > 0.3) & ~np.isin(np.arange(133), range(11, 23))]])
    mid, size = (up.min(0) + up.max(0)) / 2, (up.max(0) - up.min(0)) * 1.12
    w = max(size[0], size[1] * 0.8)                                        # panel width:height 0.8
    fig = plt.figure(figsize=(FULL, 3.45))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.42], left=0.01, right=0.99, top=0.79, bottom=0.17, wspace=0.04,
                          hspace=0.28)
    axs = [fig.add_subplot(gs[:, i]) for i in range(3)]
    off = dict(color=AXIS, lw=0.5)
    dot = lambda ax, xy, col, s=2.5, mk="o": ax.scatter(xy[:, 0], xy[:, 1], s=s, color=col, marker=mk, zorder=3, linewidths=0)
    pose_lines = lambda ax, kw: [ax.plot(pose[[a, b], 0], pose[[a, b], 1], **kw) for a, b in hol.POSE_CONNECTIONS
                                 if a <= 24 and b <= 24 and pose[a, :2].any() and pose[b, :2].any()]

    ax = axs[0]                                         # MediaPipe Holistic: 33 + 468 + 42; the models trained from scratch
    pose_lines(ax, off)                                 # use the hands
    dot(ax, pose[:25, :2], AXIS, 3)
    dot(ax, face[:, :2], AXIS, 0.4)
    for a, b in hol.POSE_CONNECTIONS:                  # SignBart's arms
        if a in SIGNBART_ARM and b in SIGNBART_ARM and pose[a, :2].any() and pose[b, :2].any():
            ax.plot(pose[[a, b], 0], pose[[a, b], 1], color=INK2, lw=0.9)
    dot(ax, pose[SIGNBART_ARM, :2], INK, 3)
    for hand, col, mk in ((rh, SERIES[0], "o"), (lh, SERIES[1], "s")):
        draw_hand(ax, hand, col, mk)
    ax.set_title("(a) MediaPipe Holistic, 543 points (x, y)\nBiLSTM: 42 hand points = 84 values\n"     # values per frame
                 "SignBart: + 6 arm points = 96 values", fontsize=7, pad=3)

    ax = axs[1]                                         # RTMW 133 keypoints; Uni-Sign uses 69
    use = np.zeros(133, bool)
    use[UNISIGN_PTS] = True
    vis = sc > 0.3
    for a, b in RT_BODY:
        if vis[a] and vis[b]:
            ax.plot(kp[[a, b], 0], kp[[a, b], 1], color=INK2 if use[a] and use[b] else AXIS, lw=0.9 if use[a] and use[b] else 0.5)
    for base, col, mk in ((112, SERIES[0], "o"), (91, SERIES[1], "s")):
        for a, b in RT_HAND:
            if vis[base + a] and vis[base + b]:
                ax.plot(kp[[base + a, base + b], 0], kp[[base + a, base + b], 1], color=col, lw=0.8)
        dot(ax, kp[[base + j for j in range(21) if vis[base + j]]], col, 3, mk)
    rest = vis & use & ~np.isin(np.arange(133), range(91, 133))
    dot(ax, kp[vis & ~use], AXIS, 2.5)
    dot(ax, kp[rest], INK, 3)
    ring = np.zeros(133, bool)
    ring[DSTA_PTS] = True
    ax.scatter(kp[ring & vis, 0], kp[ring & vis, 1], s=16, facecolors="none", edgecolors=INK, linewidths=0.5, zorder=4)
    ax.set_title(f"(b) RTMW, 133 points (x, y, score)\nUni-Sign: {len(UNISIGN_PTS)} points = {3 * len(UNISIGN_PTS)} values\n"
                 f"DSTA-SLR: {len(DSTA_PTS)} ringed = {3 * len(DSTA_PTS)} values", fontsize=7, pad=3)

    for ax in axs[:2]:
        ax.set_xlim(mid[0] - w / 2, mid[0] + w / 2)
        ax.set_ylim(mid[1] + w / 1.6, mid[1] - w / 1.6)                     # image y points down
        frame_axis(ax)

    ax = axs[2]                                         # SHuBERT: dictionary frame (not the dataset) + boxes + 7 pose points
    img, kpose, kface, klh, krh = kamus_frame()
    H, W = img.shape[:2]
    crops, boxes = shubert_crops(img, kface, klh, krh)
    ax.imshow(img)
    sh = kpose[[11, 12], :2] * [W, H]
    kx, ky = sh.mean(0)
    kh = 1.15 * np.linalg.norm(sh[0] - sh[1])
    pp = kpose[SHUBERT_POSE, :2] * [W, H]
    ax.scatter(pp[:, 0], pp[:, 1], s=9, color="#ffffff", edgecolors=INK, linewidths=0.6, zorder=3)
    for name, col, ls in (("right", SERIES[0], "-"), ("left", SERIES[1], "--"), ("face", "#ffffff", "-")):   # left dashed:
        x, y, n = boxes[name]                                                                       # black and white
        ax.add_patch(Rectangle((x, y), n, n, fill=False, edgecolor=col, lw=1.1, ls=ls))
    for a, b, c, d in boxes["inner"]:
        ax.add_patch(Rectangle((a, b), c - a, d - b, fill=False, edgecolor=SERIES[2], lw=0.7))
    top = max(ky - 1.45 * kh, 0)
    bot = min(top + 2.5 * kh, H)                                          # window as tall as the frame if the frame is shorter
    ax.set_xlim(kx - (bot - top) / 2.5, kx + (bot - top) / 2.5)
    ax.set_ylim(bot, top)
    frame_axis(ax)
    ax.set_title("(c) Image crops + body pose\nSHuBERT: 3 crops × 384 + 7 points × 2\n= 1,166 values", fontsize=7, pad=3)
    for r, (name, label, col) in enumerate((("face", "face: eyes, mouth", INK), ("right", "right hand", SERIES[0]),
                                           ("left", "left hand", SERIES[1]))):
        a = fig.add_subplot(gs[r, 3])
        a.imshow(crops[name])
        a.set_xticks([]); a.set_yticks([])
        for side in a.spines.values():
            side.set_color(col); side.set_linewidth(1.1); side.set_linestyle("--" if name == "left" else "-")
        a.set_title(label, fontsize=6, pad=1)
    pc, pk = axs[2].get_position(), a.get_position()                     # crop pipeline under panel (c)
    fig.text((pc.x0 + pk.x1) / 2, pc.y0 - 0.015, "MediaPipe landmarks → crop rule: square box, 1.5 × landmark spread\n"
             "→ 224 × 224 pixels → DINOv2 → 384 values per crop", ha="center", va="top", fontsize=6.5, color=INK2)
    fig.legend(handles=[plt.Line2D([], [], color=SERIES[0], marker="o", ms=3, lw=0.8, label="right hand"),
                        plt.Line2D([], [], color=SERIES[1], marker="s", ms=3, lw=0.8, label="left hand"),
                        plt.Line2D([], [], color=INK, marker="o", ms=3, lw=0, label="other points used"),
                        plt.Line2D([], [], color=AXIS, marker="o", ms=3, lw=0.5, label="detected, not used"),
                        plt.Line2D([], [], color=INK, marker="o", ms=4, mfc="none", mew=0.5, lw=0, label="DSTA-SLR")],
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.6, fontsize=6.5)
    save(fig, "fig_inputs")


# ---------- Fig. 5: Uni-Sign ablation variants ----------

U_FACE = [*range(23, 40, 2), *range(83, 91), 53]
U_BODY = [0, *range(3, 11)]
THUMBS = [92, 93, 94, 95, 113, 114, 115, 116]           # keypoints 1-4 of the left (91-111) and right (112-132) thumb
ABLATIONS = [("Full input", []), ("Hands only", [0, *range(3, 9), *U_FACE]), ("Thumbs masked", THUMBS),   # wrists 9 and 10
             ("No face landmarks", U_FACE)]                                                             # are kept


def draw_unisign(ax, kp, drop, vis=None):
    """The 69 Uni-Sign keypoints; keypoints in `drop` are grey. Lines: upper body, jaw, inner lips, hands. Keypoints
    outside `vis` (score <= 0.3, zeroed by Uni-Sign) are not drawn."""
    keep = np.zeros(133, bool)
    keep[UNISIGN_PTS] = True
    keep[drop] = False
    vis = np.ones(133, bool) if vis is None else vis
    col = lambda i, c: c if keep[i] else AXIS
    lines = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)] + list(zip(range(23, 39, 2), range(25, 40, 2))) + \
        list(zip(range(83, 90), range(84, 91))) + [(90, 83)]
    for a, b in lines:
        if vis[a] and vis[b]:
            ax.plot(kp[[a, b], 0], kp[[a, b], 1], color=INK2 if keep[a] and keep[b] else AXIS,
                    lw=0.8 if keep[a] and keep[b] else 0.5)
    for base, c, mk in ((112, SERIES[0], "o"), (91, SERIES[1], "s")):
        for a, b in RT_HAND:
            if vis[base + a] and vis[base + b]:
                ax.plot(kp[[base + a, base + b], 0], kp[[base + a, base + b], 1],
                        lw=0.8 if keep[base + a] and keep[base + b] else 0.5, color=c if keep[base + a] and keep[base + b] else AXIS)
        for j in range(base, base + 21):
            if vis[j] and keep[j]:
                ax.scatter(*kp[j], s=3, color=c, marker=mk, zorder=4, linewidths=0)
    for j in U_BODY + U_FACE:
        if vis[j] and keep[j]:
            ax.scatter(*kp[j], s=3, color=INK, zorder=4, linewidths=0)
    gone = [j for j in UNISIGN_PTS if vis[j] and not keep[j]]       # hidden keypoints: open circles on top, so the
    if gone:                                                        # thumbs stay visible in black and white
        ax.scatter(kp[gone, 0], kp[gone, 1], s=7, facecolors="#ffffff", edgecolors=INK2, linewidths=0.5, zorder=6)
    return int(keep.sum())


def fig_ablation_unisign():
    d = np.load(f"{DATA}/pilot/rtmw_{INPUT_SIGN.split('/')[0]}.npz")
    i = list(d["relpath"]).index(INPUT_SIGN)
    kp = d["kp"][d["off"][i] + INPUT_T].astype(float)
    sc = d["sc"][d["off"][i] + INPUT_T].astype(float)
    assert (sc[UNISIGN_PTS] > 0.3).all(), "all 69 keypoints must be detected in the example frame"
    body = kp[[5, 6]]
    cx, cy = body.mean(0)
    half = 1.15 * np.linalg.norm(body[0] - body[1])
    fig, axs = plt.subplots(1, len(ABLATIONS), figsize=(FULL, 1.95), gridspec_kw=dict(left=0.01, right=0.99, top=0.9,
                                                                                     bottom=0.1, wspace=0.04))
    for ax, (name, drop) in zip(axs, ABLATIONS):
        n = draw_unisign(ax, kp, drop)
        ax.set_title(f"{name} ({n} points)", fontsize=7, pad=3)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy + 0.8 * half, cy - 0.75 * half)                          # without the hips and the top of the head
        frame_axis(ax)
    fig.legend(handles=[plt.Line2D([], [], color=SERIES[0], marker="o", ms=3, lw=0.8, label="right hand"),
                        plt.Line2D([], [], color=SERIES[1], marker="s", ms=3, lw=0.8, label="left hand"),
                        plt.Line2D([], [], color=INK, marker="o", ms=3, lw=0.8, label="body and face"),
                        plt.Line2D([], [], color=AXIS, marker="o", ms=3.6, mfc="#ffffff", mec=INK2, mew=0.5, lw=0.5,
                                   label="removed in this variant")],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.6, fontsize=6.5)
    save(fig, "fig_ablation_unisign")


# ---------- Test-time occlusion of the full Uni-Sign on one signer (explain_unisign_run.py) ----------

E22 = f"{QC}/results/e22_report.json"
GROUP_LABEL = {"right hand, no thumb": "Right hand (no thumb)", "thumbs": "Thumbs", "body": "Body", "wrists": "Wrists",
               "face": "Face", "left hand, no thumb": "Left hand (no thumb)"}       # order = Shapley value, Overall macro-F1
E22_CATS = [("affix", "Affix", SERIES[0], "o"), ("number", "Number", SERIES[1], "s"), ("alphabet", "Alphabet", SERIES[2], "^"),
            ("kata", "Word", MUTED, "D"), ("ti+wati", "-ti/-wati", INK, "v")]


def fig_shapley(path=E22):
    """(a) Shapley values of the six groups on Overall 114-class macro-F1; (b) per-clip Shapley values on the
    log-probability of the true class, per category, with 95% bootstrap CI over clips. Values per keypoint are not
    drawn: the anchor keypoints (53, 91, 112) are already (0, 0) after loading, so occluding them only zeroes the score."""
    rep = json.load(open(path))
    f1, clip = rep["agregat"]["macro_f1"], rep["per_klip"]
    groups = list(GROUP_LABEL)
    y = np.arange(len(groups))[::-1]
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL, 3.9), gridspec_kw=dict(height_ratios=[1, 1.9], hspace=0.6))
    v = [f1["shapley"][g] for g in groups]
    a.barh(y, v, color=TEST, height=0.62)
    for yi, vi in zip(y, v):
        a.text(vi + 0.6, yi, f"{vi:.1f}", va="center", fontsize=6.5, color=INK2)
    a.set_xlim(0, 35)
    a.set_xlabel("Contribution to 114-class macro-F1 (points)")
    a.set_title(f"(a) All {clip['overall']['n']:,} clips: {f1['full']:.1f} with all keypoints, {f1['none']:.1f} with none",
                loc="left")
    off = np.linspace(0.3, -0.3, len(E22_CATS))
    for k, (c, label, col, mk) in enumerate(E22_CATS):
        m = np.array([clip[c]["shapley"][g] for g in groups])
        b.errorbar(m[:, 0], y + off[k], xerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt=mk, ms=3, color=col, lw=0.7,
                   capsize=0, label=f"{label} ({clip[c]['n']})")
    b.axvline(0, color=AXIS, lw=0.6)
    b.set_xlabel("Contribution to log-probability of the correct class, per clip")
    b.set_title("(b) By category (clips), 95% bootstrap CI over clips", loc="left")
    b.legend(loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.0)
    for ax in (a, b):
        ax.set_yticks(y, [GROUP_LABEL[g] for g in groups])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    save(fig, "fig_shapley")


# ---------- Fig. 3: Uni-Sign as fine-tuned here ----------

def fig_unisign():
    """Uni-Sign as fine-tuned here. Style after Fig. 5 of the Uni-Sign paper (flat pastel blocks without borders, a
    skeleton icon per body part, wide bars for shared stages, tensors as small boxes in dashed boxes, italic process
    labels), with numbered panels per stage under sans-serif titles, layers as 3D slabs, arrow colors per stage, a dashed
    training box and a legend below. Content follows ext/Uni-Sign/models.py:72-100, 206-295 and scripts/unisign_finetune.py:
    the hands and the face receive the wrist and nose features of the body branch without gradient, part embeddings are
    added, and both hands share weights; the RGB branch and the text decoder are not drawn."""
    from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle as R
    W, H, Y0 = 181, 88, -4.5                                        # Y0 < 0: room for a two-row legend
    HT = H - Y0
    SANS = "Arial"
    BLUE, GREEN, GREY = "#256abf", "#1baf7a", "#52514e"
    HAND = "#e4ddf4"                                                # both hands one color: shared weights
    fig = plt.figure(figsize=(FULL, FULL * HT / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(Y0, H); ax.axis("off")
    inset = lambda x, y, w, h: fig.add_axes([x / W, (y - Y0) / HT, w / W, h / HT])

    def rbox(x0, y0, x1, y1, fill="none", edge="none", lw=0.7, ls="-", r=1.2, z=1):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fill, ec=edge,
                                    lw=lw, ls=ls, zorder=z))

    def panel(x0, y0, x1, y1, num, title, col, fill, fs=7.2, sub=None):   # stage number in a circle
        rbox(x0, y0, x1, y1, fill, col, 1.0)
        if sub:
            ax.text((x0 + x1) / 2, y1 - 6.2, sub, ha="center", va="center", fontsize=6, color=col, family=SANS)
        t = ax.text((x0 + x1) / 2 + 1.6, y1 - 3.3, title, ha="center", va="center", fontsize=fs, fontweight="bold",
                    color=col, family=SANS)
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transData.inverted())
        ax.add_patch(Circle((bb.x0 - 2.0, y1 - 3.3), 1.45, fc=col, ec="none", zorder=3))
        ax.text(bb.x0 - 2.0, y1 - 3.35, str(num), ha="center", va="center", fontsize=6, fontweight="bold", color="#ffffff",
                family=SANS, zorder=4)

    def block(xc, yc, w, h, text, fill, fs=6.2):                    # flat pastel block without border (Uni-Sign style)
        ax.add_patch(R((xc - w / 2, yc - h / 2), w, h, fc=fill, ec="none", zorder=2))
        ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, color=INK, zorder=3)

    def arrow(x0, y0, x1, y1, col=GREY, label=None, dx=0.8):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, shrinkA=0, shrinkB=0,
                                                            mutation_scale=7), zorder=4)
        if label:
            ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2, label, fontsize=6, color=col, va="center")

    def tokens(x, y, n, fill, edge, vertical=False, group=True, hatch=None):   # tensor: small boxes in a dashed box
        step = 2.9
        for k in range(n):
            ax.add_patch(R((x, y + step * k) if vertical else (x + step * k, y), 2.4, 2.4, fc=fill, ec=edge, lw=0.5,
                           hatch=hatch, zorder=3))
        if group:
            w, h = (2.4, step * (n - 1) + 2.4) if vertical else (step * (n - 1) + 2.4, 2.4)
            rbox(x - 0.9, y - 0.9, x + w + 0.9, y + h + 0.9, edge="#8fb3e3", lw=0.5, ls="--", r=0.9, z=2)

    def slabs(x0, y0, n, w, h, dx, dy, gap):                        # layers as 3D slabs
        for k in range(n):
            x = x0 + gap * k
            ax.add_patch(Polygon([(x, y0 + h), (x + dx, y0 + h + dy), (x + w + dx, y0 + h + dy), (x + w, y0 + h)],
                                 fc="#bfe6d4", ec=GREEN, lw=0.4, zorder=3))
            ax.add_patch(Polygon([(x + w, y0), (x + w + dx, y0 + dy), (x + w + dx, y0 + h + dy), (x + w, y0 + h)],
                                 fc="#9fd8bf", ec=GREEN, lw=0.4, zorder=3))
            ax.add_patch(R((x, y0), w, h, fc="#e3f5ec", ec=GREEN, lw=0.4, zorder=3))

    d = np.load(f"{DATA}/pilot/rtmw_{INPUT_SIGN.split('/')[0]}.npz")
    i0 = d["off"][list(d["relpath"]).index(INPUT_SIGN)]
    frame = lambda t: d["kp"][i0 + t].astype(float)
    kp = frame(INPUT_T)
    BODY = [0, *range(3, 11)]
    PARTS = [("body", 9, BODY, "#e9e8e4"), ("left hand", 21, list(range(91, 112)), HAND),
             ("right hand", 21, list(range(112, 133)), HAND), ("face", 18, U_FACE, "#f5ecc9")]

    def part_icon(a, idx):
        """One body part from the example frame, colored as in Fig. 1, 2 and 5."""
        if idx[0] >= 91:
            col = SERIES[1] if idx[0] == 91 else SERIES[0]
            for p, q in RT_HAND:
                a.plot(kp[[idx[p], idx[q]], 0], kp[[idx[p], idx[q]], 1], color=col, lw=0.8)
        elif idx is BODY:
            col = INK
            for p, q in ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10)):
                a.plot(kp[[p, q], 0], kp[[p, q], 1], color=INK2, lw=0.8)
        else:
            col = INK
            jaw, lips = [k for k in idx if k < 40], [k for k in idx if 83 <= k <= 90]
            a.plot(kp[jaw, 0], kp[jaw, 1], color=INK2, lw=0.6)
            a.plot(kp[lips + lips[:1], 0], kp[lips + lips[:1], 1], color=INK2, lw=0.6)
        a.scatter(kp[idx, 0], kp[idx, 1], s=3, color=col, zorder=3, linewidths=0, marker="s" if idx[0] == 91 else "o")
        lo, hi = kp[idx].min(0), kp[idx].max(0)
        c, sz = (lo + hi) / 2, (hi - lo).max() * 0.62
        a.set_xlim(c[0] - sz, c[0] + sz); a.set_ylim(c[1] + sz, c[1] - sz); a.set_aspect("equal"); a.axis("off")

    def cards(frames, x0, y0, w, h, step):                          # stack of cards (video frames or skeletons)
        out = []
        for k, f in enumerate(frames):
            a = inset(x0 + step[0] * k, y0 + step[1] * (len(frames) - 1 - k), w, h)
            a.set_xticks([]); a.set_yticks([])
            for sd in a.spines.values():
                sd.set_color(AXIS); sd.set_linewidth(0.6)
            out.append(a)
        return out

    with plt.rc_context({"mathtext.fontset": "stix"}):
        # training box
        rbox(35, 75.5, 180, 87, "#ffffff", MUTED, 0.7, "--")
        ax.text(37, 84.6, "Fine-tuning", fontsize=6.6, fontweight="bold", color=INK2, va="center", family=SANS)
        ax.text(37, 81, "All modules are trained end to end except the mT5 token embeddings. AdamW, lr $3\\times10^{-4}$, "
                "cosine schedule, 20 epochs, batch 32, label smoothing 0.2.", fontsize=6, color=INK2, va="center")
        ax.text(37, 77.6, "Up to 64 frames per clip: sampled at random in training, evenly spaced in testing.", fontsize=6,
                color=INK2, va="center")

        # (1) input: SIBI dictionary video frames (not the dataset) -> RTMW -> stack of skeleton frames of signer 1
        panel(1, 12, 26, 73, 1, "Input", GREY, "#ffffff")
        import cv2
        cap = cv2.VideoCapture(os.path.join(DICT, "video", f"{KAMUS_SIGN}.webm"))
        kpose = parts(np.load(os.path.join(DICT, "keypoints", f"{KAMUS_SIGN}.npy"))[KAMUS_T])[0]
        for a, t in zip(cards(range(3), 4.5, 51.7, 9.5, 11.5, (1.8, 1.8)), (KAMUS_T - 16, KAMUS_T - 8, KAMUS_T)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, bgr = cap.read()
            assert ok, (KAMUS_SIGN, t)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            Hh, Ww = img.shape[:2]
            sh = kpose[[11, 12], :2] * [Ww, Hh]                         # window as in Fig. 2c, from the shoulders in KAMUS_T
            kx, ky = sh.mean(0)
            kh = 1.15 * np.linalg.norm(sh[0] - sh[1])
            top = int(max(ky - 1.45 * kh, 0)); bot = int(min(top + 2.5 * kh, Hh))
            a.imshow(img[top:bot, max(int(kx - (bot - top) / 2.5), 0):int(kx + (bot - top) / 2.5)])
        cap.release()
        arrow(18.0, 53.0, 21.6, 50.8, col=MUTED)
        ax.text(21.9, 48.9, "$t$", fontsize=6.3, color=INK2)
        ax.text(12.4, 48.0, "video frames", ha="center", fontsize=6, style="italic", family=SANS, color=INK2)
        rbox(3.4, 42.4, 23.6, 46.2, "#ecebe7", "none", r=1.0, z=2)
        ax.text(13.5, 44.3, "RTMW pose estimator", ha="center", va="center", fontsize=5.9, zorder=3)
        arrow(13.5, 42.4, 13.5, 40.6)
        for a, t in zip(cards(range(3), 4.5, 21.6, 12, 14.2, (2.0, 2.3)), (INPUT_T - 10, INPUT_T - 5, INPUT_T)):
            f = frame(t)
            draw_unisign(a, f, [])
            cx, cy = f[[5, 6]].mean(0)
            half = 1.15 * np.linalg.norm(f[5] - f[6])
            a.set_xlim(cx - half, cx + half); a.set_ylim(cy + 0.8 * half, cy - 0.95 * half); a.set_aspect("equal")
            a.set_facecolor("#ffffff")
        arrow(20.8, 23.0, 23.8, 21.0, col=MUTED)
        ax.text(24.2, 21.4, "$T$", fontsize=6.5, color=INK2)
        ax.text(12.2, 19.6, "keypoint sequence", ha="center", fontsize=6, style="italic", family=SANS, color=INK2)
        ax.text(13.5, 16.3, "$x_t\\in\\mathbb{R}^{69\\times3}$, $T\\leq64$", ha="center", fontsize=6.3)
        ax.text(13.5, 13.4, "normalized per part", ha="center", fontsize=6, color=INK2)
        arrow(26, 20.5, 37, 20.5)
        ax.text(30.5, 25.6, "divide\ninto\nparts", ha="center", va="center", fontsize=5.8, style="italic", family=SANS,
                color=INK2, linespacing=1.0)

        # (2) pose encoder per body part (bottom to top, as in Uni-Sign)
        panel(35, 12, 118, 73, 2, "Uni-Sign pose encoder, per body part", "#184f95", "#f7faff")   # named after the source
        xs = (47, 66.5, 86, 105.5)
        for xc, (name, n, idx, fill) in zip(xs, PARTS):
            part_icon(inset(xc - 7, 16, 14, 12.5), idx)
            ax.text(xc, 13.6, f"{name} ({n})", ha="center", fontsize=6, color=INK)
            arrow(xc, 28.8, xc, 30.7, col=BLUE)
            block(xc, 32.8, 15, 4.0, "Linear 3 → 64", fill, 5.9)
            arrow(xc, 34.8, xc, 36.5, col=BLUE)
            block(xc, 39.2, 15, 5.2, "spatial GCN", fill)
            arrow(xc, 41.8, xc, 46.3, col=BLUE)
            block(xc, 49, 15, 5.2, "temporal GCN\n(5 frames)", fill, 5.8)
            arrow(xc, 51.6, xc, 53.3, col=BLUE)
            block(xc, 55.2, 15, 3.6, "mean over points", fill, 5.8)
            arrow(xc, 57, xc, 59.4, col=BLUE, label="256")
        ax.plot([xs[0], xs[3]], [44, 44], color=GREY, lw=0.6, ls=(0, (2.5, 1.6)), zorder=3)   # models.py:240-255
        ax.plot(xs[0], 44, "o", ms=2.2, color=GREY, zorder=4)
        for xc, lab in zip(xs[1:], ("wrist", "wrist", "nose")):
            ax.add_patch(Circle((xc, 44), 1.05, fc="#ffffff", ec=GREY, lw=0.6, zorder=5))
            ax.text(xc, 44.05, "+", ha="center", va="center", fontsize=6, color=GREY, zorder=6)
            ax.text(xc + 1.6, 44, lab, ha="left", va="center", fontsize=5.8, color=GREY, zorder=6,
                    bbox=dict(fc="#f7faff", ec="none", pad=0.3))
        rbox(38, 59.6, 115, 65.6, "#ececea", "none", r=1.8, z=2)        # wide concatenation bar (like Feature Concatenation)
        ax.text(76.5, 62.6, "Concat 4 × 256 + part embedding  →  Linear  →  $e_t\\in\\mathbb{R}^{768}$",
                ha="center", va="center", fontsize=6.3, zorder=3)
        ax.plot([115, 119.5, 119.5], [62.6, 62.6, 42.7], color=GREY, lw=0.8, zorder=4)
        arrow(119.5, 42.7, 122.3, 42.7)

        # (3) mT5 encoder: text prefix from above, pose tokens from the left, layers as 3D slabs, output below
        panel(121, 12, 152, 73, 3, "mT5 encoder", "#127a55", "#f5fbf8", sub="pre-trained; decoder unused")
        tokens(123.4, 37.2, 4, "#e9f0fb", BLUE, vertical=True)
        ax.text(124.6, 33.8, "$e_{1..T}$", ha="center", fontsize=6, color=INK2)
        arrow(126.4, 42.7, 129.2, 42.7, col=GREEN)
        tokens(132.4, 58.0, 4, "#ffffff", MUTED)
        ax.text(136.8, 62.4, "text prefix (8 tokens)", ha="center", fontsize=6, color=INK2)
        arrow(136.8, 57.0, 136.8, 54.6, col=GREEN)
        slabs(129.6, 33.5, 4, 3.0, 16, 2.0, 2.4, 4.3)
        ax.text(139.8, 30.6, "12 Transformer layers", ha="center", fontsize=6, color=INK)
        ax.text(139.8, 27.6, "$d=768$", ha="center", fontsize=6, color=INK2)
        arrow(139.8, 26.2, 139.8, 23.2, col=GREEN)
        tokens(127.4, 19.2, 4, "#f2f2f0", "#c8c7c2", group=False)
        tokens(139.4, 19.2, 4, "#cfe8dc", GREEN, hatch="//////")
        ax.text(131.6, 17.4, "prefix\n(ignored)", ha="center", va="top", fontsize=6, color=MUTED, linespacing=1.0)
        ax.text(143.8, 17.4, "pose\noutputs", ha="center", va="top", fontsize=6, color=INK2, linespacing=1.0)
        arrow(151.6, 20.4, 157, 20.4)

        # (4) our classification head (bottom to top)
        panel(155, 12, 180, 73, 4, "Classifier (ours)", INK, "#f8f8f6", 6.4)
        block(167.5, 21.2, 21, 8.2, "mean over\npose tokens\n(valid frames)", "#e6e6e3", 5.8)
        arrow(167.5, 25.3, 167.5, 30.6)
        block(167.5, 34, 21, 6.6, "Linear 768 → 114", "#e6e6e3", 6)
        arrow(167.5, 37.3, 167.5, 43.6)
        for k, v in enumerate((0.92, 0.35, 0.22, 0.12, 0.06)):
            ax.add_patch(R((159.5, 57.4 - 3.1 * k), 15 * v, 2.2, fc=TEST if k == 0 else "#c9d9ef", ec="none", zorder=3))
        ax.text(167.5, 63.6, "$\\hat{y}\\in\\mathbb{R}^{114}$", ha="center", fontsize=6.4)
        ax.text(167.5, 60.8, "class scores", ha="center", fontsize=6, color=INK2)

        # two-row legend
        rbox(1, -4, 180, 9.6, "#ffffff", AXIS, 0.6)

        def row(y, items):
            x = 5
            for kind, a1, a2, lab in items:
                if kind == "mk":
                    ax.plot(x, y, marker=a1, color=a2, ms=3.5, ls="none")
                elif kind == "sq":
                    ax.add_patch(R((x - 1.2, y - 1.2), 2.4, 2.4, fc=a1, ec=a2[0], lw=0.5, hatch=a2[1]))
                else:
                    ax.plot([x - 2.2, x + 1.2], [y, y], color=GREY, lw=0.6, ls=(0, (2.5, 1.6)))
                    ax.plot(x - 2.2, y, "o", ms=2.2, color=GREY)
                ax.text(x + 2.2, y, lab, fontsize=6, va="center", family=SANS)
                x += 7.5 + len(lab) * 1.2

        row(5.8, [("mk", "o", SERIES[0], "right hand"), ("mk", "s", SERIES[1], "left hand"),
                  ("mk", "o", INK, "body and face keypoints"), ("sq", HAND, ("none", None), "both hands: shared weights"),
                  ("ln", None, None, "body feature added (no gradient)")])
        row(-0.4, [("sq", "#ffffff", (MUTED, None), "text token (frozen embedding)"), ("sq", "#e9f0fb", (BLUE, None), "pose token"),
                   ("sq", "#cfe8dc", (GREEN, "//////"), "encoder output"), ("sq", "#f2f2f0", ("#c8c7c2", None), "output not used")])
    save(fig, "fig_unisign")


def demo():
    import tempfile
    from explain_unisign import GROUPS
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


if __name__ == "__main__":
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
