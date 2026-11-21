"""Fig. 4 (evaluation protocols) and Fig. 6-7 (per-signer accuracy and affix confusion of the final
runs) (moved from figures.py)."""
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

from sibi_islr.figures.common import AFFIX, BLUE_RAMP, COL, DISPLAY, FULL, INK, INK2, MUTED, OUT, QC, TEST, TRAIN, save


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
    from sibi_islr.evaluation import metrics as analyze
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
    import json
    from sibi_islr.evaluation import metrics as analyze
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
