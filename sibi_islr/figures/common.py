"""Paths, colours, class order and the save() helper shared by all figures (moved from figures.py)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sibi_islr import config


QC = config.QC
DATA = config.DATA
PAPER = config.PAPER
DICT = config.DICT
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
