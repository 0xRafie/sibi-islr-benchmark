"""Trim the lead-in and trailing pause frames of a clip, using its cached MediaPipe keypoints (MediaPipe is not re-run).

The active span runs from the first to the last frame with a detected hand, after short blips are cleaned, plus PAD
frames on each side. A clip without any detected hand is kept whole. The blip limit (MIN_RUN) and the padding are
multiplied by fps/30 and rounded (60 fps: 6 and 4 frames; 20 fps: 2 and 1); at 30 fps they are unchanged. Other scripts
import `span()` and `span_mask()`.

The command line trims the SIBI dictionary keypoints, `<SIBI_DICT>/keypoints/*.npy` -> `<SIBI_DICT>/keypoints_trim/`,
with statistics in `<SIBI_QC>/trim_stats.csv` (SIBI_DICT defaults to the working directory, SIBI_QC to `qc`). The aim is
to compare real sign durations with the frame counts reported by Darmawan et al. (JIKI 2024: me 12-26, ber 12-16, ter 16
frames), to test whether resampling every clip to 30 frames treats affixes and content words unequally.

    python scripts/trim.py --demo
    python scripts/trim.py
"""
import csv, glob, os, sys
import numpy as np

N_HAND = 21 * 3
DICT = os.environ.get("SIBI_DICT", "")
SRC, DST = os.path.join(DICT, "keypoints"), os.path.join(DICT, "keypoints_trim")
QC = os.environ.get("SIBI_QC", "qc")
MIN_RUN = 3   # blips shorter than this are removed
PAD = 2       # keep a little padding at both ends


def hand_mask(arr):
    """True where the frame has at least one detected hand."""
    return ~np.all(arr[:, -2 * N_HAND:] == 0, axis=1)


def clean(mask, min_run=MIN_RUN):
    """Flip True/False runs shorter than min_run to the value before them (runs at either end are kept)."""
    out = mask.copy()
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if j - i < min_run and i > 0 and j < len(out):
            out[i:j] = out[i - 1]
        i = j
    return out


def span(arr, fps=30):
    """(start, end) of the active span; the whole clip if no hand is ever detected. The blip limit and the padding are
    multiplied by fps/30 and rounded (60 fps: 6 and 4 frames; 20 fps: 2 and 1); 30 fps uses MIN_RUN and PAD as they are."""
    return span_mask(hand_mask(arr), fps)


def span_mask(mask, fps=30):
    """span() from a per-frame hand mask (e.g. RTMW: hand score >= threshold)."""
    pad = round(PAD * fps / 30)
    idx = np.flatnonzero(clean(mask, round(MIN_RUN * fps / 30)))
    if len(idx) == 0:
        return 0, len(mask)
    return max(0, idx[0] - pad), min(len(mask), idx[-1] + 1 + pad)


def main():
    os.makedirs(DST, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(f"{SRC}/*.npy")):
        stem = os.path.basename(f)[:-4]
        arr = np.load(f)
        a, b = span(arr)
        np.save(f"{DST}/{stem}.npy", arr[a:b])
        rows.append(dict(kelas=stem, kategori=stem.split("__")[0],
                         frame_asli=len(arr), frame_potong=b - a,
                         sisa_pct=round(100 * (b - a) / max(len(arr), 1), 1)))
    with open(f"{QC}/trim_stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} files -> {DST}/ , statistics -> {QC}/trim_stats.csv")


def demo():
    """Self-check: a clip with hands only in the middle is trimmed to that middle."""
    arr = np.zeros((100, 1662), dtype=np.float32)
    arr[40:60, -2 * N_HAND:] = 0.5            # hands in frames 40-59
    a, b = span(arr)
    assert a == 38 and b == 62, (a, b)         # 20 frames + padding of 2 on each side
    kosong = np.zeros((10, 1662), dtype=np.float32)
    assert span(kosong) == (0, 10)             # no hands: trim nothing
    blip = np.zeros((50, 1662), dtype=np.float32)
    blip[10, -2 * N_HAND:] = 0.5               # one stray frame
    blip[30:45, -2 * N_HAND:] = 0.5
    a2, b2 = span(blip)
    assert a2 >= 25, f"blip not removed: {a2}"
    assert span(arr, 60) == (36, 64) and span(arr, 20) == (39, 61)   # padding 4 (60 fps) and 1 (20 fps)
    blip3 = np.zeros((50, 1662), dtype=np.float32)
    blip3[10:14, -2 * N_HAND:] = 0.5                                    # 4 frames: a blip at 60 fps (< 6), not at 30 fps
    blip3[30:45, -2 * N_HAND:] = 0.5
    assert span(blip3)[0] == 8 and span(blip3, 60)[0] == 26
    print("demo ok: middle span kept, empty clip safe, blip removed, fps scaling")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
