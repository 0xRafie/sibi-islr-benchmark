"""Importance of the 69 Uni-Sign input keypoints: test-time occlusion of single keypoints and exact Shapley values of six
keypoint groups (exploratory analysis).

The model is trained on the full input. At test time a dropped keypoint is made to look undetected: its row (x, y, score)
is set to 0, as Uni-Sign does for keypoints scored <= 0.3 (datasets.py:71 and :104). The zeroing is done AFTER
load_part_kp. Zeroing the score of a body keypoint before it would change the crop_scale box and shift the scale of every
part, so the measurement would reflect the normalization rather than the keypoint.

Interpretation: the values measure how much the trained model depends on a keypoint, not how much information the keypoint
carries, and occluded inputs lie outside the training distribution. The wrists (9, 10) and the nose (0) also anchor the
hand and face branches (models.py:240, :252, :255), so their values are expectedly large. Neighboring finger joints can
stand in for each other, so occluding one keypoint can look harmless even when its group matters; hence the Shapley values
per group.

    python scripts/explain_unisign.py --demo      # numpy only; needs a Uni-Sign clone in $UNISIGN_DIR (default ext/Uni-Sign)
"""
import ast
import copy
import os
import sys
from math import factorial
from pathlib import Path
from types import SimpleNamespace

import numpy as np

UNISIGN = Path(os.environ.get("UNISIGN_DIR", Path(__file__).resolve().parents[1] / "ext" / "Uni-Sign"))
PARTS = {"body": [0, *range(3, 11)], "left": list(range(91, 112)), "right": list(range(112, 133)),   # datasets.py:32-43,
         "face_all": [*range(23, 40, 2), *range(83, 91), 53]}                                         # in row order
ROW = {k: (part, r) for part, ids in PARTS.items() for r, k in enumerate(ids)}                        # COCO -> (part, row)
THUMBS = [92, 93, 94, 95, 113, 114, 115, 116]
GROUPS = {"wrists": [9, 10], "body": [0, *range(3, 9)], "face": PARTS["face_all"], "thumbs": THUMBS,
          "left hand, no thumb": [k for k in PARTS["left"] if k not in THUMBS],
          "right hand, no thumb": [k for k in PARTS["right"] if k not in THUMBS]}


def occlude(p, drop):
    """p = output of load_part_kp {part: (T, N, 3)}; COCO keypoints in drop -> row 0. In place, numpy or torch."""
    for k in drop:
        part, r = ROW[k]
        p[part][:, r] = 0
    return p


def coalition_drop(mask):
    """Bit i of mask = group i present -> keypoints to drop (all keypoints of the absent groups)."""
    return [k for i, ids in enumerate(GROUPS.values()) if not mask >> i & 1 for k in ids]


def shapley(v):
    """v[mask] = value of the coalition, length 2^n -> exact Shapley values (n,). Linear in v, so averaging over folds
    can be done before or after."""
    n = len(v).bit_length() - 1
    w = [factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)]
    return np.array([sum(w[bin(m).count("1")] * (v[m | 1 << i] - v[m]) for m in range(len(v)) if not m >> i & 1)
                     for i in range(n)])


def _load_part_kp():
    """load_part_kp and crop_scale from Uni-Sign's datasets.py with torch replaced by numpy (enough to check indices
    without torch)."""
    path = UNISIGN / "datasets.py"
    assert path.exists(), f"no Uni-Sign clone at {UNISIGN} (set UNISIGN_DIR)"
    nodes = [n for n in ast.walk(ast.parse(path.read_text(), str(path)))
             if isinstance(n, ast.FunctionDef) and n.name in ("load_part_kp", "crop_scale")]
    ns = {"np": np, "copy": copy, "torch": SimpleNamespace(tensor=np.asarray)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return lambda kp, sc: ns["load_part_kp"](kp[:, None], sc[:, None], force_ok=True)


def demo():
    pts = [0, *range(3, 11), *range(91, 133), *range(23, 40, 2), *range(83, 91), 53]   # UNISIGN_PTS in figures.py
    flat = [k for ids in GROUPS.values() for k in ids]
    assert len(ROW) == 69 and sorted(ROW) == sorted(pts) and sorted(flat) == sorted(pts)   # the groups partition the 69
    assert [len(g) for g in GROUPS.values()] == [2, 7, 18, 8, 17, 17]
    assert coalition_drop(63) == [] and sorted(coalition_drop(0)) == sorted(pts)

    # ROW indices = Uni-Sign indices: for non-body keypoints, occlusion after loading == score zeroed before loading
    load = _load_part_kp()
    rng = np.random.default_rng(0)
    kp, sc = rng.uniform(0.3, 0.7, (6, 133, 2)), np.full((6, 133), 0.9, np.float32)
    kp[:, 10, 0] = 0.95                                      # right wrist = outer side of the body box
    full = load(kp, sc)
    for k in pts:
        if k in PARTS["body"]:
            continue
        s2 = sc.copy()
        s2[:, k] = 0
        pre, post = load(kp, s2), occlude(copy.deepcopy(full), [k])
        assert all(np.array_equal(pre[q], post[q]) for q in PARTS), k
        assert not post[ROW[k][0]][:, ROW[k][1]].any() and np.count_nonzero(post[ROW[k][0]] != full[ROW[k][0]])
    # body keypoints: zeroing the score before loading shifts the scale of every part (the trap); after loading it does not
    s2 = sc.copy()
    s2[:, 10] = 0
    pre, post = load(kp, s2), occlude(copy.deepcopy(full), [10])
    assert not np.allclose(pre["left"], full["left"]) and np.array_equal(post["left"], full["left"])
    assert not post["body"][:, 8].any() and np.array_equal(post["body"][:, :8], full["body"][:, :8])

    # Shapley: additive game -> its weights; efficiency; symmetry
    a = np.array([3.0, -1.0, 0.5, 2.0, 0.0, 1.5])
    v = np.array([sum(a[i] for i in range(6) if m >> i & 1) for m in range(64)])
    assert np.allclose(shapley(v), a)
    v = rng.normal(size=64)
    assert np.isclose(shapley(v).sum(), v[63] - v[0])
    v = np.array([bin(m).count("1") ** 2 for m in range(64)], float)
    assert np.allclose(shapley(v), shapley(v)[0])
    print("demo ok: 69 keypoints = 6 groups (2+7+18+8+17+17), indices match Uni-Sign's load_part_kp, occluding a body "
          "keypoint after loading leaves the scale unchanged, exact Shapley passes additivity, efficiency and symmetry")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
