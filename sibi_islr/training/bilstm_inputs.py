"""Input features of the BiLSTM baseline and its ablations: loading the keypoint packs, face and
feature-set extras, unresampled and padded inputs (moved from train_main.py)."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; set before torch is used
import argparse, csv, json, sys, time

import numpy as np

from sibi_islr.models.bilstm import B1, fill_gaps, model, resample as resample_n
from sibi_islr.data.features import LR, features, shoulder_cw

from sibi_islr.training.common import SETTINGS


# Ablations a4/a4lips: face points appended to the 84 hand features. Lips = MediaPipe FACEMESH_LIPS (40 unique points)
LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
        308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]
FACE = {"a4": np.arange(468), "a4lips": np.array(LIPS)}
assert len(set(LIPS)) == 40
# Feature sets f1-f3 (exploratory): face subset = lips + eyes + eyebrows (MediaPipe FACEMESH_LIPS, *_EYE, *_EYEBROW),
# upper pose = points 11-22 (shoulders, elbows, wrists, three finger points)
EYES = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
        249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
BROWS = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 276, 282, 283, 285, 293, 295, 296, 300, 334, 336]
FACE_SUB = np.array(sorted(set(LIPS) | set(EYES) | set(BROWS)))
POSE_UP = np.arange(11, 23)
FX = ("f1", "f2", "f3", "f3face")
assert len(FACE_SUB) == 92


def load(setting, data_dir, folds_dir, resample=True, trim=True, face=None, how="shoulder", fx=None):
    """F0 features (N, 30, 84), fold metadata, and class names, in the row order of folds_<setting>.csv.

    resample=False (ablation a1): list of (T_i, 84) arrays at their trimmed length; shoulder normalization, gap filling,
    and mask as for F0, only the resampling to 30 frames is skipped.
    face='a4' | 'a4lips': face features from mp_face_<category>.npz appended after the 84 hand features.
    fx='f1' | 'f2' | 'f3' | 'f3face': blocks from mp_fx_<category>.npz appended after the 84 hand features."""
    rows = list(csv.DictReader(open(f"{folds_dir}/folds_{setting}.csv")))
    # load once: each NpzFile access decompresses the whole array again, and views keep it in memory
    packs = {c: dict(np.load(f"{data_dir}/mp_{c}.npz")) for c in SETTINGS[setting]}
    faces = {c: dict(np.load(f"{data_dir}/mp_face_{c}.npz")) for c in SETTINGS[setting]} if face else {}
    fxs = {c: dict(np.load(f"{data_dir}/mp_fx_{c}.npz")) for c in SETTINGS[setting]} if fx else {}
    for c, f in [*faces.items(), *fxs.items()]:                # frame-aligned with the main pack
        assert np.array_equal(f["relpath"], packs[c]["relpath"]) and np.array_equal(f["off"], packs[c]["off"]), c
    index = {c: {r: i for i, r in enumerate(p["relpath"])} for c, p in packs.items()}
    seqs, fseqs, xseqs = [], [], []
    for r in rows:
        c = r["relpath"].split("/")[0]                      # pack = relpath prefix (setting dr: external set in mp_ricky.npz)
        p = packs[c]
        i = index[c][r["relpath"]]
        a, b = p["span"][i] if trim else (0, None)          # trim=False: whole clip, no trimming
        seqs.append(p["seq"][p["off"][i]:p["off"][i + 1]][a:b])
        if face:
            f = faces[c]
            fseqs.append(f["face"][f["off"][i]:f["off"][i + 1]][a:b])   # float16 view; float32 per clip in extra_features
        if fx:
            f = fxs[c]
            xseqs.append({k: f[k][f["off"][i]:f["off"][i + 1]][a:b].astype(np.float32) for k in ("hz", "face")})
    d = dict(seq=np.concatenate(seqs), off=np.cumsum([0] + [len(s) for s in seqs]), relpath=[r["relpath"] for r in rows])
    X = features(d, how) if resample else unresampled(d)   # shoulder normalization, gap filling, 30 frames, mask; a5: how=signbart
    if face:
        X = np.concatenate([X, face_features(d, fseqs, FACE[face])], 2)
    if fx:
        X = np.concatenate([X, fx_features(d, xseqs, fx)], 2)
    meta = {k: np.array([int(r[k]) for r in rows]) for k in ("label", "subject", "cs", "cg", "cl", "cp", "shared") if k in rows[0]}
    meta["category"] = np.array([r["category"] for r in rows])
    names = {}
    for r in rows:
        names[int(r["label"])] = f"{r['category']}/{r['class']}"
    return X, meta, [names[i] for i in range(len(names))]


def extra_features(d, pts, dets, center=True, scale=True, n=30):
    """Extra points per trimmed clip, pts[i] (T, K, P, C) and dets[i] (T, K), treated like the F0 hands: minus the clip's
    shoulder midpoint (center), divided by the clip's shoulder width (scale; shoulder_cw), frames without a detection
    interpolated inside the detected range and 0 outside it (fill_gaps), resampled to n frames, then multiplied by the
    mask."""
    out = []
    for i, (f, det) in enumerate(zip(pts, dets)):
        f = f.astype(np.float32, copy=False)                  # per clip: float16 input (a4) is not copied whole to float32
        cw = shoulder_cw(d["seq"][d["off"][i]:d["off"][i + 1], :66].reshape(-1, 33, 2)) if scale else None
        if cw is not None:
            f = np.where(det[:, :, None, None], (f - (cw[0] if center else 0)) / cw[1], 0)
        h, mask = resample_n(*fill_gaps(f, det), n)
        out.append((h * mask[:, :, None, None]).reshape(n, -1).astype(np.float32))   # same values, half the memory
    return np.stack(out)


def face_features(d, fseqs, idx, n=30):
    """a4: face (T, 936) per trimmed clip -> (n, 2 x len(idx)), treated like the F0 hands (extra_features)."""
    full = len(idx) == 468 and np.array_equal(idx, np.arange(468))      # full face: a view, no indexed copy (saves RAM)
    f = [(fc.reshape(len(fc), 468, 2) if full else fc.reshape(len(fc), 468, 2)[:, idx])[:, None] for fc in fseqs]
    return extra_features(d, f, [np.any(x != 0, axis=(2, 3)) for x in f], n=n)


def fx_features(d, xseqs, fx):
    """Blocks after the 84 F0 features. f1 = hand z (42; divided by the shoulder width, not shifted, because MediaPipe z
    is already relative to the wrist). f2 = f1 + pose 11-22 xy (24). f3 = f2 + face subset xy (184), treated like F0.
    f3face = f2 + face subset relative to its per-frame center, divided by the distance between the outer eye corners
    33-263 (face shape, as in probe_pairs.lips)."""
    seq = [d["seq"][d["off"][i]:d["off"][i + 1]] for i in range(len(xseqs))]
    hdet = [np.any(s[:, 66:].reshape(-1, 2, 21, 2) != 0, axis=(2, 3)) for s in seq]
    blocks = [extra_features(d, [x["hz"].reshape(-1, 2, 21, 1) for x in xseqs], hdet, center=False)]
    if fx != "f1":
        pose = [s[:, :66].reshape(-1, 33, 2)[:, POSE_UP][:, None] for s in seq]
        blocks.append(extra_features(d, pose, [np.any(p != 0, axis=(2, 3)) for p in pose]))
    if fx in ("f3", "f3face"):
        face = [x["face"].reshape(-1, 1, 92, 2) for x in xseqs]
        fdet = [np.any(f != 0, axis=(2, 3)) for f in face]
        if fx == "f3face":
            e = np.searchsorted(FACE_SUB, [33, 263])
            face = [np.where(det[:, :, None, None], (f - f.mean(2, keepdims=True))
                             / np.maximum(np.linalg.norm(f[:, :, e[0]] - f[:, :, e[1]], axis=-1), 1e-6)[:, :, None, None], 0)
                    for f, det in zip(face, fdet)]
        blocks.append(extra_features(d, face, fdet, scale=fx == "f3"))
    return np.concatenate(blocks, 2)


def check_fx():
    """fx_features without torch: block widths of f1-f3; hands and pose unaffected by the face; the f3face block is
    unchanged when the face is shifted and scaled, while the f3 block (F0 treatment) changes."""
    g = np.random.default_rng(0)
    seq = g.uniform(0.3, 0.7, (40, 150)).astype(np.float32)
    seq[:, 22:26] = [0.6, 0.5, 0.4, 0.5]                      # shoulders (pose 11, 12): center (0.5, 0.5), width 0.2
    d = dict(seq=seq, off=np.array([0, 20, 40]))
    xs = [dict(hz=g.normal(0, .02, (20, 42)), face=g.uniform(.45, .55, (20, 184))) for _ in range(2)]
    moved = [dict(x, face=((x["face"] - .5) * 1.5 + .6)) for x in xs]
    for fx, w in (("f1", 42), ("f2", 66), ("f3", 250), ("f3face", 250)):
        a, b = fx_features(d, xs, fx), fx_features(d, moved, fx)
        assert a.shape == (2, 30, w) and np.isfinite(a).all() and np.allclose(a[..., :min(w, 66)], b[..., :min(w, 66)]), fx
    assert np.allclose(fx_features(d, xs, "f3face"), fx_features(d, moved, "f3face"), atol=1e-5)
    assert not np.allclose(fx_features(d, xs, "f3"), fx_features(d, moved, "f3"))
    print("fx ok: blocks 42/66/250/250; hands and pose unchanged; f3face invariant to face shift + scale, f3 not", flush=True)


def unresampled(d):
    """a1: like gate_variants.features(d, 'shoulder') without resampling -> list of (T, 84) float32."""
    from sibi_islr.models.bilstm import fill_gaps
    from sibi_islr.data.features import normalize
    out = []
    for i in range(len(d["relpath"])):
        seq = d["seq"][d["off"][i]:d["off"][i + 1]]
        h, mask = fill_gaps(*normalize(seq, "shoulder"))
        out.append((h * mask[:, :, None, None]).reshape(len(seq), 84).astype(np.float32))
    return out


def padded(net, xs, device):
    """BiLSTM over clips of different lengths (a1): padding + packing, mean over the real frames only -> (logits, h192).

    The batch order is set with a permutation matrix (matmul) instead of index_select, so that its backward pass is
    deterministic on CUDA. The result equals running each clip on its own."""
    import torch
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    lens = np.array([len(x) for x in xs])
    order = np.argsort(-lens, kind="stable")
    P = torch.zeros(len(xs), len(xs), device=device)
    P[torch.arange(len(xs)), torch.tensor(order)] = 1.0                 # row i = clip order[i]
    x = np.zeros((len(xs), lens.max(), xs[0].shape[1]), np.float32)
    for i, a in enumerate(xs):
        x[i, :len(a)] = a
    xs_ = torch.einsum("ij,jtf->itf", P, torch.tensor(x, device=device))
    out, _ = net.lstm(pack_padded_sequence(xs_, torch.tensor(lens[order]), batch_first=True, enforce_sorted=True))
    out, _ = pad_packed_sequence(out, batch_first=True)
    h = torch.einsum("ij,if->jf", P, out.sum(1) / torch.tensor(lens[order], device=device, dtype=out.dtype)[:, None])
    return net.head(h), h
