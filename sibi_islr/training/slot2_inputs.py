"""Inputs and recipes of SignBart and DSTA-SLR as run by training/slot2.py: official repositories,
replication packs, the F0, rtmw27 and mp48 inputs (moved from slot2.py)."""
import csv
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; set before torch is imported
import numpy as np

from sibi_islr.data.features import L_SH, R_SH, features


REPOS = {"signbart": (os.environ.get("SIGNBART_DIR", "ext/SignBart"), "https://github.com/TinhNguyen2312/SignBart", "1b7d3d1"),
         "dsta": (os.environ.get("DSTA_SLR_DIR", "ext/DSTA-SLR"), "https://github.com/hulianyuyy/DSTA-SLR", "e7e5ee2")}
PACKS = dict(affix="data/pilot/gate_raw_am7.npz", number="data/pilot/gate_raw_number_am7.npz",
             alphabet="data/pilot/gate_raw_alphabet_am7.npz")  # replication packs (subjects 1-20), used without proto=
RAW = PACKS["affix"]
# Repo recipes. SignBart: scripts/train_LSA-64.sh (lr 2e-4; the README and the main.py default say 2e-5) +
# configs/LSA-64.yaml + utils.train_epoch + dataset.Datasets. DSTA-SLR: config/train.yaml + main.py + feeders/feeder.py
SIGNBART = dict(epochs=200, lr=2e-4, batch=1, factor=0.1, patience=5, p_aug=0.4)
DSTA = dict(epochs=250, lr=0.1, batch=24, momentum=0.9, wd=1e-4, warmup=20, step=(150, 200), keep_rate=0.9,
            shift_px=10, frame_px=512, model=dict(groups=16, block_size=41, inner_dim=64, depth=4, drop_layers=2))
# MediaPipe hand, 21 points, anatomical bones (child -> parent, towards the wrist). F0: left 0-20, right 21-41, two
# components without an edge between the hands
HAND = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7), (9, 5), (10, 9), (11, 10), (12, 11),
        (13, 9), (14, 13), (15, 14), (16, 15), (17, 13), (17, 0), (18, 17), (19, 18), (20, 19)]
FLIP = np.r_[21:42, 0:21]  # DSTA-SLR mirror (flip_index): swap the left and right hands
# Original DSTA-SLR input (rtmw27): SAM-SLR-v2 points (data_gen/sign_gendata.py:13-14) = COCO-WholeBody nose, shoulders,
# elbows, wrists + 10 points per hand (base and tip of the thumb, base and tip of the four fingers). RTMW outputs the same
# COCO-WholeBody points, divided by (W, H); the dataset videos are square, so x 512 = pixels of the repo's 512 x 512
# frame (feeder.py:253).
HAND10 = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
SIGN27 = np.r_[0, 5:11, [91 + k for k in HAND10], [112 + k for k in HAND10]]
FLIP27 = np.r_[[0, 2, 1, 4, 3, 6, 5], 17:27, 7:17]  # feeder.py:15-22 (flip_index)
WINDOW = 30         # 30 frames to reduce cost; repo: 120, "100 for AUTSL"


class Graph42:
    """F0 graph for DSTA-SLR, replacing graph/sign_27.py. Loaded through import_class("slot2.Graph42")."""

    def __init__(self, labeling_mode="spatial"):
        from graph import tools                                     # DSTA-SLR/graph/tools.py
        self.num_node = 42
        self.self_link = [(i, i) for i in range(42)]
        self.inward = [(c + o, p + o) for o in (0, 21) for c, p in HAND]
        self.outward = [(p, c) for c, p in self.inward]
        self.neighbor = self.inward + self.outward
        assert labeling_mode == "spatial"
        self.A = tools.get_spatial_graph(42, self.self_link, self.inward, self.outward)


def load(raw=RAW):
    """F0 (N, 30, 42, 2), labels, folds, number of classes, and the median shoulder width (MediaPipe coordinate units)."""
    d = dict(np.load(raw))
    X = features(d, "shoulder").reshape(-1, 30, 42, 2)
    fmap = dict(csv.reader(open(str(d["fold_csv"]))))
    return X, d["y"], np.array([int(fmap[r]) for r in d["relpath"]]), len(d["classes"]), shoulder_width(d)


def load_t2(cat, proto):
    """Like load(), for the 22-signer data and the fold column `proto` of folds_<cat>.csv. F0 is exactly the BiLSTM input
    (`train_main.load`); s follows load() over the trimmed clips of the setting (the fold-CSV clips from the packs of all
    its categories; the combined setting = four packs). Also returns meta and class names (train_main metrics)."""
    from sibi_islr.training.common import SETTINGS
    from sibi_islr.training.bilstm_inputs import load as load_e1
    data, folds = os.environ.get("SLOT2_DATA", "data/pilot"), os.environ.get("SLOT2_FOLDS", "qc")
    X, meta, names = load_e1(cat, data, folds)
    keep = {r["relpath"] for r in csv.DictReader(open(f"{folds}/folds_{cat}.csv"))}   # number: the CSV leaves out 'ribuan' of signer 17
    seqs = []
    for c in SETTINGS[cat]:
        p = dict(np.load(f"{data}/mp_{c}.npz"))
        seqs += [p["seq"][p["off"][i]:p["off"][i + 1]][slice(*p["span"][i])] for i, r in enumerate(p["relpath"]) if r in keep]
    d = dict(seq=np.concatenate(seqs), off=np.cumsum([0] + [len(q) for q in seqs]))
    return X.reshape(-1, 30, 42, 2), meta["label"], meta[proto], len(names), shoulder_width(d), meta, names


def rtmw27_clips(relpaths, mps, rtmws):
    """Clips (T, 27, 3) float32 per relpath: RTMW x, y times 512 and the score, cut to the F0 span (`span` in the
    MediaPipe pack of that category). mps/rtmws: {category: dict(relpath, off[, span] | kp, sc)}; both must have the same
    number of frames."""
    pos = {c: ({r: i for i, r in enumerate(mps[c]["relpath"])}, {r: i for i, r in enumerate(rtmws[c]["relpath"])}) for c in mps}
    out = np.empty(len(relpaths), object)
    for n, r in enumerate(relpaths):
        c = r.split("/")[0]
        m, w, (i, j) = mps[c], rtmws[c], (pos[c][0][r], pos[c][1][r])
        lo, hi = w["off"][j], w["off"][j + 1]
        assert hi - lo == m["off"][i + 1] - m["off"][i], f"RTMW frame count != MediaPipe: {r}"
        a, b = m["span"][i]
        kp, sc = w["kp"][lo:hi][a:b][:, SIGN27], w["sc"][lo:hi][a:b][:, SIGN27]
        out[n] = np.concatenate([kp.astype(np.float32) * DSTA["frame_px"], sc.astype(np.float32)[..., None]], 2)
    return out


def load_rtmw27(cat, proto):
    """Like load_t2(), with rtmw27_clips() at their original length (after the span) as X. Meta, labels, and class names
    come from train_main.load, so the order and the metrics are exactly those of the F0 runs."""
    from sibi_islr.training.common import SETTINGS
    from sibi_islr.training.bilstm_inputs import load as load_e1
    data, folds = os.environ.get("SLOT2_DATA", "data/pilot"), os.environ.get("SLOT2_FOLDS", "qc")
    dirs = (os.environ.get("SLOT2_RTMW") or data).split(os.pathsep)  # e.g. rtmw_kata.npz in a separate folder
    _, meta, names = load_e1(cat, data, folds)
    rel = [r["relpath"] for r in csv.DictReader(open(f"{folds}/folds_{cat}.csv"))]
    mps = {c: {k: np.load(f"{data}/mp_{c}.npz")[k] for k in ("relpath", "off", "span")} for c in SETTINGS[cat]}
    rtmws = {c: dict(np.load(next(f"{d}/rtmw_{c}.npz" for d in dirs if os.path.exists(f"{d}/rtmw_{c}.npz"))))
             for c in SETTINGS[cat]}
    return rtmw27_clips(rel, mps, rtmws), meta["label"], meta[proto], len(names), None, meta, names



def mp75(seq):
    """(T, 150) MediaPipe pack -> (T, 75, 2) as in the data files of the SignBart repo: pose 33, left hand 21, right 21."""
    return seq.reshape(len(seq), 75, 2).astype(np.float32)


def load_mp75(cat, proto):
    """SignBart with its own input: mp75() clips at their original length after the F0 span (trimming), MediaPipe
    coordinates normalized to the frame. Meta, labels, and class names as in load_rtmw27()."""
    from sibi_islr.training.common import SETTINGS
    from sibi_islr.training.bilstm_inputs import load as load_e1
    data, folds = os.environ.get("SLOT2_DATA", "data/pilot"), os.environ.get("SLOT2_FOLDS", "qc")
    _, meta, names = load_e1(cat, data, folds)
    rel = [r["relpath"] for r in csv.DictReader(open(f"{folds}/folds_{cat}.csv"))]
    packs = {c: dict(np.load(f"{data}/mp_{c}.npz")) for c in SETTINGS[cat]}
    pos = {c: {r: i for i, r in enumerate(p["relpath"])} for c, p in packs.items()}
    X = np.empty(len(rel), object)
    for n, r in enumerate(rel):
        p, i = packs[r.split("/")[0]], pos[r.split("/")[0]][r]
        a, b = p["span"][i]
        X[n] = mp75(p["seq"][p["off"][i]:p["off"][i + 1]][a:b])
    return X, meta["label"], meta[proto], len(names), None, meta, names


def shoulder_width(d):
    """Median over clips of the mean shoulder width per clip, without labels. Converts recipe magnitudes to F0 units."""
    w = []
    for i in range(len(d["off"]) - 1):
        sh = d["seq"][d["off"][i]:d["off"][i + 1], :66].reshape(-1, 33, 2)[:, [L_SH, R_SH]]
        ok = np.all(sh != 0, axis=(1, 2))
        if ok.any():
            w.append(np.linalg.norm(sh[ok, 0] - sh[ok, 1], axis=1).mean())
    return float(np.median(w))


def signbart_aug(kp, s):
    """SignBart Datasets.apply_augment in F0 units: noise divided by the shoulder width s, mirror x -> -x (without
    swapping hands, as flip_keypoints), rotation around the F0 origin (shoulder midpoint). Box normalization and
    clipping to [0, 1] are skipped."""
    import augmentations as A                                       # SignBart/augmentations.py
    aug = False
    while not aug:
        if np.random.uniform(0, 1) < 0.5:
            kp = A.rotate_keypoints(kp, (0, 0), np.random.uniform(-15, 15))
            aug = True
        if np.random.uniform(0, 1) < 0.5:
            kp = A.noise_injection(kp, np.random.uniform(0.01, 0.2) / s)
            aug = True
        if np.random.uniform(0, 1) < 0.5:
            n_f = kp.shape[0]
            kp = A.clip_frame(kp, np.random.randint(n_f // 2, n_f), np.random.uniform(0, 1) < 0.5)
            aug = True
        if np.random.uniform(0, 1) < 0.5:
            kp = A.time_warp_uniform(kp, np.random.uniform(1.1, 1.5))
            aug = True
        if np.random.uniform(0, 1) < 0.5:
            kp = kp.copy()
            kp[..., 0] = -kp[..., 0]
            aug = True
    return kp


def dsta_input(xb, s, train):
    """DSTA-SLR Feeder.__getitem__ for F0: (B, 30, 42, 2) -> (B, C=2, T=30, V=42, M=1). The window of 30 = the F0
    length, so random temporal sampling has no effect. The normalization by the mean of point 0 is skipped (F0 instead)."""
    x = xb.transpose(0, 3, 1, 2)[..., None].copy()
    if train:
        for b in range(len(x)):
            if random.random() > 0.5:                               # random_mirror, p 0.5
                x[b] = x[b][:, :, FLIP]
                x[b, 0] = -x[b, 0]                                  # 512 - x in pixels
            for c in (0, 1):                                        # random_shift: +-10 px of a 512 frame
                x[b, c] += (random.random() * 2 * DSTA["shift_px"] - DSTA["shift_px"]) / DSTA["frame_px"] / s
    return x


def dsta_tools():
    """feeders/tools.py of the DSTA-SLR repo, loaded from its file: feeders/__init__ also imports posenc
    (torch_geometric)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("dsta_tools", f"{REPOS['dsta'][0]}/feeders/tools.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def dsta27_input(clips, train, tools):
    """DSTA-SLR Feeder.__getitem__ (joint stream, is_vector False) for clips (T, 27, 3) -> (B, 3, WINDOW, 27, 1), in the
    repo order: temporal sampling (random_sample_np in training, uniform_sample_np at test), mirror with p 0.5
    (training), subtract the mean nose x and y (training and test), shift +-10 px (training). The score channel is only
    swapped along when mirroring."""
    out = []
    for kp in clips:
        d = np.array(kp.transpose(2, 0, 1)[..., None], np.float32)   # C, T, V, M
        d = tools.random_sample_np(d, WINDOW) if train else tools.uniform_sample_np(d, WINDOW)
        if train and random.random() > 0.5:
            d = d[:, :, FLIP27]
            d[0] = DSTA["frame_px"] - d[0]
        for c in (0, 1):
            d[c] -= d[c, :, 0, 0].mean(axis=0)
        if train:
            for c in (0, 1):
                d[c] += random.random() * 2 * DSTA["shift_px"] - DSTA["shift_px"]
        out.append(d)
    return np.stack(out)
