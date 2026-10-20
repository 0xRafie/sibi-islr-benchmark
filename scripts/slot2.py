"""SignBart and DSTA-SLR with the training recipes of their official repositories (source of the reported results).

The official repositories are used as they are, at the pinned commits (REPOS), and cloned on first use into
$SIGNBART_DIR (default ext/SignBart) and $DSTA_SLR_DIR (default ext/DSTA-SLR), relative to the working directory. All
adaptations are in this file. Both repositories have a module named `model`, so one process loads only one of them.
DSTA-SLR creates its parameters on device="cuda", so it runs on a GPU only.

    python scripts/slot2.py --demo
    python scripts/slot2.py probe signbart|dsta [epoch] [batch=B] [epochs=E] [nodet] [amp]  # fold 0, seed 0, test data unused
    python scripts/slot2.py full signbart|dsta [fold ...] [batch=B] [epochs=E] [cat=K] [meanloss]  # seed 0, last epoch
    ... dsta [fold ...] stop=E | resume  # pause after epoch E (checkpoint to $SLOT2_CKPT_OUT), resume in a later session

Reported runs, one fold (0-21) per process:

    python scripts/slot2.py full signbart <fold> batch=128 meanloss mp48 proto=cl cat=combined
    python scripts/slot2.py full dsta <fold> epochs=250 rtmw27 proto=cl cat=combined

cat = affix (default) | number | alphabet without proto=: the replication packs in PACKS (subjects 1-20, C-S folds).
meanloss (SignBart): ReduceLROnPlateau is stepped on the mean loss of the epoch instead of the loss of its last batch
(utils.train_epoch); with mp48 it only changes the file name and `loss_kind`. The JSON is written to $SLOT2_OUT (default:
working directory) after each fold.

proto=cs|cg|cl: the 22-signer data (`mp_<category>.npz` for each category of setting `cat`, `train_main.SETTINGS`, e.g.
cat=combined, in $SLOT2_DATA, default data/pilot) and the fold column `proto` of `folds_<cat>.csv` (in $SLOT2_FOLDS,
default qc) instead of the replication packs. The JSON name contains the fold list, and the JSON holds the CSV row
indices of the test clips, their predictions, and the metrics of `train_main.metrics`.

rtmw27 (DSTA-SLR only, needs proto): the original DSTA-SLR input instead of F0. 27 COCO-WholeBody points from
`rtmw_<category>.npz` in $SLOT2_RTMW (folders separated by os.pathsep; default $SLOT2_DATA), channels x, y, score, cut
to the same span as F0, then 30 frames through the repo sampler (random and sorted in training, uniform at test). Graph
graph/sign_27.py of the repo.

mp48 (SignBart only, needs proto): the input, data pipeline, and model of the SignBart repo (signbart_mp48).

stop=E / resume (DSTA-SLR full only): the checkpoint holds all random states, so a resumed run equals an uninterrupted
one. resume reads the checkpoint from the folders in $SLOT2_CKPT (separated by os.pathsep).
"""
import csv
import json
import os
import random
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; set before torch is imported
import numpy as np

from gate_variants import L_SH, R_SH, features

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
    from train_main import SETTINGS, load as load_e1
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
    from train_main import SETTINGS, load as load_e1
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
    from train_main import SETTINGS, load as load_e1
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


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def env(det=True):
    import torch
    from train_main import versions
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = det, not det
    torch.use_deterministic_algorithms(det, warn_only=True)
    cuda = torch.cuda.is_available()
    return dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0) if cuda else "", device="cuda" if cuda else "cpu",
                deterministic=torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled(), **versions())


def use_repo(name):
    path, url, commit = REPOS[name]
    if not os.path.isdir(path):
        subprocess.run(["git", "clone", "-q", url, path], check=True)
        subprocess.run(["git", "-C", path, "checkout", "-q", commit], check=True)
    head = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    assert head.strip().startswith(commit), (name, head)
    sys.path.insert(0, path)
    return commit


def import_signbart():
    """SignBart's decoder.py uses relative imports and its other files absolute ones, so the repo cannot be imported as
    it is. It is loaded as a package, and the absolute names are aliased to the package modules. Files are unchanged."""
    import importlib
    import types
    pkg = types.ModuleType("signbart_repo")
    pkg.__path__ = [os.path.abspath(REPOS["signbart"][0])]
    sys.modules["signbart_repo"] = pkg
    for m in ("utils", "layers", "attention", "encoder", "decoder", "model", "augmentations"):
        sys.modules[m] = importlib.import_module(f"signbart_repo.{m}")
    return sys.modules["model"].SignBart


def signbart(n_cls, s, device, batch=SIGNBART["batch"], epochs=SIGNBART["epochs"], meanloss=False):
    import torch
    import yaml
    SignBart = import_signbart()                                    # SignBart/model.py

    with open(f"{REPOS['signbart'][0]}/configs/LSA-64.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg.update(joint_idx=list(range(42)), num_labels=n_cls)
    net = SignBart(cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=SIGNBART["lr"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=SIGNBART["factor"], patience=SIGNBART["patience"])

    def epoch(X, y, gen):
        net.train()
        loss, order, total = None, torch.randperm(len(X), generator=gen).tolist(), []
        for j in range(0, len(order), batch):                       # repo: batch 1
            idx = order[j:j + batch]
            kps = []
            for i in idx:
                kp = X[i].copy()
                if np.random.uniform(0, 1) < SIGNBART["p_aug"]:
                    kp = signbart_aug(kp, s)
                kps.append(kp)
            k = np.zeros((len(kps), max(len(kp) for kp in kps), 42, 2), np.float32)
            m = np.zeros(k.shape[:2], np.float32)
            for b, kp in enumerate(kps):                            # data_collator: zeros at the end + mask
                k[b, :len(kp)], m[b, :len(kp)] = kp, 1
            opt.zero_grad()
            loss, _ = net(keypoints=torch.tensor(k, device=device), attention_mask=torch.tensor(m, device=device),
                          labels=torch.tensor(y[idx], device=device))
            loss.backward()
            opt.step()
            total.append(loss.item())
        stepped = float(np.mean(total)) if meanloss else loss.item()  # repo (utils.train_epoch): loss of the last batch
        sch.step(stepped)
        return stepped

    def predict(X):
        net.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(X), 64):                          # same T (30), no augmentation
                k = torch.tensor(X[b:b + 64], device=device)
                out.append(net(keypoints=k, attention_mask=torch.ones(k.shape[:2], device=device))[1])
        return torch.cat(out).float().cpu().numpy()                 # logits; argmax in the caller

    return net, epoch, predict, epochs



def signbart_mp48(n_cls, device, batch=128, epochs=SIGNBART["epochs"]):
    """SignBart as in its repo: 48 points (`configs/LSA-64.yaml`), the repo's `dataset.py` (clip to [0, 1], at most 64
    evenly spaced frames, augmentation with p 0.4, box normalization per body part, padding + mask) through the repo's
    methods, AdamW 2e-4. Batch 128 and cosine schedule with a 10-epoch warmup from the paper (the repo's batch 1 is too
    slow)."""
    import importlib
    import math
    import torch
    import yaml
    SignBart = import_signbart()
    Datasets, U = importlib.import_module("signbart_repo.dataset").Datasets, sys.modules["utils"]
    clip_frame = sys.modules["augmentations"].clip_frame

    class Clips(Datasets):                                          # the repo's __getitem__ without pickle files
        def __init__(self, X, y, augment):
            self.X, self.y, self.augment = X, y, augment
            self.joint_idxs = [U.body_idx, U.lefthand_idx, U.righthand_idx]   # as in the repo's main.py

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            kp = np.clip(self.X[i].copy(), 0, 1)
            if kp.shape[0] > 64:
                kp = clip_frame(kp, 64, True)
            if self.augment and np.random.uniform(0, 1) < SIGNBART["p_aug"]:
                kp = self.apply_augment(kp)
            return torch.from_numpy(self.normalize_keypoints(kp)).float(), torch.tensor(self.y[i], dtype=torch.long)

    with open(f"{REPOS['signbart'][0]}/configs/LSA-64.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg.update(num_labels=n_cls)                                    # the repo's joint_idx (48 points) stays
    net = SignBart(cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=SIGNBART["lr"])    # default weight decay 0.01 = paper
    warm = 10
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / warm if e < warm else
                                            0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, epochs - warm))))

    def loader(X, y, gen=None):
        ds = Clips(X, y, gen is not None)
        return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=gen is not None, generator=gen,
                                           collate_fn=ds.data_collator, num_workers=2)

    def epoch(X, y, gen):
        net.train()
        total = []
        for d in loader(X, y, gen):
            opt.zero_grad()
            loss, _ = net(**{k: v.to(device) for k, v in d.items()})
            loss.backward()
            opt.step()
            total.append(loss.item())
        sch.step()
        return float(np.mean(total))

    def predict(X):
        net.eval()
        out = []
        with torch.no_grad():
            for d in loader(X, np.zeros(len(X), int)):
                out.append(net(keypoints=d["keypoints"].to(device), attention_mask=d["attention_mask"].to(device))[1])
        return torch.cat(out).float().cpu().numpy()                 # logits; argmax in the caller

    return net, epoch, predict, epochs


def dsta_schedule(e, epochs=DSTA["epochs"]):
    """DSTA-SLR lr and keep_prob (main.py) at epoch e. With fewer epochs, the warmup, the lr steps, and the keep_prob
    ramp (100 epochs) are scaled proportionally."""
    f = epochs / DSTA["epochs"]
    warm, steps, ramp = round(DSTA["warmup"] * f), [round(t * f) for t in DSTA["step"]], round(100 * f)
    lr = DSTA["lr"] * ((e + 1) / warm if e < warm else 0.1 ** sum(e >= t for t in steps))
    return lr, (-(1 - DSTA["keep_rate"]) / ramp * e + 1.0 if e < ramp else DSTA["keep_rate"])


def dsta(n_cls, s, device, amp=False, epochs=DSTA["epochs"], rtmw=False):
    import torch
    from model.fstgan import Model                                  # DSTA-SLR/model/fstgan.py

    if rtmw:                                                        # original input: repo graph, channels x, y, score
        tools = dsta_tools()
        net = Model(num_class=n_cls, num_point=27, num_person=1, graph="graph.sign_27.Graph",
                    graph_args=dict(labeling_mode="spatial"), in_channels=3, window_size=WINDOW, **DSTA["model"]).to(device)
        dsta_in = lambda xb, train: dsta27_input(xb, train, tools)
    else:
        net = Model(num_class=n_cls, num_point=42, num_person=1, graph="slot2.Graph42",
                    graph_args=dict(labeling_mode="spatial"), in_channels=2, window_size=30, **DSTA["model"]).to(device)
        dsta_in = lambda xb, train: dsta_input(xb, s, train)
    params = [dict(params=v, lr=DSTA["lr"], weight_decay=DSTA["wd"]) for _, v in net.named_parameters()]
    opt = torch.optim.SGD(params, momentum=DSTA["momentum"], nesterov=True)  # main.py load_optimizer
    lossf, state = torch.nn.CrossEntropyLoss(), dict(e=0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)              # amp: not the repo recipe, only for speed tests

    def epoch(X, y, gen):
        lr, keep = dsta_schedule(state["e"], epochs)
        for g in opt.param_groups:
            g["lr"] = lr
        net.train()
        perm = torch.randperm(len(X), generator=gen)
        for b in perm[:len(X) // DSTA["batch"] * DSTA["batch"]].split(DSTA["batch"]):  # drop_last=True
            x = torch.tensor(dsta_in(X[b.numpy()], True), dtype=torch.float32, device=device)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = lossf(net(x, keep), torch.tensor(y[b.numpy()], device=device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        state["e"] += 1
        return loss.item()

    def predict(X):
        net.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(X), DSTA["batch"]):
                x = torch.tensor(dsta_in(X[b:b + DSTA["batch"]], False), dtype=torch.float32, device=device)
                out.append(net(x))
        return torch.cat(out).float().cpu().numpy()                 # logits; argmax in the caller

    # pause/resume (main: stop, resume): training state other than the RNG
    epoch.state = lambda: dict(net=net.state_dict(), opt=opt.state_dict(), e=state["e"])
    epoch.load = lambda d: (net.load_state_dict(d["net"]), opt.load_state_dict(d["opt"]), state.update(e=d["e"]))
    return net, epoch, predict, epochs


def main(mode, name, rest):
    import torch
    from sklearn.metrics import f1_score
    opts, nums = [a for a in rest if not a.isdigit()], [int(a) for a in rest if a.isdigit()]
    val = lambda key, default: int(next((o.split("=")[1] for o in opts if o.startswith(key + "=")), default))
    batch, n_epochs = val("batch", SIGNBART["batch"]), val("epochs", (SIGNBART if name == "signbart" else DSTA)["epochs"])
    det, amp, meanloss = "nodet" not in opts, "amp" in opts, "meanloss" in opts  # default = repo recipe + deterministic
    cat = next((o.split("=")[1] for o in opts if o.startswith("cat=")), "affix")
    proto = next((o.split("=")[1] for o in opts if o.startswith("proto=")), None)
    seed = val("seed", 0)                                         # extra seeds
    # runs longer than one session. stop=E: checkpoint after epoch E, then the next fold; resume: must continue from a
    # checkpoint of an earlier session ($SLOT2_CKPT). All random states are saved, so the result equals an uninterrupted run
    stop, resume = val("stop", 0), "resume" in opts
    assert set(opts) <= {"nodet", "amp", "meanloss", f"batch={batch}", f"epochs={n_epochs}", f"cat={cat}", f"proto={proto}",
                         f"seed={seed}", "rtmw27", "mp48", f"stop={stop}", "resume"}, opts
    assert not (stop or resume) or (name == "dsta" and mode == "full" and not amp), "stop/resume: DSTA-SLR full only"
    assert not (name == "dsta" and (batch != 1 or meanloss)), "DSTA-SLR keeps batch 24 and has no plateau schedule"
    rtmw = "rtmw27" in opts
    assert not rtmw or (name == "dsta" and proto), "rtmw27: DSTA-SLR with proto= only"
    mp48 = "mp48" in opts                                           # SignBart with the repo's input and recipe
    assert not mp48 or (name == "signbart" and proto), "mp48: SignBart with proto= only"
    info = env(det)
    commit = use_repo(name)
    if mp48:
        X, y, fold, n_cls, s, meta, names = load_mp75(cat, proto)
    elif rtmw:
        X, y, fold, n_cls, s, meta, names = load_rtmw27(cat, proto)
    elif proto:
        X, y, fold, n_cls, s, meta, names = load_t2(cat, proto)
    else:
        X, y, fold, n_cls, s = load(PACKS[cat])
    out_dir = os.environ.get("SLOT2_OUT", ".")
    folds = nums if mode == "full" and nums else [0]
    out, tag = [], "".join(f"_{o.replace('=', '')}" for o in opts if not o.startswith(("stop=", "resume"))) + (
        f"_k{'-'.join(map(str, folds))}" if proto else "")
    ck_in = [d for d in os.environ.get("SLOT2_CKPT", "").split(os.pathsep) if d]
    ck_out = os.environ.get("SLOT2_CKPT_OUT", out_dir)
    for k in folds:
        seed_all(seed)
        net, epoch, predict, n_ep = (signbart_mp48(n_cls, info["device"], batch, n_epochs) if mp48 else
                                     signbart(n_cls, s, info["device"], batch, n_epochs, meanloss) if name == "signbart"
                                     else dsta(n_cls, s, info["device"], amp, n_epochs, rtmw))
        if mode == "probe":
            n_ep = nums[0] if nums else 5
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        gen, sec, losses, t0 = torch.Generator().manual_seed(seed), [], [], time.time()
        curve, every, start, prev = [], {"dsta": 5, "signbart": 1}[name], 0, 0.0   # test score every few epochs, for reference only
        ck = f"slot2_full_{name}{tag}_f{k:02d}.ckpt"
        if resume:
            path = next((os.path.join(d, ck) for d in ck_in if os.path.exists(os.path.join(d, ck))), None)
            assert path, f"resume: {ck} not found in $SLOT2_CKPT {ck_in}"
            st = torch.load(path, map_location="cpu", weights_only=False)   # RNG states must be CPU tensors
            epoch.load(st["model"])
            random.setstate(st["py"]), np.random.set_state(st["np"]), torch.set_rng_state(st["cpu"])
            torch.cuda.set_rng_state_all(st["cuda"]), gen.set_state(st["gen"])
            start, sec, losses, curve, prev = st["epoch"], st["sec"], st["losses"], st["curve"], st["elapsed"]
            print(f"{name} fold {k}: resuming from epoch {start} ({path})", flush=True)
        for e in range(start, n_ep):
            t = time.time()
            losses.append(epoch(X[tr], y[tr], gen))
            sec.append(round(time.time() - t, 2))
            print(f"{name} fold {k} epoch {e + 1}/{n_ep}: loss {losses[-1]:.3f} ({sec[-1]:.1f} s)", flush=True)
            if mode == "full" and ((e + 1) % every == 0 or e + 1 == n_ep):
                st = random.getstate(), np.random.get_state()      # evaluation must not move the training RNG
                with torch.random.fork_rng():                       # one GPU per process (CUDA_VISIBLE_DEVICES)
                    p_ = predict(X[te]).argmax(1)
                random.setstate(st[0]); np.random.set_state(st[1])
                curve.append([e + 1, float((p_ == y[te]).mean()), float(f1_score(y[te], p_, average="macro"))])
            if e + 1 == stop < n_ep:
                os.makedirs(ck_out, exist_ok=True)
                torch.save(dict(model=epoch.state(), py=random.getstate(), np=np.random.get_state(), cpu=torch.get_rng_state(),
                                cuda=torch.cuda.get_rng_state_all(), gen=gen.get_state(), epoch=e + 1, sec=sec,
                                losses=losses, curve=curve, elapsed=prev + time.time() - t0), os.path.join(ck_out, ck))
                break
        if len(losses) < n_ep:                                      # paused: this fold's result comes from a later session
            print(f"PAUSED {name} fold {k} after epoch {len(losses)}/{n_ep}: {ck}", flush=True)
            continue
        r = dict(model=name, mode=mode, options=opts, commit=commit, code_commit=open("kode_commit.txt").read().strip()
                 if os.path.exists("kode_commit.txt") else None, cat=cat, fold=k, seed=seed, epochs=n_ep, n_train=len(tr),
                 n_test=len(te), n_classes=n_cls, loss_kind="epoch_mean" if meanloss else "last_batch",
                 params=sum(p.numel() for p in net.parameters()), shoulder_width=s, sec_per_epoch=sec,
                 total_sec=round(prev + time.time() - t0, 1), last_loss=losses, test_curve=curve, **info)
        r["train_acc"] = float((predict(X[tr]).argmax(1) == y[tr]).mean())
        if mode == "full":                                          # last epoch, no checkpoint selection
            sc = predict(X[te])
            pred = sc.argmax(1)
            r.update(acc=float((pred == y[te]).mean()), macro_f1=float(f1_score(y[te], pred, average="macro")))
            if proto:                                               # other metrics are recomputed from the predictions
                from train_main import metrics                      # + train_main metrics, including within-category
                r.update(proto=proto, test_idx=te.tolist(), pred=pred.tolist(),
                         metrics=metrics(sc, y[te], {k2: v[te] for k2, v in meta.items()}, names)[0])
                np.savez_compressed(os.path.join(out_dir, f"slot2_{mode}_{name}{tag}_f{k:02d}_scores.npz"), **{name: sc})
        print({k2: v for k2, v in r.items() if k2 not in ("sec_per_epoch", "last_loss", "test_idx", "pred", "metrics")}, flush=True)
        out.append(r)
        with open(os.path.join(out_dir, f"slot2_{mode}_{name}{tag}.json"), "w") as f:
            json.dump(out, f, indent=1)
        # one write, so the result can be recovered from the log even if the job is killed
        print(f"RESULT_JSON slot2_{mode}_{name}{tag}.json {json.dumps(out)}\n", end="", flush=True)


def demo():
    from pack_mp import frame150                                    # 75-point layout = data files of the SignBart repo
    a = np.zeros((1, 1662), np.float32)
    a[0, :132].reshape(33, 4)[:, :2] = np.c_[np.arange(33), 100 + np.arange(33)]
    a[0, -126:].reshape(2, 21, 3)[:, :, :2] = np.stack([np.c_[200 + np.arange(21), 300 + np.arange(21)],
                                                        np.c_[400 + np.arange(21), 500 + np.arange(21)]])
    m = mp75(frame150(a))[0]
    assert (m[:33] == np.c_[np.arange(33), 100 + np.arange(33)]).all()
    assert (m[33:54] == np.c_[200 + np.arange(21), 300 + np.arange(21)]).all() and (m[54:, 1] == 500 + np.arange(21)).all()
    g = np.zeros((42, 42), int)
    for c, p in [(c + o, p + o) for o in (0, 21) for c, p in HAND]:
        g[c, p] = g[p, c] = 1
    assert g.sum() == 2 * 42 and not g[:21, 21:].any()             # 21 bones per hand, two separate components
    reach = np.linalg.matrix_power(g[:21, :21] + np.eye(21, dtype=int), 21)
    assert (reach > 0).all()                                        # each hand is fully connected
    rng = np.random.default_rng(0)
    xb = rng.normal(size=(3, 30, 42, 2)).astype(np.float32)
    assert np.array_equal(dsta_input(xb, 0.25, False)[..., 0], xb.transpose(0, 3, 1, 2))
    random.seed(1)
    x = dsta_input(xb, 0.25, True)
    for b in range(3):                                              # exactly one: original or mirrored, then a constant shift
        cands = (xb[b].transpose(2, 0, 1), (xb[b][:, FLIP] * [-1, 1]).transpose(2, 0, 1))
        ok = [x[b, ..., 0] - c for c in cands]
        ok = [dd for dd in ok if all(np.allclose(dd[c], dd[c].flat[0], atol=1e-5) for c in (0, 1))]
        assert len(ok) == 1 and np.abs(ok[0][:, 0, 0]).max() <= DSTA["shift_px"] / DSTA["frame_px"] / 0.25 + 1e-6
    assert np.array_equal(FLIP[FLIP], np.arange(42))
    close = lambda a, b: np.allclose(a, b, rtol=0, atol=1e-12)
    assert close(dsta_schedule(0), (0.005, 1.0)) and close(dsta_schedule(19)[0], 0.1) and close(dsta_schedule(150)[0], 0.01)
    assert close(dsta_schedule(99)[1], 0.901) and close(dsta_schedule(200), (0.001, 0.9))
    lr50 = [dsta_schedule(e, 50)[0] for e in range(50)]             # 50 epochs: warmup 4, lr steps at 30 and 40
    assert close(lr50[2], 0.075) and close(lr50[3], 0.1) and close(lr50[29], 0.1) and close(lr50[30], 0.01)
    assert close(lr50[40], 0.001) and dsta_schedule(19, 50)[1] > 0.9 and close(dsta_schedule(20, 50)[1], 0.9)
    d = dict(relpath=np.array(["a", "b", "c"]), off=np.array([0, 2, 4, 5]), seq=np.zeros((5, 150), np.float32))
    p = d["seq"][:, :66].reshape(5, 33, 2)
    p[:, L_SH], p[:, R_SH] = (0.4, 0.5), (0.6, 0.5)                 # width 0.2
    p[4, R_SH] = (0.7, 0.5)                                         # clip c: width 0.3
    d["seq"][:, :66] = p.reshape(5, 66)
    assert abs(shoulder_width(d) - 0.2) < 1e-6                      # median of (0.2, 0.2, 0.3)
    # rtmw27: points, span cut, then the repo feeder
    assert SIGN27.tolist() == [0, 5, 6, 7, 8, 9, 10, 91, 95, 96, 99, 100, 103, 104, 107, 108, 111, 112, 116, 117, 120, 121,
                               124, 125, 128, 129, 132] and np.array_equal(FLIP27[FLIP27], np.arange(27))
    mps = dict(a=dict(relpath=np.array(["a/1", "a/2"]), off=np.array([0, 4, 10]), span=np.array([[1, 3], [0, 6]])))
    kp, sc = rng.random((10, 133, 2)).astype(np.float16), rng.random((10, 133)).astype(np.float16)
    rtmws = dict(a=dict(relpath=np.array(["a/2", "a/1"]), off=np.array([0, 6, 10]), kp=kp, sc=sc))   # different order
    c = rtmw27_clips(["a/1", "a/2"], mps, rtmws)
    assert c[0].shape == (2, 27, 3) and c[1].shape == (6, 27, 3)    # a/1 = RTMW frames 6-9, span [1, 3) -> 7, 8
    assert np.array_equal(c[0][..., :2], kp[7:9][:, SIGN27].astype(np.float32) * 512)
    assert np.array_equal(c[0][..., 2], sc[7:9][:, SIGN27].astype(np.float32))
    try:
        rtmw27_clips(["a/1"], dict(a=dict(mps["a"], off=np.array([0, 5, 10]))), rtmws)
        raise RuntimeError("frame count mismatch not caught")
    except AssertionError:
        pass
    if not os.path.isdir(REPOS["dsta"][0]):
        print("demo ok (rtmw27 feeder skipped: no DSTA-SLR clone)")
        return
    import ast
    tools = dsta_tools()
    tree = ast.parse(open(f"{REPOS['dsta'][0]}/feeders/feeder.py").read())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "flip_index")
    assert np.array_equal(FLIP27, np.concatenate([ast.literal_eval(e) for e in node.value.args[0].elts]))
    sys.path.insert(0, REPOS["dsta"][0])
    from graph.sign_27 import Graph
    g27 = Graph()
    assert g27.num_node == 27 and len(g27.inward) == 26
    assert {(FLIP27[i], FLIP27[j]) for i, j in g27.inward} == set(g27.inward)   # mirroring maps bones to bones
    v = np.arange(27, dtype=np.float32)
    for T in (20, 97):                                              # shorter and longer than the window
        t = np.arange(T, dtype=np.float32)[:, None]
        clip = np.stack([256 + v + 0 * t, 100 + t + 0 * v, 100 * t + v], 2)   # x = point, y = time, score = code
        te = dsta27_input([clip], False, tools)[0, ..., 0]          # C, 30, 27
        assert te.shape == (3, WINDOW, 27)
        assert np.array_equal(te[2, :, 0] // 100, [int(i * (T / WINDOW)) for i in range(WINDOW)])
        assert np.allclose(te[0], v) and np.array_equal(te[2] % 100, np.tile(v, (WINDOW, 1)))   # test: no mirroring
        assert abs(te[0, :, 0].mean()) < 1e-4 and abs(te[1, :, 0].mean()) < 1e-4   # nose at the origin
        random.seed(T)
        seen = set()
        for _ in range(20):
            tr = dsta27_input([clip], True, tools)[0, ..., 0]
            tt, src = tr[2] // 100, (tr[2] % 100).astype(int)
            assert (np.diff(tt[:, 0]) >= 0).all() and (tt == tt[:, :1]).all()   # time sorted, one frame per row
            mir = bool(np.array_equal(src[0], FLIP27))
            assert mir or np.array_equal(src[0], np.arange(27))
            assert np.allclose(tr[0] - tr[0][:, :1], -FLIP27 if mir else v)     # mirror: 512 - x, nodes swapped
            assert all(abs(tr[c, :, 0].mean()) <= DSTA["shift_px"] + 1e-4 for c in (0, 1))   # shift +-10 px
            seen.add(mir)
        assert seen == {True, False}
    print("demo ok: two-hand graph, DSTA-SLR input (mirror, shift), shoulder width, rtmw27 (points, span, repo feeder)")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3:])
