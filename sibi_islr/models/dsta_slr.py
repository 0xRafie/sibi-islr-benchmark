# sibi_islr/models/dsta_slr.py and dsta_slr_network.py are adapted from DSTA-SLR:
#   https://github.com/hulianyuyy/DSTA-SLR, commit e7e5ee225511488039a4f68ab3a146cf8f01312d
#   Hu et al., "Dynamic Spatial-Temporal Aggregation for Skeleton-Aware Sign Language Recognition",
#   LREC-COLING 2024 (arXiv:2403.12519).
# DSTA-SLR is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0),
# https://creativecommons.org/licenses/by-nc/4.0/. This modified version is distributed under the same license, for
# non-commercial use only. The MIT license of the rest of this repository does not apply to these two files.
#
# Modifications by the SIBI-ISLR Benchmark authors (2026):
# - copied from model/fstgan.py, model/attention.py, model/dropSke.py, model/dropT.py, graph/sign_27.py, and
#   graph/tools.py into one file (now split into dsta_slr.py and
#   dsta_slr_network.py); only the joint stream is kept;
# - in_channels = 2 (xy input; the HRNet score channel is dropped) and the number of classes is an argument;
# - parameters are created on the module's device (the original forces device="cuda");
# - classes not used by Model (unit_tan, global_tan, Edge_conv, MHSA, ...) are not copied;
# - einops.rearrange is replaced by the same permute/reshape (tok, untok), and timm's trunc_normal_ by
#   torch.nn.init.trunc_normal_ (same algorithm);
# - graph/sign_27.py and graph/tools.py are reduced to EDGES and graph_a() (labeling mode "spatial" only);
# - DropBlock_Ske keeps only the branch used for 27 points, without its warning; console prints are removed;
# - new input code (prepare, make_batch) that follows feeders/feeder.py; comments rewritten.
"""DSTA-SLR (Hu et al., LREC-COLING 2024) for scripts/train_slot2.py: joint stream only, 27 points, xy input.

The reported DSTA-SLR results come from slot2.py (rtmw27), which runs the official code at the pinned commit; this
module is an exploratory adapted copy.

- Points: COCO-WholeBody [0, 5..10] + hands 91/112 + [0, 4, 5, 8, 9, 12, 13, 16, 17, 20] (SAM-SLR-v2
  data_gen/sign_gendata.py:13-14) = MediaPipe pose [0, 11..16] + left/right hand with the same numbers. Left = the
  signer's anatomical left in both systems. The SAM-SLR extractor flips every frame before HRNet (data-prepare
  wholepose/demo.py:150), so in the official data the "left" nodes hold the signer's right hand. That global flip is not
  reproduced: with 50% random mirroring in training, the training data has the same distribution.
- Time: clips are looped to 150 frames (sign_gendata.py:50-63), then 120 frames are drawn at random without replacement
  in training (tools.random_sample_np) or int(i * 150 / 120) at test (tools.uniform_sample_np), feeder.py:234-237.
- Training augmentation (feeder.py:245-281): mirror with p 0.5 (swap left and right, flip x), then a global shift
  U(-10, 10) pixels of a 512 frame in x and y, the same for all points and frames (paper: "random noise of maximum value
  of 20"). No rotation, scaling, or per-point noise (random_move is off in the config).

The demo reads real clips from $MP_AFFIX (default data/pilot/mp_affix.npz in the repository root):

    cd scripts && python -c "import models.dsta_slr as m; m.demo()"
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sibi_islr.models.common import f0

from sibi_islr.models.dsta_slr_network import EDGES, Model, graph_a


NAME = "dsta_slr"
SOURCE = ("https://github.com/hulianyuyy/DSTA-SLR @ e7e5ee225511488039a4f68ab3a146cf8f01312d (config/train.yaml, "
          "WLASL2000 joint); Hu et al., Dynamic Spatial-Temporal Aggregation for Skeleton-Aware Sign Language "
          "Recognition, LREC-COLING 2024")

HAND10 = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]                      # hand numbers, the same in COCO-WholeBody and MediaPipe
IDX27 = [0, 11, 12, 13, 14, 15, 16] + [33 + k for k in HAND10] + [54 + k for k in HAND10]   # indices into the f0() output
FLIP = np.r_[[0, 2, 1, 4, 3, 6, 5], 17:27, 7:17]                 # feeder.py:15-22
PAD, WINDOW = 150, 120                                            # sign_gendata.py:18, config/train.yaml:9
FRAME = 512                                                       # feeder.py:253, "input size 512*512"
# SW_OFF: typical shoulder width of the official data (WLASL2000) in pixels of a 512 frame, from data_bn of the official
# checkpoint (0.4732 x 256)
SW_OFF = 121.0
SHIFT = 10 / SW_OFF  # shift U(-10, 10) pixels (feeder.py:274-281) in shoulder widths = the unit of both inputs

# config/train.yaml:44-59; SGD in main.py:346-368 (momentum 0.9, wd 1e-4 for all parameters, nesterov from the config)
RECIPE = dict(optimizer="sgd", lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-4,
              schedule=("multistep", [150, 200], 0.1), epochs=250, batch=24, label_smoothing=0.0, clip_grad=None,
              warmup=20,                   # main.py:406-416: lr = 0.1 * (e + 1) / 20 for epoch e < 20 (per epoch)
              keep_prob=(1.0, 0.9, 100),   # main.py:471-475: 2nd forward argument, 1.0 -> 0.9 linearly until epoch 100
              drop_last=True,              # main.py:266
              aug=f"loop {PAD}, take {WINDOW}, mirror 0.5, shift U(+-{SHIFT:.4f}) x and y")


def prepare(seq, how):
    """seq150 -> (T, 27, 2) float32, full length.

    f0: models.f0, in units of the clip's shoulder width. native: the repo normalization of the raw coordinates.
    MediaPipe coordinates x 512 = pixels of the 512 x 512 frame the repo assumes (the dataset videos are square), minus
    the clip's mean nose position (feeder.py:256-272), then divided by SW_OFF so that one shift constant fits both
    inputs. data_bn normalizes each channel, so this constant scale does not change the model. Missing points (0) are
    used as they are, as in the repo: HRNet always outputs coordinates, so the repo has no rule for missing points. The
    mean nose position is taken over the clip's frames, not over 120 frames drawn from the 150 looped frames as in the
    repo (difference <= 0.6 pixel@512, median 0.07, against a shift of +-10 pixels).
    """
    if how == "f0":
        return f0(seq)[:, IDX27]
    if how == "native":
        x = seq.reshape(len(seq), 75, 2)[:, IDX27] * FRAME
        return ((x - x[:, 0].mean(0)) / SW_OFF).astype(np.float32)
    raise ValueError(how)


def make_batch(xs, train, rng):
    """List of (T, 27, 2) -> (B, 2, 120, 27, 1). Test: deterministic, no augmentation."""
    out = np.empty((len(xs), WINDOW, 27, 2), np.float32)
    for i, x in enumerate(xs):
        x = x[np.arange(PAD) % len(x)]                                  # loop to 150; T >= 150: first 150
        if train:
            x = x[np.sort(rng.choice(PAD, WINDOW, replace=False))]    # random_sample_np, interval ceil(120/150) = 1
            if rng.random() > 0.5:
                x = x[:, FLIP] * np.float32([-1, 1])                    # 512 - x, then centered = -x centered
            x = x + rng.uniform(-SHIFT, SHIFT, 2).astype(np.float32)
        else:
            x = x[(np.arange(WINDOW) * (PAD / WINDOW)).astype(int)]    # uniform_sample_np
        out[i] = x
    return torch.from_numpy(out).permute(0, 3, 1, 2)[..., None].contiguous()


def build(n_cls):
    """Model of config/train.yaml:30-42, in_channels 2."""
    return Model(n_cls)


def demo():
    torch.use_deterministic_algorithms(True)
    z = dict(np.load(os.environ.get("MP_AFFIX", os.path.join(os.path.dirname(__file__), "../../data/pilot/mp_affix.npz"))))
    pick = [0, 1000, 2000, 3000]
    seqs = []
    for i in pick:
        a, b = z["span"][i]
        seqs.append(np.array(z["seq"][z["off"][i]:z["off"][i + 1]][a:b]))
    s = seqs[0]
    T = len(s)
    xf, xn = prepare(s, "f0"), prepare(s, "native")
    assert xf.shape == xn.shape == (T, 27, 2) and xf.dtype == xn.dtype == np.float32
    raw = s.reshape(T, 75, 2)
    assert np.array_equal(raw[:, 12], s[:, 24:26]) and np.array_equal(raw[:, 33 + 4], s[:, 66 + 8:66 + 10])
    assert np.array_equal(raw[:, 54 + 20], s[:, 148:150])        # IDX27 points to the right seq150 columns
    assert np.allclose(xn[:, 0].mean(0), 0, atol=1e-5)            # native: mean nose position at 0
    assert np.allclose(xn * SW_OFF, (raw[:, IDX27] - raw[:, 0].mean(0)) * FRAME, atol=1e-3)
    assert np.all(xn[:, 1, 0] > xn[:, 2, 0])                       # MediaPipe left shoulder on the image right, as in COCO
    assert np.array_equal(FLIP[FLIP], np.arange(27))                # mirroring twice = identity
    A = graph_a()
    sup = A.sum(0) > 0
    assert A.shape == (3, 27, 27) and len(EDGES) == 26 and np.array_equal(sup, sup.T) and sup.sum() == 27 + 2 * 26
    xs = [prepare(x, "f0") for x in seqs]
    rg = np.random.default_rng(0)
    bt, be, be2 = make_batch(xs, True, rg), make_batch(xs, False, rg), make_batch(xs, False, rg)
    assert bt.shape == be.shape == (4, 2, WINDOW, 27, 1) and bt.dtype == torch.float32 and torch.equal(be, be2)
    loop = xs[0][np.arange(PAD) % T]                                 # T < 150: the clip is looped
    assert np.array_equal(loop[T:2 * T], xs[0]) and loop.shape == (PAD, 27, 2)
    assert torch.equal(be[0, ..., 0], torch.from_numpy(loop[(np.arange(WINDOW) * 1.25).astype(int)]).permute(2, 0, 1))
    flips = set()
    for seed in range(8):                                            # replay the rng draws: frames, mirror, shift
        r = np.random.default_rng(seed)
        ex = loop[np.sort(r.choice(PAD, WINDOW, replace=False))]
        flip = r.random() > 0.5
        ex = (ex[:, FLIP] * np.float32([-1, 1]) if flip else ex) + r.uniform(-SHIFT, SHIFT, 2).astype(np.float32)
        got = make_batch([xs[0]], True, np.random.default_rng(seed))[0, ..., 0].permute(1, 2, 0).numpy()
        assert np.allclose(got, ex)
        flips.add(flip)
    assert flips == {True, False}
    torch.manual_seed(0)
    net = build(18)
    n_par = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in net.parameters())
    assert not any("DecoupleA" in k for k, _ in net.named_parameters())     # only_train_epoch has no effect
    y = torch.tensor([0, 5, 10, 17])
    net.train()
    loss = F.cross_entropy(net(bt, 0.9), y)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in net.parameters() if p.requires_grad)
    net.eval()
    with torch.no_grad():
        o1, o2 = net(be), net(be)
    assert o1.shape == (4, 18) and torch.equal(o1, o2)
    bn = make_batch([prepare(x, "native") for x in seqs], False, rg)
    with torch.no_grad():
        assert torch.isfinite(net(bn)).all()
    print(f"demo ok: prepare f0/native (T, 27, 2), batch {tuple(bt.shape)}, loss {loss.item():.3f}, "
          f"{n_par:,} trainable parameters ({n_all:,} in total, including the frozen A), deterministic test")
