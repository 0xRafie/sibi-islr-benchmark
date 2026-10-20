"""SignBart (Nguyen and Phan Tran, 2025) for scripts/train_slot2.py: our input pipeline around the official model.

The official repository (https://github.com/TinhNguyen2312/SignBart) has no license file, so none of its code is
included here. build() imports the model (model.py with encoder.py, decoder.py, attention.py, layers.py, utils.py) at run
time from a local clone at the commit used in this work:

    git clone https://github.com/TinhNguyen2312/SignBart ext/SignBart
    git -C ext/SignBart checkout 1b7d3d1a1df11ee9147464143ea002d2b8688970

$SIGNBART_DIR points to the clone (default ext/SignBart in the repository root). The official utils.py imports cv2 and
tqdm, and the config is read with PyYAML, so opencv-python, tqdm, and pyyaml must be installed.

The input pipeline in this file (prepare, make_batch, augment, normalize) is our own re-implementation of the official
dataset.py and augmentations.py, with a seeded generator and an F0 variant. Model input: 48 points, MediaPipe pose
11-16 (shoulders, elbows, wrists), left hand 21, right hand 21. x goes to the encoder, y to the decoder.

    how = "f0":     models.f0(seq)[:, IDX]. Official augmentation in F0 units: mirror at x = 0, noise divided by W_SH.
    how = "native": order of dataset.py. prepare: 75 points in image coordinates, clipped to [0, 1]. make_batch: cut to
                    64 frames, augment, normalize a box per body part. The official code normalizes AFTER augmentation,
                    so make_batch recognizes native input by its number of points (75).

The reported SignBart results come from slot2.py (mp48), which runs the official dataset.py and model unchanged; this
module is exploratory. The demo reads real clips from $MP_AFFIX (default data/pilot/mp_affix.npz in the repository root):

    cd scripts && python -c "import models.signbart as m; m.demo()"
"""
import importlib, os, subprocess, sys, types

import numpy as np
import torch
import torch.nn.functional as F

from models import f0

NAME = "signbart"
COMMIT = "1b7d3d1a1df11ee9147464143ea002d2b8688970"
SOURCE = (f"github.com/TinhNguyen2312/SignBart @ {COMMIT}; Nguyen & Phan Tran, "
          "SignBart: New approach with the skeleton sequence for Isolated Sign Language Recognition, Springer 2026, "
          "doi 10.1007/978-3-032-10209-6_7 (arXiv 2506.21592)")
# AdamW with torch's default weight decay 0.01 (main.py:106; paper §4.1), lr 2e-4 and 200 epochs
# (scripts/train_LSA-64.sh:13-14), batch 1 (main.py:91-92), cross-entropy without smoothing (model.py:19), no gradient
# clipping (utils.py:73-77). ReduceLROnPlateau every epoch on the loss of the last batch (main.py:107-108,
# utils.py:88-89, train_LSA-64.sh:16-17). The repo picks the best epoch on the test split (main.py:90, 142-148); we use
# the last epoch.
RECIPE = dict(optimizer="adamw", lr=2e-4, weight_decay=0.01, epochs=200, batch=1, label_smoothing=0.0, clip_grad=None,
              reduce_on_plateau=dict(factor=0.1, patience=5, metric="loss of the last batch of the epoch"))
# configs/LSA-64.yaml at COMMIT (encoder = decoder), checked by build(). layerdrop 0.2 and pe "learn" are in the config
# but unused by the code
CONFIG = dict(d_model=128, encoder_layers=2, decoder_layers=2, encoder_attention_heads=16, decoder_attention_heads=16,
              encoder_ffn_dim=256, decoder_ffn_dim=256, dropout=0.0, attention_dropout=0.0, activation_dropout=0.1,
              classifier_dropout=0.4, max_position_embeddings=256)
IDX = list(range(11, 17)) + list(range(33, 75))      # joint_idx of LSA-64.yaml = utils.py:11-16
PARTS = (slice(0, 6), slice(6, 27), slice(27, 48))   # body, left hand, right hand, each with its own box (main.py:101)
MAX_T = 64                                           # dataset.py:42-43
W_SH = 0.35                                          # median per-clip shoulder width in mp_affix 0.348 (p5-p95 0.30-0.40)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIGNBART_DIR = os.environ.get("SIGNBART_DIR", os.path.join(ROOT, "ext", "SignBart"))


def prepare(seq, how):
    """seq150 -> f0: (T, 48, 2) F0-normalized; native: (T, 75, 2) image coordinates clipped to [0, 1].

    seq150 already has the (T, 75, 2) layout of the official repo: pose 33, left hand 21, right hand 21, 0 = not
    detected. Clipping to [0, 1] = dataset.py:41.
    """
    if how == "f0":
        return f0(seq)[:, IDX]
    assert how == "native", how
    return np.clip(seq.reshape(len(seq), 75, 2), 0, 1)


def lin(t, n):
    """n evenly spaced frame indices out of t frames, rounded down (augmentations.py:31, 43)."""
    return np.clip(np.linspace(0, t - 1, n).astype(int), 0, t - 1)


def normalize(x):
    """dataset.py:90-126, in place, per frame and body part. A square box with side 1.1 x the longest side (5% margin)
    and corners clipped to [0, 1], then (p - start) / (end - start) per axis, unless the box has width 0. Missing
    points (0) take part in the box; a part that is all 0 stays 0."""
    for p in PARTS:
        q = x[:, p]
        lo, hi = q.min(1, keepdims=True), q.max(1, keepdims=True)
        ext = hi - lo
        m = ext.max(-1, keepdims=True)
        d = 0.05 * m + (m - ext) / 2
        s, e = np.clip(lo - d, 0, 1), np.clip(hi + d, 0, 1)
        x[:, p] = np.where(e != s, (q - s) / np.where(e != s, e - s, 1), q)
    return x


def augment(x, rng, mid, k):
    """dataset.py:55-81 and augmentations.py. Repeated until at least one augmentation is picked. The order is fixed and
    each augmentation has probability 0.5:

    rotation U(-15, 15) degrees around the origin; Gaussian noise with sd U(0.01, 0.2) x k on all points, including the
    zeros; cut to U{T//2..T-1} frames (evenly spaced, or random with replacement and then sorted); stretch by
    U(1.1, 1.5) with repeated frames; mirror x -> 2 mid - x without swapping the left and right hands.

    Native (image coordinates): mid 0.5 and k 1, exactly as the official code. F0: the origin is the shoulder midpoint,
    so the rotation equals an image rotation followed by F0; mid 0 and k = 1 / W_SH, so that mirroring and the noise
    size are the same relative to the body.
    """
    done = False
    while not done:
        if rng.uniform(0, 1) < 0.5:
            a = np.radians(rng.uniform(-15, 15))
            x = np.einsum("ij,klj->kli", np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]), x)
            done = True
        if rng.uniform(0, 1) < 0.5:
            x = x + rng.normal(0, rng.uniform(0.01, 0.2) * k, x.shape)
            done = True
        if rng.uniform(0, 1) < 0.5:
            n = len(x)
            tgt = rng.integers(n // 2, n)
            x = x[lin(n, tgt) if rng.uniform(0, 1) < 0.5 else np.sort(rng.choice(np.arange(n), tgt, replace=True))]
            done = True
        if rng.uniform(0, 1) < 0.5:
            x = x[lin(len(x), int(len(x) * rng.uniform(1.1, 1.5)))]
            done = True
        if rng.uniform(0, 1) < 0.5:
            x = x * (-1, 1) + (2 * mid, 0)
            done = True
    return x


def make_batch(xs, train, rng):
    """Clips -> (points (B, T, 48, 2), mask (B, T)) float32, T = longest clip in the batch.

    dataset.py:42-48: clips longer than 64 frames are cut evenly to 64; in training, augmentation with probability 0.4;
    native input is then normalized. data_collator (dataset.py:136-164): zero padding at the end, mask 1 for real frames.
    """
    out = []
    for x in xs:
        raw = x.shape[1] == 75
        x = np.asarray(x, np.float64)
        if len(x) > MAX_T:
            x = x[lin(len(x), MAX_T)]
        if train and rng.uniform(0, 1) < 0.4:
            x = augment(x, rng, *((0.5, 1.0) if raw else (0.0, 1 / W_SH)))
        out.append(normalize(x[:, IDX]) if raw else x)
    T = max(map(len, out))
    kp, mask = np.zeros((len(out), T, 48, 2), np.float32), np.zeros((len(out), T), np.float32)
    for i, x in enumerate(out):
        kp[i, :len(x)], mask[i, :len(x)] = x, 1
    return torch.from_numpy(kp), torch.from_numpy(mask)


def official():
    """SignBart class of the official clone (model.py), files unchanged. decoder.py uses relative imports and the other
    files absolute ones, so the clone is loaded as a package and its absolute module names are aliased to the package
    modules (as slot2.import_signbart does)."""
    if "signbart_repo" not in sys.modules:
        head = subprocess.run(["git", "-C", SIGNBART_DIR, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != COMMIT:
            raise ImportError(
                f"SignBart model code is not included in this repository (the official repository has no license).\n"
                f"Clone it at the commit used in this work:\n"
                f"  git clone https://github.com/TinhNguyen2312/SignBart {SIGNBART_DIR}\n"
                f"  git -C {SIGNBART_DIR} checkout {COMMIT}\n"
                f"or set SIGNBART_DIR to such a clone (found: {head or 'no git checkout'} at {SIGNBART_DIR}).")
        pkg = types.ModuleType("signbart_repo")
        pkg.__path__ = [os.path.abspath(SIGNBART_DIR)]
        sys.modules["signbart_repo"] = pkg
        for m in ("utils", "layers", "attention", "encoder", "decoder", "model"):
            sys.modules[m] = importlib.import_module(f"signbart_repo.{m}")
    return sys.modules["signbart_repo.model"].SignBart


def build(n_cls):
    """Official SignBart with configs/LSA-64.yaml, n_cls classes, on the 48 points already selected by make_batch.

    forward(kp, mask) -> logits chains the official modules as model.py does, but finds the last real frame with an
    integer cumsum: the float cumsum of model.py:38 is not deterministic on CUDA."""
    import yaml
    base = official()
    with open(os.path.join(SIGNBART_DIR, "configs", "LSA-64.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert {k: cfg[k] for k in CONFIG} == CONFIG, f"configs/LSA-64.yaml differs from CONFIG: {cfg}"
    cfg.update(joint_idx=list(range(len(IDX))), num_labels=n_cls)

    class SignBart(base):
        def forward(self, kp, mask):
            x, y = self.projection(kp)
            enc = self.encoder(x_embed=x, attention_mask=mask)
            dec = self.decoder(encoder_hidden_states=enc, encoder_attention_mask=mask, attention_mask=mask, y_embed=y)
            last = (mask == 1).long().cumsum(1).argmax(1)
            return self.classification_head(dec[torch.arange(len(dec), device=dec.device), last])

    return SignBart(cfg)


def demo():
    """Real mp_affix clips: shapes of prepare and make_batch, normalization, augmentation, forward + backward,
    determinism. Needs torch, the official clone, and $MP_AFFIX."""
    z = dict(np.load(os.environ.get("MP_AFFIX", f"{ROOT}/data/pilot/mp_affix.npz")))
    seqs = [z["seq"][z["off"][i]:z["off"][i + 1]][slice(*z["span"][i])] for i in range(8)]
    y = torch.tensor(np.unique(z["cls"], return_inverse=True)[1][:8])
    lens, T0 = [min(len(q), MAX_T) for q in seqs], min(len(seqs[0]), MAX_T)
    # right hand: longest side [0.2, 0.6] -> [0.05/1.1, 1.05/1.1], short side centered in the box; the missing left
    # hand stays 0; six coincident body points (box of width 0) are unchanged
    s = np.zeros((1, 150), np.float32)
    s[0, 108::2], s[0, 109::2], s[0, 22:34] = np.linspace(0.2, 0.6, 21), 0.5, 0.5
    k = make_batch([prepare(s, "native")], False, None)[0][0, 0].numpy()
    assert np.isclose(k[27:, 0].min(), 0.05 / 1.1) and np.isclose(k[27:, 0].max(), 1.05 / 1.1)
    assert np.allclose(k[27:, 1], 0.5) and np.all(k[6:27] == 0) and np.all(k[:6] == 0.5)
    assert np.array_equal(prepare(seqs[0], "f0"), f0(seqs[0])[:, IDX])
    rng = np.random.default_rng(0)
    for how, n in (("f0", 48), ("native", 75)):
        xs = [prepare(q, how) for q in seqs]
        before = [x.copy() for x in xs]
        assert xs[0].shape == (len(seqs[0]), n, 2) and xs[0].dtype == np.float32
        kp, mask = make_batch(xs, False, rng)
        assert kp.shape == (8, max(lens), 48, 2) and mask.sum(1).tolist() == lens
        assert how == "f0" or 0 <= kp.min() <= kp.max() <= 1                     # image points inside their box
        same = 0
        for _ in range(300):
            kt, mt = make_batch(xs[:1], True, rng)
            assert T0 // 2 <= mt.shape[1] <= int(1.5 * T0) and mt.all() and torch.isfinite(kt).all()
            same += torch.equal(kt, kp[:1, :T0])
        assert 0.5 < same / 300 < 0.7, same                                       # no augmentation: probability 0.6
        a, b = make_batch(xs, True, np.random.default_rng(1)), make_batch(xs, True, np.random.default_rng(1))
        assert all(torch.equal(u, v) for u, v in zip(a, b)) and all(map(np.array_equal, xs, before))
    torch.manual_seed(0)
    net = build(18)
    n_par = sum(p.numel() for p in net.parameters())
    assert n_par == 743954 and sum(p.numel() for p in build(64).parameters()) == 749888   # 749,888: SignBart README, paper
    out = net(*make_batch(xs, True, rng))
    F.cross_entropy(out, y).backward()
    assert out.shape == (8, 18) and all(torch.isfinite(p.grad).all() for p in net.parameters())
    torch.manual_seed(0)
    twin, j = build(18).eval(), int(np.argmin(lens))
    with torch.no_grad():
        e1, e2 = net.eval()(kp, mask), twin(kp, mask)
        alone = net(kp[j:j + 1, :lens[j]], mask[j:j + 1, :lens[j]])
    assert torch.equal(e1, e2) and torch.equal(e1, net(kp, mask).detach())         # same seed -> same eval output
    assert torch.allclose(alone, e1[j:j + 1], atol=1e-5)                            # padding + mask = clip on its own
    print(f"demo ok: {n_par:,} parameters (18 classes; 749,888 for the 64 classes of LSA-64), prepare f0 (T, 48, 2) and "
          f"native (T, 75, 2), batch {tuple(kp.shape)}, padding = clip on its own, deterministic eval")
