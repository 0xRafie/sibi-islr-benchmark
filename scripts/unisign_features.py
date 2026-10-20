"""Frozen Uni-Sign features for the linear/MLP heads of train_heads.py. This script is the encoder side only: RTMW
keypoints -> two 768-d vectors per clip, in one pass, without gradients.

Input `rtmw_<category>.npz` (RTMW keypoint extraction): kp float16 (frames, 133, 2) divided by (W, H), sc float16
(frames, 133), off (clips + 1), relpath. The Uni-Sign code is used directly from a clone in $UNISIGN_DIR (default
ext/Uni-Sign in the repository root), not copied:

    git clone https://github.com/ZechengLi19/Uni-Sign ext/Uni-Sign
    git -C ext/Uni-Sign checkout eed438bcb49e30405cd6ccdfcccca330c134e830

- datasets.py: load_part_kp + crop_scale (body/left/right/face_all, threshold 0.3) and Base_Dataset.collate_fn, executed
  from their source through ast, because importing datasets.py pulls in deepspeed (through utils.py), decord, and
  torchvision;
- models.py: Uni_Sign with the ISLR WLASL pose-only args (hidden_dim 256, rgb_support False, dataset WLASL), loaded
  through importlib under its own name, because `models` is taken by the scripts/models package;
- weights: wlasl_pose_only_islr.pth (Hugging Face ZechengLi19/Uni-Sign), strict. The checkpoint holds the pose encoder
  and the whole mT5, so only the config and the tokenizer are needed from google/mt5-base.
Maximum length 64 frames (ISLR example, script/train_stage3.sh:29; utils.py default 256). Uni-Sign draws 64 random
frames and sorts them, also at test time (datasets.py:472-473); here 64 evenly spaced frames keep the features
deterministic.

Two representations per clip, averaged over the valid frames:
  pose  output of pose_proj, i.e. the pose tokens before the text prefix (models.py:266-267)
  mt5   last hidden state of the mT5 encoder over [prefix + pose tokens] (models.py:269-295), pose-token positions only
Uni-Sign's own ISLR inference has no classification head: the mT5 decoder writes a gloss (beam 4) while cross-attending
to the whole mT5 encoder output, and the gloss is matched as a string (fine_tuning.py:263-266, 302-303).

Check: full Uni-Sign ISLR inference (forward + beam 4) through parts() and collate_fn of this file, on 24 randomly drawn
official WLASL test poses: 19/24 correct (paper, pose-only WLASL2000 P-I: 63.13%). transformers 4.40.0 vs 5.18.0:
identical pose features, mt5 max difference 2e-7.

    python scripts/unisign_features.py --demo [ckpt]
    python scripts/unisign_features.py extract <rtmw.npz> <out.npz> [--device cuda|cpu] [--limit N] [--max-len 64]
                                               [--ckpt P] [--mt5 P] [--drop face|thumb] [--spans mp_<cat>.npz]

--ckpt default: $UNISIGN_CKPT, else downloaded from Hugging Face. --mt5 default: $UNISIGN_MT5 or google/mt5-base (folder
or Hugging Face id). Needs transformers, sentencepiece, protobuf, einops, and torchvision, e.g. outside the project
environment:
    uv run --no-project --python 3.12 --with torch==2.10.0 --with numpy==2.0.2 --with transformers \\
        --with sentencepiece --with protobuf --with einops --with torchvision python scripts/unisign_features.py --demo
The demo reads real clips from $RTMW_AFFIX (default data/pilot/rtmw_affix.npz in the repository root).
"""
import argparse, ast, collections, copy, functools, importlib.util, os, sys, time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; set before torch is imported
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
UNISIGN = Path(os.environ.get("UNISIGN_DIR", ROOT / "ext" / "Uni-Sign"))
HF_REPO, CKPT_FILE = "ZechengLi19/Uni-Sign", "wlasl_pose_only_islr.pth"
MT5 = os.environ.get("UNISIGN_MT5", "google/mt5-base")
MAX_LEN = 64
# args read by Uni_Sign for ISLR WLASL pose-only (utils.get_args_parser, script/train_stage3.sh:19-30)
ARGS = SimpleNamespace(hidden_dim=256, rgb_support=False, dataset="WLASL", task="ISLR", label_smoothing=0.2,
                       max_length=MAX_LEN)
PARTS = ("body", "left", "right", "face_all")
PREFIX = "Translate sign language video to English: "      # models.py:270; English because "CSL" is not in "WLASL"
# ▁Translate ▁sign ▁language ▁video ▁to ▁English : </s>, the same in T5Tokenizer(legacy=False) 4.40.0 (Uni-Sign's pin) and 5.18.0
PREFIX_IDS = [89349, 11329, 17896, 1552, 288, 5413, 267, 1]


@functools.cache
def _ds():
    """load_part_kp, crop_scale, and collate_fn of Uni-Sign's datasets.py, executed from its source (original line
    numbers)."""
    path = UNISIGN / "datasets.py"
    want = {"load_part_kp", "crop_scale", "collate_fn"}
    nodes = [n for n in ast.walk(ast.parse(path.read_text(), str(path)))
             if isinstance(n, ast.FunctionDef) and n.name in want]
    assert sorted(n.name for n in nodes) == sorted(want), "Uni-Sign functions not found in datasets.py"
    ns = {"np": np, "copy": copy, "torch": torch, "pad_sequence": pad_sequence}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return SimpleNamespace(**{k: ns[k] for k in want})


def frame_index(n, max_len=MAX_LEN):
    """All frames if n <= max_len (datasets.py:474-475); otherwise max_len evenly spaced frames, replacing the sorted
    random.sample (datasets.py:472-473)."""
    return np.arange(n) if n <= max_len else np.linspace(0, n - 1, max_len).round().astype(int)


def parts(kp, sc, max_len=MAX_LEN):
    """One clip kp (T, 133, 2) + sc (T, 133) -> {body (T', 9, 3), left/right (T', 21, 3), face_all (T', 18, 3)} float64
    tensors, through Uni-Sign's load_part_kp with force_ok=True as in S2T_Dataset.load_pose (datasets.py:490). Data types
    follow Uni-Sign's pkl files: keypoints float64 (divided by (W, H)), scores float32; the person axis (1) is added."""
    i = frame_index(len(kp), max_len)
    return _ds().load_part_kp(np.asarray(kp, np.float64)[i, None], np.asarray(sc, np.float32)[i, None], force_ok=True)


THUMBS = [92, 93, 94, 95, 113, 114, 115, 116]                 # points 1-4 of the left (91-111) and right (112-132) hand


def ablate(kp, sc, off, rel, drop=None, spans=None):
    """Ablations without changing the input file. drop="face": COCO-WholeBody face scores (23-90) = 0; drop="thumb":
    thumb scores = 0. spans=mp_<cat>.npz: each clip is cut to its MediaPipe span [a, b)."""
    sc = sc.copy()
    if drop == "face":
        sc[:, 23:91] = 0
    elif drop == "thumb":
        sc[:, THUMBS] = 0
    if spans:
        m = np.load(spans)
        pos = {p: i for i, p in enumerate(m["relpath"])}
        j = np.array([pos[p] for p in rel])
        assert (np.diff(m["off"])[j] == np.diff(off)).all(), "RTMW frame count != MediaPipe"
        sp = m["span"][j]
        keep = np.concatenate([np.arange(off[i] + a, off[i] + b) for i, (a, b) in enumerate(sp)])
        kp, sc = kp[keep], sc[keep]
        off = np.concatenate([[0], np.cumsum(sp[:, 1] - sp[:, 0])]).astype(off.dtype)
    return kp, sc, off


def check_ablate():
    """numpy check of ablate(): face/thumb scores zero, other scores intact, span cut, on 2 synthetic clips."""
    import tempfile
    rng = np.random.default_rng(0)
    off, rel = np.array([0, 5, 12]), np.array(["a.mp4", "b.mp4"])
    kp, sc = rng.uniform(0, 1, (12, 133, 2)), rng.uniform(0.5, 1, (12, 133))
    for drop, idx in (("face", list(range(23, 91))), ("thumb", THUMBS)):
        k2, s2, o2 = ablate(kp, sc, off, rel, drop)
        rest = np.setdiff1d(np.arange(133), idx)
        assert (s2[:, idx] == 0).all() and (s2[:, rest] == sc[:, rest]).all() and (sc > 0).all()
        assert k2 is kp and (o2 == off).all()
    with tempfile.TemporaryDirectory() as t:
        f = f"{t}/mp.npz"
        np.savez(f, relpath=rel[::-1], off=np.array([0, 7, 12]), span=np.array([[2, 6], [1, 4]]))
        k3, s3, o3 = ablate(kp, sc, off, rel, None, f)
    assert (o3 == [0, 3, 7]).all() and (k3[:3] == kp[1:4]).all() and (k3[3:] == kp[7:11]).all() and (s3[3:] == sc[7:11]).all()
    print("ablate ok")


def _models():
    """Uni-Sign's models.py under its own module name; stgcn_layers, deformable_attention_2d, config come from Uni-Sign."""
    sys.path.insert(0, str(UNISIGN))
    dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True      # keep the Uni-Sign clone clean (no .pyc)
    try:
        spec = importlib.util.spec_from_file_location("unisign_models", UNISIGN / "models.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(UNISIGN))
        sys.dont_write_bytecode = dwb
    return mod


class _Embeds(Exception):
    """Raised by a pre-hook just before mT5 is called (models.py:291), carrying inputs_embeds and attention_mask."""


def _stop(module, args, kwargs):
    raise _Embeds(kwargs["inputs_embeds"], kwargs["attention_mask"])


class UniSignEncoder(torch.nn.Module):
    """Original Uni_Sign + wlasl_pose_only_islr.pth, frozen, float32. The mT5 decoder and lm_head are dropped after the
    key check: the features only need the pose encoder, the mT5 embedding (prefix), and the mT5 encoder."""

    def __init__(self, ckpt=None, mt5=MT5, device="cpu"):
        super().__init__()
        from transformers import MT5Config, MT5ForConditionalGeneration
        um = _models()
        um.mt5_path = mt5                                    # config.py: ./pretrained_weight/mt5-base

        class FromConfig:                                    # mT5 weights come from the checkpoint, not from mt5-base
            @staticmethod
            def from_pretrained(path):
                return MT5ForConditionalGeneration(MT5Config.from_pretrained(path))

        um.MT5ForConditionalGeneration = FromConfig
        with torch.device("meta"):                           # no allocation; weights are assigned from the checkpoint
            uni = um.Uni_Sign(ARGS)
        if ckpt is None:
            from huggingface_hub import hf_hub_download
            ckpt = hf_hub_download(HF_REPO, CKPT_FILE)
        try:
            sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)["model"]
        except RuntimeError:                                 # not a zip file: load without mmap
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
        want = uni.state_dict()
        top = collections.Counter()
        for k, v in sd.items():
            top[k.split(".")[0]] += v.numel()
        self.report = dict(ckpt=str(ckpt), n_ckpt=len(sd), n_model=len(want), params=dict(top),
                           dtypes=sorted({str(v.dtype) for v in sd.values()}),
                           missing=sorted(set(want) - set(sd)), unexpected=sorted(set(sd) - set(want)),
                           shape=sorted(k for k in set(want) & set(sd) if want[k].shape != sd[k].shape))
        self.report["strict"] = not (self.report["missing"] or self.report["unexpected"] or self.report["shape"])
        uni.load_state_dict(sd, strict=self.report["strict"], assign=True)
        del sd
        # assign unties shared/encoder.embed_tokens in transformers 5; the prefix only goes through encoder.embed_tokens
        uni.mt5_model.shared = uni.mt5_model.decoder = uni.mt5_model.lm_head = None
        meta = [k for k, v in uni.state_dict().items() if v.is_meta]
        assert not meta, f"weights in use are missing from the checkpoint: {meta[:5]}"
        ids = list(uni.mt5_tokenizer([PREFIX])["input_ids"][0])
        assert ids == PREFIX_IDS, f"prefix tokenization differs from transformers 4.40.0: {ids}"
        self.n_prefix = len(ids)
        self.uni = uni.float().eval().requires_grad_(False).to(device)
        self.uni.mt5_model.register_forward_pre_hook(_stop, with_kwargs=True)
        self.device = torch.device(device)

    @torch.inference_mode()
    def forward(self, src, tgt):
        """src, tgt from Uni-Sign's collate_fn -> (pose, mt5), each numpy (B, 768) float32."""
        src = {k: (v.to(self.device, torch.float32) if k in PARTS else v.to(self.device)) if torch.is_tensor(v) else v
               for k, v in src.items()}
        try:
            self.uni(src, tgt)                               # Uni_Sign.forward up to just before mT5
            raise RuntimeError("mT5 pre-hook was not called")
        except _Embeds as e:
            x, mask = e.args                                 # [prefix + pose tokens], mask [prefix + valid frames]
        assert x.shape[1] == self.n_prefix + src["attention_mask"].shape[1]
        h = self.uni.mt5_model.encoder(inputs_embeds=x, attention_mask=mask).last_hidden_state  # what the decoder reads
        w = src["attention_mask"][..., None].float()
        pool = lambda z: ((z[:, self.n_prefix:] * w).sum(1) / w.sum(1)).float().cpu().numpy()
        return pool(x), pool(h)


def features(model, batch):
    """batch: list of parts() outputs -> (pose, mt5) numpy (B, 768). Different lengths are padded with the last frame as
    in Uni-Sign's collate_fn (datasets.py:353-359), and that padding also shifts the temporal convolution output at the
    end of a clip. extract() groups clips of equal length, so the result equals batch 1 (demo online_inference.py)."""
    src, tgt = _ds().collate_fn(SimpleNamespace(rgb_support=False),
                                [(str(i), p, "", "", {}) for i, p in enumerate(batch)])
    return model(src, tgt)


def _print_report(r):
    print(f"checkpoint {Path(r['ckpt']).name}: {r['n_ckpt']} keys, dtype {r['dtypes']}, elements per module (tied "
          "tensors counted per name) " + ", ".join(f"{k} {v / 1e6:.2f} M" for k, v in r["params"].items()))
    print(f"  model {r['n_model']} keys; strict={r['strict']}; missing {len(r['missing'])} {r['missing'][:5]}; "
          f"unexpected {len(r['unexpected'])} {r['unexpected'][:5]}; shape mismatch {r['shape'][:5]}")


def extract(src, out, device=None, limit=None, ckpt=None, mt5=MT5, max_len=MAX_LEN, batch=64, drop=None, spans=None):
    """The first `limit` clips (all if None) -> npz pose (n, 768), mt5 (n, 768) float32, relpath."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested, but CUDA is not available")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    import transformers
    print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "device",
          device, torch.cuda.get_device_name(0) if device == "cuda" else "")
    d = np.load(src)
    kp, sc, off, rel = d["kp"], d["sc"], d["off"], d["relpath"]   # once: each NpzFile access decompresses again
    if drop or spans:
        check_ablate()
        kp, sc, off = ablate(kp, sc, off, rel, drop, spans)
        print(f"ablation drop={drop} spans={spans}: {len(kp)} frames")
    n = len(rel) if limit is None else min(limit, len(rel))
    t0 = time.time()
    model = UniSignEncoder(ckpt, mt5, device)
    _print_report(model.report)
    lens = np.minimum(np.diff(off)[:n], max_len)
    pose, enc = np.zeros((n, 768), np.float32), np.zeros((n, 768), np.float32)
    t1, done = time.time(), 0
    for L in np.unique(lens):                                # equal length: no padding, result = batch 1
        ids = np.flatnonzero(lens == L)
        for j in range(0, len(ids), batch):
            b = ids[j:j + batch]
            pose[b], enc[b] = features(model, [parts(kp[off[i]:off[i + 1]], sc[off[i]:off[i + 1]], max_len)
                                               for i in b])
            done += len(b)
        if L % 8 == 0:
            print(f"  {done}/{n} clips, {(time.time() - t1) / done:.3f} s/clip", flush=True)
    dt = time.time() - t1
    assert np.isfinite(pose).all() and np.isfinite(enc).all()
    np.savez(out, pose=pose, mt5=enc, relpath=rel[:n])
    print(f"{n} clips in {dt:.1f} s = {dt / n:.3f} s/clip ({device}); model loaded in {t1 - t0:.1f} s -> {out}")


def demo(ckpt=None):
    # 1. parts() on a synthetic 80-frame clip: sampling, shapes, crop_scale of the body, relative hands/face, mask
    rng = np.random.default_rng(0)
    T = 80
    kp = rng.uniform(0.2, 0.8, (T, 133, 2)).astype(np.float16)
    sc = rng.uniform(0.0, 1.3, (T, 133)).astype(np.float16)  # RTMW gives scores > 1; Uni-Sign clips them at 1
    i = frame_index(T)
    assert len(i) == MAX_LEN and i[0] == 0 and i[-1] == T - 1 and (np.diff(i) > 0).all()
    assert (frame_index(50) == np.arange(50)).all()
    p = parts(kp, sc)
    shapes = {k: tuple(v.shape) for k, v in p.items()}
    assert shapes == {"body": (64, 9, 3), "left": (64, 21, 3), "right": (64, 21, 3), "face_all": (64, 18, 3)}, shapes
    assert all(v.dtype == torch.float64 and v.abs().max() <= 1 for v in p.values())
    k, s = kp[i].astype(np.float64), sc[i].astype(np.float32)
    idx = {"body": [0] + list(range(3, 11)), "left": list(range(91, 112)), "right": list(range(112, 133)),
           "face_all": list(range(23, 40, 2)) + list(range(83, 91)) + [53]}
    ok = {q: s[:, j] > 0.3 for q, j in idx.items()}
    xy = k[:, idx["body"]][ok["body"]]
    side = xy.max(0) - xy.min(0)
    scale = side.max()                                       # crop_scale: longest side of the box of valid points in the clip
    b = p["body"].numpy()[ok["body"]]
    assert np.allclose(b[:, side.argmax()].min(), -1) and np.allclose(b[:, side.argmax()].max(), 1)
    for q, root in (("left", 0), ("right", 0), ("face_all", -1)):   # wrist; nose tip (point 53)
        rel_xy = np.clip((k[:, idx[q]] - k[:, idx[q]][:, [root]]) / scale, -1, 1)
        got = p[q].numpy()
        assert np.allclose(got[ok[q]][:, :2], rel_xy[ok[q]]), q
        assert np.allclose(got[ok[q]][:, 2], np.minimum(s[:, idx[q]][ok[q]], 1)) and (got[~ok[q]] == 0).all(), q
    assert all((v == 0).all() for v in parts(kp, np.zeros_like(sc)).values())   # no valid points: scale 0
    print("parts ok:", shapes)

    # 2. model + checkpoint
    t0 = time.time()
    model = UniSignEncoder(ckpt or os.environ.get("UNISIGN_CKPT"), MT5, "cpu")
    _print_report(model.report)
    print(f"  loaded in {time.time() - t0:.1f} s; prefix {model.n_prefix} tokens; float32 parameters in use "
          f"{sum(v.numel() for v in model.uni.parameters()) / 1e6:.1f} M")

    # 3. three real clips > 64 frames: all 64 after sampling, so a batch of 3 without padding
    d = np.load(os.environ.get("RTMW_AFFIX", ROOT / "data" / "pilot" / "rtmw_affix.npz"))
    kp, sc, off, rel = d["kp"], d["sc"], d["off"], d["relpath"]
    pick = np.flatnonzero(np.diff(off) > MAX_LEN)[:3]
    batch = [parts(kp[off[j]:off[j + 1]], sc[off[j]:off[j + 1]]) for j in pick]
    t0 = time.time()
    a = features(model, batch)
    dt = (time.time() - t0) / len(batch)
    again = features(model, batch)
    for x, y in zip(a, again):
        assert x.shape == (3, 768) and x.dtype == np.float32 and np.isfinite(x).all() and np.array_equal(x, y)
        assert not np.allclose(x[0], x[1])
    one = [features(model, [q]) for q in batch]              # batch 1 = batch 3 for equal lengths
    for r in range(2):
        gap = np.abs(np.concatenate([o[r] for o in one]) - a[r]).max()
        assert gap < 1e-4, gap
    got = {}                                                 # `pose` = pose_proj output, averaged
    hook = model.uni.pose_proj.register_forward_hook(lambda m, inp, o: got.update(o=o))
    single = features(model, batch[:1])
    hook.remove()
    assert np.allclose(got["o"].mean(1).float().cpu().numpy(), single[0], atol=1e-6)
    print(f"features ok: pose {a[0].shape}, mt5 {a[1].shape}, deterministic, batch 1 = batch 3 (max difference {gap:.1e}); "
          f"clips {[str(rel[j]) for j in pick]}")
    print(f"CPU {torch.get_num_threads()} threads: {dt:.3f} s/clip (64 frames)")
    print("demo ok: parts() = Uni-Sign load_part_kp, Uni_Sign + wlasl_pose_only_islr.pth, 768-d pose and mt5 features")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--demo":
        demo(a[1] if len(a) > 1 else None)
    elif a[0] == "extract":
        ap = argparse.ArgumentParser(prog="unisign_features.py extract")
        ap.add_argument("src")
        ap.add_argument("out")
        ap.add_argument("--device", choices=["cuda", "cpu"])
        ap.add_argument("--limit", type=int)
        ap.add_argument("--max-len", type=int, default=MAX_LEN)
        ap.add_argument("--ckpt", default=os.environ.get("UNISIGN_CKPT"))
        ap.add_argument("--mt5", default=MT5)
        ap.add_argument("--drop", choices=["face", "thumb"])
        ap.add_argument("--spans")
        o = ap.parse_args(a[1:])
        extract(o.src, o.out, o.device, o.limit, o.ckpt, o.mt5, o.max_len, drop=o.drop, spans=o.spans)
    else:
        sys.exit(__doc__)
