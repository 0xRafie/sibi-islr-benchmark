"""Frozen SHuBERT features (Gueuwou et al., ACL 2025): video -> 12 layers x 768 per clip, averaged over time.

Two stages, because the fairseq bundled with SHuBERT fails to import on Python >= 3.11 (dataclasses with mutable
defaults):
  a  Python 3.12 (uv environment): video -> MediaPipe landmarks -> 224 hand and face crops -> DINOv2 face/hand
     (CLS 384-d) and body pose (14-d) per frame. Output npz: face, lhand, rhand float16 (frames, 384), pose float32
     (frames, 14), det uint8 (frames, 5) = DET, sec float32 (clips, 5) = SEC, off (clips + 1), relpath, meta.
  b  Python 3.10 + SHuBERT's bundled fairseq (uv environment): per-frame features -> SHuBERT encoder -> `layers` float16
     (clips, 12, 768) = time average of the output of each layer, + relpath, meta.

The authors' code is used directly from its source at pinned revisions (PINS; `fetch` downloads it to $SHUBERT_EXT,
default ext/ in the repository root), not copied:
- Hugging Face Space ShesterG/TTIC-SHuBERT-ASLVideo-to-EnglishText (the authors' inference path):
  kpe_mediapipe.HolisticDetector (legacy Holistic min_detection_confidence 0.1; FaceLandmarker num_faces 6;
  HandLandmarker num_hands 6, min_hand_detection_confidence 0.05), crop_hands.HandExtractor, crop_face.FaceExtractor,
  dinov2_features.DINOEmbedder;
- GitHub ShesterG/SHuBERT: features/body_features.keypoints_to_numpy and examples/shubert/models/shubert.SHubertModel
  on its bundled fairseq;
- weights from Hugging Face ShesterG/SHuBERT/models: checkpoint_836_400000.pt, dinov2face.pth, dinov2hand.pth (used for
  both hands, as in the Space's features.py), face_landmarker_v2_with_blendshapes.task, hand_landmarker.task;
- DINOv2 facebookresearch/dinov2 through a local torch.hub, at the commit current when the Space was last changed
  (12 Jul 2025); the Space's patched attention.py and block.py are identical to it except for type annotations.
Rewritten here, small and marked in the code: load_model and process_sample of features/shubert_inference.py (any
device, weights_only=False, key and cfg checks), video reading, the per-clip loop, detection flags.

Differences from the authors' path (reasons in the code):
- frames go to DINOv2 as RGB, like the pretraining data (GitHub: crops written with cv2.VideoWriter, then read with
  decord). The Space's features.py sends BGR crops: cvtColor(BGR2RGB) on decord frames that are already RGB. Space mode:
  space_bgr=True;
- crops do not go through mp4v -> decord as in the GitHub pretraining data, but stay in memory as in the Space;
- no signer crop with YOLOv8 (dataset/clips_bbox.py). The Space has none either; our videos show one signer, square;
- a clean Holistic per clip through reset() (the Space creates a new detector per video; the GitHub script uses one
  Holistic for a batch of videos, so tracking carries over between videos);
- frames without a pose take the previous pose (GitHub, pretraining data); the Space's PoseProcessor gives -9999;
- default frame step 2 (pretraining at about 15 fps); the Space uses stride = 1.

Checks on a Tesla T4 (decord; stage a torch 2.10.0, stage b torch 2.3.0):
- Parity on two example videos of the Space (L5hUxT5YbnY_crop_2, 196 frames; rDUefZVPfmU_crop_3, 268 frames; stride 1).
  Space path = steps 1-3 of features.py with the Space functions, then SignHubertModel of the Space's shubert.py with the
  logic of SignHubertAdapter; this path = StageA(space_bgr=True) + encoder() + layers(). Crops identical byte for byte,
  DINOv2 identical, pose difference 1.5e-8 (float32); the 12 x 768 mean differs by at most 1.4e-6 (max scale 3.2), 6.1e-4
  when the per-frame features are stored in fp16. RGB (default) vs the Space's BGR: max 0.53 and 0.21; stride 2 vs 1:
  max 10.0 and 0.44, largest in layers 10-12. Earlier on CPU: difference exactly 0; Holistic.reset() = a new Holistic.
- Smoke test on 20 affix + 20 number clips (stride 2, 4 processes, DINOv2 on the T4), seconds per clip per process:
  affix 6.0 (landmarks 4.85, DINOv2 0.50, decoding 0.28, cropping 0.23, pose 0.17; 20.9 frames), number 9.2 (30.2
  frames); in total 13.9 and 13.1 frames/s. Stage b <= 0.05 s/clip. Frames with a pose 100%, face 100%, >= 1 hand
  73% / 64%, left hand matched 27% / 9%, right 62% / 64% (affix / number); in the clips checked, the left wrist of the
  pose lies below the frame (y about 1.04) during one-handed signs.

    python scripts/shubert_features.py fetch                   # clone the code + download the pinned weights (ext/, HF cache)
    python scripts/shubert_features.py --demo [ckpt]           # the parts whose modules are installed in this interpreter
    python scripts/shubert_features.py a <clips.json> <out.npz> [--stride 2] [--shard i/n] [--device D]
    python scripts/shubert_features.py merge <out.npz> <shard.npz>...
    python scripts/shubert_features.py b <frames.npz> <out.npz> [--device cuda|cpu] [--ckpt P]

clips.json = [[relpath, video path], ...]; relpath `<category>/<class>/<stem>` as in folds_*.csv. Word clips (category
kata): [relpath, path of the neutral sentence video, start, end, video frame count] (kata_clips.csv), read as clips of
their own: [start, end) with step stride from start (frame_ids). --ckpt default: $SHUBERT_CKPT, else the pinned download.

Environments (uv): a = Python 3.12 + torch 2.10.0 (cu126), torchvision 0.25.0, numpy 2.0.2, mediapipe 0.10.14, decord
0.6.0, huggingface_hub; b = Python 3.10 + torch 2.3.0 (cu121), numpy 1.26.4, hydra-core 1.0.7, omegaconf 2.0.6, bitarray,
sacrebleu, huggingface_hub (the authors' environment).
"""
import argparse, functools, importlib.util, json, os, subprocess, sys, tempfile, time, types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT = Path(os.environ.get("SHUBERT_EXT", ROOT / "ext"))
GITHUB, DINOV2, SPACE = EXT / "SHuBERT", EXT / "dinov2", EXT / "shubert-space"
PINS = dict(github=("https://github.com/ShesterG/SHuBERT", "cc1929326075bfbad7ad73159b2acf84356059bb"),
            dinov2=("https://github.com/facebookresearch/dinov2", "592541c8d842042bb5ab29a49433f73b544522d5"),
            space=("ShesterG/TTIC-SHuBERT-ASLVideo-to-EnglishText", "69d3d77aa4a4fec89048f5417d44fa717e36f6f8"),
            weights=("ShesterG/SHuBERT", "578a0233e770c8ce4dc75d859b91fdea7c34f5aa"))
SPACE_FILES = ("kpe_mediapipe.py", "crop_hands.py", "crop_face.py", "dinov2_features.py", "body_features.py",
               "shubert.py")                               # the last two only for the parity check
WEIGHTS = ("checkpoint_836_400000.pt", "dinov2face.pth", "dinov2hand.pth", "face_landmarker_v2_with_blendshapes.task",
           "hand_landmarker.task")
DET = ("pose", "face", "hand", "left", "right")   # Holistic pose, Tasks face, >= 1 Tasks hand, hand matched to a pose wrist
SEC = ("decode", "landmark", "crop", "dino", "pose")
# checkpoint cfg fields that may differ from the default SHubertConfig(): unused at inference (eval, mask=False, no
# wav2vec convolutional extractor)
TRAIN_ONLY = {"_name", "extractor_mode", "encoder_layerdrop", "mask_length", "mask_prob", "latent_temp", "max_update"}


def _retry(fn, tries=6):
    """Hugging Face sometimes answers 429: wait 30, 60, ... s."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:                               # noqa: BLE001 (HfHubHTTPError, connection)
            if k == tries - 1:
                raise
            print(f"  retry {k + 1}: {str(e)[:120]}", flush=True)
            time.sleep(30 * (k + 1))


def weight(name):
    """Path of a weight file in the Hugging Face cache at the pinned revision (downloaded if missing)."""
    from huggingface_hub import hf_hub_download
    return Path(_retry(lambda: hf_hub_download(PINS["weights"][0], f"models/{name}", revision=PINS["weights"][1])))


def fetch():
    """The authors' code at the pinned commits -> $SHUBERT_EXT, weights -> the Hugging Face cache. Idempotent."""
    from huggingface_hub import hf_hub_download
    for key, d in (("github", GITHUB), ("dinov2", DINOV2)):
        url, rev = PINS[key]
        if not (d / ".git").exists():
            subprocess.run(["git", "clone", "-q", url, str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", rev], check=True)
    repo, rev = PINS["space"]
    for f in SPACE_FILES:
        _retry(lambda: hf_hub_download(repo, f, repo_type="space", revision=rev, local_dir=SPACE))
    return {w: weight(w) for w in WEIGHTS}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.cache
def authors():
    """Space modules + GitHub body_features (all import decord at the top of the file: environment a only)."""
    m = {k: _load(SPACE / f"{k}.py", f"shubert_space_{k}") for k in ("kpe_mediapipe", "crop_hands", "crop_face",
                                                                      "dinov2_features")}
    m["body"] = _load(GITHUB / "features" / "body_features.py", "shubert_github_body")
    return types.SimpleNamespace(**m)


def frame_ids(n, stride=2, start=0, end=None, total=None):
    """Frame indices [start, end) with step stride from a video of n frames; by default the whole video. Word clips
    (ranges in the neutral sentence videos, kata_clips.csv) carry total = the video frame count when their bounds were
    made (OpenCV decoding during MediaPipe extraction). If decord counts differently, the indices could shift, so the
    clip fails instead of passing silently."""
    end = n if end is None else end
    assert total is None or n == total, f"decord reads {n} frames, the clip bounds were made on {total} frames"
    assert 0 <= start < end <= n, f"range [{start}, {end}) outside a video of {n} frames"
    return list(range(start, end, stride))


def read_frames(path, stride=2, *rng):
    """RGB uint8 frames (T, H, W, 3) with step stride, as in the Space's features.py (decord get_batch). rng = (start,
    end, total) for word clips (frame_ids); without rng the whole video."""
    import decord
    vr = decord.VideoReader(str(path))
    return vr.get_batch(frame_ids(len(vr), stride, *rng)).asnumpy()


def body(lm):
    """GitHub features/body_features.keypoints_to_numpy (the pretraining-data path) through a temporary json file:
    joints [0, 11..16] normalized to a signing space scaled by the shoulder width, 14-d; frames without a pose = the
    previous pose, -9999 before the first pose."""
    with tempfile.TemporaryDirectory() as d:
        json.dump(lm, open(f"{d}/c.json", "w"))
        authors().body.keypoints_to_numpy(f"{d}/c.json", d)
        return np.load(f"{d}/c.npy").astype(np.float32)


def flags(lm, shape):
    """Rewritten (statistics only): DET flags per frame; left/right hand through the Space's select_hands."""
    sel = authors().crop_hands.HandExtractor().select_hands
    out = np.zeros((len(lm), len(DET)), np.uint8)
    for i in range(len(lm)):
        f = lm[i]
        if f is None:
            continue
        hands = sel(f["pose_landmarks"][0], f["hand_landmarks"], shape) if f["pose_landmarks"] else (None, None)
        out[i] = (f["pose_landmarks"] is not None, f["face_landmarks"] is not None, f["hand_landmarks"] is not None,
                  hands[0] is not None, hands[1] is not None)
    return out


class StageA:
    """Landmarks -> crops -> DINOv2 + pose for one clip, with the Space classes as they are."""

    def __init__(self, device=None):
        import torch
        a = authors()
        self.det = a.kpe_mediapipe.HolisticDetector(str(weight(WEIGHTS[3])), str(weight(WEIGHTS[4])))
        hub = torch.hub.load                                 # the Space patches torch.hub too (app.py); here it
        torch.hub.load = lambda repo, model, **kw: hub(str(DINOV2), model, source="local", **kw)   # points to the pinned clone
        try:
            self.hand = a.dinov2_features.DINOEmbedder(str(weight("dinov2hand.pth")), device=device)
            self.face = a.dinov2_features.DINOEmbedder(str(weight("dinov2face.pth")), device=device)
        finally:
            torch.hub.load = hub

    def __call__(self, frames, space_bgr=False):
        """RGB frames (T, H, W, 3) -> dict of per-frame float32 features + det + sec (see SEC)."""
        a, t = authors(), [time.time()]
        # a clean Holistic per clip, like the Space's video_holistic (new detector per video). reset() gives landmarks
        # identical to a new Holistic (checked on affix clips), 0.02 vs 0.18 s per clip
        self.det.mp_holistic.reset()
        lm = self.det.process_video(frames)
        t.append(time.time())
        # the Space extractors apply cvtColor(BGR2RGB) to their input: BGR frames -> RGB crops (as the pretraining data)
        src = frames if space_bgr else np.ascontiguousarray(frames[..., ::-1])
        lh, rh = a.crop_hands.HandExtractor().extract_hand_frames(src, lm)
        fc = a.crop_face.FaceExtractor().extract_face_frames(src, lm)
        t.append(time.time())
        r = dict(face=self.face.extract_embeddings_from_frames(fc), lhand=self.hand.extract_embeddings_from_frames(lh),
                 rhand=self.hand.extract_embeddings_from_frames(rh))
        t.append(time.time())
        r.update(pose=body(lm), det=flags(lm, frames.shape[1:]))
        t.append(time.time())
        r["sec"] = np.diff(t)
        r["crops"] = (fc, lh, rh)
        return r


def stage_a(clips, out, stride=2, shard=(0, 1), device=None):
    """[(relpath, video)] -> npz of per-frame features (module docstring). Failed clips are recorded in meta; the run
    continues."""
    clips = sorted(map(tuple, clips))[shard[0]::shard[1]]
    import mediapipe as mp, torch
    t0, model = time.time(), StageA(device)
    load = time.time() - t0
    keep, rel, failed = {k: [] for k in ("face", "lhand", "rhand", "pose", "det", "sec")}, [], []
    for k, (r, path, *rng) in enumerate(clips):
        try:
            t1 = time.time()
            f = model(read_frames(path, stride, *rng))
            f["sec"] = np.r_[time.time() - t1 - f["sec"].sum(), f["sec"]][None]
        except Exception as e:                               # noqa: BLE001
            failed.append([r, repr(e)[:300]])
            print("FAILED", r, repr(e)[:300], flush=True)
            continue
        for key in keep:
            keep[key].append(f[key])
        rel.append(r)
        if k % 50 == 0:
            s = np.concatenate(keep["sec"]).sum(0)
            print(f"  {k + 1}/{len(clips)} clips, {s.sum() / len(rel):.2f} s/clip (" +
                  ", ".join(f"{n} {v / len(rel):.2f}" for n, v in zip(SEC, s)) + ")", flush=True)
    meta = dict(stride=stride, pins=PINS, mediapipe=mp.__version__, torch=torch.__version__, device=str(model.hand.device),
                failed=failed, sec_model_load=round(load, 1))
    np.savez(out, **{k: np.concatenate(v).astype(np.float16 if k in ("face", "lhand", "rhand") else
                                              np.uint8 if k == "det" else np.float32) for k, v in keep.items()},
             off=np.cumsum([0] + [len(d) for d in keep["det"]]), relpath=np.array(rel), meta=json.dumps(meta))
    print(f"{len(rel)} clips ({len(failed)} failed) in {time.time() - t0:.0f} s -> {out}", flush=True)


def merge(out, srcs):
    """Stage-a shard npz files -> one npz, sorted by relpath."""
    ds = [dict(np.load(s)) for s in srcs]                    # once: each NpzFile access decompresses again
    rows = sorted((str(r), j, i) for j, d in enumerate(ds) for i, r in enumerate(d["relpath"]))
    cut = lambda d, i, k: d[k][d["off"][i]:d["off"][i + 1]]
    keys = ("face", "lhand", "rhand", "pose", "det")
    res = {k: np.concatenate([cut(ds[j], i, k) for _, j, i in rows]) for k in keys}
    res["sec"] = np.concatenate([ds[j]["sec"][i:i + 1] for _, j, i in rows])
    res["off"] = np.cumsum([0] + [ds[j]["off"][i + 1] - ds[j]["off"][i] for _, j, i in rows])
    metas = [json.loads(str(d["meta"])) for d in ds]
    meta = dict(metas[0], failed=sum((m["failed"] for m in metas), []), shards=len(ds))
    np.savez(out, **res, relpath=np.array([r for r, _, _ in rows]), meta=json.dumps(meta))
    print(f"{len(rows)} clips, {res['off'][-1]} frames -> {out}")


# ---------------------------------------------------------------- stage b (Python 3.10, SHuBERT's bundled fairseq)

def encoder(ckpt=None, device="cpu"):
    """load_model of features/shubert_inference.py, rewritten: default SHubertConfig() and strict=False as the authors
    do, but torch.load(weights_only=False) (omegaconf cfg; the torch >= 2.6 default refuses it), any device (originally
    .cuda()), then checked: no missing keys, and the checkpoint cfg only differs in TRAIN_ONLY fields."""
    import dataclasses, torch
    sys.path.insert(0, str(GITHUB / "fairseq"))
    from examples.shubert.models.shubert import SHubertConfig, SHubertModel
    ckpt = ckpt or weight(WEIGHTS[0])
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg, saved = SHubertConfig(), ck["cfg"]["model"]
    model = SHubertModel(cfg)
    res = model.load_state_dict(ck["model"], strict=False)
    diff = sorted(f.name for f in dataclasses.fields(cfg) if f.name in saved and getattr(cfg, f.name) != saved[f.name])
    report = dict(ckpt=str(ckpt), n=len(ck["model"]), missing=res.missing_keys, unexpected=res.unexpected_keys,
                  cfg_diff=diff, params=sum(p.numel() for p in model.parameters()))
    assert not res.missing_keys, f"keys missing from the checkpoint: {res.missing_keys[:5]}"
    assert set(diff) <= TRAIN_ONLY, f"checkpoint cfg differs in inference fields: {set(diff) - TRAIN_ONLY}"
    return model.eval().to(device), report


def layers(model, face, lhand, rhand, pose):
    """process_sample of features/shubert_inference.py, rewritten without .cuda(): one clip (B = 1, no padding)
    -> (12, T, 768) float32, each layer = layer_results[i][-1] (FFN output before the residual, the authors' choice)."""
    import torch
    dev = next(model.parameters()).device
    x = [torch.as_tensor(np.asarray(v, np.float32), device=dev) for v in (face, lhand, rhand, pose)]
    z = torch.zeros((len(x[0]), 1), device=dev)
    src = [dict(face=x[0], left_hand=x[1], right_hand=x[2], body_posture=x[3], label_face=z, label_left_hand=z,
                label_right_hand=z, label_body_posture=z)]
    with torch.no_grad():
        res = model.extract_features(src, padding_mask=None, kmeans_labels=None, mask=False)
    return torch.stack([lr[-1].squeeze(1) for lr in res["layer_results"]]).float().cpu().numpy()


def stage_b(src, out, device=None, ckpt=None):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    d = dict(np.load(src))
    t0 = time.time()
    model, rep = encoder(ckpt, device)
    print({k: v for k, v in rep.items() if k != "ckpt"}, "| torch", torch.__version__, device, flush=True)
    off, n = d["off"], len(d["relpath"])
    res, t1 = np.zeros((n, 12, 768), np.float16), time.time()
    for i in range(n):
        s = slice(off[i], off[i + 1])
        res[i] = layers(model, d["face"][s], d["lhand"][s], d["rhand"][s], d["pose"][s]).mean(1)
        if i % 500 == 0:
            print(f"  {i + 1}/{n} clips, {(time.time() - t1) / (i + 1):.3f} s/clip", flush=True)
    assert np.isfinite(res).all()
    meta = dict(json.loads(str(d["meta"])), encoder=rep, device_b=device, torch_b=torch.__version__)
    np.savez(out, layers=res, relpath=d["relpath"], meta=json.dumps(meta))
    print(f"{n} clips in {time.time() - t1:.0f} s ({device}); loading {t1 - t0:.0f} s -> {out}", flush=True)


# ---------------------------------------------------------------- demo

def _synthetic():
    """One random 300x400 (H, W) frame + made-up landmarks: a pose with wrists 15 (0.30, 0.60) and 16 (0.70, 0.60), two
    hands around those wrists (in reverse detection order), one face of 478 points."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    pose = [[0.5, 0.3, 0.0] for _ in range(33)]
    pose[0], pose[2], pose[11], pose[12] = [0.5, 0.30, 0], [0.47, 0.27, 0], [0.62, 0.45, 0], [0.38, 0.45, 0]
    pose[13], pose[14], pose[15], pose[16] = [0.66, 0.55, 0], [0.34, 0.55, 0], [0.30, 0.60, 0], [0.70, 0.60, 0]
    hand = lambda cx, cy: [[cx + 0.04 * np.cos(a), cy + 0.06 * np.sin(a), 0.0] for a in np.linspace(0, 6, 21)]
    face = [[0.5 + 0.05 * np.cos(a), 0.3 + 0.07 * np.sin(a), 0.0] for a in np.linspace(0, 6.2, 478)]
    lm = {"face_landmarks": [face], "hand_landmarks": [hand(0.70, 0.60), hand(0.30, 0.60)], "pose_landmarks": [pose]}
    return frame, lm


def demo(ckpt=None):
    done = []
    try:
        import cv2, decord, mediapipe, torchvision                      # noqa: F401 (environment a)
        part_a = True
    except ImportError:
        part_a = False
    if part_a:
        a = authors()
        frame, lm = _synthetic()
        frame[130:230, 70:170] = (255, 0, 0)                 # hand of pose 15 (signer's left) painted red in RGB
        none = {"face_landmarks": None, "hand_landmarks": None, "pose_landmarks": None}
        seq = {0: none, 1: lm, 2: dict(lm, hand_landmarks=None, face_landmarks=None), 3: None}
        frames = np.stack([frame] * 4)
        src = np.ascontiguousarray(frames[..., ::-1])
        lh, rh = a.crop_hands.HandExtractor().extract_hand_frames(src, seq)
        fc = a.crop_face.FaceExtractor().extract_face_frames(src, seq)
        assert all(len(c) == 4 and all(x.shape == (224, 224, 3) and x.dtype == np.uint8 for x in c) for c in (lh, rh, fc))
        assert not lh[0].any() and not rh[0].any() and not fc[0].any()          # before any detection: black
        for c in (lh, rh, fc):                               # no hand/face, or landmarks None: the previous crop
            assert c[2] is c[1] and c[3] is c[1]
        H, W = frame.shape[:2]                               # get_bounding_box(scale 1.5) + adjust_bounding_box
        for crop, (cx, cy) in ((lh[1], (0.30, 0.60)), (rh[1], (0.70, 0.60))):
            px = np.array([(int(p[0] * W), int(p[1] * H)) for p in
                           [[cx + 0.04 * np.cos(t), cy + 0.06 * np.sin(t)] for t in np.linspace(0, 6, 21)]])
            side = max(np.ptp(px, 0) + 1)
            pad = int(0.5 * side / 2)
            x, y = px.mean(0).astype(int) - side // 2 - pad
            want = cv2.resize(frame[y:y + side + 2 * pad, x:x + side + 2 * pad], (224, 224), interpolation=cv2.INTER_AREA)
            assert np.array_equal(crop, want), (cx, cy)
        assert lh[1][..., 0].mean() > 200 and lh[1][..., 1:].mean() < 100 and rh[1][..., 0].mean() < 200   # RGB
        assert (fc[1][0, 0] == 128).all() and (fc[1][-1, -1] == 128).all()       # grey background outside the eyes/mouth
        lh_b, _ = a.crop_hands.HandExtractor().extract_hand_frames(frames, seq)   # Space mode: RGB in -> BGR
        assert np.array_equal(lh_b[1], lh[1][..., ::-1])
        p = body(seq)
        assert p.shape == (4, 14) and (p[0] == -9999).all() and np.array_equal(p[2], p[1]) and np.array_equal(p[3], p[1])
        h = (0.62 - 0.38) / 2                                # half the shoulder width; signing space 6h x 7h, top = eyes - h/2
        ry = lambda y: (y - (0.27 - h / 2)) / (7 * h) - 0.5
        assert np.allclose(p[1, :2], [0, ry(0.30)])          # nose centered horizontally
        assert np.allclose(p[1, 10:], [(0.30 - 0.5) / (6 * h), ry(0.60), (0.70 - 0.5) / (6 * h), ry(0.60)])  # wrists
        fl = flags(seq, frame.shape)
        assert fl.tolist() == [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], fl.tolist()
        print("crops ok: box = longest landmark side x 1.5 at the landmark center, INTER_AREA 224, left = pose 15, "
              "RGB, black, then the previous crop; face grey 128; 14-d pose in signing space, -9999, then the previous one")
        t0 = time.time()
        m = StageA("cpu")
        print(f"  StageA loaded in {time.time() - t0:.1f} s")
        for w in (m.hand.model, m.face.model):
            assert w.pos_embed.shape == (1, 257, 384) and w.num_register_tokens == 4
        f = m(np.full((3, 300, 400, 3), 128, np.uint8))      # flat grey: no detections
        assert {k: v.shape for k, v in f.items() if k not in ("crops", "sec")} == dict(
            face=(3, 384), lhand=(3, 384), rhand=(3, 384), pose=(3, 14), det=(3, 5))
        assert (f["pose"] == -9999).all() and not f["det"].any() and np.allclose(f["lhand"], f["rhand"])
        e = m.hand.extract_embeddings_from_frames([lh[1], rh[1], lh[1]])
        assert e.shape == (3, 384) and np.isfinite(e).all() and np.allclose(e[0], e[2], atol=1e-5)
        assert not np.allclose(e[0], e[1])
        print(f"DINOv2 ok: 384-d CLS from dinov2hand/dinov2face, pos_embed (1, 257, 384); empty 3-frame clip "
              f"{f['sec'].sum():.2f} s")
        done.append("a")
    try:
        import torch                                                     # noqa: F401
        sys.path.insert(0, str(GITHUB / "fairseq"))
        import fairseq                                                   # noqa: F401
        part_b = True
    except Exception:                                                    # noqa: BLE001 (Python >= 3.11, no fairseq)
        part_b = False
    if part_b:
        t0 = time.time()
        model, rep = encoder(ckpt or os.environ.get("SHUBERT_CKPT"))
        print(f"encoder ok ({time.time() - t0:.0f} s): {rep['n']} keys, missing {rep['missing']}, unused "
              f"{rep['unexpected']}, cfg differences {rep['cfg_diff']}, {rep['params'] / 1e6:.1f} M parameters")
        rng = np.random.default_rng(1)
        T = 37                                               # odd: padding path of required_seq_len_multiple = 2
        x = [rng.normal(size=(T, 384)) for _ in range(3)] + [rng.uniform(-1, 1, (T, 14))]
        got = {}
        hook = model.encoder.layers[5].fc2.register_forward_hook(lambda mod, i, o: got.update(o=o))
        y = layers(model, *x)
        hook.remove()
        assert y.shape == (12, T, 768) and np.isfinite(y).all() and np.array_equal(y, layers(model, *x))
        assert np.allclose(got["o"][:T, 0].numpy(), y[5], atol=1e-6)            # = fc2 output, without the residual
        assert not np.allclose(y[5], y[6])
        print("encoder forward ok: (12, T, 768), layer = FFN output (fc2), deterministic")
        done.append("b")
    assert done, "nothing to test: run fetch, then --demo in environment a and/or b"
    print("demo ok:", "+".join(done), "(for the other part, run --demo in the other environment)" if len(done) < 2 else "")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--demo":
        demo(a[1] if len(a) > 1 else None)
    elif a[0] == "fetch":
        print(json.dumps({k: str(v) for k, v in fetch().items()}, indent=1))
    elif a[0] == "a":
        ap = argparse.ArgumentParser(prog="shubert_features.py a")
        ap.add_argument("clips"), ap.add_argument("out")
        ap.add_argument("--stride", type=int, default=2)
        ap.add_argument("--shard", default="0/1")
        ap.add_argument("--device")
        o = ap.parse_args(a[1:])
        stage_a(json.load(open(o.clips)), o.out, o.stride, tuple(map(int, o.shard.split("/"))), o.device)
    elif a[0] == "merge":
        merge(a[1], a[2:])
    elif a[0] == "b":
        ap = argparse.ArgumentParser(prog="shubert_features.py b")
        ap.add_argument("src"), ap.add_argument("out")
        ap.add_argument("--device", choices=["cuda", "cpu"])
        ap.add_argument("--ckpt", default=os.environ.get("SHUBERT_CKPT"))
        o = ap.parse_args(a[1:])
        stage_b(o.src, o.out, o.device, o.ckpt)
    else:
        sys.exit(__doc__)
