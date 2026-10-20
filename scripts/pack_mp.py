"""Pack the cached MediaPipe keypoints of one category into a single npz for training.

Each frame keeps pose x, y (33 points) and left then right hand x, y (2 x 21 points), the same 150 columns as
gate_variants.pack(), for all 22 signers and the clips kept by protocol_counts.clips(). Clips are stored untrimmed,
together with `span` from trim.span: models that need the full sequence can use it as is, and cutting each clip to its
`span` gives the trimmed input. Only landmark arrays are written, no video.

Outputs:
- mp_<cat>.npz: seq (frames, 150) float32, off, span, relpath, cls, subject, rep, cat.
- mp_face_<cat>.npz (`--face`): face x, y (468 x 2 = 936) float16 per frame, aligned with mp_<cat>.npz (face ablation).
- mp_fx_<cat>.npz (`--fx`): hand z (42) and the x, y of the face subset train_main.FACE_SUB (184) float16 per frame.

Paths: SIBI_QC (default `qc`) holds darmawan_files.csv. Under SIBI_DATA (default `data`), keypoints are read from
keypoints/<category>/<class>/<clip>.npy (extract.py --darmawan) and packages are written to pilot/.

    python scripts/pack_mp.py --demo
    python scripts/pack_mp.py                # -> data/pilot/mp_affix.npz, mp_number.npz, mp_alphabet.npz
    python scripts/pack_mp.py --face affix   # -> data/pilot/mp_face_affix.npz
    python scripts/pack_mp.py --fx affix     # -> data/pilot/mp_fx_affix.npz
"""
import csv
import os
import sys

import numpy as np

from gate import N_HAND
from gate_variants import POSE_XY
from protocol_counts import clips
from trim import span

DATA = os.environ.get("SIBI_DATA", "data")
QC = os.environ.get("SIBI_QC", "qc")
CATS = ("affix", "number", "alphabet")
FACE = slice(132, 132 + 468 * 3)   # face block of the 1662 layout (extract.py)


def frame150(arr):
    """(T, 1662) -> (T, 150): pose xy 66 + left then right hand xy 84, as in gate_variants.pack()."""
    hands = arr[:, -2 * N_HAND:].reshape(len(arr), 2, 21, 3)[..., :2].reshape(len(arr), 84)
    return np.concatenate([arr[:, POSE_XY], hands], 1).astype(np.float32)


def pack(cl, cat, src=f"{DATA}/keypoints", out_dir=f"{DATA}/pilot"):
    rel = sorted(r for r, v in cl.items() if v[0] == cat)
    seqs, spans = [], []
    for r in rel:
        arr = np.load(f"{src}/{r}.npy")
        seqs.append(frame150(arr))
        spans.append(span(arr))
    off = np.cumsum([0] + [len(s) for s in seqs])
    out = f"{out_dir}/mp_{cat}.npz"
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(out, seq=np.concatenate(seqs), off=off, span=np.array(spans, np.int64), relpath=np.array(rel),
                        cls=np.array([cl[r][1] for r in rel]), subject=np.array([cl[r][2] for r in rel]),
                        rep=np.array([cl[r][3] for r in rel]), cat=cat)
    print(f"{cat}: {len(rel)} clips, {off[-1]} frames (untrimmed) -> {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    return out


def pack_face(cl, cat, src=f"{DATA}/keypoints", out_dir=f"{DATA}/pilot"):
    """Face xy (468 x 2 = 936) per frame, float16, aligned frame by frame with mp_<cat>.npz (same clip order and `off`,
    untrimmed). An undetected face stays 0. Used by the face ablation."""
    rel = sorted(r for r, v in cl.items() if v[0] == cat)
    faces = [np.load(f"{src}/{r}.npy")[:, FACE].reshape(-1, 468, 3)[..., :2].reshape(-1, 936) for r in rel]
    off = np.cumsum([0] + [len(f) for f in faces])
    out = f"{out_dir}/mp_face_{cat}.npz"
    np.savez_compressed(out, face=np.concatenate(faces).astype(np.float16), off=off, relpath=np.array(rel))
    print(f"{cat}: face {len(rel)} clips, {off[-1]} frames -> {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    return out


def pack_fx(cl, cat, src=f"{DATA}/keypoints", out_dir=f"{DATA}/pilot"):
    """Extra features: left then right hand z (42) and xy of the face subset `train_main.FACE_SUB` (92 points, 184) per
    frame, float16, aligned frame by frame with mp_<cat>.npz. Files are read one at a time to keep memory small."""
    from train_main import FACE_SUB
    rel = sorted(r for r, v in cl.items() if v[0] == cat)
    hz, fs = [], []
    for r in rel:
        a = np.load(f"{src}/{r}.npy")
        hz.append(a[:, -2 * N_HAND:].reshape(-1, 2, 21, 3)[..., 2].reshape(-1, 42).astype(np.float16))
        fs.append(a[:, FACE].reshape(-1, 468, 3)[:, FACE_SUB, :2].reshape(-1, 184).astype(np.float16))
    off = np.cumsum([0] + [len(h) for h in hz])
    out = f"{out_dir}/mp_fx_{cat}.npz"
    np.savez_compressed(out, hz=np.concatenate(hz), face=np.concatenate(fs), off=off, relpath=np.array(rel))
    print(f"{cat}: fx {len(rel)} clips, {off[-1]} frames -> {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    return out


def main(fn=None, cats=CATS):
    with open(f"{QC}/darmawan_files.csv", newline="") as fh:
        cl = clips(csv.DictReader(fh))
    for cat in cats:
        (fn or pack)(cl, cat)


def demo():
    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        cl = {}
        for i in range(3):
            rel = f"affix/an/subject-{i + 1}-affix-an-rep-1"
            os.makedirs(f"{d}/kp/affix/an", exist_ok=True)
            arr = np.zeros((40, 1662), np.float32)
            arr[5:35, -126:] = rng.uniform(0.3, 0.7, (30, 126))              # hands in frames 5-34
            arr[:, :132] = rng.uniform(0.2, 0.8, (40, 132))
            arr[:, FACE] = rng.uniform(0.3, 0.7, (40, 1404))
            np.save(f"{d}/kp/{rel}.npy", arr)
            cl[rel] = ("affix", "an", i + 1, 1)
        z = np.load(pack(cl, "affix", f"{d}/kp", d))
        assert z["seq"].shape == (120, 150) and list(z["off"]) == [0, 40, 80, 120] and z["span"].shape == (3, 2)
        a = np.load(f"{d}/kp/affix/an/subject-1-affix-an-rep-1.npy")
        assert np.array_equal(z["seq"][:40], frame150(a)) and tuple(z["span"][0]) == span(a)
        f = np.load(pack_face(cl, "affix", f"{d}/kp", d))
        assert f["face"].shape == (120, 936) and f["face"].dtype == np.float16
        assert np.array_equal(f["off"], z["off"]) and np.array_equal(f["relpath"], z["relpath"])
        assert np.allclose(f["face"][:40].reshape(40, 468, 2), a[:, FACE].reshape(40, 468, 3)[..., :2], atol=1e-3)
        x = np.load(pack_fx(cl, "affix", f"{d}/kp", d))
        assert x["hz"].shape == (120, 42) and x["face"].shape == (120, 184) and np.array_equal(x["off"], z["off"])
        assert np.allclose(x["hz"][:40], a[:, -126:].reshape(40, 2, 21, 3)[..., 2].reshape(40, 42), atol=1e-3)
    print("demo ok: 150 columns per frame, offsets, trim span, metadata; aligned face package (936 columns, float16); "
          "fx package")


if __name__ == "__main__":
    a = sys.argv[1:]
    flag = next((f for f in ("--face", "--fx") if f in a), None)
    demo() if "--demo" in a else main(*((pack_face if flag == "--face" else pack_fx, a[a.index(flag) + 1:]) if flag else ()))
