"""Cut the 1,100 neutral sentence videos into clips of the 29 word classes (one video is dropped as a duplicate, BUANG).

Cut points: videos with a complete manual reference (words_manual.csv) use the manual points; every other video uses
the selected automatic candidate (A, B, C: dynamic programming over a boundary signal; D, E1, E2, E3, F+: points read
from the CSV of that method). Word k of a sentence occupies frames [point k, point k+1). The sign sequence of a sentence
comes from segment_words.gloss_of(). Input keypoints are the MediaPipe keypoints of the full sentence videos.

Outputs: mp_kata.npz (same format as pack_mp.pack) and kata_clips.csv, one row per clip: relpath, kelas (class),
subject, rep, video, k (word index in the sentence), awal and akhir (first and end frame, end exclusive), sumber (source
of the points: `rujukan` = manual reference, or the candidate name).

The benchmark clips come from `cut_words.py F+` (points from seg_fplus.py).

Paths: under SIBI_DATA (default `data`): kata/keypoints/sentence/neutral/*.npy (input keypoints), kata/*.csv (candidate
points), segfull/, segsample/, segpilot/ (candidate signals), pilot/ (outputs). SIBI_QC (default `qc`) holds
words_manual.csv and receives kata_clips.csv. kata/keypoints/ has the layout that `extract.py --darmawan
'sentence/neutral/*.mp4'` writes to keypoints/ instead, so link it once: `ln -s ../keypoints data/kata/keypoints`
(align_words.py reads the same folder).

    python scripts/cut_words.py --demo
    python scripts/cut_words.py <selected candidate: A|B|C|D|E1|E2|E3|F+>
    python scripts/cut_words.py rtmw [data/pilot/rtmw_neutral.npz] [data/pilot/rtmw_kata.npz]   # cut_rtmw
    python scripts/cut_words.py face [data/pilot/mp_face_kata.npz]   # face package, aligned with mp_kata.npz
"""
import csv
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_mp import frame150
from segment_words import boundaries, gloss_of, span

DATA = os.environ.get("SIBI_DATA", "data")
QC = os.environ.get("SIBI_QC", "qc")
KATA = ("apa bapak beli buah dapat guru ibu ikan jadwal jam juara kamu kue lahir lihat masak nama olahraga pagi pasar "
        "pulang pulpen punya saudara saya sekarang selamat siang tahun").split()
SIG = {"A": ("motion", 1), "B": ("segprobs", -1), "C": ("fastprobs", -1)}
CSV = {"D": f"{DATA}/kata/align_D.csv", "E1": f"{DATA}/kata/hal_points.csv", "E2": f"{DATA}/kata/atba_points.csv",
       "E3": f"{DATA}/kata/ctc_points.csv", "F+": f"{DATA}/kata/fplus_points.csv"}
ROOTS = (f"{DATA}/segfull", f"{DATA}/segsample", f"{DATA}/segpilot")      # signals A/B/C: 880 + 110 sample + 110 pilot
# Duplicate content (checked visually and with keypoints): 12 short takes plus copies of rep-2..rep-5 of the same
# sentence, so no file can be identified as the original rep-1. Dropped under the rule for duplicates.
BUANG = {"sentence/neutral/subject-4-sentence-7-neutral-rep-1.mp4"}


def signal(d, rel):
    for root in ROOTS:
        f = f"{root}/{d}/{rel[:-4]}.npy"
        if os.path.exists(f):
            return np.load(f)
    raise FileNotFoundError(f"{d} {rel}")


def winner_points(win, rels):
    """{relpath: N+1 points} of the selected candidate for every video."""
    if win in CSV:
        return {r["relpath"]: [int(x) for x in r["titik"].split()] for r in csv.DictReader(open(CSV[win]))}
    d, sgn = SIG[win]
    out = {}
    for rel in rels:
        a, b = span(np.load(f"{DATA}/kata/keypoints/{rel[:-4]}.npy"))
        bd = boundaries(sgn * signal(d, rel), len(gloss_of(rel)), a, b)
        out[rel] = None if bd is None else [a] + bd + [b]
    return out


def clips(gl, pts):
    """[(k, class, start, end)] for the word tokens of a sentence; word k occupies [pts[k], pts[k+1])."""
    return [(k, t.strip().lower(), pts[k], pts[k + 1]) for k, t in enumerate(gl) if t.strip().lower() in KATA]


def main(win, manual=f"{QC}/words_manual.csv", out=f"{DATA}/pilot/mp_kata.npz", man_out=f"{QC}/kata_clips.csv"):
    rels = sorted("sentence/neutral/" + os.path.basename(p)[:-4] + ".mp4" for p in glob.glob(f"{DATA}/kata/keypoints/sentence/neutral/*.npy"))
    assert len(rels) == 1100, len(rels)
    rels = [r for r in rels if r not in BUANG]
    ref = {r["relpath"]: [int(x) for x in r["titik"].split()] for r in csv.DictReader(open(manual))
           if r["flag"] != "1" and int(r["n_isyarat"]) == len(gloss_of(r["relpath"]))}
    wp = winner_points(win, rels)
    seqs, rows = [], []
    for rel in rels:
        gl = gloss_of(rel)
        pts, src = (ref[rel], "rujukan") if rel in ref else (wp.get(rel), win)
        assert pts is not None and len(pts) == len(gl) + 1 and all(np.diff(pts) > 0), (rel, src, pts)
        raw = np.load(f"{DATA}/kata/keypoints/{rel[:-4]}.npy").reshape(-1, 1662)
        subj, rep = (int(x) for x in re.search(r"subject-(\d+)-sentence-\d+-[A-Za-z]+-rep-(\d+)", rel).groups())
        for k, cls, s, e in clips(gl, pts):
            seqs.append(frame150(raw[s:e]))
            rows.append((f"kata/{cls}/{os.path.basename(rel)[:-4]}__{k}", cls, subj, rep, rel, k, s, e, src))
    off = np.cumsum([0] + [len(x) for x in seqs])
    rp, cls, subj, rep = (np.array([r[i] for r in rows]) for i in range(4))
    np.savez_compressed(out, seq=np.concatenate(seqs).astype(np.float32), off=off,
                        span=np.array([[0, len(x)] for x in seqs], np.int64), relpath=rp, cls=cls, subject=subj, rep=rep,
                        cat=np.array("kata"))
    with open(man_out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["relpath", "kelas", "subject", "rep", "video", "k", "awal", "akhir", "sumber"])
        w.writerows(rows)
    n = {c: int((cls == c).sum()) for c in KATA}
    print(f"{out}: {len(rows)} clips, {len(KATA)} classes (saya {n['saya']}, smallest {min(n.values())}); "
          f"source: manual reference {sum(r[-1] == 'rujukan' for r in rows)}, {win} {sum(r[-1] == win for r in rows)}; -> {man_out}")


def cut_rtmw(src=f"{DATA}/pilot/rtmw_neutral.npz", out=f"{DATA}/pilot/rtmw_kata.npz", table=f"{QC}/kata_clips.csv",
             ref=f"{DATA}/pilot/mp_kata.npz", mp_dir=f"{DATA}/kata/keypoints"):
    """RTMW keypoints of the full neutral videos (extract_rtmw.py; relpath <category>/<folder>/<stem>, matched by stem)
    -> word clips [awal, akhir) from `table`, in the order and with the relpaths of mp_kata.npz, in the rtmw_<cat>.npz
    format (kp, sc, off, relpath). The cut points are MediaPipe frame indices, so the RTMW frame count of every video must
    equal that of its MediaPipe keypoints."""
    rows, m, d = list(csv.DictReader(open(table))), np.load(ref), dict(np.load(src))
    pos = {str(r).rsplit("/", 1)[1]: i for i, r in enumerate(d["relpath"])}
    assert len(pos) == len(d["relpath"]), "duplicate video stem in " + src
    n = np.diff(d["off"])
    vids = {r["video"][:-4] for r in rows}                     # sentence/neutral/<stem>
    i = {v: pos[v.rsplit("/", 1)[1]] for v in vids}
    bad = sorted(v for v in vids if n[i[v]] != np.load(f"{mp_dir}/{v}.npy", mmap_mode="r").shape[0])
    assert not bad, f"{len(bad)} videos differ in frame count between RTMW and MediaPipe: {bad[:3]}"
    a, b = (np.array([int(r[k]) for r in rows]) for k in ("awal", "akhir"))
    v = np.array([i[r["video"][:-4]] for r in rows])
    assert ((0 <= a) & (a < b) & (b <= n[v])).all()           # no clip spills into the next video
    assert list(m["relpath"]) == [r["relpath"] for r in rows] and (np.diff(m["off"]) == b - a).all()
    idx = np.concatenate([np.arange(s, e) for s, e in zip(d["off"][v] + a, d["off"][v] + b)])
    np.savez_compressed(out, kp=d["kp"][idx], sc=d["sc"][idx], off=np.cumsum([0, *(b - a)]), relpath=m["relpath"])
    print(f"{out}: {len(rows)} clips, {len(idx)} frames from {len(vids)} videos (RTMW frame count = MediaPipe per video)")


def cut_face(out=f"{DATA}/pilot/mp_face_kata.npz", table=f"{QC}/kata_clips.csv", ref=f"{DATA}/pilot/mp_kata.npz",
             mp_dir=f"{DATA}/kata/keypoints"):
    """Face ablation: face xy (468 x 2 = 936) float16 of the word clips, from the MediaPipe keypoints of the neutral
    videos, [awal, akhir) from `table`; pack_mp.pack_face format, with the relpath order and `off` of mp_kata.npz (table
    order, not alphabetical)."""
    from pack_mp import FACE
    rows, m = list(csv.DictReader(open(table))), np.load(ref)
    faces = [np.load(f"{mp_dir}/{r['video'][:-4]}.npy", mmap_mode="r")[int(r["awal"]):int(r["akhir"]), FACE]
             .reshape(-1, 468, 3)[..., :2].reshape(-1, 936).astype(np.float16) for r in rows]
    off = np.cumsum([0] + [len(f) for f in faces])
    assert list(m["relpath"]) == [r["relpath"] for r in rows] and np.array_equal(off, m["off"]), "not aligned with mp_kata"
    np.savez_compressed(out, face=np.concatenate(faces), off=off, relpath=m["relpath"])
    print(f"{out}: face {len(rows)} clips, {off[-1]} frames")


def demo_face():
    import tempfile
    v = np.random.default_rng(0).random((6, 1662), np.float32)
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(f"{t}/kp/sentence/neutral")
        np.save(f"{t}/kp/sentence/neutral/s1.npy", v)
        with open(f"{t}/clips.csv", "w", newline="") as f:
            csv.writer(f).writerows([["relpath", "video", "awal", "akhir"], ["kata/b/s1__1", "sentence/neutral/s1.mp4", 3, 6],
                                     ["kata/a/s1__0", "sentence/neutral/s1.mp4", 0, 3]])
        np.savez(f"{t}/mp.npz", relpath=np.array(["kata/b/s1__1", "kata/a/s1__0"]), off=np.array([0, 3, 6]))
        cut_face(f"{t}/f.npz", f"{t}/clips.csv", f"{t}/mp.npz", f"{t}/kp")
        z = np.load(f"{t}/f.npz")
        assert list(z["relpath"]) == ["kata/b/s1__1", "kata/a/s1__0"] and z["face"].shape == (6, 936)   # table order
        assert np.allclose(z["face"][0, :3], v[3, [132, 133, 135]], atol=1e-3)   # first clip starts at frame 3; no z
    print("demo face ok: face of the word clips aligned with mp_kata (table order, [awal, akhir))")


def demo_rtmw():
    import tempfile
    stems = ["subject-1-sentence-1-neutral-rep-1", "subject-2-sentence-3-neutral-rep-2", "subject-3-sentence-2-neutral-rep-1"]
    T = [10, 8, 5]                                             # the third video has no word clip
    order = [1, 2, 0]                                          # relpath order of the RTMW npz differs from the table
    off = np.cumsum([0] + [T[j] for j in order])
    kp = np.repeat(np.arange(off[-1], dtype=np.float16)[:, None, None], 133, 1).repeat(2, 2)   # value = global frame
    rows = [("kata/saya/" + stems[0] + "__0", 2, 5, 0), ("kata/beli/" + stems[0] + "__2", 5, 9, 0),
            ("kata/ibu/" + stems[1] + "__1", 0, 3, 1)]
    with tempfile.TemporaryDirectory() as t:
        np.savez(f"{t}/neutral.npz", kp=kp, sc=kp[..., 0], off=off,
                 relpath=np.array([f"neutral/s{j + 1:02d}/{stems[j]}" for j in order]))
        with open(f"{t}/clips.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["relpath", "kelas", "subject", "rep", "video", "k", "awal", "akhir", "sumber"])
            w.writerows([(r, r.split("/")[1], j + 1, 1, f"sentence/neutral/{stems[j]}.mp4", 0, s, e, "F+")
                         for r, s, e, j in rows])
        np.savez(f"{t}/mp.npz", relpath=np.array([r[0] for r in rows]), off=np.array([0, 3, 7, 10]))
        os.makedirs(f"{t}/kp/sentence/neutral")
        for s, n in zip(stems, T):
            np.save(f"{t}/kp/sentence/neutral/{s}.npy", np.zeros((n, 1), np.float32))
        args = (f"{t}/neutral.npz", f"{t}/out.npz", f"{t}/clips.csv", f"{t}/mp.npz", f"{t}/kp")
        cut_rtmw(*args)
        z = np.load(f"{t}/out.npz")
        g = off[order.index(0)], off[order.index(1)]           # global first frame of videos 1 and 2
        want = [*range(g[0] + 2, g[0] + 5), *range(g[0] + 5, g[0] + 9), *range(g[1], g[1] + 3)]
        assert list(z["off"]) == [0, 3, 7, 10] and list(z["relpath"]) == [r[0] for r in rows]
        assert (z["kp"][:, 0, 0] == want).all() and (z["sc"][:, 0] == want).all() and z["kp"].shape == (10, 133, 2)
        np.save(f"{t}/kp/sentence/neutral/{stems[1]}.npy", np.zeros((9, 1), np.float32))   # MediaPipe 9, RTMW 8 frames
        try:
            cut_rtmw(*args)
        except AssertionError as e:
            assert "differ in frame count" in str(e)
        else:
            raise AssertionError("frame count mismatch not rejected")
    print("demo rtmw ok: word clips from full videos matched by stem, table order, videos without clips skipped, "
          "RTMW vs MediaPipe frame count mismatch rejected")


def demo():
    demo_rtmw()
    demo_face()
    gl = ["saya", "me ", "beli", "buah", "di", "pasar"]
    assert clips(gl, [10, 40, 55, 80, 110, 125, 160]) == [(0, "saya", 10, 40), (2, "beli", 55, 80), (3, "buah", 80, 110),
                                                         (5, "pasar", 125, 160)]
    assert len(KATA) == 29 and "ribu" not in KATA and "di" not in KATA
    print("demo ok: slices [point k, point k+1), 29 classes without affix or number tokens")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--demo":
        demo()
    elif a[0] == "rtmw":
        cut_rtmw(*a[1:3])
    elif a[0] == "face":
        cut_face(*a[1:2])
    else:
        main(a[0])
