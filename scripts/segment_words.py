"""Word segmentation of the sentence videos when the number of signs N is known: candidates A and B, and shared helpers.

Both candidates use the same dynamic programming (DP): it places N-1 boundaries at the lowest values of a per-frame
boundary signal, with every segment at least MIN_SEG = 15 frames (0.5 s), inside the active span of trim.py.
- A `jiki`: number of pixels with Farneback flow magnitude > 5.0 on frames resized to 360 pixels wide (Darmawan et al.,
  JIKI 2024). Cached in <SIBI_DATA>/motion/.
- B `pose`: -log p(sign start) from sign-language-processing/segmentation, read from <SIBI_DATA>/segprobs/ (written by a
  separate run of that model).
The command line prints only aggregate statistics, never per-video boundaries, so the manual annotation of the
validation sample stays blind to the automatic output.

Helpers used by other scripts: gloss_of() (sign sequence of a sentence video from the composition table of the dataset),
boundaries(), sample() (validation sample: 10% of the videos of each signer, seed 0; half 'pilih' selects the word
cutter, half 'gerbang' tests it), word_iou(), iou_summary(), and gate() (selection and test of the word cutter).

Paths: under SIBI_DATA (default `data`): darmawan/ (videos and the composition table `Dataset Kalimat - DIB.xlsx`),
keypoints/, motion/, segprobs/, kata/ (points of the word-cutter candidates), segsample/ and segpilot/ (keypoints/,
motion/, segprobs/, fastprobs/ of the sample and pilot videos, in the layout of the folders above). Under SIBI_QC
(default `qc`): words_manual.csv (manual boundaries, see seg_fplus.py), neutral_pilot.csv, and the IoU summary.
relabel_s18.csv is read from SIBI_QC, by default the `qc` folder of the repository (see relabel()).

    python scripts/segment_words.py --demo
    python scripts/segment_words.py jiki 'sentence/fear/*-rep-1.mp4'
    python scripts/segment_words.py pose 'sentence/fear/*-rep-1.mp4'
    python scripts/segment_words.py gate     # only once every video of the sample has manual boundaries
    python scripts/segment_words.py iou      # IoU of the selected cutter vs the manual points, aggregated per half
"""
import csv, glob, multiprocessing, os, re, sys, time, zipfile
import functools, random
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
import numpy as np

from trim import span

DATA = os.environ.get("SIBI_DATA", "data")
QC = os.environ.get("SIBI_QC", "qc")
FB = dict(pyr_scale=0.5, levels=3, winsize=25, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
THRESH, WIDTH, MIN_SEG = 5.0, 360, 15
XLSX = f"{DATA}/darmawan/Dataset Kalimat - DIB.xlsx"
MANUAL = f"{DATA}/annotations/words_manual.csv"
# words come from the neutral sentences only; the validation sample (110 videos) needs every neutral video
EXPRESSIONS = ("neutral",)
EMO = {"Netral": "neutral", "Senang": "happy", "Sedih": "sad", "Marah": "anger",
       "Jijik": "disgust", "Takut": "fear", "Kaget": "surprise"}
M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


@functools.lru_cache(maxsize=1)
def glosses():
    """{(emotion folder, sentence number in the file name): [signs]}. The file number is the row order of the emotion
    sheet."""
    z = zipfile.ZipFile(XLSX)
    ss = ["".join(t.text or "" for t in si.iter(M + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(M + "si")]
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    sh = {}
    for s in ET.fromstring(z.read("xl/workbook.xml")).find(M + "sheets"):
        rows = []
        for r in ET.fromstring(z.read("xl/" + rels[s.get(R + "id")].replace("xl/", "").lstrip("/"))).iter(M + "row"):
            row = {}
            for c in r.findall(M + "c"):
                v = c.find(M + "v")
                if v is not None and v.text is not None:
                    row[re.match(r"[A-Z]+", c.get("r")).group()] = ss[int(v.text)] if c.get("t") == "s" else v.text
            rows.append(row)
        sh[s.get("name")] = rows
    comp = {int(float(r["A"])): [r[k] for k in sorted(r) if k not in ("A", "B") and r[k].strip()]
            for r in sh["komposisi isyarat"][2:] if r.get("A")}
    # composition 38 = neutral sentence 8: the table writes 2012 as 2 0 1 2, but the signers sign 2 ribu 12 (checked
    # visually against the isolated number/ribu and ribuan clips). The assert flags a changed table for a new check.
    assert comp[38] == ["saya", "lahir", "tahun", "2.0", "0.0", "1.0", "2.0"], comp[38]
    comp[38] = ["saya", "lahir", "tahun", "2.0", "ribu", "12.0"]
    return {(folder, k): comp[int(float(r["A"]))]
            for name, folder in EMO.items() for k, r in enumerate([r for r in sh[name][1:] if r.get("A")], 1)}


def motion(path):
    """Motion signal per frame: number of pixels with |flow| > THRESH on frames 360 pixels wide."""
    cap, prev, out = cv2.VideoCapture(path), None, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        g = cv2.cvtColor(cv2.resize(f, (WIDTH, round(h * WIDTH / w))), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, **FB)
            out.append(int(np.count_nonzero(np.hypot(fl[..., 0], fl[..., 1]) > THRESH)))
        prev = g
    cap.release()
    return np.array([out[0] if out else 0] + out, float)  # frame 0 takes the value of frame 1


def boundaries(sig, n, a, b, min_seg=MIN_SEG):
    """n-1 boundaries in [a, b) with the minimum total signal; every segment >= min_seg. None if they do not fit."""
    k, T = n - 1, b - a
    if k <= 0:
        return []
    if T < n * min_seg:
        return None
    s = sig[a:b]
    cost, arg = np.full((k, T), np.inf), np.zeros((k, T), int)
    cost[0, min_seg:] = s[min_seg:]
    for j in range(1, k):
        best, bi = np.inf, -1
        for t in range(T):
            p = t - min_seg
            if p >= 0 and cost[j - 1, p] < best:
                best, bi = cost[j - 1, p], p
            if bi >= 0:
                cost[j, t], arg[j, t] = best + s[t], bi
    t = int(np.argmin(cost[k - 1, :T - min_seg + 1]))
    if not np.isfinite(cost[k - 1, t]):
        return None
    out = [t]
    for j in range(k - 1, 0, -1):
        t = int(arg[j, t])
        out.append(t)
    return [a + x for x in reversed(out)]


def pure_segments(sig, min_seg=MIN_SEG):
    """Plain JIKI 2024 diagnostic without N: a segment is a run of moving frames; segments < min_seg join a neighbour."""
    moving, runs, i = sig > 0, [], 0
    while i < len(moving):
        j = i
        while j < len(moving) and moving[j] == moving[i]:
            j += 1
        if moving[i]:
            runs.append([i, j])
        i = j
    merged = []
    for r in runs:
        if merged and (r[1] - r[0] < min_seg or merged[-1][1] - merged[-1][0] < min_seg):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return len(merged)


def run_one(path):
    rel = os.path.relpath(path, f"{DATA}/darmawan")
    cache = f"{DATA}/motion/" + rel[:-4] + ".npy"
    t0 = time.time()
    if os.path.exists(cache):
        sig = np.load(cache)
    else:
        sig = motion(path)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, sig)
    return rel, sig, time.time() - t0


def hits(auto, ref, tol=6):
    """Number of `auto` boundaries with a match in `ref` within ±tol frames (one-to-one, nearest first)."""
    pairs = sorted((abs(x - y), i, j) for i, x in enumerate(auto) for j, y in enumerate(ref) if abs(x - y) <= tol)
    ua, ur = set(), set()
    for _, i, j in pairs:
        if i not in ua and j not in ur:
            ua.add(i)
            ur.add(j)
    return len(ua)


def sample():
    """Validation sample: 10% of the videos of each signer (seed 0). The first half is 'pilih', the rest 'gerbang'."""
    rng, by = random.Random(0), defaultdict(list)
    for e in EXPRESSIONS:
        for p in sorted(glob.glob(f"{DATA}/darmawan/sentence/{e}/*.mp4")):
            by[int(re.search(r"subject-(\d+)", p).group(1))].append(os.path.relpath(p, f"{DATA}/darmawan"))
    n = sum(len(v) for v in by.values())
    if n != 1100 * len(EXPRESSIONS):  # the sample would change if drawn from incomplete data
        sys.exit(f"{'+'.join(EXPRESSIONS)} has only {n}/{1100 * len(EXPRESSIONS)} videos: the validation sample needs the complete data")
    out = []
    for s in sorted(by):
        pick = rng.sample(by[s], max(1, round(0.1 * len(by[s]))))
        out += [(rel, "pilih" if i < (len(pick) + 1) // 2 else "gerbang") for i, rel in enumerate(pick)]
    return out


def pilot(k=5, seed=1):
    """Pilot comparison of the cutters A, B, FAST (exploratory, aggregates only): k videos per signer outside the
    validation sample, so the annotation stays blind."""
    held = {rel for rel, _ in sample()}
    rng, by = random.Random(seed), defaultdict(list)
    for p in sorted(glob.glob(f"{DATA}/darmawan/sentence/neutral/*.mp4")):
        rel = os.path.relpath(p, f"{DATA}/darmawan")
        if rel not in held:
            by[int(re.search(r"subject-(\d+)", p).group(1))].append(rel)
    return [rel for s in sorted(by) for rel in rng.sample(by[s], k)]


def pilot_report(root=f"{DATA}/segpilot", seeds=20):
    """Pilot of A, B, C (FAST) over neutral_pilot.csv, aggregates only: number of segments without N vs N, and boundary
    agreement with N between candidates (±6 frames). Baseline: random boundaries that obey the same DP rule, averaged
    over `seeds` draws."""
    rels = [r["relpath"] for r in csv.DictReader(open(f"{QC}/neutral_pilot.csv"))]
    own_b = {r["relpath"][:-4] + ".mp4": int(r["n_segmen_model"]) for r in csv.DictReader(open(f"{root}/segprobs/own_segments.csv"))}
    own_c = {r["relpath"]: int(r["n_onset"]) for r in csv.DictReader(open(f"{root}/fast_own.csv"))}
    sig = {"A": lambda rel: np.load(f"{root}/motion/{rel[:-4]}.npy"),
           "B": lambda rel: -np.load(f"{root}/segprobs/{rel[:-4]}.npy"),
           "C": lambda rel: -np.load(f"{root}/fastprobs/{rel[:-4]}.npy")}
    rng, diff, bounds = np.random.default_rng(0), defaultdict(list), defaultdict(dict)
    for rel in rels:
        n = len(gloss_of(rel))
        a, b = span(np.load(f"{root}/keypoints/{rel[:-4]}.npy"))
        own = {"A": pure_segments(sig["A"](rel)), "B": own_b[rel], "C": own_c[rel]}
        for m in "ABC":
            diff[m].append(own[m] - n)
            bounds[m][rel] = boundaries(sig[m](rel), n, a, b)
        bounds["random"][rel] = [boundaries(rng.random(b), n, a, b) for _ in range(seeds)]
    for m in "ABC":
        d = np.array(diff[m])
        print(f"{m} without N: segments = N in {np.mean(d == 0):.0%} of videos; median difference {np.median(d):+.0f}, "
              f"fewer {np.mean(d < 0):.0%}, more {np.mean(d > 0):.0%} | failures with N: {sum(v is None for v in bounds[m].values())}")
    for x, y in (("A", "B"), ("A", "C"), ("B", "C"), ("A", "random"), ("B", "random"), ("C", "random")):
        h = t = 0
        for rel in rels:
            ref = bounds[y][rel] if y != "random" else [r for r in bounds[y][rel] if r]
            if bounds[x][rel] and ref:
                h += hits(bounds[x][rel], ref) if y != "random" else np.mean([hits(bounds[x][rel], r) for r in ref])
                t += len(bounds[x][rel])
        print(f"boundary agreement {x}–{y} (with N): {h / max(t, 1):.0%} within ±6 frames ({t} boundaries)")


RELABEL = os.path.join(os.environ.get("SIBI_QC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qc")),
                       "relabel_s18.csv")


@functools.lru_cache(maxsize=1)
def relabel():
    """{relpath: true sentence number} for mislabeled files: neutral videos of signer 18 whose content is another
    sentence than the number in the file name (checked visually). relabel_s18.csv has the columns relpath,
    kalimat_berkas (number in the file name), kalimat_sebenarnya (true number). Missing file -> empty."""
    if not os.path.exists(RELABEL):
        return {}
    return {r["relpath"]: int(r["kalimat_sebenarnya"]) for r in csv.DictReader(open(RELABEL))
            if r["kalimat_sebenarnya"] != r["kalimat_berkas"]}


def gloss_of(rel):
    """Sign sequence of a sentence video from the composition table; sentence number from the file name, unless relabeled."""
    emo, k = re.search(r"sentence-(\d+)-([A-Za-z]+)-rep", rel).groups()[::-1]
    return glosses()[(emo.lower(), relabel().get(rel, int(k)))]


def word_iou(bounds, points, a, b):
    """IoU per word. Automatic: N-1 boundaries in the span [a, b). Manual: N+1 points (start, N-1 boundaries, exclusive
    end)."""
    A = [a] + list(bounds) + [b]
    return [max(0, min(A[k + 1], points[k + 1]) - max(A[k], points[k])) /
            max(1, max(A[k + 1], points[k + 1]) - min(A[k], points[k])) for k in range(len(A) - 1)]


GATE = ("A", "B", "C", "D", "E1", "E2", "E3", "F+")   # tie-break order
FILES = {"D": f"{DATA}/kata/align_D.csv", "E1": f"{DATA}/kata/hal_points.csv", "E2": f"{DATA}/kata/atba_points.csv",
         "E3": f"{DATA}/kata/ctc_points.csv", "F+": f"{DATA}/kata/fplus_loso_points.csv"}


def iou_summary(points=f"{DATA}/kata/fplus_loso_points.csv", manual=f"{QC}/words_manual.csv", out=f"{QC}/kata_iou_ringkasan.csv"):
    """IoU per word of the selected cutter vs the manual points, per half of the sample (pilih, gerbang). Aggregates only,
    so the annotation stays blind to the automatic boundaries of each video. A video with the wrong number of points
    counts as IoU 0."""
    ref = {r["relpath"]: [int(x) for x in r["titik"].split()] for r in csv.DictReader(open(manual))}
    pts = {r["relpath"]: [int(x) for x in r["titik"].split()] for r in csv.DictReader(open(points))}
    iou = {"pilih": [], "gerbang": []}
    for rel, half in sample():
        n, c = len(gloss_of(rel)), pts.get(rel)
        iou[half] += [0.0] * n if c is None or len(c) != n + 1 else word_iou(c[1:-1], ref[rel], c[0], c[-1])
    edges, rows = [0, .5, .6, .7, .8, .9, 1.0001], []
    for h, v in iou.items():
        v, k = np.array(v), np.histogram(v, edges)[0]
        rows.append(dict(paruh=h, kata=len(v), rata_rata=round(float(v.mean()), 3), median=round(float(np.median(v)), 3),
                         persentil_10=round(float(np.percentile(v, 10)), 3), persentil_25=round(float(np.percentile(v, 25)), 3),
                         iou_min_0_5=round(float(np.mean(v >= 0.5)), 4),
                         **{f"iou_{a:.1f}_{min(b, 1):.1f}": int(x) for a, b, x in zip(edges, edges[1:], k)}))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    return rows


def gate(root=f"{DATA}/segsample", manual=f"{QC}/words_manual.csv", files=FILES):
    """Selects the word cutter (A JIKI, B pose, C FAST, D forced alignment, E1 HAL, E2 ATBA, E3 CTC, F+): the highest mean
    IoU per word on the 'pilih' half (ties < 0.01 follow GATE), then the test requires >= 80% of words with IoU >= 0.5 on
    the 'gerbang' half. The reference is the manual boundaries (`manual`). A, B, C use the signals under `root`; the
    other candidates read their point CSVs (F+ its leave-one-signer-out predictions)."""
    smp = sample()
    rows = {r["relpath"]: r for r in csv.DictReader(open(manual))}
    done = sum(rel in rows and rows[rel]["flag"] != "1" and int(rows[rel]["n_isyarat"]) == len(gloss_of(rel)) for rel, _ in smp)
    if done < len(smp):
        sys.exit(f"only {done}/{len(smp)} reference videos are complete (not flagged, current N). The test needs all of them.")
    pts = {k: {r["relpath"]: [int(x) for x in r["titik"].split()] for r in csv.DictReader(open(f))} for k, f in files.items()}
    sig = {"A": lambda rel: np.load(f"{root}/motion/{rel[:-4]}.npy"),
           "B": lambda rel: -np.load(f"{root}/segprobs/{rel[:-4]}.npy"),
           "C": lambda rel: -np.load(f"{root}/fastprobs/{rel[:-4]}.npy")}
    iou = {k: {"pilih": [], "gerbang": []} for k in GATE}
    for rel, half in smp:
        n = len(gloss_of(rel))
        a, b = span(np.load(f"{root}/keypoints/{rel[:-4]}.npy"))
        ref = [int(x) for x in rows[rel]["titik"].split()]
        cut = {k: None if (bd := boundaries(f(rel), n, a, b)) is None else [a] + bd + [b] for k, f in sig.items()}
        cut.update({k: pts[k].get(rel) for k in pts})
        for k, c in cut.items():
            # failed segmentation or not N+1 points: every word gets IoU 0
            iou[k][half] += [0.0] * n if c is None or len(c) != n + 1 else word_iou(c[1:-1], ref, c[0], c[-1])
    m = {k: float(np.mean(iou[k]["pilih"])) for k in GATE}
    win = next(k for k in GATE if m[k] > max(m.values()) - 0.01)
    ok = float(np.mean(np.array(iou[win]["gerbang"]) >= 0.5))
    print("half 'pilih': mean IoU per word " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()) + f" -> selected {win}")
    print(f"half 'gerbang' ({win}): {ok:.1%} of words IoU >= 0.5; threshold 80% -> {'PASS' if ok >= 0.8 else 'FAIL: words exploratory only'}")
    return win, m, ok


def signal(kind, rel):
    """Cached boundary signal, or None. A: Farneback motion; B: -log p(sign start)."""
    f = (f"{DATA}/motion/" if kind == "jiki" else f"{DATA}/segprobs/") + rel[:-4] + ".npy"
    if not os.path.exists(f):
        return None
    return np.load(f) if kind == "jiki" else -np.load(f)


def main(kind, *patterns):
    files = sorted(f for p in patterns for f in glob.glob(f"{DATA}/darmawan/{p}"))
    rels = [os.path.relpath(p, f"{DATA}/darmawan") for p in files]
    secs = []
    if kind == "jiki":
        with multiprocessing.Pool(6) as pool:
            secs = [dt for _, _, dt in pool.map(run_one, files)]
        own = {rel: pure_segments(signal("jiki", rel)) for rel in rels}
    else:
        with open(f"{DATA}/segprobs/own_segments.csv") as fh:
            own = {r["relpath"][:-4] + ".mp4": int(r["n_segmen_model"]) for r in csv.DictReader(fh)}
    other = "pose" if kind == "jiki" else "jiki"
    fail, diff, spans, agree, total = 0, [], [], 0, 0
    for rel in rels:
        n = len(gloss_of(rel))
        a, b = span(np.load(f"{DATA}/keypoints/" + rel[:-4] + ".npy"))
        spans.append((b - a) / n)
        mine = boundaries(signal(kind, rel), n, a, b)
        fail += mine is None
        diff.append(own[rel] - n)
        ref = signal(other, rel)
        if mine and ref is not None and (theirs := boundaries(ref, n, a, b)):
            agree, total = agree + hits(mine, theirs), total + len(mine)
    d = np.array(diff)
    if secs:
        print(f"{len(rels)} videos, mean {np.mean(secs):.1f} s/video (optical flow)")
    print(f"candidate {kind} (N known): {len(rels) - fail} segmented, {fail} failed (active span < N x {MIN_SEG} frames)")
    print(f"  active frames per sign: median {np.median(spans):.1f}, bottom 10% {np.percentile(spans, 10):.1f}")
    print(f"diagnostic without N ({'plain JIKI' if kind == 'jiki' else 'own model segments'}): segments = signs "
          f"in {np.sum(d == 0)}/{len(rels)} videos; median difference {np.median(d):+.0f}, fewer {np.mean(d < 0):.0%}, more {np.mean(d > 0):.0%}")
    if total:
        print(f"boundary agreement {kind} vs {other}: {agree}/{total} = {agree / total:.0%} within ±6 frames")


def demo():
    s = np.ones(60)
    s[[20, 40]] = 0
    assert boundaries(s, 3, 0, 60) == [20, 40]
    assert boundaries(s, 5, 0, 60) is None                     # 5 signs x 15 frames > 60
    assert boundaries(s, 2, 5, 60) == [20]                     # search starts at the span start; result in video frames
    assert pure_segments(np.array([0, 5, 5, 0] + [5] * 20 + [0] * 3 + [5] * 20, float)) == 2  # 2-frame blip merged
    assert hits([10, 20, 30], [14, 27, 40]) == 2                # a reference boundary is matched only once
    assert word_iou([20], [0, 20, 40], 0, 40) == [1.0, 1.0]     # identical cuts
    assert word_iou([30], [0, 20, 40], 0, 40) == [20 / 30, 10 / 20]
    img = np.zeros((40, 40), np.uint8)
    img[10:20, 10:20] = 255
    fl = cv2.calcOpticalFlowFarneback(img, np.roll(img, 8, axis=1), None, **FB)
    assert np.count_nonzero(np.hypot(fl[..., 0], fl[..., 1]) > THRESH) > 0   # an 8-pixel shift is detected
    print("demo ok: boundary DP, short span, blip merging, Farneback motion detection")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    elif sys.argv[1] == "gate":
        gate()
    elif sys.argv[1] == "iou":                                # iou [points.csv]: IoU summary per half -> qc/kata_iou_ringkasan.csv
        iou_summary(*sys.argv[2:3])
    else:
        main(sys.argv[1], *sys.argv[2:])
