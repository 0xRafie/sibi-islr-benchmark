"""Extract MediaPipe Holistic keypoints, 1662 values per frame, and cache them as .npy.

Extraction runs once; training reads the cache. MediaPipe is pinned to 0.10.14 because the legacy
`mp.solutions.holistic` API is gone in 1.x. Holistic runs with model_complexity=1 and detection and tracking confidence
0.5. A landmark set that is not detected in a frame is stored as zeros.

Modes:
- default: the SIBI dictionary videos `<SIBI_DICT>/video/*.webm` -> `<SIBI_DICT>/keypoints/<stem>.npy`, plus the
  per-video hand-detection rate in `<SIBI_QC>/keypoint_stats.csv`. Missed hands are a candidate cause of the lower
  accuracy on affixes, and SIBI papers do not report this rate.
- `--darmawan PATTERN`: videos `<SIBI_DATA>/darmawan/PATTERN` -> `<SIBI_DATA>/keypoints/<relative path>.npy`. The relative
  path `<category>/<class>/<file>` is the clip key. Keypoints are derived from the dataset and stay outside this repository.
- `--jobs JOBS.json OUT.npz`: JOBS = [[video, relpath], ...] -> one npz (raw, off, relpath, w, h, fps), with the
  per-clip arrays cached in `OUT.npz.d/`.

Paths (environment variables): SIBI_DATA (default `data`), SIBI_QC (default `qc`), SIBI_DICT (default: working directory).

    python scripts/extract.py --demo
    python scripts/extract.py
    python scripts/extract.py --darmawan 'sentence/fear/*-rep-1.mp4'
    python scripts/extract.py --jobs jobs.json out.npz
"""
import csv, glob, json, multiprocessing, os, sys
import cv2
import numpy as np
import mediapipe as mp

DATA = os.environ.get("SIBI_DATA", "data")
QC = os.environ.get("SIBI_QC", "qc")
DICT = os.environ.get("SIBI_DICT", "")
OUT = os.path.join(DICT, "keypoints")
# 1662 features (pose 132 + face 1404 + two hands 126): the layout of Ho & Santoso, so results stay directly comparable.
N_POSE, N_FACE, N_HAND = 33 * 4, 468 * 3, 21 * 3


def frame_to_vec(res):
    """One frame -> 1662-vector. Missing landmarks become zeros."""
    pose = (np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks.landmark]).flatten()
            if res.pose_landmarks else np.zeros(N_POSE))
    face = (np.array([[p.x, p.y, p.z] for p in res.face_landmarks.landmark]).flatten()
            if res.face_landmarks else np.zeros(N_FACE))
    lh = (np.array([[p.x, p.y, p.z] for p in res.left_hand_landmarks.landmark]).flatten()
          if res.left_hand_landmarks else np.zeros(N_HAND))
    rh = (np.array([[p.x, p.y, p.z] for p in res.right_hand_landmarks.landmark]).flatten()
          if res.right_hand_landmarks else np.zeros(N_HAND))
    return np.concatenate([pose, face, lh, rh]), (res.left_hand_landmarks is not None,
                                                  res.right_hand_landmarks is not None)


def extract(path, holistic):
    cap = cv2.VideoCapture(path)
    vecs, lh_hit, rh_hit = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        vec, (l, r) = frame_to_vec(res)
        vecs.append(vec)
        lh_hit += l
        rh_hit += r
    cap.release()
    return np.array(vecs, dtype=np.float32), lh_hit, rh_hit


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DICT, "video", "*.webm")))
    stats = []
    with mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as holistic:
        for i, path in enumerate(files, 1):
            stem = os.path.basename(path)[:-5]
            npy = f"{OUT}/{stem}.npy"
            if os.path.exists(npy):
                arr = np.load(npy)
                kat = stem.split("__")[0]
                stats.append(dict(kelas=stem, kategori=kat, frame=len(arr),
                                  lh_pct=-1, rh_pct=-1, any_hand_pct=-1))
                continue
            arr, lh, rh = extract(path, holistic)
            np.save(npy, arr)
            n = max(len(arr), 1)
            kat = stem.split("__")[0]
            # a frame without any hand = both hand vectors are zero
            no_hand = int(np.sum(np.all(arr[:, -2 * N_HAND:] == 0, axis=1)))
            stats.append(dict(kelas=stem, kategori=kat, frame=len(arr),
                              lh_pct=round(100 * lh / n, 1),
                              rh_pct=round(100 * rh / n, 1),
                              any_hand_pct=round(100 * (n - no_hand) / n, 1)))
            print(f"[{i:3d}/{len(files)}] {stem:40s} {len(arr):4d}f  "
                  f"hand detected {stats[-1]['any_hand_pct']:5.1f}%", flush=True)

    with open(f"{QC}/keypoint_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)
    print(f"\nDone. {len(stats)} files -> {OUT}/ , statistics -> {QC}/keypoint_stats.csv")


def demo():
    """Self-check: the feature layout is right and an empty frame still gives a 1662-vector."""
    class Empty:
        pose_landmarks = face_landmarks = left_hand_landmarks = right_hand_landmarks = None
    vec, hits = frame_to_vec(Empty())
    assert vec.shape == (1662,), vec.shape
    assert not any(hits)
    assert N_POSE + N_FACE + 2 * N_HAND == 1662
    print("demo ok: 1662-vector, empty frame safe")


def extract_one(job):
    """One dataset video. A new Holistic per video, so tracking does not carry over between videos."""
    src, out = job
    if os.path.exists(out):
        return out, -1
    with mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as holistic:
        arr, _, _ = extract(src, holistic)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, arr)
    return out, len(arr)


def main_darmawan(pattern, workers=6):
    files = sorted(glob.glob(f"{DATA}/darmawan/{pattern}"))
    jobs = [(p, f"{DATA}/keypoints/" + os.path.relpath(p, f"{DATA}/darmawan")[:-4] + ".npy") for p in files]
    with multiprocessing.Pool(workers) as pool:
        for i, (out, n) in enumerate(pool.imap_unordered(extract_one, jobs), 1):
            print(f"[{i}/{len(jobs)}] {out} {'exists' if n < 0 else f'{n}f'}", flush=True)


def main_jobs(jobs_json, out_npz, workers=4):
    """Same settings as `--darmawan` (extract_one). w, h = size of the decoded frame (what MediaPipe sees), used to make
    coordinates isotropic; fps is used to scale trimming to the frame rate."""
    jobs = json.load(open(jobs_json))
    work = [(src, f"{out_npz}.d/{rel}.npy") for src, rel in jobs]
    with multiprocessing.Pool(workers) as pool:
        for i, (out, n) in enumerate(pool.imap_unordered(extract_one, work), 1):
            print(f"[{i}/{len(work)}] {out} {n}f", flush=True)
    arrs, props = [np.load(o).reshape(-1, 1662) for _, o in work], []
    for src, _ in jobs:
        cap = cv2.VideoCapture(src)
        ok, frame = cap.read()
        props.append((*frame.shape[1::-1], cap.get(cv2.CAP_PROP_FPS)) if ok else (0, 0, 0))
        cap.release()
    p = np.array(props, dtype=np.float64)
    np.savez_compressed(out_npz, raw=np.concatenate(arrs), off=np.cumsum([0] + [len(a) for a in arrs]),
                        relpath=np.array([r for _, r in jobs]), w=p[:, 0], h=p[:, 1], fps=p[:, 2])
    print(f"{out_npz}: {len(arrs)} clips, {sum(map(len, arrs))} frames, {sum(len(a) == 0 for a in arrs)} empty")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    elif "--jobs" in sys.argv:
        i = sys.argv.index("--jobs")
        main_jobs(sys.argv[i + 1], sys.argv[i + 2])
    elif "--darmawan" in sys.argv:
        main_darmawan(sys.argv[sys.argv.index("--darmawan") + 1])
    else:
        main()
