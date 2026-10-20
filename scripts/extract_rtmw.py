"""Extract RTMW whole-body keypoints (the Uni-Sign input) from a folder of videos.

rtmlib `Wholebody(mode="balanced")` (YOLOX-m detector + RTMW 256x192) runs on every frame, as in Uni-Sign
`demo/pose_extraction.py` (Li et al., Uni-Sign, ICLR 2025; rtmlib by Tau-J). Of six extractors compared with the
released Uni-Sign pose files, this one came closest. The first person is kept and the keypoints are divided by the
frame width and height; a frame without a person gets zeros.

Every `*.mp4` under VIDEO_ROOT is one clip. Its class is the name of its parent folder, and its relpath is
`<category>/<class>/<video stem>`, with the category from --category, else the name of VIDEO_ROOT. OUT.npz holds the
clips sorted by relpath:
    kp       float16 (frames, 133, 2)   COCO-WholeBody keypoints, x / W and y / H
    sc       float16 (frames, 133)      keypoint scores
    off      (clips + 1,)               clip i is frames off[i]:off[i + 1]
    relpath  (clips,)
Only keypoints are written, never video.

The clips are split over --procs processes per GPU (2 on a T4); each process writes a shard, and the shards are merged
at the end. Every frame is processed on its own, so the output does not depend on the number of processes.

Setup (CUDA 12): `pip uninstall -y onnxruntime && pip install onnxruntime-gpu==1.22.0 && pip install --no-deps rtmlib`
(--no-deps, because rtmlib would pull in the CPU build of onnxruntime). torch is imported only to load the CUDA
libraries for onnxruntime and to count the GPUs.

    python scripts/extract_rtmw.py --demo
    python scripts/extract_rtmw.py data/darmawan/affix data/pilot/rtmw_affix.npz [--category affix] [--procs 2] [--limit 40]
"""
import argparse, glob, json, os, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np


def list_clips(root, cat, limit=None):
    """[(category, class, path)] of every *.mp4 under `root`, class = parent folder; `limit` keeps an evenly spaced subset
    (smoke test)."""
    clips = sorted({(cat, v.parent.name, str(v)) for v in Path(root).rglob('*.mp4')})
    return clips[::max(1, len(clips) // limit)][:limit] if limit else clips


def frame_kp(wb, fr):
    """First person in one BGR frame -> keypoints (133, 2) divided by (W, H) and scores (133,), float16; zeros if there
    is no person."""
    H, W = fr.shape[:2]
    p, s = wb(fr)
    if len(p):
        return (p[0] / np.array([W, H])).astype(np.float16), s[0].astype(np.float16)
    return np.zeros((133, 2), np.float16), np.zeros(133, np.float16)


def extract(clips, wb, shard=0):
    """Every frame of every clip -> shard arrays (kp, sc, off, rel)."""
    import cv2
    kp, sc, off, rel, t0, nf = [], [], [0], [], time.time(), 0
    for k, (cat, cls, path) in enumerate(clips):
        cap = cv2.VideoCapture(path)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            p, s = frame_kp(wb, fr)
            kp.append(p); sc.append(s)
            nf += 1
        off.append(len(kp)); rel.append(cat + '/' + cls + '/' + path.rsplit('/', 1)[1][:-4])
        if k % 200 == 0:
            print('shard', shard, k, '/', len(clips), 'clips', round(nf / (time.time() - t0), 1), 'fps', flush=True)
    print('shard', shard, 'done:', len(clips), 'clips', nf, 'frames', round(nf / (time.time() - t0), 1), 'fps', flush=True)
    return dict(kp=np.stack(kp), sc=np.stack(sc), off=np.array(off), rel=np.array(rel))


def worker(spec, clips_json, out):
    """One process: shard i of n (spec "i/n") of the clip list -> `out` npz."""
    import torch, onnxruntime as ort   # torch first: its CUDA libraries serve onnxruntime
    ort.preload_dlls()
    from rtmlib import Wholebody
    i, n = map(int, spec.split('/'))
    clips = json.load(open(clips_json))[i::n]
    wb = Wholebody(mode='balanced', backend='onnxruntime', device='cuda')
    np.savez(out, **extract(clips, wb, i))


def merge(shards, out):
    """Shard npz files -> `out` with the clips sorted by relpath."""
    sh = [dict(np.load(f)) for f in shards]   # load once: every NpzFile access decompresses again
    items = {}
    for d in sh:
        for j, r in enumerate(d['rel']):
            a, b = d['off'][j], d['off'][j + 1]
            items[str(r)] = (d['kp'][a:b], d['sc'][a:b])
    keys = sorted(items)
    np.savez_compressed(out, kp=np.concatenate([items[k][0] for k in keys]),
                        sc=np.concatenate([items[k][1] for k in keys]),
                        off=np.cumsum([0] + [len(items[k][0]) for k in keys]), relpath=np.array(keys))
    print(out, len(keys), 'clips', round(os.path.getsize(out) / 1e6), 'MB')


def main(root, out, cat=None, procs=2, limit=None):
    import torch, onnxruntime as ort
    ort.preload_dlls()
    print(torch.__version__, ort.__version__, ort.get_available_providers(), torch.cuda.device_count())
    assert 'CUDAExecutionProvider' in ort.get_available_providers()
    from rtmlib import Wholebody
    Wholebody(mode='balanced', backend='onnxruntime', device='cuda')   # download the ONNX models once; parallel processes would race for the cache
    clips = list_clips(root, cat or Path(root).name, limit)
    assert clips, f'no *.mp4 under {root}'
    print(len(clips), 'clips')
    work = os.path.dirname(os.path.abspath(out))
    os.makedirs(work, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as t:
        json.dump(clips, open(f'{t}/clips.json', 'w'))
        g = max(1, torch.cuda.device_count()); n = g * procs
        ps = [subprocess.Popen([sys.executable, os.path.abspath(__file__), '--shard', f'{i}/{n}', f'{t}/clips.json',
                                f'{t}/shard{i}.npz'],
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % g), OMP_NUM_THREADS='1'),
                               stdout=open(f'{t}/shard{i}.log', 'w'), stderr=subprocess.STDOUT) for i in range(n)]
        codes = [p.wait() for p in ps]
        for i in range(n):
            print(open(f'{t}/shard{i}.log').read()[-600:])
        print('exit codes:', codes)
        merge(sorted(glob.glob(f'{t}/shard*.npz')), out)


def demo():
    import cv2

    def fake(fr):   # stands in for rtmlib Wholebody: two people in a bright frame, nobody in a dark one
        H, W = fr.shape[:2]
        if fr.mean() < 128:
            return np.zeros((0, 133, 2)), np.zeros((0, 133))
        return np.stack([np.tile([W / 2, H / 4], (133, 1)), np.zeros((133, 2))]), np.stack([np.full(133, 0.9), np.zeros(133)])

    with tempfile.TemporaryDirectory() as t:
        for cls, n in (('an', 3), ('ber', 2)):   # first frame dark, the rest bright
            os.makedirs(f'{t}/affix/{cls}')
            w = cv2.VideoWriter(f'{t}/affix/{cls}/subject-1-affix-{cls}-rep-1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 48))
            for j in range(n):
                w.write(np.full((48, 64, 3), 255 if j else 0, np.uint8))
            w.release()
        clips = list_clips(f'{t}/affix', 'affix')
        assert [c[:2] for c in clips] == [('affix', 'an'), ('affix', 'ber')] and list_clips(f'{t}/affix', 'affix', 1) == clips[:1]
        np.savez(f'{t}/shard0.npz', **extract(clips[::-1], fake))   # reversed: the merge must sort
        merge([f'{t}/shard0.npz'], f'{t}/out.npz')
        z = np.load(f'{t}/out.npz')
        assert list(z['relpath']) == ['affix/an/subject-1-affix-an-rep-1', 'affix/ber/subject-1-affix-ber-rep-1']
        assert list(z['off']) == [0, 3, 5] and z['kp'].shape == (5, 133, 2) and z['sc'].shape == (5, 133)
        assert z['kp'].dtype == z['sc'].dtype == np.float16
        assert not z['kp'][[0, 3]].any() and not z['sc'][[0, 3]].any()                        # dark frames: zeros
        assert np.allclose(z['kp'][[1, 2, 4]], [0.5, 0.25]) and np.allclose(z['sc'][[1, 2, 4]], 0.9, atol=1e-3)
    print('demo ok: clip list and relpath, first person divided by (W, H), zeros without a person, float16, merge sorted')


if __name__ == '__main__':
    if '--demo' in sys.argv:
        demo()
    elif sys.argv[1:2] == ['--shard']:   # internal: one worker process started by main()
        worker(*sys.argv[2:5])
    else:
        ap = argparse.ArgumentParser(description='RTMW keypoints (rtmlib Wholebody, balanced) of every *.mp4 under video_root')
        ap.add_argument('video_root')
        ap.add_argument('out', help='output .npz')
        ap.add_argument('--category', help='relpath prefix (default: name of video_root)')
        ap.add_argument('--procs', type=int, default=2, help='processes per GPU')
        ap.add_argument('--limit', type=int, help='evenly spaced subset of this many clips (smoke test)')
        a = ap.parse_args()
        main(a.video_root, a.out, a.category, a.procs, a.limit)
