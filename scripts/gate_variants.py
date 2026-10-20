"""Replication check, round 2 (exploratory): spatial normalizations and learning-rate schedules for the BiLSTM baseline.

The aim was the accuracy of Ho & Santoso (>= 68.17% on affixes), not their exact normalization or mask. VARIANTS lists
each variant as (normalization, schedule); the normalizations from the literature are implemented in extra(). All
variants use the folds in gate_folds_<category>.csv, resampling to 30 frames, the last epoch, and no validation set.
Other scripts reuse POSE_XY, DROP_ALPHABET, DUP_WRONG, LR, features(), normalize(), and shoulder_cw().

Data rules: DUP_WRONG (used with `am7`, and by protocol_counts.clips()) drops only the wrongly labeled side of a content
duplicate across classes or signers; DUP_BOTH (the default here) drops both sides.

Paths: under SIBI_DATA (default `data`), keypoints are read from keypoints/ (keypoints_hs/ with `hs`, see gate.py) and
packages are written to pilot/gate_raw*.npz. SIBI_QC (default `qc`) holds gate_folds_<category>.csv. GATE_FOLDS
overrides the fold file and GATE_OUT the result file.

    python scripts/gate_variants.py --demo
    python scripts/gate_variants.py pack [hs] [number|alphabet] [am7]   # data/pilot/gate_raw[_cat][_hs][_am7].npz + qc/gate_folds_<cat>[_am7].csv
    python scripts/gate_variants.py screen V0 V1 ...      # seed 0, 5 folds (GPU)
    python scripts/gate_variants.py full V3               # 5 seeds, 5 folds
"""
import csv, glob, json, os, sys, time
import numpy as np

from trim import span
from gate import DROP, NAME, N_HAND, fill_gaps, resample, B1, model

DATA = os.environ.get("SIBI_DATA", "data")
QC = os.environ.get("SIBI_QC", "qc")
# columns kept from the 1662 layout: pose xy of 33 points + hands xy of 2 x 21
POSE_XY = np.array([[i * 4, i * 4 + 1] for i in range(33)]).ravel()
L_SH, R_SH = 11, 12
FACE = list(range(11))                           # pose 0-10 (nose, eyes, ears, mouth): stand-in for face landmarks
BODY = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]   # upper body of Ko 2019 / Kim & Baek 2023, without the neck (= shoulder centre)

VARIANTS = {
    # name: (normalization, LR schedule)
    "V0": ("raw", "b1"),            # control; should be close to the 35.75% of round 1
    "V1": ("shoulder", "b1"),       # shift to the shoulder centre, divide by the shoulder width
    "V2": ("shoulder_wrist", "b1"), # V1 for the wrist; other points relative to the wrist, divided by the hand size
    "V3": ("shoulder", "slow"),     # V1 + Adam 1e-3, StepLR step 20 gamma 0.5
    "V4": ("shoulder_wrist", "slow"),
    # one variant per normalization from the literature, with the V3 schedule; defined in extra()
    "V5": ("spoter", "slow"),
    "V6": ("gislr1", "slow"),
    "V7": ("raw", "slow"),          # control: no spatial normalization (Ho & Santoso, JAIC 2025), V3 schedule
    "V8": ("signspace", "slow"),
    "V9": ("signbart", "slow"),
    "V10": ("unisign", "slow"),
    "V11": ("kimbaek", "slow"),
    "V12": ("ko", "slow"),
    "V13": ("yaslc", "slow"),
    "V14": ("yaslf", "slow"),
    "V15": ("com", "slow"),
    "V16": ("nose", "slow"),
    "V17": ("gislr6", "slow"),
    "V18": ("maxdist", "slow"),
    # Ho & Santoso (JUTIF 2026): raw coordinates standardized with training-data statistics only, per fold in run()
    "V19": ("zscore_train", "b1"),
    "V20": ("zscore_train", "slow"),
}
EXTRA = {how for how, _ in VARIANTS.values()} - {"raw", "shoulder", "shoulder_wrist", "zscore_train"}  # handled by extra()
LR = {"b1": (B1["lr"], B1["step"], B1["gamma"]), "slow": (1e-3, 20, 0.5)}


# Content duplicates across classes or signers: both sides dropped.
# Affix pe/se of signer 10 is NOT dropped here, so the affix replication subset stays as it was first recorded.
DUP_BOTH = {("number", "17", 10), ("number", "18", 10), ("number", "60", 10), ("number", "70", 10),
            ("number", "20", 22), ("number", "30", 21)}


# Final rule: only the wrongly labeled side is dropped (checked visually).
DUP_WRONG = {("affix", "se", 10), ("number", "18", 10), ("number", "70", 10), ("number", "30", 21),
             # alphabet: f/g/k of signer 20 = copies of signer 19; x of signer 20 = w
             ("alphabet", "f", 20), ("alphabet", "g", 20), ("alphabet", "k", 20), ("alphabet", "x", 20),
             # number: ribuan of signer 17 = copy of jutaan of signer 17
             ("number", "ribuan", 17)}
# duplicates within the same class and signer: the lowest-numbered rep is kept
DROP_ALPHABET = {"alphabet/h/subject-11-alphabet-h-rep-9", "alphabet/h/subject-12-alphabet-h-rep-9"}


def pack(src=f"{DATA}/keypoints", suffix="", cat="affix", am7=False):
    """Signers 1-20 of one category -> trimmed sequences (pose xy 66 + hands xy 84), concatenated, plus offsets."""
    rows, seqs = [], []
    assert os.path.isdir(f"{src}/{cat}"), f"{src}/{cat} not found: run extract.py --darmawan first"
    for f in sorted(glob.glob(f"{src}/{cat}/*/*.npy")):
        if f.endswith(".reex.npy"):
            continue
        rel = os.path.relpath(f, src)[:-4]
        m = NAME.search(rel)
        if rel in DROP or rel in DROP_ALPHABET or m is None or int(m.group(1)) > 20:  # m is None for rep--N/sep-N, skipped here
            continue
        if (cat, rel.split("/")[1], int(m.group(1))) in (DUP_WRONG if am7 else DUP_BOTH):
            continue
        arr = np.load(f)
        a, b = span(arr)
        arr = arr[a:b]
        hands = arr[:, -2 * N_HAND:].reshape(len(arr), 2, 21, 3)[..., :2].reshape(len(arr), 84)
        seqs.append(np.concatenate([arr[:, POSE_XY], hands], 1).astype(np.float32))
        rows.append((rel, rel.split("/")[1]))
    rel, cls = map(np.array, zip(*rows))
    classes = sorted(set(cls))
    if cat == "affix":
        assert len(rel) == (3588 if am7 else 3598), len(rel)
    suffix += "_am7" if am7 else ""
    off = np.cumsum([0] + [len(s) for s in seqs])
    y = np.array([classes.index(c) for c in cls])
    out = f"{DATA}/pilot/gate_raw{'' if cat == 'affix' else '_' + cat}{suffix}.npz"
    np.savez_compressed(out, seq=np.concatenate(seqs), off=off, relpath=rel, y=y, classes=np.array(classes), cat=cat,
                        fold_csv=fold_file(cat, suffix))
    print(f"{cat}: {len(rel)} clips, {len(classes)} classes, {off[-1]} frames -> {out}")
    fold_csv = fold_file(cat, suffix)
    if cat != "affix" and not os.path.exists(fold_csv):  # the affix folds are fixed by gate.folds()
        from sklearn.model_selection import StratifiedKFold
        fold = np.empty(len(y), int)
        for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y)):
            fold[te] = k
        with open(fold_csv, "w", newline="") as fh:
            csv.writer(fh).writerows([("relpath", "fold")] + sorted(zip(rel, fold)))
        print(fold_csv, np.bincount(fold).tolist(), "clips per fold")


def fold_file(cat, suffix=""):
    """Affix always uses the fixed folds (a subset is matched by relpath); other categories have folds per data variant."""
    return f"{QC}/gate_folds_affix.csv" if cat == "affix" else f"{QC}/gate_folds_{cat}{suffix.replace('_hs', '')}.csv"


def shoulder_cw(pose):
    """pose (T, 33, 2) -> shoulder centre c and shoulder width w, clip means over the frames where both shoulders are
    detected. None if there is no such frame. Used to normalize the hands (and the face in the face ablation)."""
    sh = pose[:, [L_SH, R_SH]]
    ok = np.all(sh != 0, axis=(1, 2))
    if not ok.any():
        return None
    return sh[ok].mean(1).mean(0), np.linalg.norm(sh[ok, 0] - sh[ok, 1], axis=1).mean() + 1e-6


def normalize(seq, how):
    """(T, 150) pose xy + hands xy -> (T, 2, 21, 2) hands, detection (T, 2)."""
    pose = seq[:, :66].reshape(len(seq), 33, 2)
    h = seq[:, 66:].reshape(len(seq), 2, 21, 2).copy()
    det = np.any(h != 0, axis=(2, 3))
    if how in EXTRA:
        h = extra(pose, h, det, how)
        h[~det] = 0
        return h, det
    if how in ("raw", "zscore_train"):                          # zscore_train is standardized per fold in run()
        return h, det
    cw = shoulder_cw(pose)
    if cw is None:
        return h, det
    c, w = cw
    wrist = (h[:, :, :1] - c) / w
    if how == "shoulder":
        h = (h - c) / w
    else:  # shoulder_wrist
        # hand size = median wrist-to-middle-MCP distance over detected frames, per hand and clip
        dist = np.linalg.norm(h[:, :, 9] - h[:, :, 0], axis=-1)
        size = np.array([np.median(dist[det[:, k], k]) if det[:, k].any() else 1.0 for k in range(2)]) + 1e-6
        rel = np.clip((h - h[:, :, :1]) / size[None, :, None, None], -5, 5)
        h = np.concatenate([wrist, rel[:, :, 1:]], 2)
    h[~det] = 0
    return h, det


def fill_pose(v, ok):
    """Per-frame reference (T, ...): frames without pose are filled by linear interpolation from the frames with pose."""
    if ok.all():
        return v
    t, idx, flat = np.arange(len(v)), np.flatnonzero(ok), v.reshape(len(v), -1)
    return np.stack([np.interp(t, idx, flat[idx, j]) for j in range(flat.shape[1])], 1).reshape(v.shape)


def square_box(p, m):
    """Square box over points p (..., N, 2): centre of the bounding box, side (1 + 2m) x the longest side (SPOTER)."""
    lo, hi = p.min(-2, keepdims=True), p.max(-2, keepdims=True)
    return (lo + hi) / 2, (1 + 2 * m) * (hi - lo).max(-1, keepdims=True) + 1e-6


def extra(pose, h, det, how):
    """Normalizations from the literature. Statistics come from this clip only (per frame or per clip).

    The pose is only a reference, not a feature. Methods with a body channel (wrist position, global frame) plus a local
    hand channel are adapted as in V2: point 0 (wrist) in the global frame, points 1-20 in the local frame of the hand.
    """
    T = len(h)
    hp = np.where(np.repeat(det, 21, 1)[..., None], h.reshape(T, 42, 2), np.nan)  # hand points, NaN when missing
    if how == "maxdist":
        # coordinate baseline of arXiv 2603.09213, per frame and hand: relative to the wrist, divided by the largest
        # point-to-point distance. The wrist is always 0, so the hand position relative to the body is lost
        dmax = np.linalg.norm(h[:, :, :, None] - h[:, :, None], axis=-1).max((2, 3))
        return (h - h[:, :, :1]) / (dmax[..., None, None] + 1e-6)
    ok = np.any(pose != 0, axis=(1, 2))
    if not ok.any():
        return h                                                    # no pose at all: raw, as in V1
    P, Q = fill_pose(pose, ok), pose[ok]                            # per-frame reference; frames with pose for per-clip statistics
    sh = (P[:, L_SH] + P[:, R_SH]) / 2
    d = np.linalg.norm(P[:, L_SH] - P[:, R_SH], axis=-1) + 1e-6     # shoulder distance per frame
    if how == "spoter":
        # SPOTER (Bohacek & Hruz 2022, as defined in the paper), per frame. Wrist in the signing space: u = d/2, width 6u
        # centred on the nose, height 7u starting 0.5u above the left eye, then -0.5. Hands: own box, 10% margin
        u = d / 2
        o = np.stack([P[:, 0, 0] - 3 * u, P[:, 2, 1] - u / 2], -1)
        _, s = square_box(h, 0.1)
        out = (h - h[:, :, :1]) / s                                  # relative to the wrist
        out[:, :, 0] = (h[:, :, 0] - o[:, None]) / np.stack([6 * u, 7 * u], -1)[:, None] - 0.5
    elif how == "signspace":
        # SignSpace (Zelezny et al. 2025), per frame. Global: shoulder centre, box 3d -> [-1, 1]. Local: hand box,
        # 10% margin
        _, s = square_box(h, 0.1)
        out = 2 * (h - h[:, :, :1]) / s
        out[:, :, 0] = (h[:, :, 0] - sh[:, None]) / (1.5 * d[:, None, None])
    elif how == "signbart":
        # SignBart (2025, official code), per frame: square box with 5% margin -> [0, 1], one over the pose shoulders,
        # elbows, and wrists (for the wrist), one per hand (points 1-20)
        _, s = square_box(h, 0.05)
        cb, sb = square_box(P[:, 11:17], 0.05)
        out = (h - h[:, :, :1]) / s + 0.5
        out[:, :, 0] = (h[:, :, 0] - cb) / sb + 0.5
    elif how == "unisign":
        # Uni-Sign (ICLR 2025, datasets.py), per clip: square box over the pose nose, ears, shoulders, elbows, and wrists
        # -> [-1, 1]; hand points relative to the wrist divided by the same box scale; everything clipped to [-1, 1]
        cb, sb = square_box(Q[:, [0, 7, 8, 11, 12, 13, 14, 15, 16]].reshape(-1, 2), 0)
        out = (h - h[:, :, :1]) / sb
        out[:, :, 0] = 2 * (h[:, :, 0] - cb) / sb
        out = np.clip(out, -1, 1)
    elif how in ("kimbaek", "ko"):
        body = np.concatenate([P[:, BODY], sh[:, None]], 1)         # 12 upper-body points, neck = shoulder centre
        if how == "kimbaek":
            # Kim & Baek (Sensors 2023), per frame. The wrist belongs to the arm part: (x - c) / |c - elbow|, c = mean
            # of all points (body + hands). Hands: min-max per axis - 0.5
            c = np.nanmean(np.concatenate([body, hp], 1), 1)
            r = np.linalg.norm(c[:, None] - P[:, [13, 14]], axis=-1)[..., None] + 1e-6   # left, right elbow
            lo, hi = h.min(2, keepdims=True), h.max(2, keepdims=True)
            out = (h - h[:, :, :1]) / (hi - lo + 1e-6)
            out[:, :, 0] = (h[:, :, 0] - c[:, None]) / r
        else:
            # Ko et al. (2019) "object 2D", per frame: z-score per object and axis. The wrist belongs to the body object
            out = (h - h.mean(2, keepdims=True)) / (h.std(2, keepdims=True) + 1e-6)
            out[:, :, 0] = (h[:, :, 0] - body.mean(1)[:, None]) / (body.std(1)[:, None] + 1e-6)
    elif how in ("yaslc", "yaslf"):
        # YouTube-ASL (Uthus et al. 2023): min-max per axis over shoulders, elbows, hips, face, and hands. yaslc: one
        # box per clip; yaslf (Zelezny et al. 2025): one box per frame
        ref = [11, 12, 13, 14, 23, 24] + FACE
        if how == "yaslc":
            pts = np.concatenate([Q[:, ref].reshape(-1, 2), h[det].reshape(-1, 2)])
            lo, hi = pts.min(0), pts.max(0)
        else:
            pts = np.concatenate([P[:, ref], hp], 1)
            lo, hi = np.nanmin(pts, 1)[:, None, None], np.nanmax(pts, 1)[:, None, None]
        out = (h - lo) / (hi - lo + 1e-6)
    elif how == "com":
        # Perkins et al. (2025), per frame: subtract the centre of mass of hand + face points, divide by the largest
        # deviation -> [-1, 1]
        pts = np.concatenate([P[:, FACE], hp], 1)
        c = np.nanmean(pts, 1)
        s = np.nanmax(np.abs(pts - c[:, None]), axis=(1, 2)) + 1e-6
        out = (h - c[:, None, None]) / s[:, None, None, None]
    elif how == "nose":
        # Perkins et al. (2025) "global nose-anchored": the nose of the frame is the origin, no scaling
        out = h - P[:, 0][:, None, None]
    elif how in ("gislr1", "gislr6"):
        # Kaggle GISLR 2023 competition, z-score per clip and axis. 1st place (hoyso48): centre = mean of one face point
        # (here the nose), deviation over hands + nose, eyes, mouth. 6th place (Viel & Dieter): mean and SD of all points
        idx = [0, 1, 2, 3, 4, 5, 6, 9, 10] if how == "gislr1" else list(range(23))   # gislr6: face + arms
        pts = np.concatenate([Q[:, idx].reshape(-1, 2), h[det].reshape(-1, 2)])
        c = Q[:, 0].mean(0) if how == "gislr1" else pts.mean(0)
        out = (h - c) / (np.sqrt(((pts - c) ** 2).mean(0)) + 1e-6)
    else:
        raise ValueError(how)
    return out


def features(d, how, n=30):
    X = []
    for i in range(len(d["relpath"])):
        seq = d["seq"][d["off"][i]:d["off"][i + 1]]
        h, det = normalize(seq, how)
        h, mask = resample(*fill_gaps(h, det), n)
        X.append((h * mask[:, :, None, None]).reshape(n, 84))
    return np.stack(X).astype(np.float32)


def zscore_fit(Xtr):
    """Mean and SD per feature (84) over the masked entries of the training clips only. Entry 0 = missing point, not
    counted."""
    m = Xtr != 0
    n = np.maximum(m.sum((0, 1)), 1)
    mu = (Xtr * m).sum((0, 1)) / n
    return mu, np.sqrt((((Xtr - mu) * m) ** 2).sum((0, 1)) / n) + 1e-6


def zscore_apply(X, mu, sd):
    return np.where(X != 0, (X - mu) / sd, 0).astype(np.float32)


def run(X, y, fold, k, n_cls, seed, sched, device, zscore=False):
    import torch
    from sklearn.metrics import f1_score
    lr, step, gamma = LR[sched]
    tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    if zscore:                                                  # statistics from the training folds only: no test leakage
        X = zscore_apply(X, *zscore_fit(X[tr]))
    torch.manual_seed(seed)
    net = model(n_cls).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step, gamma)
    Xtr, ytr = torch.tensor(X[tr], device=device), torch.tensor(y[tr], device=device)
    g, lossf = torch.Generator().manual_seed(seed), torch.nn.CrossEntropyLoss()
    for _ in range(B1["epochs"]):
        net.train()
        for b in torch.randperm(len(tr), generator=g).split(B1["batch"]):
            opt.zero_grad()
            lossf(net(Xtr[b]), ytr[b]).backward()
            opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        tr_acc = (net(Xtr).argmax(1) == ytr).float().mean().item()
        pred = net(torch.tensor(X[te], device=device)).argmax(1).cpu().numpy()
    return float((pred == y[te]).mean()), float(f1_score(y[te], pred, average="macro")), tr_acc


def evaluate(names, seeds, raw=f"{DATA}/pilot/gate_raw.npz"):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = dict(np.load(raw))  # load once; every NpzFile access decompresses again
    cat = str(d["cat"]) if "cat" in d else "affix"
    # GATE_FOLDS: another fold file (e.g. StratifiedKFold without shuffle)
    fmap = dict(csv.reader(open(os.environ.get("GATE_FOLDS") or (str(d["fold_csv"]) if "fold_csv" in d else fold_file(cat)))))
    fold = np.array([int(fmap[r]) for r in d["relpath"]])
    out = []
    for name in names:
        how, sched = VARIANTS[name]
        t0 = time.time()
        X = features(d, how)
        t_feat = time.time() - t0
        print(f"{os.path.basename(raw)} {name}: features {t_feat:.0f} s", flush=True)
        for seed in seeds:
            for k in range(5):
                t0 = time.time()
                acc, f1, tr = run(X, d["y"], fold, k, len(d["classes"]), seed, sched, device, how == "zscore_train")
                out.append(dict(data=os.path.basename(raw), variant=name, seed=seed, fold=k, acc=acc, macro_f1=f1, train_acc=tr,
                                sec=round(time.time() - t0, 1), feat_sec=round(t_feat, 1), device=device,
                                gpu=torch.cuda.get_device_name(0) if device == "cuda" else ""))
                print(f"{os.path.basename(raw)} {name} seed {seed} fold {k}: test {100*acc:.2f}%  F1 {100*f1:.2f}  "
                      f"train {100*tr:.1f}%  ({time.time()-t0:.0f} s)", flush=True)
        r = [x for x in out if x["variant"] == name and x["data"] == os.path.basename(raw)]
        print(f"== {name} ({how}, {sched}): mean test {100*np.mean([x['acc'] for x in r]):.2f}%  "
              f"train {100*np.mean([x['train_acc'] for x in r]):.1f}%  target 68.17", flush=True)
    return out


def demo():
    seq = np.zeros((10, 150), np.float32)
    pose = np.zeros((33, 2)); pose[L_SH] = [0.4, 0.5]; pose[R_SH] = [0.6, 0.5]
    seq[:, :66] = pose.ravel()
    seq[:, 66:66 + 42] = 0.5                      # left hand at (0.5, 0.5) = shoulder centre
    h, det = normalize(seq, "shoulder")
    assert np.allclose(h[:, 0], 0) and det[:, 0].all() and not det[:, 1].any()
    h2, _ = normalize(seq, "shoulder_wrist")
    assert h2.shape == (10, 2, 21, 2) and np.all(h2[:, 1] == 0)
    assert set(VARIANTS) == {f"V{i}" for i in range(21)} and len(EXTRA) == 13
    assert all(VARIANTS[f"V{i}"][1] == "slow" for i in range(5, 19)) and VARIANTS["V19"][1] == "b1" and VARIANTS["V20"][1] == "slow"
    # literature normalizations. Random clip: pose missing in frame 5, right hand missing in frames 3 and 7, left hand
    # never present
    rng = np.random.default_rng(0)
    s = np.zeros((12, 150))
    s[:, :66], s[:, 108:] = rng.uniform(0.2, 0.8, (12, 66)), rng.uniform(0.3, 0.7, (12, 42))
    s[5, :66] = s[[3, 7], 108:] = 0
    moved = np.where(s != 0, 2.5 * s + 0.3, 0)               # whole scene shifted and scaled uniformly
    no_pose = np.where(np.arange(150) < 66, 0, s)
    for how in EXTRA:
        a, da = normalize(s, how)
        assert a.shape == (12, 2, 21, 2) and np.isfinite(a).all() and np.all(a[~da] == 0), how
        assert how == "nose" or np.allclose(a, normalize(moved, how)[0], rtol=1e-3, atol=1e-4), how
        assert np.isfinite(normalize(seq, how)[0]).all(), how                   # 21 coincident points: divisors guarded
        assert how == "maxdist" or np.array_equal(normalize(no_pose, how)[0], normalize(no_pose, "raw")[0]), how
    p = s[:, :66].reshape(12, 33, 2)
    assert np.allclose(fill_pose(p, np.any(p != 0, axis=(1, 2)))[5], (p[4] + p[6]) / 2)
    nose = normalize(s, "nose")[0]                            # only shifts, so the scale is kept
    assert np.allclose(nose, normalize(np.where(s != 0, s + 0.3, 0), "nose")[0])
    assert np.allclose(nose[0, 1, 0], s[0, 108:110] - s[0, :2])
    sp, ss = normalize(s, "spoter")[0], normalize(s, "signspace")[0]
    assert np.allclose(ss[:, :, 1:], 2 * sp[:, :, 1:])        # SignSpace hand box = 2 x SPOTER
    s2 = s.copy()
    s2[0, 108], s2[1, 108:110] = s2[0, 0], (p[1, L_SH] + p[1, R_SH]) / 2   # wrist at the nose x / at the shoulder centre
    assert np.isclose(normalize(s2, "spoter")[0][0, 1, 0, 0], 0, atol=1e-6)
    assert np.allclose(normalize(s2, "signspace")[0][1, 1, 0], 0, atol=1e-6)
    for how, lo, hi, k in (("spoter", -1 / 1.2, 1 / 1.2, 1), ("signbart", .5 - 1 / 1.1, .5 + 1 / 1.1, 1), ("kimbaek", -1, 1, 1),
                           ("unisign", -1, 1, 0), ("yaslc", 0, 1, 0), ("yaslf", 0, 1, 0), ("com", -1, 1, 0)):
        a, da = normalize(s, how)
        assert lo - 1e-6 <= a[da][:, k:].min() and a[da][:, k:].max() <= hi + 1e-6, how
    # the wrist moves inside the hand (not an extreme point) -> points 1-20 must change too
    s3 = s.copy()
    rows = [i for i in range(12) if i not in (3, 7)]
    s3[rows, 108:110] = s3[rows, 110:150].reshape(len(rows), 20, 2).mean(1)
    s4 = s3.copy()
    s4[rows, 108] += 0.005
    for how in ("spoter", "signspace", "signbart", "kimbaek"):
        assert not np.allclose(normalize(s3, how)[0][rows, 1, 1:], normalize(s4, how)[0][rows, 1, 1:]), how
    # V19/V20: statistics from training clips only; missing entries stay 0; test clips use the training statistics
    Z = rng.uniform(0.2, 0.8, (4, 3, 84)).astype(np.float32)
    Z[0, 0, 5], Z[2:] = 0, 9.0
    mu, sd = zscore_fit(Z[:2])
    zs = zscore_apply(Z, mu, sd)
    v = zs[:2, :, 7][Z[:2, :, 7] != 0]
    assert zs[0, 0, 5] == 0 and abs(v.mean()) < 1e-5 and abs(v.std() - 1) < 1e-3
    assert np.isclose(zs[2, 0, 0], (9.0 - mu[0]) / sd[0]) and mu.max() < 1
    md = normalize(s, "maxdist")[0][0, 1]
    assert np.allclose(md[0], 0) and np.isclose(np.linalg.norm(md[:, None] - md[None], axis=-1).max(), 1, atol=1e-4)
    print(f"demo ok: shoulder and wrist normalization, variants, {len(EXTRA)} literature normalizations")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--demo"
    if cmd == "--demo":
        demo()
    elif cmd == "pack":
        hs = "hs" in sys.argv[2:]
        cat = next((a for a in sys.argv[2:] if a in ("affix", "number", "alphabet")), "affix")
        pack(f"{DATA}/keypoints_hs" if hs else f"{DATA}/keypoints", "_hs" if hs else "", cat, am7="am7" in sys.argv[2:])
    else:
        raws = sorted(glob.glob(f"{DATA}/pilot/gate_raw*.npz"))
        names = [a for a in sys.argv[2:] if a in VARIANTS]
        res = []
        for raw in raws:
            res += evaluate(names, [0] if cmd == "screen" else [0, 1, 2, 3, 4], raw)
        json.dump(res, open(os.environ.get("GATE_OUT", f"gate_variants_{cmd}.json"), "w"), indent=1)  # GATE_OUT: one file per parallel run
