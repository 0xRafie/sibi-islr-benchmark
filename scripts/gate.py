"""Replication check: the BiLSTM of Ho & Santoso on affixes only, signers 1-20, stratified 5-fold cross-validation.

It only validates the pipeline; its accuracy is not a benchmark result. Target: 68.17% +- 1.47% (Ho & Santoso, JAIC
2025). Choices that the paper leaves open are recorded in PILOT. Other scripts reuse DROP, N_HAND, NAME, fill_gaps(),
resample(), B1, and model() (the BiLSTM baseline).

Paths: under SIBI_DATA (default `data`), keypoints are read from keypoints/<category>/<class>/<clip>.npy
(extract.py --darmawan) and packages are written to pilot/. SIBI_QC (default `qc`) receives gate_folds_affix.csv and
gate_affix.json. GATE_VARIANT (exploratory) selects other keypoints: "" = extract.py (default); "hs" and "hs_noreex" =
keypoints_hs/ from an alternative extraction that is not part of this repository ("hs_noreex" zeroes again the frames
that its re-extraction recovered, listed in <clip>.reex.npy).

    python scripts/gate.py --demo
    python scripts/gate.py build affix      # data/keypoints/affix -> data/pilot/affix.npz
    python scripts/gate.py folds            # qc/gate_folds_affix.csv (fixed before training)
    python scripts/gate.py train [seed ...] # qc/gate_affix.json + summary; no argument = 5 seeds
    python scripts/gate.py build number     # extraction + preprocessing only
"""
import csv, glob, json, os, re, sys, time
import numpy as np

from trim import span

DATA = os.environ.get("SIBI_DATA", "data")
QC = os.environ.get("SIBI_QC", "qc")
PILOT = dict(
    # Ho & Santoso: "interpolating keypoints to address missing data, and applying a binary mask".
    # Our reading: per hand, missing frames BETWEEN the first and last detection are filled by linear interpolation;
    # outside that range zero and mask 0. A hand that is never detected: zero, mask 0.
    gap_fill="linear, within the detection range of each hand",
    # Resampling the trimmed length -> 30: linear coordinates at 30 evenly spaced points over [0, T-1], nearest mask.
    resample="linear coordinates, nearest mask, 30 evenly spaced points over [0, T-1]",
    frames=30,
    features="x,y of 21 points, left then right hand = 84, multiplied by the mask",
    preprocessing_frame="no CLAHE/resize 224 (Ho & Santoso use them; extract.py does not)",
    fold_seed=0,
    seeds=[0, 1, 2, 3, 4],
    metric="test-fold accuracy at the last epoch (Ho & Santoso use no separate validation set)",
)
B1 = dict(hidden=96, layers=2, fc=256, dropout=0.3, lr=1e-4, step=5, gamma=0.5, epochs=100, batch=16)

# Dropped before use: 10 byte-identical affix re-exports of signer 3 (`rep--N`), 2 duplicate `ti` takes (the
# lowest-numbered rep is kept), and 2 broken number files (a copy of rep-7, and a recording of 1 frame)
DROP = {
    "affix/di/subject-3-affix-di-rep--9", "affix/kah/subject-3-affix-kah-rep--9",
    "affix/lah/subject-3-affix-lah-rep--8", "affix/me/subject-3-affix-me-rep--6",
    "affix/nya/subject-3-affix-nya-rep--8", "affix/nya/subject-3-affix-nya-rep--9",
    "affix/pe/subject-3-affix-pe-rep--8", "affix/ter/subject-3-affix-ter-rep--7",
    "affix/ter/subject-3-affix-ter-rep--8", "affix/ti/subject-3-affix-ti-rep--9",
    "affix/ti/subject-17-affix-ti-rep-10", "affix/ti/subject-18-affix-ti-rep-10",
    "number/triliun/subject-12-number-triliun-rep-10", "number/20/subject-8-number-20-rep-4",
}
N_HAND = 21 * 3
VARIANT = os.environ.get("GATE_VARIANT", "")
SRC = f"{DATA}/keypoints_hs" if VARIANT.startswith("hs") else f"{DATA}/keypoints"
SUF = f"_{VARIANT}" if VARIANT else ""
NAME = re.compile(r"subject-(\d+)-(affix|number|alphabet)-.+-rep-(\d+)$")


def hand_xy(arr):
    """(T, 1662) -> (T, 2, 21, 2) xy of the left and right hand, and (T, 2) detection per hand."""
    h = arr[:, -2 * N_HAND:].reshape(len(arr), 2, 21, 3)[..., :2]
    return h, np.any(h != 0, axis=(2, 3))


def fill_gaps(h, det):
    """Linear interpolation of missing frames inside the detection range of each hand. Mask = that range.
    h (T, K, N, 2), det (T, K): K = 2 hands x 21 points, or K = 1 face x N points (face ablation)."""
    h, mask = h.copy(), np.zeros_like(det)
    t = np.arange(len(h))
    for k in range(det.shape[1]):
        idx = np.flatnonzero(det[:, k])
        if len(idx) == 0:
            continue
        a, b = idx[0], idx[-1] + 1
        mask[a:b, k] = True
        flat = h[:, k].reshape(len(h), -1)
        for j in range(flat.shape[1]):
            flat[a:b, j] = np.interp(t[a:b], idx, flat[idx, j])
        h[:, k] = flat.reshape(h[:, k].shape)
    return h, mask


def resample(h, mask, n=30):
    """Length T -> n: linear coordinates, nearest mask."""
    T = len(h)
    pos = np.linspace(0, T - 1, n)
    flat = h.reshape(T, -1)
    out = np.stack([np.interp(pos, np.arange(T), flat[:, j]) for j in range(flat.shape[1])], 1)
    return out.reshape((n,) + h.shape[1:]), mask[np.rint(pos).astype(int)]


def preprocess(arr, n=30):
    """Raw (T, 1662) -> (n, 84) float32, trimmed by trim.py and multiplied by the mask."""
    a, b = span(arr)
    h, det = hand_xy(arr[a:b])
    h, mask = resample(*fill_gaps(h, det), n)
    return (h * mask[:, :, None, None]).reshape(n, 84).astype(np.float32)


def build(cat):
    files = sorted(f for f in glob.glob(f"{SRC}/{cat}/*/*.npy") if not f.endswith(".reex.npy"))
    assert files, f"no keypoints in {SRC}/{cat}: run extract.py --darmawan first"
    rows, X = [], []
    for f in files:
        rel = os.path.relpath(f, SRC)[:-4]
        if rel in DROP:
            continue
        m = NAME.search(rel)
        if m is None and "-rep--" in rel:  # rep--N re-exports in number, not handled here
            print("skipped, rep--N pattern:", rel)
            continue
        assert m, rel
        arr = np.load(f)
        if VARIANT == "hs_noreex":
            reex = np.load(f[:-4] + ".reex.npy")
            for k in range(2):
                a = arr.shape[1] - 2 * N_HAND + k * N_HAND
                arr[reex[:, k], a:a + N_HAND] = 0
        if len(arr) < 2:
            print("skipped, < 2 frames:", rel)
            continue
        X.append(preprocess(arr))
        rows.append((rel, rel.split("/")[1], int(m.group(1)), int(m.group(3))))
    rel, cls, subj, rep = map(np.array, zip(*rows))
    classes = sorted(set(cls))
    os.makedirs(f"{DATA}/pilot", exist_ok=True)
    np.savez_compressed(f"{DATA}/pilot/{cat}{SUF}.npz", X=np.stack(X), y=np.array([classes.index(c) for c in cls]),
                        classes=np.array(classes), subject=subj, rep=rep, relpath=rel)
    print(f"{cat}: {len(rel)} clips, {len(classes)} classes, {len(set(subj))} signers -> {DATA}/pilot/{cat}{SUF}.npz")


def gate_subset():
    d = np.load(f"{DATA}/pilot/affix{SUF}.npz")
    keep = d["subject"] <= 20
    assert keep.sum() == 3598, f"the replication subset must have 3,598 clips, got {keep.sum()}"
    return {k: d[k][keep] for k in ("X", "y", "relpath")}, d["classes"]


def folds():
    from sklearn.model_selection import StratifiedKFold
    d, _ = gate_subset()
    skf = StratifiedKFold(5, shuffle=True, random_state=PILOT["fold_seed"])
    fold = np.empty(len(d["y"]), int)
    for k, (_, te) in enumerate(skf.split(d["X"], d["y"])):
        fold[te] = k
    with open(f"{QC}/gate_folds_affix.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("relpath", "fold")] + sorted(zip(d["relpath"], fold)))
    print(f"{QC}/gate_folds_affix.csv:", np.bincount(fold).tolist(), "clips per fold")


def model(n_cls, n_in=84):
    """BiLSTM baseline (B1). n_in = features per frame: 84 for the hands, more when the face ablation adds the face."""
    import torch.nn as nn

    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_in, B1["hidden"], B1["layers"], batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Linear(2 * B1["hidden"], B1["fc"]), nn.ReLU(),
                                      nn.Dropout(B1["dropout"]), nn.Linear(B1["fc"], n_cls))

        def forward(self, x):
            return self.head(self.lstm(x)[0].mean(1))  # mean of the 192 dimensions over time

    return BiLSTM()


def run_fold(X, y, tr, te, n_cls, seed, device):
    import torch
    from sklearn.metrics import f1_score
    tr, te = np.flatnonzero(tr), np.flatnonzero(te)  # boolean mask -> indices
    torch.manual_seed(seed)
    net = model(n_cls).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=B1["lr"])
    sched = torch.optim.lr_scheduler.StepLR(opt, B1["step"], B1["gamma"])
    Xtr, ytr = torch.tensor(X[tr], device=device), torch.tensor(y[tr], device=device)
    g = torch.Generator().manual_seed(seed)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(B1["epochs"]):
        net.train()
        for b in torch.randperm(len(tr), generator=g).split(B1["batch"]):
            opt.zero_grad()
            lossf(net(Xtr[b]), ytr[b]).backward()
            opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(X[te], device=device)).argmax(1).cpu().numpy()
    return float((pred == y[te]).mean()), float(f1_score(y[te], pred, average="macro"))


def train(seeds=None):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d, classes = gate_subset()
    fmap = dict(csv.reader(open(f"{QC}/gate_folds_affix.csv")))
    fold = np.array([int(fmap[r]) for r in d["relpath"]])
    res = []
    for seed in seeds or PILOT["seeds"]:
        for k in range(5):
            t0 = time.time()
            acc, f1 = run_fold(d["X"], d["y"], fold != k, fold == k, len(classes), seed, device)
            res.append(dict(seed=seed, fold=k, acc=acc, macro_f1=f1))
            print(f"seed {seed} fold {k}: accuracy {100 * acc:.2f}%  macro-F1 {100 * f1:.2f}  ({time.time() - t0:.0f} s)",
                  flush=True)
    acc = np.array([[r["acc"] for r in res if r["seed"] == s] for s in sorted({r["seed"] for r in res})])
    per_seed = 100 * acc.mean(1)
    summary = dict(pilot=PILOT, b1=B1, device=device, torch=torch.__version__, n_clip=len(d["y"]),
                   target="68.17 ± 1.47", per_seed_mean_acc=per_seed.round(2).tolist(),
                   fold_sd_acc=(100 * acc.std(1, ddof=1)).round(2).tolist(), runs=res)
    summary["variant"] = VARIANT
    json.dump(summary, open(f"{QC}/gate_affix{SUF}.json", "w"), indent=1)
    print(f"\nmean per seed: {per_seed.round(2).tolist()}  target 68.17 ± 1.47  -> {QC}/gate_affix{SUF}.json")


def demo():
    arr = np.zeros((40, 1662), np.float32)
    arr[5:35, -2 * N_HAND:-N_HAND] = 0.5          # left hand in frames 5-34
    arr[20, -2 * N_HAND:-N_HAND] = 0              # one missing frame in the middle
    arr[5:35, -N_HAND:] = 0                       # right hand never detected
    h, det = hand_xy(arr)
    hf, mask = fill_gaps(h, det)
    assert np.allclose(hf[20, 0], 0.5) and mask[20, 0] and not mask[:, 1].any()
    x = preprocess(arr)
    assert x.shape == (30, 84) and np.all(x[:, 42:] == 0)
    ramp = np.arange(10, dtype=float)[:, None, None, None] * np.ones((10, 2, 21, 2))
    r, m = resample(ramp, np.ones((10, 2), bool), 4)
    assert np.allclose(r[:, 0, 0, 0], [0, 3, 6, 9])
    assert NAME.search("affix/me/subject-2-affix-me-rep-10").groups() == ("2", "affix", "10")
    import torch
    assert model(18)(torch.zeros(3, 30, 84)).shape == (3, 18)
    print("demo ok: gap filling, resampling, mask, model shape")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--demo"
    seeds = [int(a) for a in sys.argv[2:]] if cmd == "train" else []  # `train 0` = seed 0 only
    {"--demo": demo, "folds": folds, "train": lambda: train(seeds or None)}.get(cmd, lambda: build(sys.argv[2]))()
