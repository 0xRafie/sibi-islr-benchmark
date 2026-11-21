"""BiLSTM baseline (Ho and Santoso, JUTIF 2026) and the B2 classifiers on its frozen embeddings.

One run = one (setting, protocol, fold, seed). Each run writes one JSON with the metrics of all models, the predictions
per test clip, the device, and the duration, plus <run>_scores.npz with the full test scores. Runs whose JSON already
exists are skipped, so an interrupted session can be resumed.

Input F0: the 84 hand coordinates (2 hands x 21 MediaPipe landmarks x xy) of each trimmed clip, shifted to the mean
shoulder midpoint, divided by the mean shoulder width, gap-filled, and resampled to 30 frames (gate_variants.features).
Keypoint packs mp_<category>.npz are read from --data and fold tables folds_<setting>.csv from --folds-dir.

Protocols (fold columns of folds_<setting>.csv): cs = stratified 5-fold over clips (signer-dependent, C-S),
cl = leave-one-signer-out over 22 signers (C-L). Exploratory: cg = stratified group 5-fold (group = category, class,
signer), cp = one fold per recording-session group, cr = one paired signer held out and one random other signer removed
from training (folds_cr.csv), csn = cs without shuffling (affix only), all = train on all clips and test on an external
set (--external), rl/dr/rd/drr = cross-source splits with an external number dataset (setting dr).

    python train.py --model bilstm --demo
    python train.py --model bilstm --check-fx
    python train.py --model bilstm --setting affix --protocol cl --folds 0-21 --seeds 0-4 \
        --data <dir with mp_*.npz> --folds-dir <dir with folds_*.csv> --out <dir> [--max-hours 10.5] [--shard 0/2]
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; set before torch is used
import argparse, csv, json, sys, time

import numpy as np

from sibi_islr.models.bilstm import B1, fill_gaps, model, resample as resample_n
from sibi_islr.data.features import LR, features, shoulder_cw

from sibi_islr.training.common import CROSS, PROTOCOLS, SETTINGS, device_info, metrics, rng, split
from sibi_islr.training.bilstm_inputs import FACE, FX, check_fx, load, padded


AUG_SHIFT = 10 / 512 / 0.3477   # DSTA-SLR random shift (+-10 px of a 512 frame) in F0 units (shoulder widths, slot2.py)


def augment(xb, gen):
    """DSTA-SLR augmentation (as slot2.dsta_input) for an F0 batch (B, 30, 84): mirror with p 0.5 (swap hands, negate
    x), then a random shift per sample in x and y. Zero points are shifted too, as in DSTA-SLR."""
    import torch
    b = len(xb)
    x = xb.view(b, 30, 2, 21, 2).clone()
    flip = torch.rand(b, generator=gen) < 0.5
    x[flip] = x[flip].flip(2) * torch.tensor([-1.0, 1.0], device=x.device)
    shift = (torch.rand(b, 2, generator=gen) * 2 - 1) * AUG_SHIFT
    return (x + shift.to(x.device)[:, None, None, None, :]).view(b, 30, 84)


def check_augment():
    """Check of augment(), run before training: shape unchanged, each sample shifted by a constant <= AUG_SHIFT, and
    mirrored samples = hands swapped + x negated."""
    import torch
    xb = torch.rand(64, 30, 84)
    out = augment(xb, torch.Generator().manual_seed(0)).view(64, 30 * 2 * 21, 2)
    x = xb.view(64, 30, 2, 21, 2)
    flips = 0
    for k in range(64):
        for cand, is_flip in ((x[k].reshape(-1, 2), False), ((x[k].flip(1) * torch.tensor([-1.0, 1.0])).reshape(-1, 2), True)):
            d = out[k] - cand
            if torch.allclose(d, d[0].expand_as(d), atol=1e-6):
                assert d[0].abs().max() <= AUG_SHIFT + 1e-6
                flips += is_flip
                break
        else:
            raise AssertionError(f"sample {k}: neither a constant shift nor mirror + shift")
    assert 16 < flips < 48, flips
    print(f"augment ok: {flips}/64 mirrored, shift <= {AUG_SHIFT:.4f}", flush=True)


def train_bilstm(X, y, tr, te, n_cls, seed, device, epochs=B1["epochs"], aug=False):
    """As gate_variants.run, but returns the test scores and the 192/256-d embeddings for B2.
    If X is a list of variable-length arrays (ablation a1), clips are padded and packed; otherwise unchanged."""
    import torch
    lr, step, gamma = LR["slow"]
    torch.manual_seed(seed)
    net = model(n_cls, X[0].shape[-1]).to(device)             # 84 hand features; more when a4 appends the face
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step, gamma)
    var = isinstance(X, list)
    Xtr = None if var else torch.tensor(X[tr], device=device)
    ytr = torch.tensor(y[tr], device=device)
    g, lossf = torch.Generator().manual_seed(seed), torch.nn.CrossEntropyLoss()
    ga = torch.Generator().manual_seed(10_000 + seed)        # augmentation stream, separate from the batch order
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(tr), generator=g).split(B1["batch"]):
            opt.zero_grad()
            out = padded(net, [X[tr[i]] for i in b.tolist()], device)[0] if var else net(augment(Xtr[b], ga) if aug else Xtr[b])
            lossf(out, ytr[b]).backward()
            opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        def embed(idx):
            e192, e256, out = [], [], []
            for b in torch.tensor(idx).split(512):
                h = (padded(net, [X[i] for i in b.tolist()], device)[1] if var
                     else net.lstm(torch.tensor(X[b.numpy()], device=device))[0].mean(1))
                e192.append(h.cpu())
                e256.append(torch.relu(net.head[0](h)).cpu())
                out.append(net.head(h).cpu())
            return [torch.cat(t).numpy() for t in (e192, e256, out)]
        tr192, tr256, tr_out = embed(tr)
        te192, te256, te_out = embed(te)
    train_acc = float((tr_out.argmax(1) == y[tr]).mean())
    return te_out, {192: (tr192, te192), 256: (tr256, te256)}, train_acc


def b2(emb, y, tr, seed):
    """B2 (Ho and Santoso, JUTIF 2026): MLP, SVM, and RF on frozen BiLSTM embeddings. Test scores per classifier."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    out = {}
    for dim, (etr, ete) in emb.items():
        clfs = {"mlp": make_pipeline(StandardScaler(), MLPClassifier((100,), max_iter=1000, random_state=seed)),
                "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
                "rf": RandomForestClassifier(100, random_state=seed)}
        for name, clf in clfs.items():
            clf.fit(etr, y[tr])
            s = clf.decision_function(ete) if name == "svm" else clf.predict_proba(ete)
            full = np.full((len(ete), y.max() + 1), -np.inf)      # classes absent from the training data: lowest score
            full[:, clf.classes_] = s
            out[f"b2_{name}_{dim}"] = full
    return out


def run(X, meta, names, protocol, fold, seed, folds_dir, epochs=B1["epochs"], ablation=None, dump=None, scores_out=None):
    """With an ablation (e.g. 'a1' without resampling, 'notrim' without trimming): BiLSTM only (bilstm_<ablation>), no
    B2. dump: .npy file for the 192-d embeddings of the test clips (probe). scores_out: .npz file with the full scores
    of the first model per test clip, in test_index order."""
    info = device_info()
    tr, te = split(meta, protocol, fold, seed, folds_dir)
    y = meta["label"]
    t0 = time.time()
    te_out, emb, train_acc = train_bilstm(X, y, tr, te, len(names), seed, info["device"], epochs, aug=ablation == "aug")
    if protocol in CROSS:                                  # cross-source: each model only predicts its training labels
        te_out[:, np.setdiff1d(np.arange(len(names)), y[tr])] = -np.inf
    t1 = time.time()
    if dump:
        np.save(dump, emb[192][1].astype(np.float32))
    scores = {f"bilstm_{ablation}": te_out} if ablation else {"bilstm": te_out, **b2(emb, y, tr, seed)}
    if scores_out:
        np.savez_compressed(scores_out, **{next(iter(scores)): te_out.astype(np.float32)})
    meta_te = {k: v[te] for k, v in meta.items()}
    res = dict(protocol=protocol, fold=fold, seed=seed, n_train=int(len(tr)), n_test=int(len(te)),
               test_subjects=sorted(map(int, np.unique(meta["subject"][te]))), bilstm_train_acc=train_acc,
               sec_bilstm=round(t1 - t0, 1), sec_b2=round(time.time() - t1, 1), epochs=epochs, **info,
               label="train_main", models={}, test_index=te.tolist(), **({"ablation": ablation} if ablation else {}))
    for name, sc in scores.items():
        m, pred = metrics(sc, y[te], meta_te, names)
        res["models"][name] = dict(metrics=m, pred=pred.tolist())
    res["models"][next(iter(scores))]["top5"] = np.argsort(-te_out, 1)[:, :5].tolist()
    return res


def main(argv):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    p = argparse.ArgumentParser()
    p.add_argument("--setting", choices=SETTINGS, required=True)
    p.add_argument("--protocol", choices=PROTOCOLS, required=True)
    p.add_argument("--folds", nargs="+", default=["0-99"])
    p.add_argument("--seeds", nargs="+", default=["0-4"])
    p.add_argument("--data", required=True)
    p.add_argument("--folds-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-hours", type=float, default=10.5)
    p.add_argument("--epochs", type=int, default=B1["epochs"], help="for smoke tests only; reported runs use the B1 default")
    p.add_argument("--shard", default="0/1", help="i/n: run only every n-th job, starting at job i (one shard per GPU)")
    p.add_argument("--ablation", choices=("a1", "notrim", "a4", "a4lips", "a2", "a5", "aug", *FX),
                   help="a1: no resampling; notrim: no trimming; a4/a4lips: + face/lips; "
                        "a2: thumb points set to zero; a5: SignBart box normalization per body part; "
                        "aug: DSTA-SLR augmentation, mirror + shift; f1/f2/f3/f3face: feature sets f1-f3")
    p.add_argument("--dump-emb", action="store_true", help="save the 192-d embeddings of the test clips (probe)")
    p.add_argument("--external", help="npz with an external test set (X, label, subject, category), used with --protocol all")
    a = p.parse_args(argv)
    i, n = map(int, a.shard.split("/"))
    jobs = [(f, s) for s in rng(a.seeds, 5) for f in rng(a.folds, PROTOCOLS[a.protocol])][i::n]
    os.makedirs(a.out, exist_ok=True)
    t_start, X = time.time(), None
    for fold, seed in jobs:
        path = f"{a.out}/{a.setting}_{a.protocol}_f{fold:02d}_s{seed}{'_' + a.ablation if a.ablation else ''}.json"
        if os.path.exists(path):
            continue
        if (time.time() - t_start) / 3600 > a.max_hours:
            print(f"stopping: exceeded --max-hours {a.max_hours}", flush=True)
            break
        if X is None:
            if a.ablation == "aug":
                check_augment()
            elif a.ablation in FX:
                check_fx()
            X, meta, names = load(a.setting, a.data, a.folds_dir, resample=a.ablation != "a1", trim=a.ablation != "notrim",
                                  face=a.ablation if a.ablation in FACE else None,
                                  how="signbart" if a.ablation == "a5" else "shoulder",   # a5: box per body part
                                  fx=a.ablation if a.ablation in FX else None)
            if a.external:                               # external set appended at the end; labels are class indices of this setting
                e = dict(np.load(a.external))
                X = np.concatenate([X, e["X"]])
                meta = {k: np.concatenate([v, e[k] if k in ("label", "subject", "category") else np.full(len(e["X"]), -1)])
                        for k, v in meta.items()}
            if a.ablation == "a2":                       # a2: thumb points (MediaPipe 1-4) of both hands set to zero
                X = X.reshape(len(X), 30, 2, 21, 2).copy()
                X[:, :, :, 1:5] = 0
                X = X.reshape(len(X), 30, 84)
        res = dict(setting=a.setting, n_classes=len(names), code_commit=open("kode_commit.txt").read().strip()   # optional code commit file
                   if os.path.exists("kode_commit.txt") else None,
                   **run(X, meta, names, a.protocol, fold, seed, a.folds_dir, epochs=a.epochs, ablation=a.ablation,
                         dump=path[:-5] + "_emb192.npy" if a.dump_emb else None, scores_out=path[:-5] + "_scores.npz"))
        json.dump(res, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
        b = next(iter(res["models"].values()))["metrics"]
        print(f"{a.setting} {a.protocol} fold {fold} seed {seed}: BiLSTM acc {100 * b['acc']:.2f}  F1 {100 * b['macro_f1']:.2f}  "
              f"train {100 * res['bilstm_train_acc']:.1f}  ({res['sec_bilstm']:.0f}+{res['sec_b2']:.0f} s, {res['gpu'] or 'cpu'})",
              flush=True)


def demo():
    import tempfile
    from sibi_islr.data.pack import frame150
    cross = dict(subject=np.array([1, 2, 101, 101, 102, 102]), shared=np.array([1, 0, 1, 0, 1, 1]))   # cross-source splits
    assert [x.tolist() for x in split(cross, "dr", 0, 0, None)] == [[0, 1], [2, 4, 5]]
    assert [x.tolist() for x in split(cross, "rd", 0, 0, None)] == [[2, 3, 4, 5], [0]]
    assert [x.tolist() for x in split(cross, "rl", 1, 0, None)] == [[2, 3], [4, 5]]
    assert [x.tolist() for x in split(cross, "drr", 0, 0, None)] == [[0, 1, 4, 5], [2]]
    rng_ = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        rows, seqs, spans = [], [], []
        for k, cls in enumerate(("an", "ber", "di")):
            for s in (1, 2, 3, 4):
                for rep in range(1, 4):
                    arr = np.zeros((20, 1662), np.float32)
                    arr[:, 11 * 4:11 * 4 + 2], arr[:, 12 * 4:12 * 4 + 2] = (0.4, 0.5), (0.6, 0.5)   # shoulders
                    arr[:, -126:-63] = rng_.normal(0.5 + 0.1 * k, 0.02, (20, 63))                 # left hand, per class
                    rows.append(dict(relpath=f"affix/{cls}/subject-{s}-affix-{cls}-rep-{rep}", category="affix",
                                     **{"class": cls}, label=k, subject=s, rep=rep, cs=rep % 3, cg=s % 3, cl=s - 1, cp=s - 1))
                    seqs.append(frame150(arr))
                    spans.append((0, 20))
        np.savez(f"{d}/mp_affix.npz", seq=np.concatenate(seqs), off=np.cumsum([0] + [20] * len(seqs)),
                 span=np.array(spans), relpath=np.array([r["relpath"] for r in rows]))
        with open(f"{d}/folds_affix.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        X, meta, names = load("affix", d, d)
        assert X.shape == (36, 30, 84) and names == ["affix/an", "affix/ber", "affix/di"]
        tr, te = split(meta, "cl", 0, 0, d)
        assert set(meta["subject"][te]) == {1} and 1 not in set(meta["subject"][tr])
        res = run(X, meta, names, "cl", 0, 0, d, epochs=30, scores_out=f"{d}/s.npz")
        assert np.array_equal(np.load(f"{d}/s.npz")["bilstm"].argmax(1), res["models"]["bilstm"]["pred"])
        m = res["models"]["bilstm"]["metrics"]
        assert res["n_test"] == 9 and m["acc"] == 1.0 and set(res["models"]) == {"bilstm"} | {
            f"b2_{c}_{k}" for c in ("mlp", "svm", "rf") for k in (192, 256)}
        assert 0 <= m["macro_f1"] <= 1 and m["top5"] == 1.0 and list(m["acc_per_subject"]) == ["1"]
        Xa, _, _ = load("affix", d, d, resample=False)                 # a1: original length (20 frames), no resampling
        assert isinstance(Xa, list) and Xa[0].shape == (20, 84)
        ra = run(Xa, meta, names, "cl", 0, 0, d, epochs=30, ablation="a1", dump=f"{d}/e.npy")
        assert set(ra["models"]) == {"bilstm_a1"} and ra["models"]["bilstm_a1"]["metrics"]["acc"] == 1.0
        assert np.load(f"{d}/e.npy").shape == (9, 192) and ra["ablation"] == "a1"
        fc = rng_.uniform(0.3, 0.7, (len(seqs) * 20, 936)).astype(np.float16)
        fc[5:8] = 0                                                # clip 0: face missing mid-clip -> interpolated
        fc[20:24] = 0                                              # clip 1: face missing at the start -> 0 (outside the range)
        np.savez(f"{d}/mp_face_affix.npz", face=fc, off=np.cumsum([0] + [20] * len(seqs)),
                 relpath=np.array([r["relpath"] for r in rows]))
        X4, _, _ = load("affix", d, d, face="a4")
        X4l, _, _ = load("affix", d, d, face="a4lips")
        assert X4.shape == (36, 30, 1020) and X4l.shape == (36, 30, 164)
        assert np.array_equal(X4[..., :84], X) and np.array_equal(X4l[..., :84], X)   # hand part = F0 without a4
        assert np.any(X4[0, :, 84:] != 0, axis=1).all()                               # middle gap filled
        assert np.all(X4[1, 0, 84:] == 0) and np.any(X4[1, -1, 84:] != 0)              # outside the range = 0
        r4 = run(X4l, meta, names, "cl", 0, 0, d, epochs=3, ablation="a4lips")
        assert set(r4["models"]) == {"bilstm_a4lips"} and r4["ablation"] == "a4lips"
    print("demo ok: F0 features loaded, C-L split without the test signer, BiLSTM + 6 B2 classifiers, full metrics, a1,"
          " embedding dump, a4 (face 30x1020, lips 30x164, hand part identical)")


def cli():
    check_fx() if "--check-fx" in sys.argv else demo() if "--demo" in sys.argv else main(sys.argv[1:])


if __name__ == "__main__":
    cli()
