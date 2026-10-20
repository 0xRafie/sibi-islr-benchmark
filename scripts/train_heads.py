"""Linear and MLP heads on frozen encoder features, per (protocol, fold, seed). The JSON has the same format as
train_main.run, so analyze.py reads it directly. CPU, scikit-learn.

Models per run: unisign_<pose|mt5>_<lin|mlp> (npz keys `pose`, `mt5` from unisign_features.py) or
shubert_<last|all>_<lin|mlp> (npz key `layers` (clips, 12, 768) from shubert_features.py; `last` = layer 12, `all` = the
12 layers concatenated). Features are standardized with the mean and SD of the training clips of the fold.
lin = LogisticRegression (lbfgs, C 1, deterministic); mlp = MLPClassifier 1x256 ReLU, Adam, random_state = seed.
Fold tables folds_<setting>.csv are read from qc/ if it exists, else from the working directory.

    python scripts/train_heads.py --demo
    python scripts/train_heads.py <setting> <features.npz[,features2.npz ...]> <out_dir> [cs cg cl ...] [rev] [folds=a-b]
"""
import csv, json, os, sys, time

import numpy as np

from train_main import PROTOCOLS, metrics, split

QC = "qc" if os.path.isdir("qc") else "."   # fold CSVs: qc/, or the working directory when the scripts sit in one flat folder


def heads(seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    return {"lin": LogisticRegression(max_iter=5000),
            "mlp": MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=seed)}


def table(setting, folds_dir=QC):
    """Rows of folds_<setting>.csv -> relpaths, meta (as in train_main.load), class names."""
    rows = list(csv.DictReader(open(f"{folds_dir}/folds_{setting}.csv")))
    meta = {k: np.array([int(r[k]) for r in rows]) for k in ("label", "subject", "cs", "cg", "cl", "cp")}
    meta["category"] = np.array([r["category"] for r in rows])
    names = {int(r["label"]): f"{r['category']}/{r['class']}" for r in rows}
    return [r["relpath"] for r in rows], meta, [names[i] for i in range(len(names))]


def run(F, meta, names, protocol, fold, seed, folds_dir=QC, prefix="unisign", scores_out=None):
    """F = {feature: (N, d)} in fold-CSV row order -> result dict in the train_main JSON format."""
    tr, te = split(meta, protocol, fold, seed, folds_dir)
    y, t0 = meta["label"], time.time()
    meta_te = {k: v[te] for k, v in meta.items()}
    res = dict(protocol=protocol, fold=fold, seed=seed, n_train=int(len(tr)), n_test=int(len(te)),
               test_subjects=sorted(map(int, np.unique(meta["subject"][te]))), device="cpu", label=f"train_heads ({prefix})",
               models={}, test_index=te.tolist())
    full = {}
    for feat, X in F.items():
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        for kind, clf in heads(seed).items():
            clf.fit(Xtr, y[tr])
            sc = np.full((len(te), len(names)), -np.inf)
            sc[:, clf.classes_] = clf.predict_proba(Xte)       # classes absent from the training data are never predicted
            m, pred = metrics(sc, y[te], meta_te, names)
            full[f"{prefix}_{feat}_{kind}"] = sc.astype(np.float32)
            res["models"][f"{prefix}_{feat}_{kind}"] = dict(metrics=m, pred=pred.tolist(),
                                                           train_acc=float(clf.score(Xtr, y[tr])))
    if scores_out:                                     # full scores per test clip, in test_index order
        np.savez_compressed(scores_out, **full)
    res["sec"] = round(time.time() - t0, 1)
    return res


def main(setting, src, out, protocols=("cs", "cg", "cl"), seeds=range(5), rev=False, folds=None):
    rel, meta, names = table(setting)
    ds = [dict(np.load(f)) for f in src.split(",")]          # combined setting: one npz per category, comma-separated
    d = {k: np.concatenate([x[k] for x in ds]) for k in ("relpath", "layers", "pose", "mt5") if k in ds[0]}
    pos = {r: i for i, r in enumerate(d["relpath"])}
    order = [pos[r] for r in rel]                                            # a missing clip raises KeyError, not silence
    if "layers" in d:                                                        # SHuBERT
        L, prefix = d["layers"][order].astype(np.float64), "shubert"
        F = {"last": L[:, -1], "all": L.reshape(len(L), -1)}
    else:
        F, prefix = {k: d[k][order].astype(np.float64) for k in ("pose", "mt5")}, "unisign"
    os.makedirs(out, exist_ok=True)
    for p in protocols:
        for fold in (reversed if rev else iter)(folds or range(PROTOCOLS[p])):   # rev: a second process starts at the other end
            for seed in seeds:
                path = f"{out}/{setting}_{p}_f{fold:02d}_s{seed}.json"
                if os.path.exists(path):
                    continue
                res = dict(setting=setting, n_classes=len(names), code_commit=open("kode_commit.txt").read().strip()   # optional code commit file
                           if os.path.exists("kode_commit.txt") else None, **run(F, meta, names, p, fold, seed, prefix=prefix,
                                                                       scores_out=path[:-5] + "_scores.npz"))
                json.dump(res, open(path + ".tmp", "w"))
                os.replace(path + ".tmp", path)
                print(f"{setting} {p} fold {fold} seed {seed}: " + "  ".join(
                    f"{k.split('_', 1)[1]} {100 * v['metrics']['acc']:.2f}" for k, v in res["models"].items()) + f"  ({res['sec']:.0f} s)",
                    flush=True)


def demo():
    """Needs folds_affix.csv (from folds.py): synthetic features on the real affix folds."""
    rng = np.random.default_rng(0)
    rel, meta, names = table("affix")
    n = len(rel)
    signal = np.eye(len(names))[meta["label"]] @ rng.normal(size=(len(names), 16))   # separable synthetic features
    F = {"pose": signal + rng.normal(scale=0.5, size=(n, 16)), "mt5": rng.normal(size=(n, 16))}
    r = run(F, meta, names, "cl", 9, 0, scores_out="/tmp/heads_scores.npz")   # signer 10 has no `se`, so 17 test classes
    acc = {k: v["metrics"]["acc"] for k, v in r["models"].items()}
    assert r["n_test"] == 170 and r["test_subjects"] == [10], (r["n_test"], r["test_subjects"])
    assert acc["unisign_pose_lin"] > 0.9 and acc["unisign_mt5_lin"] < 0.2, acc   # signal is read, noise stays at chance
    s = np.load("/tmp/heads_scores.npz")
    assert set(s.files) == set(r["models"]) and all(np.array_equal(s[k].argmax(1), v["pred"]) for k, v in r["models"].items())
    assert run(F, meta, names, "cl", 9, 0)["models"] == r["models"]                  # deterministic per seed
    print("demo ok: C-L fold from folds_affix.csv, two features x two heads, train_main metrics, deterministic", acc)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--demo":
        demo()
    else:
        ps = tuple(x for x in a[3:] if x in PROTOCOLS)
        fr = next((x[6:] for x in a[3:] if x.startswith("folds=")), None)   # folds=a-b: a slice for parallel jobs
        main(a[0], a[1], a[2], ps or ("cs", "cg", "cl"), rev="rev" in a[3:],
             folds=fr and range(int(fr.split("-")[0]), int(fr.split("-")[1]) + 1))
