"""Test-time occlusion of the fine-tuned combined Uni-Sign on one leave-one-signer-out fold (exploratory; the paper uses
fold 4, test signer 5), retrained with --state-dir so that its weights are saved. Conditions: the 64 coalitions of the
six groups in explain_unisign.GROUPS (exact Shapley values) and each of the 69 keypoints alone; keypoints are dropped
after load_part_kp (explain_unisign.occlude).

    python explain_unisign_run.py <setting> <rtmw.npz,...> <state.pt> <out_dir> --fold 4 [--seed 0] [--shard 0/2]   # GPU
    python scripts/explain_unisign_run.py --report <result.json> ...   # from the repo root, numpy: Shapley, per keypoint,
                                                                     #   per-clip CI (JSON on stdout)
    python explain_unisign_run.py --demo                               # numpy only

Folds are read from $SIBI_QC (default qc) if that folder exists, else from the working directory.
"""
import argparse, json, os, sys

import numpy as np

from explain_unisign import GROUPS, ROW, coalition_drop, occlude, shapley

QC = os.environ.get("SIBI_QC", "qc")
CATS = ("affix", "number", "alphabet", "kata")


def conditions():
    """(name, dropped keypoints): c00-c63 = coalitions (bit i = group i present; c63 = full input), then p<COCO index>
    for each keypoint."""
    return [(f"c{m:02d}", coalition_drop(m)) for m in range(64)] + [(f"p{k}", [k]) for k in sorted(ROW)]


def run(setting, src, state, out, fold, seed, shard):
    import torch
    from train_heads import table
    from train_main import metrics, split
    from unisign_finetune import MT5, FineTune, load_rtmw, scores
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    rel, meta, names = table(setting)
    kp, sc, off, rp = load_rtmw(src.split(","))
    pos = {r: i for i, r in enumerate(rp)}
    order = np.array([pos[r] for r in rel])
    _, te = split(meta, "cl", fold, seed, QC if os.path.isdir(QC) else ".")
    s = torch.load(state, map_location="cpu", weights_only=False)
    assert s["epoch"] == 20, f"checkpoint from epoch {s['epoch']}, not 20"
    net = FineTune(len(names), None, MT5, "cuda")
    miss = set(net.load_state_dict(s["net"], strict=False).missing_keys)
    need = {k for k, p in net.named_parameters() if p.requires_grad} | {k for k, _ in net.named_buffers()}
    assert not need & miss, f"checkpoint lacks trained parameters or BatchNorm buffers: {sorted(need & miss)[:5]}"
    y, sub = meta["label"][te], {k: v[te] for k, v in meta.items()}
    i_sh, n_sh = map(int, shard.split("/"))
    res, raw = {}, {}
    for name, drop in conditions()[i_sh::n_sh]:
        st = scores(net, (kp, sc, off), order[te], post=lambda p, d=drop: occlude(p, d))
        m, _ = metrics(st, y, sub, names)
        lp = (st - np.logaddexp.reduce(st, axis=1, keepdims=True))[np.arange(len(y)), y]   # log p of the true class, 114 classes
        m.update({f"logp_{c}": float(lp[sub["category"] == c].mean()) for c in CATS})
        res[name], raw[name] = dict(drop=drop, metrics=m), st.astype(np.float32)   # pred = argmax of the raw scores
        print(name, " ".join(f"{c} {100 * m[f'acc_within_{c}']:.2f}" for c in CATS), flush=True)
    tag = f"{out}/{setting}_cl_f{fold:02d}_s{seed}_e22_{i_sh}"
    json.dump(dict(fold=fold, seed=seed, test_index=te.tolist(), conditions=res), open(tag + ".json", "w"))
    np.savez_compressed(tag + "_scores.npz", **raw)


def report(paths):
    """Merge the JSONs of all shards -> Shapley values of the six groups and the drop per keypoint (points), per category."""
    cond = {k: v for p in paths for k, v in json.load(open(p))["conditions"].items()}
    assert len(cond) == 133, f"{len(cond)} conditions, expected 133"
    keys = ["acc", "macro_f1"] + [f"acc_within_{c}" for c in CATS] + [f"logp_{c}" for c in CATS]   # acc, macro_f1 = Overall
    k100 = lambda k: 100 if k.startswith(("acc", "macro")) else 1
    out = {}
    for k in keys:
        v = np.array([cond[f"c{m:02d}"]["metrics"][k] for m in range(64)]) * k100(k)
        out[k] = dict(full=v[63], none=v[0], shapley=dict(zip(GROUPS, shapley(v).round(3).tolist())),
                      point={n: round(v[63] - cond[n]["metrics"][k] * k100(k), 3) for n in cond if n[0] == "p"})
    return out


def clip_ci(paths, folds_csv=f"{QC}/folds_combined.csv", reps=10000, seed=0):
    """From the raw scores: per-clip Shapley values on the log-probability of the true class (114 classes) and the
    per-keypoint drop in log p, each with its mean and 95% bootstrap CI over clips, per category and for the -ti/-wati
    clips. An exploratory tool (one fold), not a pre-specified rule."""
    import csv
    te = json.load(open(paths[0]))["test_index"]
    raw = {k: v for p in paths for k, v in dict(np.load(p[:-5] + "_scores.npz")).items()}
    rows = list(csv.DictReader(open(folds_csv)))
    y, cat, cls = (np.array([rows[i][f] for i in te]) for f in ("label", "category", "class"))
    y = y.astype(int)
    lp = {k: (s - np.logaddexp.reduce(s, axis=1, keepdims=True))[np.arange(len(y)), y] for k, s in raw.items()}
    phi = shapley(np.stack([lp[f"c{m:02d}"] for m in range(64)]))             # (6 groups, clips)
    drop = {f"p{k}": lp["c63"] - lp[f"p{k}"] for k in sorted(ROW)}             # drop in log p per keypoint, per clip
    rng, out = np.random.default_rng(seed), {}
    subsets = {"overall": np.ones(len(y), bool), **{c: cat == c for c in CATS},
               "ti+wati": (cat == "affix") & np.isin(cls, ["ti", "wati"])}
    for name, sel in subsets.items():
        b = rng.integers(0, sel.sum(), (reps, sel.sum()))
        ci = lambda x: [round(float(v), 4) for v in (x.mean(), *np.percentile(x[b].mean(1), [2.5, 97.5]))]
        out[name] = dict(n=int(sel.sum()), shapley={g: ci(phi[i][sel]) for i, g in enumerate(GROUPS)},
                         point={n: ci(d[sel]) for n, d in drop.items()})
    return out


def demo():
    c = conditions()
    assert len(c) == 133 and len({n for n, _ in c}) == 133
    assert c[63] == ("c63", []) and sorted(c[0][1]) == sorted(ROW)                   # full input; all 69 keypoints dropped
    assert all(d == [int(n[1:])] for n, d in c[64:])                                  # one keypoint per p condition
    assert sorted(x for i in range(2) for x in c[i::2]) == sorted(c)                  # two shards = all conditions, once each
    # report: additive game, keypoint 9 costs 2 points -> Shapley = weights, drop of p9 = 2
    a = np.array([5.0, 1.0, 3.0, 2.0, 0.5, 0.5]) / 100
    met = lambda acc: {"acc": acc, "macro_f1": acc, **{f"acc_within_{x}": acc for x in CATS}, **{f"logp_{x}": -acc for x in CATS}}
    cond = {f"c{m:02d}": dict(metrics=met(0.5 + sum(a[i] for i in range(6) if m >> i & 1))) for m in range(64)}
    cond.update({f"p{k}": dict(metrics=met(0.62 - (0.02 if k == 9 else 0))) for k in ROW})
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        json.dump(dict(conditions=cond), open(f"{t}/r.json", "w"))
        r = report([f"{t}/r.json"])["acc_within_affix"]
    assert np.allclose(list(r["shapley"].values()), a * 100) and np.isclose(r["point"]["p9"], 2) and r["point"]["p0"] == 0
    # clip_ci: random scores for 133 conditions x 40 clips; per-clip Shapley efficiency (sum = log p full - log p empty),
    # and the CI brackets the mean
    rng = np.random.default_rng(1)
    raw = {n: rng.normal(size=(40, 114)).astype(np.float32) for n, _ in c}
    raw["p9"] = raw["c63"]                                                            # keypoint 9 has no effect
    cats = ["affix"] * 10 + ["number"] * 10 + ["alphabet"] * 10 + ["kata"] * 10
    with tempfile.TemporaryDirectory() as t:
        with open(f"{t}/f.csv", "w") as f:
            f.write("label,category,class\n" + "".join(f"{i % 114},{x},{'ti' if i < 3 else 'an'}\n" for i, x in enumerate(cats)))
        json.dump(dict(test_index=list(range(40)), conditions={}), open(f"{t}/r.json", "w"))
        np.savez(f"{t}/r_scores.npz", **raw)
        r = clip_ci([f"{t}/r.json"], f"{t}/f.csv", reps=500)
    lp = lambda s: (s - np.logaddexp.reduce(s, axis=1, keepdims=True))[np.arange(40), np.arange(40) % 114]
    eff = (lp(raw["c63"]) - lp(raw["c00"]))[:10].mean()
    assert np.isclose(sum(v[0] for v in r["affix"]["shapley"].values()), eff, atol=1e-3)
    assert r["affix"]["n"] == 10 and r["ti+wati"]["n"] == 3 and r["overall"]["n"] == 40 and r["affix"]["point"]["p9"] == [0, 0, 0]
    assert all(lo <= m <= hi for v in r["number"]["shapley"].values() for m, lo, hi in [v])
    print("demo ok: 133 conditions (64 coalitions + 69 keypoints), two shards without duplicates, report Shapley passes "
          "the additivity check, clip_ci passes per-clip efficiency and brackets the mean with its bootstrap CI")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--demo":
        demo()
    elif a[0] == "--report":
        print(json.dumps(dict(agregat=report(a[1:]), per_klip=clip_ci(a[1:])), indent=1))
    else:
        ap = argparse.ArgumentParser()
        for x in ("setting", "src", "state", "out"):
            ap.add_argument(x)
        ap.add_argument("--fold", type=int, required=True)
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--shard", default="0/1")
        o = ap.parse_args(a)
        os.makedirs(o.out, exist_ok=True)
        run(o.setting, o.src, o.state, o.out, o.fold, o.seed, o.shard)
