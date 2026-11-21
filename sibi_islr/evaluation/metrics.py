"""Loading the result JSONs and computing per-fold, per-class and per-category scores (moved from
analyze.py)."""
import csv, functools, glob, json, os, sys
from collections import defaultdict

import numpy as np

from sibi_islr import config


QC = config.QC
DATA = config.DATA
FINAL, RESULTS = f"{QC}/final", f"{QC}/results"


@functools.cache
def labels(setting, folds_dir=QC):
    return np.array([int(r["label"]) for r in csv.DictReader(open(f"{folds_dir}/folds_{setting}.csv"))])


def s4_name(x):
    """Model name of a slot2.py run: <model>_am21 for runs on the model's original input (option rtmw27 or mp48),
    otherwise <model>_am7 (earlier runs on the shared input). The suffixes are run tags kept in the result files."""
    return f"{x['model']}_{'am21' if {'rtmw27', 'mp48'} & set(x.get('options', ())) else 'am7'}"


def s4_cells(rows, folds_dir=QC):
    """slot2.py result JSON (proto=, one entry per fold) -> {(setting, protocol, model_am7 | model_am21):
    {(fold, seed): (acc, macro_f1)}}. Macro-F1 over the classes present in the fold, recomputed from the predictions."""
    from sklearn.metrics import f1_score
    out = defaultdict(dict)
    for x in rows:
        y, pred = labels(x["cat"], folds_dir)[x["test_idx"]], np.array(x["pred"])
        name, key = s4_name(x), (x["fold"], x["seed"])   # runs on the original input get their own name
        out[(x["cat"], x["proto"], name)][key] = (
            float((pred == y).mean()), float(f1_score(y, pred, labels=np.unique(y), average="macro", zero_division=0)))
        for nm, val in within(name, x.get("metrics", {})).items():    # from the full scores, computed by slot2.py
            out[(x["cat"], x["proto"], nm)][key] = val
    return out


def within(name, mt):
    """Combined setting: per-category scores, within-category (`model|cat`, argmax restricted to the category's classes)
    and from the scores over all classes (`model|cat*`)."""
    vals = {}
    for k in mt:
        if k.startswith("acc_within_"):
            c = k[len("acc_within_"):]
            vals[f"{name}|{c}"] = (mt[k], mt.get(f"f1_within_{c}", np.nan))
            vals[f"{name}|{c}*"] = (mt[f"acc_{c}"], mt.get(f"f1_{c}", np.nan))
    return vals


def result_paths(root):
    """Result JSONs under root. Paths containing `colab` sort last, so when a run exists in two folders the other copy
    is read first and used."""
    return sorted(glob.glob(f"{root}/**/*.json", recursive=True), key=lambda p: ("colab" in p, p))


def load(root, seeds=None):
    """-> {(setting, protocol, model): {(fold, seed): (acc, macro_f1)}}. A run found twice must give identical scores
    (a warning is printed otherwise). seeds: keep only these seeds (the paper uses seed 0 for every model)."""
    runs = defaultdict(dict)
    for p in result_paths(root):
        r = json.load(open(p))
        if isinstance(r, list) and r and isinstance(r[0], dict) and "proto" in r[0]:
            for k, cells in s4_cells(r).items():
                runs[k].update({key: v for key, v in cells.items() if seeds is None or key[1] in seeds})
            continue
        if not isinstance(r, dict) or "models" not in r or "protocol" not in r:
            continue
        if "recipe" in r and "epochs" in r and str(r["epochs"]) != r["recipe"].get("epochs"):
            continue                                   # smoke test of train_slot2.py --epochs, not a real run
        if seeds is not None and r["seed"] not in seeds:
            continue
        for name, m in r["models"].items():
            key, mt = (r["fold"], r["seed"]), m["metrics"]
            vals = {name: (mt["acc"], mt["macro_f1"]), **within(name, mt)}
            for nm, val in vals.items():
                old = runs[(r["setting"], r["protocol"], nm)].setdefault(key, val)
                if old != val and not (np.isnan(old[1]) and np.isnan(val[1])):
                    print(f"WARNING: {nm} {r['setting']} {r['protocol']} fold {key[0]} seed {key[1]} differs between runs: "
                          f"{old} vs {val}", file=sys.stderr)
    return runs


def scores(root, model, proto, setting="combined"):
    """Per-clip scores of each run -> ((fold, seed), test indices, scores over all classes); a duplicate run uses the
    first copy, as in load()."""
    seen = set()
    for p in result_paths(root):
        r = json.load(open(p))
        if isinstance(r, list):                                      # slot2.py: one list per fold, scores in <stem>_fNN_scores.npz
            items = [(e, f"{p[:-5]}_f{int(e['fold']):02d}_scores.npz", e["model"], e.get("test_idx")) for e in r
                     if isinstance(e, dict) and e.get("cat") == setting and e.get("proto") == proto and "model" in e
                     and s4_name(e) == model]
        elif isinstance(r, dict) and r.get("setting") == setting and r.get("protocol") == proto and model in r.get("models", {}):
            items = [(r, p[:-5] + "_scores.npz", model, r.get("test_index"))]
        else:
            continue
        for e, zp, key, te in items:
            k = (int(e["fold"]), int(e["seed"]))
            if k in seen or te is None or not os.path.exists(zp) or key not in np.load(zp).files:
                continue
            seen.add(k)
            yield k, np.array(te), np.load(zp)[key]


def class_acc(y, pred):
    """Per-class accuracy (P-C, as in WLASL and MS-ASL): mean of the accuracies of the classes present."""
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def per_fold(cells, k=0):
    """Mean over seeds per fold for metric k (0 = accuracy, 1 = macro-F1) -> {fold: value}."""
    by = defaultdict(list)
    for (f, s), v in cells.items():
        by[f].append(v[k])
    return {f: round(float(np.mean(sorted(v))), 12) for f, v in by.items()}   # rounded: a zero difference stays 0 for Wilcoxon


def summary(runs):
    rows = []
    for (st, pr, name), cells in sorted(runs.items()):
        seeds = defaultdict(list)
        for (f, s), v in cells.items():
            seeds[s].append(v[0])
        folds = per_fold(cells)
        rows.append(dict(setting=st, protocol=pr, model=name, runs=len(cells), folds=len(folds), seeds=len(seeds),
                         acc=100 * np.mean([v[0] for v in cells.values()]),
                         sd_seed=100 * np.std([np.mean(v) for v in seeds.values()], ddof=1) if len(seeds) > 1 else np.nan,
                         sd_fold=100 * np.std(list(folds.values()), ddof=1) if len(folds) > 1 else np.nan,
                         f1=100 * np.mean([v[1] for v in cells.values()])))
    return rows
