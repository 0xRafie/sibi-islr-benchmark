"""Summaries and statistical tests over the result JSONs, and the inputs of the paper's Tables III and IV.

A result JSON from train_main.py or train_slot2.py holds one (setting, protocol, fold, seed) run, possibly with several
models; slot2.py writes one list per fold instead. Accuracy is the main metric, macro-F1 is reported alongside. Two
models are compared fold by fold on the mean over seeds: two-sided Wilcoxon signed-rank test, 95% bootstrap CI of the
mean difference (10,000 resamples of folds, seed 0), Cliff's delta, and the Hodges-Lehmann effect with its exact CI.

    python scripts/analyze.py --demo                            # self-check on synthetic data
    python scripts/analyze.py qc/final                          # table per (setting, protocol, model)
    python scripts/analyze.py qc/final --vs bilstm [--family m1,m2]   # each model against bilstm, Holm over the family
    python scripts/analyze.py final [qc/final]                  # tests of the final runs -> qc/results/final_tests.csv,
                                                                #   final_summary.csv, final_per_signer.csv, final_a2.json
    python scripts/analyze.py perclass [qc/final]               # Tables III and IV -> final_per_class.csv,
                                                                #   final_tests_seed0.csv, final_ablasi_acc.csv
    python scripts/analyze.py ablasi [qc/final]                 # Uni-Sign ablation tests on macro-F1 -> final_ablasi_unisign.csv

Paths are relative to the working directory; environment variables move them:
    SIBI_QC    (default qc)    folds_<setting>.csv; final runs in $SIBI_QC/final; outputs in $SIBI_QC/results
    SIBI_DATA  (default data)  MediaPipe keypoint packs (pilot/mpk/mp_*.npz, pilot/mp_kata.npz) for the thumb-ablation subset
"""
import csv, functools, glob, json, os, sys
from collections import defaultdict

import numpy as np

QC = os.environ.get("SIBI_QC", "qc")
DATA = os.environ.get("SIBI_DATA", "data")
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


def per_class(root=FINAL, out=f"{RESULTS}/final_per_class.csv", seeds=(0,)):
    """Per-instance (pi) and per-class (pc) accuracy of the combined model: Overall (114 classes) and within each
    category, averaged over all runs (fold x seed) as in final_summary.csv. Table III uses seed 0 for every model
    (the BiLSTM also has seeds 1-4)."""
    rows = list(csv.DictReader(open(f"{QC}/folds_combined.csv")))
    y = np.array([int(r["label"]) for r in rows])
    lab = {c: np.array(sorted({int(r["label"]) for r in rows if r["category"] == c})) for c in CATS}
    res = []
    for m in ("bilstm", *FAMILY_A, *(v for _, vs in ABLASI for v in vs)):          # + Uni-Sign ablation variants (Table IV)
        for pr in ("cs", "cl") if m == "bilstm" else ("cl",):
            acc = defaultdict(list)
            for (_, sd), te, sc in scores(root, m, pr):
                if seeds is not None and sd not in seeds:
                    continue
                yt, pred = y[te], sc.argmax(1)
                acc["overall"].append(((pred == yt).mean(), class_acc(yt, pred)))
                for c in CATS:
                    i = np.isin(yt, lab[c])
                    pc = lab[c][sc[i][:, lab[c]].argmax(1)]                  # within category
                    acc[c].append(((pc == yt[i]).mean(), class_acc(yt[i], pc)))
            res += [dict(protocol=pr, model=m, category=c, runs=len(v), pi=100 * np.mean([a for a, _ in v]),
                         pc=100 * np.mean([b for _, b in v])) for c, v in acc.items()]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(res)
    return res


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


def cliffs(a, b):
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float((a > b).mean() - (a < b).mean())


def hodges_lehmann(d, alpha=0.05):
    """Hodges-Lehmann estimate and its exact CI, dual to the exact Wilcoxon signed-rank test (no ties, n >= 6)."""
    n = len(d)
    w = np.sort((d[:, None] + d[None, :])[np.triu_indices(n)] / 2)             # n(n+1)/2 Walsh averages
    c = np.zeros(n * (n + 1) // 2 + 1)
    c[0] = 1
    for k in range(1, n + 1):                                                  # null distribution of W+, dynamic programming
        c[k:] = c[k:] + c[:-k]
    k = int(np.searchsorted(np.cumsum(c) / 2 ** n, alpha / 2, side="right")) - 1  # largest k with P(W+ <= k) <= alpha/2
    assert k >= 0, n
    return float(np.median(w)), float(w[k]), float(w[-1 - k])


# Leave-one-signer-out fold = subject - 1. Session pairs (subjects 4-5, 11-12, ..., 21-22): second fold -> first fold
SESSION = {4: 3, 11: 10, 13: 12, 15: 14, 17: 16, 19: 18, 21: 20}


def compare(a_cells, b_cells, n_boot=10000, matched=False, sessions=False, k=0):
    """Model a against reference b, per fold.

    Main analysis: per fold, the mean over all seeds of each model, so a candidate with seed 0 is compared with the
    mean of the BiLSTM's five seeds. matched=True (sensitivity): only (fold, seed) pairs present in both.
    sessions=True (sensitivity, leave-one-signer-out only): the folds of a session pair are averaged into one unit,
    22 folds -> 15 units. k: the tested metric, 0 = accuracy (family a), 1 = macro-F1 (family b); `f1_diff` is always
    macro-F1.
    """
    from scipy.stats import wilcoxon
    if matched:
        common = set(a_cells) & set(b_cells)
        a_cells, b_cells = ({k: c[k] for k in common} for c in (a_cells, b_cells))
    A, B = ([per_fold(c, i) for i in (0, 1)] for c in (a_cells, b_cells))
    folds = sorted(set(A[0]) & set(B[0]))
    unit = (lambda f: SESSION.get(f, f)) if sessions else (lambda f: f)
    units = sorted({unit(f) for f in folds})
    agg = lambda M: np.array([np.mean([M[f] for f in folds if unit(f) == u]) for u in units])
    a, b = agg(A[k]), agg(B[k])
    d = a - b
    rg = np.random.default_rng(0)
    boot = d[rg.integers(0, len(d), (n_boot, len(d)))].mean(1)
    p = float(wilcoxon(a, b).pvalue) if len(d) > 1 and np.any(d != 0) else 1.0
    hl = hodges_lehmann(100 * d) if len(d) >= 6 else (np.nan,) * 3
    return dict(folds=len(units), diff=100 * d.mean(), lo=100 * np.percentile(boot, 2.5), hi=100 * np.percentile(boot, 97.5),
                p=p, cliff=cliffs(a, b), f1_diff=100 * (agg(A[1]) - agg(B[1])).mean(), acc_diff=100 * (agg(A[0]) - agg(B[0])).mean(),
                hl=hl[0], hl_lo=hl[1], hl_hi=hl[2])


FAMILY_A = ("signbart_am21", "dsta_am21", "unisign_ft", "shubert_lora")   # family (a): the four models against the BiLSTM
# (earlier signbart_am7, dsta_am7 and shubert_all_lin runs are still loaded but not tested)
FAMILY_B = ("bilstm_a1", "bilstm_a2", "bilstm_a4", "bilstm_a5")   # family (b), BiLSTM ablations: no resampling, thumbs
# masked, + face points, per-part box normalization
CATS = ("affix", "number", "alphabet", "kata")


def final(root=FINAL, ref="bilstm", sub=None, seeds=None):
    """Confirmatory tests of the final 114-class runs, per category on the within-category metric (`model|category`).
    Family (a): the four models against the BiLSTM on accuracy, Holm per category. Family (b): BiLSTM ablations against
    the BiLSTM on macro-F1, decision for affixes only (the other categories are controls, descriptive). Two-sided
    Wilcoxon test over folds (mean of seeds per fold); a decision needs leave-one-signer-out (cl) and Holm p < 0.05,
    reported with the Hodges-Lehmann effect and its exact 95% CI. Stratified 5-fold (cs) is descriptive. Sensitivity:
    15 session units; for family (a) also matched (fold, seed) pairs. `sub` = subset_runs(): the thumb-masked BiLSTM
    (bilstm_a2) and its reference are scored on the hand-detected subset, with the full set as sensitivity (`*_full`).
    seeds: see load()."""
    runs = load(root, seeds)
    out = []
    for fam, models, k, dec in (("a", FAMILY_A, 0, CATS), ("b", FAMILY_B, 1, ("affix",))):
        for pr in ("cl", "cs"):
            for c in CATS:
                base = runs.get(("combined", pr, f"{ref}|{c}"))
                rows = [(m, runs[("combined", pr, f"{m}|{c}")]) for m in models if ("combined", pr, f"{m}|{c}") in runs]
                if not base or not rows:
                    continue
                src = lambda m: sub if sub and m == "bilstm_a2" else runs
                pair = lambda m: (src(m)[("combined", pr, f"{m}|{c}")], src(m)[("combined", pr, f"{ref}|{c}")])
                cs = [compare(*pair(m), k=k) for m, _ in rows]
                ph = holm([x["p"] for x in cs]) if pr == "cl" and c in dec else [np.nan] * len(cs)
                for (m, cells), x, h in zip(rows, cs, ph):
                    ses = compare(*pair(m), k=k, sessions=True) if pr == "cl" else {}
                    extra = {}
                    if fam == "a":
                        mt = compare(*pair(m), k=k, matched=True)
                        extra.update(diff_matched=mt["diff"], hl_matched=mt["hl"], p_matched=mt["p"], folds_matched=mt["folds"])
                    if src(m) is sub:
                        fl = compare(cells, base, k=k)
                        extra.update(diff_full=fl["diff"], hl_full=fl["hl"], p_full=fl["p"])
                    out.append(dict(family=fam, protocol=pr, category=c, model=m, metric="acc" if k == 0 else "macro_f1",
                                    complete=x["folds"] == (22 if pr == "cl" else 5), holm=h, subset=src(m) is sub,
                                    decision=bool(pr == "cl" and c in dec and h < 0.05), p_sessions=ses.get("p"), **extra, **x))
    return out


ABLASI = (("am18", ("unisign_ft_hands",)), ("am19", ("unisign_ft_thumbs", "unisign_ft_face")))   # one Holm family each;
# the labels are the family tags written to the CSV


def ablasi(root=FINAL, ref="unisign_ft", k=1):
    """Uni-Sign ablation: each variant against the full Uni-Sign (seed 0), leave-one-signer-out. Decision on Overall
    macro-F1 (114 classes, all test clips; k=1) with Overall accuracy alongside; two-sided Wilcoxon test over the 22
    folds, Hodges-Lehmann effect with exact 95% CI, Holm per family (a variant that was not run counts as p = 1); a
    decision needs all 22 folds. Per category (within-category) descriptive; sensitivity: 15 session units. k=0 tests
    Overall accuracy instead, as in the paper (Table IV)."""
    runs = load(root)
    base, out = runs[("combined", "cl", ref)], []
    for fam, models in ABLASI:
        xs = {m: compare(runs[("combined", "cl", m)], base, k=k) for m in models if ("combined", "cl", m) in runs}
        for m, h in zip(models, holm([xs[m]["p"] if m in xs else 1.0 for m in models])):
            if m not in xs:
                out.append(dict(family=fam, model=m, category="overall", ran=False, holm=h, decision=False))
                continue
            x, ses = xs[m], compare(runs[("combined", "cl", m)], base, k=k, sessions=True)
            out.append(dict(family=fam, model=m, category="overall", ran=True, complete=x["folds"] == 22, holm=h,
                            decision=bool(x["folds"] == 22 and h < 0.05), p_sessions=ses["p"], **x))
            for c in CATS:
                if ("combined", "cl", f"{m}|{c}") in runs:
                    y = compare(runs[("combined", "cl", f"{m}|{c}")], runs[("combined", "cl", f"{ref}|{c}")], k=k)
                    out.append(dict(family=fam, model=m, category=c, ran=True, complete=y["folds"] == 22, **y))
    return out


def subset_runs(root, keep, models=("bilstm", "bilstm_a2"), setting="combined", folds_dir=QC):
    """Rescores every main and thumb-masked (A2) BiLSTM run on the subset `keep` from the per-clip scores; same shape
    as load(). Duplicate runs: the first path in sorted order is used (see result_paths)."""
    runs = defaultdict(dict)
    for p in result_paths(root):
        r = json.load(open(p))
        if not isinstance(r, dict) or r.get("setting") != setting or "models" not in r:
            continue
        name = next((m for m in models if m in r["models"]), None)
        if name is None or (r["fold"], r["seed"]) in runs[(setting, r["protocol"], name)]:
            continue
        mt = subset_metrics(p, keep, setting, folds_dir)
        for nm, val in {name: (mt["acc"], mt["macro_f1"]), **within(name, mt)}.items():
            runs[(setting, r["protocol"], nm)][(r["fold"], r["seed"])] = val
    return runs


def a2_control(sub, ref="bilstm", m="bilstm_a2", pr="cl"):
    """Control test for the thumb-masked BiLSTM (A2): drop in within-category macro-F1 (BiLSTM - A2, mean of seeds per
    fold, on the subset), then two-sided Wilcoxon tests over folds of the affix drop minus the number drop and minus the
    word drop, with Hodges-Lehmann effect and exact 95% CI. The thumb-position claim is supported if both are positive
    with p < 0.05."""
    from scipy.stats import wilcoxon
    drop = {}
    for c in ("affix", "number", "kata"):
        A, B = (per_fold(sub[("combined", pr, f"{x}|{c}")], 1) for x in (ref, m))
        drop[c] = {f: A[f] - B[f] for f in set(A) & set(B)}
    out = {}
    p_ = lambda d: float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    for c in ("number", "kata"):
        folds = sorted(set(drop["affix"]) & set(drop[c]))
        d = 100 * np.array([drop["affix"][f] - drop[c][f] for f in folds])
        units = sorted({SESSION.get(f, f) for f in folds})                     # sensitivity: 15 session units
        du = np.array([np.mean([x for f, x in zip(folds, d) if SESSION.get(f, f) == u]) for u in units])
        hl, lo, hi = hodges_lehmann(d)
        out[f"affix-{c}"] = dict(folds=len(folds), drop_affix=100 * np.mean([drop["affix"][f] for f in folds]),
                                 drop_other=100 * np.mean([drop[c][f] for f in folds]), diff=float(d.mean()), hl=hl,
                                 hl_lo=lo, hl_hi=hi, p=p_(d), p_sessions=p_(du))
    out["supported"] = all(v["hl"] > 0 and v["p"] < 0.05 for v in out.values())
    return out


def a2_confusions(root, keep, setting="combined", folds_dir=QC, pr="cl"):
    """Secondary, descriptive A2 analysis: within-affix confusions me- -> se-/ter- and se-/ter- -> me- on the subset,
    summed over folds and seeds, main BiLSTM against A2. A duplicate run is counted once."""
    rows = list(csv.DictReader(open(f"{folds_dir}/folds_{setting}.csv")))
    y = np.array([int(r["label"]) for r in rows])
    lab = {r["class"]: int(r["label"]) for r in rows if r["category"] == "affix"}
    affix = np.array(sorted(lab.values()))
    me, other = lab["me"], [lab["se"], lab["ter"]]
    out, seen = {}, set()
    for p in result_paths(root):
        r = json.load(open(p))
        if not isinstance(r, dict) or r.get("setting") != setting or r.get("protocol") != pr or "models" not in r:
            continue
        name = next((m for m in ("bilstm", "bilstm_a2") if m in r["models"]), None)
        if name is None or (name, r["fold"], r["seed"]) in seen:
            continue
        seen.add((name, r["fold"], r["seed"]))
        te = np.array(r["test_index"])
        s = np.load(p[:-5] + "_scores.npz")
        sel = keep[te] & np.isin(y[te], affix)
        t, pred = y[te][sel], affix[s[s.files[0]][sel][:, affix].argmax(1)]
        c = out.setdefault(name, dict(me_to_se_ter=0, se_ter_to_me=0, runs=0))
        c["me_to_se_ter"] += int(np.sum((t == me) & np.isin(pred, other)))
        c["se_ter_to_me"] += int(np.sum(np.isin(t, other) & (pred == me)))
        c["runs"] += 1
    return out


def holm(ps):
    """Holm-Bonferroni adjusted p-values, in input order."""
    order, out, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out.tolist()


def main(root, vs=None, family=None):
    runs = load(root)
    print(f"{'setting':8} {'prot':4} {'model':16} {'run':>4} {'fold':>4} {'seed':>4} {'acc':>7} {'sd_seed':>7} {'sd_fold':>7} {'F1':>7}")
    for r in summary(runs):
        print(f"{r['setting']:8} {r['protocol']:4} {r['model']:16} {r['runs']:4d} {r['folds']:4d} {r['seeds']:4d} "
              f"{r['acc']:7.2f} {r['sd_seed']:7.2f} {r['sd_fold']:7.2f} {r['f1']:7.2f}")
    if vs:
        print(f"\nvs {vs}: accuracy difference (points), 95% bootstrap CI, Wilcoxon p, Holm p (family = models printed per "
              f"setting and protocol), Cliff's delta, F1 difference. 'same-seed' rows = sensitivity on matched (fold, seed) pairs")
        groups = defaultdict(list)
        for (st, pr, name), cells in sorted(runs.items()):
            ref = runs.get((st, pr, vs))
            if name == vs or not ref or (family and name not in family):
                continue
            groups[(st, pr)].append((name, compare(cells, ref), compare(cells, ref, matched=True) if set(cells) & set(ref) else None))
        for (st, pr), rows in groups.items():
            ph = holm([c["p"] for _, c, _ in rows])
            for (name, c, m), h in zip(rows, ph):
                print(f"{st:8} {pr:4} {name:16} fold {c['folds']:2d}  {c['diff']:+6.2f} [{c['lo']:+6.2f}, {c['hi']:+6.2f}]  "
                      f"p {c['p']:.4f}  Holm {h:.4f}  delta {c['cliff']:+.2f}  F1 {c['f1_diff']:+6.2f}")
                if m and m["folds"] and (m["diff"], m["p"]) != (c["diff"], c["p"]):
                    print(f"{'':30}same-seed {m['diff']:+6.2f} [{m['lo']:+6.2f}, {m['hi']:+6.2f}]  p {m['p']:.4f}")


FINAL_PACKS = {"affix": f"{DATA}/pilot/mpk/mp_affix.npz", "number": f"{DATA}/pilot/mpk/mp_number.npz",
               "alphabet": f"{DATA}/pilot/mpk/mp_alphabet.npz", "kata": f"{DATA}/pilot/mp_kata.npz"}   # MediaPipe packs, final runs


def a2_subset(setting="combined", folds_dir=QC, packs=FINAL_PACKS, thr=0.5):
    """Evaluation subset for A2: clips with at least one hand detected in >= thr of the frames of the trimmed span
    (`span` of the pack). -> bool per row of <folds_dir>/folds_<setting>.csv."""
    rows = list(csv.DictReader(open(f"{folds_dir}/folds_{setting}.csv")))
    frac = {}
    for c in {r["relpath"].split("/")[0] for r in rows}:
        p = np.load(packs[c])
        off, hand = p["off"], np.any(p["seq"][:, 66:150] != 0, 1)   # left and right hand x, y (pack_mp.frame150)
        frac.update({r: hand[off[i] + a:off[i] + b].mean() for i, (r, (a, b)) in enumerate(zip(p["relpath"], p["span"]))})
    return np.array([frac[r["relpath"]] >= thr for r in rows])


def subset_metrics(run_json, keep, setting="combined", folds_dir=QC):
    """train_main.metrics (including within-category) over the test clips of a run that are in `keep`, from the
    per-clip scores <run>_scores.npz. Used to compare A2 with the main BiLSTM on the same subset."""
    import train_main
    rows = list(csv.DictReader(open(f"{folds_dir}/folds_{setting}.csv")))
    names = [n for _, n in sorted({int(r["label"]): f"{r['category']}/{r['class']}" for r in rows}.items())]
    y, subj = (np.array([int(r[k]) for r in rows]) for k in ("label", "subject"))
    cat = np.array([r["category"] for r in rows])
    te = np.array(json.load(open(run_json))["test_index"])
    z = np.load(run_json[:-5] + "_scores.npz")
    sel = te[keep[te]]
    m, _ = train_main.metrics(z[z.files[0]][keep[te]], y[sel], dict(subject=subj[sel], category=cat[sel]), names)
    return m


def friedman_nemenyi(scores, alpha=0.05):
    """Descriptive Friedman test over folds with the Nemenyi post-hoc test, and the cliques of a critical-difference
    diagram built from the SAME critical difference, so the diagram cannot contradict the pairwise tests (the
    Nemenyi-versus-CD inconsistency in Ho and Santoso). scores: {model: {fold: value}}; only folds complete for every
    model are used. Rank 1 = best. -> chi2, p, number of folds, mean ranks, CD, pairs that differ (rank difference > CD),
    cliques (rank range <= CD)."""
    from scipy.stats import friedmanchisquare, rankdata, studentized_range
    models = sorted(scores)
    folds = sorted(set.intersection(*(set(scores[m]) for m in models)))
    X = np.array([[scores[m][f] for m in models] for f in folds])
    chi2, p = friedmanchisquare(*X.T)
    R = np.mean([rankdata(-row) for row in X], 0)
    k, n = len(models), len(folds)
    cd = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    order = np.argsort(R)
    diff = [(models[i], models[j]) for a, i in enumerate(order) for j in order[a + 1:] if R[j] - R[i] > cd]
    cliques = []
    for a, i in enumerate(order):
        grp = [models[j] for j in order[a:] if R[j] - R[i] <= cd]
        if len(grp) > 1 and not any(set(grp) <= set(c) for c in cliques):
            cliques.append(grp)
    return dict(chi2=float(chi2), p=float(p), n=n, ranks=dict(zip(models, map(float, R))), cd=float(cd), different=diff,
                cliques=cliques)


def report(root=FINAL, out=RESULTS):
    """Full final analysis: completeness, duplicate runs, A2 subset, family (a) and (b) tests, A2 control and secondary
    analysis, descriptive cells. Writes <out>/final_tests.csv, final_a2.json, final_per_signer.csv, final_summary.csv,
    and <SIBI_QC>/final_duplikat.csv."""
    import os
    os.makedirs(out, exist_ok=True)
    runs = load(root)
    for m in ("bilstm", *FAMILY_A, *FAMILY_B):                              # leave-one-signer-out completeness before testing
        cells = runs.get(("combined", "cl", m), {})
        seeds = sorted({s for _, s in cells})
        print(f"complete? {m:16} seed {seeds} runs {len(cells)} missing {[(f, s) for f in range(22) for s in seeds if (f, s) not in cells]}")
    grp = defaultdict(list)                                                  # duplicate runs
    for p in result_paths(root):
        r = json.load(open(p))
        if isinstance(r, dict) and r.get("setting") == "combined" and "models" in r:
            grp[(r["protocol"], next(iter(r["models"])), r["fold"], r["seed"])].append(p)
    with open(f"{QC}/final_duplikat.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "model", "fold", "seed", "dipakai", "lain", "skor_identik"])
        for k, ps in sorted(grp.items()):
            if len(ps) > 1:
                z = [np.load(p[:-5] + "_scores.npz") for p in ps]
                same = all(np.array_equal(z[0][z[0].files[0]], x[x.files[0]]) for x in z[1:])
                w.writerow([*k, ps[0].split("/")[-3], ";".join(p.split("/")[-3] for p in ps[1:]), same])
    keep = a2_subset()
    rows = list(csv.DictReader(open(f"{QC}/folds_combined.csv")))
    cat, fold = np.array([r["category"] for r in rows]), np.array([int(r["cl"]) for r in rows])
    print("A2 subset per category:", {c: f"{int(keep[cat == c].sum())}/{int((cat == c).sum())}" for c in CATS},
          "smallest per fold:", {c: int(min(keep[(cat == c) & (fold == f)].sum() for f in range(22))) for c in CATS})
    sub = subset_runs(root, keep)
    res = final(root, sub=sub)
    with open(f"{out}/final_tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in res for k in r)))
        w.writeheader()
        w.writerows(res)
    a2 = dict(control=a2_control(sub), confusions=a2_confusions(root, keep))
    json.dump(a2, open(f"{out}/final_a2.json", "w"), indent=1)
    with open(f"{out}/final_per_signer.csv", "w", newline="") as f:    # spread over signers (accuracy per held-out signer)
        w = csv.writer(f)
        w.writerow(["model", "category", "folds", "mean", "median", "q1", "q3", "min", "max", "sd"])
        for (st, pr, name), cells in sorted(runs.items()):
            if st == "combined" and pr == "cl" and "|" in name and not name.endswith("*"):
                v = 100 * np.array(list(per_fold(cells).values()))     # mean of seeds per fold = per held-out signer
                w.writerow([*name.split("|"), len(v), *np.round([v.mean(), np.median(v), *np.percentile(v, [25, 75]),
                                                                 v.min(), v.max(), v.std(ddof=1)], 2)])
    summ = [r for r in summary(runs) if r["setting"] == "combined"]
    with open(f"{out}/final_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    return res, a2


def demo():
    assert class_acc(np.array([0, 0, 1]), np.array([0, 1, 1])) == 0.75          # (0.5 + 1) / 2
    import os, tempfile
    with tempfile.TemporaryDirectory() as d:                 # A2 subset: hand detected in >= 50% of the trimmed span
        rows = [("affix/an/x1", "affix", "an", 0, 1), ("affix/ber/x2", "affix", "ber", 1, 2), ("kata/ibu/x3", "kata", "ibu", 2, 1)]
        with open(f"{d}/folds_combined.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["relpath", "category", "class", "label", "subject"])
            w.writerows(rows)
        packs = {}
        for c, rels, dets in (("affix", ["affix/an/x1", "affix/ber/x2"], [[1, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0]]),
                              ("kata", ["kata/ibu/x3"], [[0, 1, 1, 0, 0, 0]])):
            seq = np.zeros((6 * len(rels), 150), np.float32)
            seq[:, 108] = np.concatenate(dets)               # right hand only; frame 4 of x1 lies outside the span
            packs[c] = f"{d}/mp_{c}.npz"
            np.savez(packs[c], seq=seq, off=np.arange(0, 6 * len(rels) + 1, 6), span=[[0, 4]] * len(rels), relpath=rels)
        keep = a2_subset(folds_dir=d, packs=packs)
        assert keep.tolist() == [True, False, True], keep    # 2/4, 1/4, 2/4 frames
        json.dump(dict(test_index=[0, 1, 2]), open(f"{d}/run.json", "w"))
        np.savez(f"{d}/run_scores.npz", bilstm=np.array([[2, 1, 0], [2, 1, 0], [0, 0, 1]], np.float32))
        m = subset_metrics(f"{d}/run.json", keep, folds_dir=d)
        assert m["acc"] == 1.0 and m["acc_within_affix"] == 1.0, m   # x2 (misclassified) is outside the subset
        json.dump(dict(setting="combined", protocol="cl", fold=0, seed=0, models={"bilstm": {}}, test_index=[0, 1, 2]),
                  open(f"{d}/r.json", "w"))
        np.savez(f"{d}/r_scores.npz", bilstm=np.array([[2, 1, 0], [2, 1, 0], [0, 0, 1]], np.float32))
        sr = subset_runs(d, keep, folds_dir=d)
        assert sr[("combined", "cl", "bilstm")][(0, 0)] == (1.0, 1.0) and sr[("combined", "cl", "bilstm|affix")][(0, 0)][0] == 1.0
    with tempfile.TemporaryDirectory() as d:                 # secondary A2 analysis: me <-> se/ter confusions among affixes
        with open(f"{d}/folds_combined.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["relpath", "category", "class", "label", "subject"])
            w.writerows([("a/me/1", "affix", "me", 0, 1), ("a/se/1", "affix", "se", 1, 1), ("a/ter/1", "affix", "ter", 2, 1),
                         ("k/ibu/1", "kata", "ibu", 3, 1)])
        for name, sc in (("bilstm", [[0, 2, 1, 9], [3, 1, 0, 0], [0, 1, 2, 0], [0, 0, 0, 1]]), ("bilstm_a2", np.eye(4))):
            json.dump(dict(setting="combined", protocol="cl", fold=0, seed=0, models={name: {}}, test_index=[0, 1, 2, 3]),
                      open(f"{d}/{name}.json", "w"))
            np.savez(f"{d}/{name}_scores.npz", **{name: np.array(sc, np.float32)})
        cf = a2_confusions(d, np.ones(4, bool), folds_dir=d)
        assert cf["bilstm"] == dict(me_to_se_ter=1, se_ter_to_me=1, runs=1) and cf["bilstm_a2"]["me_to_se_ter"] == 0, cf
    for affix_drop, ok in ((lambda f, s: 0.05 + 0.001 * f, True), (lambda f, s: 0.01 + 0.0001 * (-1) ** f * (f + 1), False)):
        sub = defaultdict(dict)                              # A2 control test: affix drop against number and word drops
        for f in range(22):
            for s in range(5):
                for c, dr in (("affix", affix_drop(f, s)), ("number", 0.01), ("kata", 0.01 + 0.0005 * s)):
                    sub[("combined", "cl", f"bilstm|{c}")][(f, s)] = (0.9, 0.9)
                    sub[("combined", "cl", f"bilstm_a2|{c}")][(f, s)] = (0.9, 0.9 - dr)
        ctl = a2_control(sub)
        assert ctl["supported"] == ok and ctl["affix-number"]["folds"] == 22 and "p_sessions" in ctl["affix-kata"], ctl
        assert not ok or (ctl["affix-number"]["hl"] > 3 and ctl["affix-number"]["p"] < 1e-4), ctl
    with tempfile.TemporaryDirectory() as d:                 # final(): family (a) on accuracy, (b) on macro-F1, affixes only
        rg = np.random.default_rng(0)
        for f in range(22):
            for s in range(5):
                mt = lambda acc, f1: dict(acc=acc, macro_f1=f1, acc_within_affix=acc, f1_within_affix=f1,
                                          acc_affix=acc, f1_affix=f1, acc_within_number=acc, f1_within_number=f1,
                                          acc_number=acc, f1_number=f1)
                b0 = 0.8 + 0.005 * f + rg.normal(0, 0.002)
                ms = {"bilstm": mt(b0, b0), "bilstm_a2": mt(b0, b0 - 0.03)}
                if s == 0:
                    ms["signbart_am21"] = mt(b0 + 0.05, b0)
                json.dump(dict(setting="combined", protocol="cl", fold=f, seed=s,
                               models={k: dict(metrics=v) for k, v in ms.items()}), open(f"{d}/r{f}_{s}.json", "w"))
        res = {(r["family"], r["category"], r["model"]): r for r in final(d)}
        assert res[("a", "affix", "signbart_am21")]["decision"] and res[("a", "affix", "signbart_am21")]["diff"] > 4
        assert res[("b", "affix", "bilstm_a2")]["decision"] and res[("b", "affix", "bilstm_a2")]["metric"] == "macro_f1"
        assert not res[("b", "number", "bilstm_a2")]["decision"] and np.isnan(res[("b", "number", "bilstm_a2")]["holm"])
        res2 = {(r["family"], r["category"], r["model"]): r for r in final(d, sub=load(d))}   # A2 on the subset; matched pairs
        x = res2[("b", "affix", "bilstm_a2")]
        assert x["subset"] and x["p_full"] == x["p"] and x["acc_diff"] == 0 and not res2[("b", "affix", "bilstm_a2")].get("p_matched")
        assert res2[("a", "affix", "signbart_am21")]["folds_matched"] == 22 and not res2[("a", "affix", "signbart_am21")]["subset"]
    with tempfile.TemporaryDirectory() as d:                 # ablasi(): Overall macro-F1, Holm per family, not run = p 1
        for f in range(22):
            mt = lambda v: dict(acc=v, macro_f1=v, acc_within_affix=v, f1_within_affix=v, acc_affix=v, f1_affix=v)
            b0 = 0.9 + 0.003 * f
            ms = {"unisign_ft": mt(b0), "unisign_ft_hands": mt(b0 - 0.05 - 0.001 * f), "unisign_ft_thumbs": mt(b0 - 0.02 - 0.0005 * f)}
            json.dump(dict(setting="combined", protocol="cl", fold=f, seed=0, models={k: dict(metrics=v) for k, v in ms.items()}),
                      open(f"{d}/r{f}.json", "w"))
        ab = {(r["model"], r["category"]): r for r in ablasi(d)}
        h, t, fc = ab[("unisign_ft_hands", "overall")], ab[("unisign_ft_thumbs", "overall")], ab[("unisign_ft_face", "overall")]
        assert h["decision"] and h["folds"] == 22 and h["hl"] < -5 and h["holm"] == h["p"], h      # one-variant family
        assert not fc["ran"] and fc["holm"] == 1.0 and t["holm"] == min(1.0, 2 * t["p"]) and t["decision"], t
        assert ab[("unisign_ft_hands", "affix")]["folds"] == 22 and "decision" not in ab[("unisign_ft_hands", "affix")]
    rg = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as d:
        for f in range(22):
            for s in range(5):
                base = 0.6 + 0.01 * f
                r = dict(setting="affix", protocol="cl", fold=f, seed=s, models={
                    "bilstm": dict(metrics=dict(acc=base, macro_f1=base - 0.05)),
                    "better": dict(metrics=dict(acc=base + 0.05 + rg.normal(0, 0.005), macro_f1=base))})
                json.dump(r, open(f"{d}/r_{f}_{s}.json", "w"))
        json.dump(json.load(open(f"{d}/r_0_0.json")), open(f"{d}/dup.json", "w"))   # duplicate run: counted once
        json.dump(dict(setting="affix", protocol="cl", fold=0, seed=9, epochs=1, recipe=dict(epochs="200"),
                       models={"bilstm": dict(metrics=dict(acc=0.0, macro_f1=0.0))}), open(f"{d}/smoke.json", "w"))
        json.dump(dict(setting="an", protocol="cs", fold=0, seed=0, models={"bilstm": dict(metrics=dict(
            acc=0.9, macro_f1=0.8, acc_within_affix=0.7, f1_within_affix=0.6, acc_affix=0.5, f1_affix=0.4))}),
            open(f"{d}/an.json", "w"))
        runs = load(d)
        assert runs[("an", "cs", "bilstm|affix")][(0, 0)] == (0.7, 0.6) and runs[("an", "cs", "bilstm|affix*")][(0, 0)] == (0.5, 0.4)
        del runs[("an", "cs", "bilstm")], runs[("an", "cs", "bilstm|affix")], runs[("an", "cs", "bilstm|affix*")]
        assert len(runs[("affix", "cl", "bilstm")]) == 110
        rows = {r["model"]: r for r in summary(runs)}
        assert abs(rows["bilstm"]["acc"] - 100 * np.mean([0.6 + 0.01 * f for f in range(22)])) < 1e-9
        assert rows["bilstm"]["sd_seed"] == 0 and rows["bilstm"]["folds"] == 22
        c = compare(runs[("affix", "cl", "better")], runs[("affix", "cl", "bilstm")])
        assert c["folds"] == 22 and 4 < c["diff"] < 6 and c["lo"] > 0 and c["p"] < 1e-4 and c["f1_diff"] > 0, c
        cu = compare(runs[("affix", "cl", "better")], runs[("affix", "cl", "bilstm")], sessions=True)
        assert cu["folds"] == 15 and abs(cu["f1_diff"] - c["f1_diff"]) < 1e-9 and cu["p"] > c["p"], cu   # 15 session units
        seed0 = {k: v for k, v in runs[("affix", "cl", "better")].items() if k[1] == 0}         # candidate with seed 0 only
        assert compare(seed0, runs[("affix", "cl", "bilstm")])["folds"] == 22
        assert compare(seed0, runs[("affix", "cl", "bilstm")], matched=True)["folds"] == 22
        from scipy.stats import wilcoxon
        assert hodges_lehmann(np.array([1., 2, 3, 4, 5, 6])) == (3.5, 1.0, 6.0)               # n = 6: the 96.9% CI is [min, max]
        for _ in range(300):                                                                   # duality with the exact test
            x = rg.normal(0.4, 1, 22)
            _, lo, hi = hodges_lehmann(x)
            assert (wilcoxon(x, method="exact").pvalue <= 0.05) == (lo > 0 or hi < 0)
        assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06] and holm([0.0205] + [0.5] * 5)[0] == 0.123
        fn = friedman_nemenyi({m: {f: 0.9 - 0.02 * i + rg.normal(0, 0.005) for f in range(22)} for i, m in enumerate("ABCDE")})
        assert fn["ranks"]["A"] < fn["ranks"]["E"] and ("A", "E") in fn["different"] and fn["p"] < 1e-6, fn
        assert all(fn["ranks"][y] - fn["ranks"][x] <= fn["cd"] for c in fn["cliques"] for x in c for y in c), fn   # cliques = CD
        assert not any(x in c and y in c for x, y in fn["different"] for c in fn["cliques"]), fn
        with open(f"{d}/folds_affix.csv", "w", newline="") as fh:   # slot2.py format; fold 9 has no clip of class 13
            w = csv.writer(fh)
            w.writerow(["label", "cl"])
            w.writerows([(c, f) for f in (8, 9) for c in range(18) if (f, c) != (9, 13)])
        y = labels("affix", d)
        te = np.flatnonzero(np.array([int(r["cl"]) for r in csv.DictReader(open(f"{d}/folds_affix.csv"))]) == 9)
        c4 = s4_cells([dict(cat="affix", proto="cl", model="dsta", fold=9, seed=0, test_idx=te.tolist(), pred=y[te].tolist(),
                            metrics=dict(acc_within_kata=0.7, f1_within_kata=0.6, acc_kata=0.5, f1_kata=0.4)),
                       dict(cat="affix", proto="cl", model="sb", fold=9, seed=0, test_idx=te.tolist(),
                            pred=np.where(np.arange(len(te)) == 0, 13, y[te]).tolist()),
                       dict(cat="affix", proto="cl", model="dsta", fold=9, seed=0, test_idx=te.tolist(), pred=y[te].tolist(),
                            options=["epochs=250", "rtmw27", "resume"])], folds_dir=d)   # original input: does not overwrite
        assert c4[("affix", "cl", "dsta_am7")][(9, 0)] == (1.0, 1.0) and c4[("affix", "cl", "dsta_am21")][(9, 0)] == (1.0, 1.0)
        assert c4[("affix", "cl", "dsta_am7|kata")][(9, 0)] == (0.7, 0.6) and c4[("affix", "cl", "dsta_am7|kata*")][(9, 0)] == (0.5, 0.4)
        acc_sb, f1_sb = c4[("affix", "cl", "sb_am7")][(9, 0)]
        assert abs(acc_sb - (len(te) - (y[te][0] != 13)) / len(te)) < 1e-12 and f1_sb < 1   # class 13 is not among the F1 classes
        assert cliffs([1, 2], [0, 0]) == 1.0 and cliffs([0], [1]) == -1.0
    print("demo ok: JSON loading (duplicate runs counted once), per-seed and per-fold summary, Wilcoxon + bootstrap + "
          "Cliff's delta against the reference, Hodges-Lehmann CI, Holm, tests of the final runs and ablations")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--demo":
        demo()
    elif a[0] == "final":                                    # analyze.py final [qc/final]
        res, a2 = report(*a[1:2])
        for r in res:
            if r["protocol"] == "cl":
                print(f"({r['family']}) {r['category']:8} {r['model']:16} {r['metric']:8} mean {r['diff']:+6.2f} HL {r['hl']:+6.2f} "
                      f"[{r['hl_lo']:+6.2f}, {r['hl_hi']:+6.2f}] p {r['p']:.4f} Holm {r['holm']:.4f} "
                      f"{'DECISION' if r['decision'] else '-'} sessions {r['p_sessions']:.4f} accuracy {r['acc_diff']:+6.2f}"
                      + (f" matched p {r['p_matched']:.4f}" if "p_matched" in r else "") + (f" full p {r['p_full']:.4f}" if "p_full" in r else ""))
        print(json.dumps(a2, indent=1))
    elif a[0] == "ablasi":                                   # analyze.py ablasi [qc/final]
        res = ablasi(*a[1:2])
        for r in res:
            if not r["ran"]:
                print(f"({r['family']}) {r['model']:18} not run, p = 1, Holm {r['holm']:.4f}")
                continue
            print(f"({r['family']}) {r['model']:18} {r['category']:8} {r['folds']:2d} folds | macro-F1 mean {r['diff']:+6.2f} HL {r['hl']:+6.2f} "
                  f"[{r['hl_lo']:+6.2f}, {r['hl_hi']:+6.2f}] p {r['p']:.4f}"
                  + (f" Holm {r['holm']:.4f} {'DECISION' if r['decision'] else '-'} sessions {r['p_sessions']:.4f}" if r["category"] == "overall" else "")
                  + f" | accuracy mean {r['acc_diff']:+6.2f}")
        os.makedirs(RESULTS, exist_ok=True)
        with open(f"{RESULTS}/final_ablasi_unisign.csv", "w", newline="") as f:
            keys = sorted({k for r in res for k in r})
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(res)
    elif a[0] == "perclass":                                 # analyze.py perclass [qc/final]
        print(len(per_class(*a[1:2])), "rows")
        res = [r for r in final(*a[1:2], seeds=(0,)) if r["family"] == "a"]     # Table III: tests against BiLSTM seed 0
        with open(f"{RESULTS}/final_tests_seed0.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in res for k in r)), lineterminator="\n")
            w.writeheader()
            w.writerows(res)
        ab = ablasi(*a[1:2], k=0)                                            # Table IV: tests on Overall accuracy
        with open(f"{RESULTS}/final_ablasi_acc.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in ab for k in r}), lineterminator="\n")
            w.writeheader()
            w.writerows(ab)
    else:
        main(a[0], a[a.index("--vs") + 1] if "--vs" in a else None,
             a[a.index("--family") + 1].split(",") if "--family" in a else None)
