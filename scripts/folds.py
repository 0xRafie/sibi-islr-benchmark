"""Fold assignments of every protocol, one CSV per setting.

The clips are those kept by `protocol_counts.clips()` (the data rules applied to the file inventory) plus the word clips
in `kata_clips.csv` (written by cut_words.py; the same clips as `mp_kata.npz`). The setting `combined` is the benchmark
(affix + number + alphabet + kata, 114 classes, 22,671 clips). The settings `affix` (18 classes, the set of Ho &
Santoso), `number`, `alphabet`, and `an` (affix + number, 59 classes) cover single categories.

Columns of folds_<setting>.csv:
- cs: StratifiedKFold(5, shuffle, random_state=0) over the classes (signer-dependent).
- cg: StratifiedGroupKFold(5, shuffle, random_state=0) with group = (category, class, signer), so all takes of one
  (signer, class) pair fall in the same fold.
- cl: one fold per signer (22), leave-one-signer-out.
- cp: one fold per recording-session group (7 signer pairs + 8 single signers = 15).
folds_cr.csv: for each paired signer s and each seed, a signer r drawn from the 20 signers other than s and its partner.
Training without s and r keeps the partner in training with as many training signers (20) as in cp.

Paths: SIBI_QC (default `qc`) holds the inputs darmawan_files.csv (from inventory_darmawan.py) and kata_clips.csv, and
receives the outputs.

    python scripts/folds.py --demo
    python scripts/folds.py [setting ...]   # default combined -> qc/folds_combined.csv (+ qc/folds_cr.csv, same for any setting)
"""
import csv
import os
import sys
from collections import Counter

import numpy as np

from protocol_counts import PAIRS, SUBJECTS, clips, groups

QC = os.environ.get("SIBI_QC", "qc")
SETTINGS = {"affix": ("affix",), "an": ("affix", "number"), "number": ("number",), "alphabet": ("alphabet",),
            "combined": ("affix", "number", "alphabet", "kata")}             # the 114-class benchmark
SEEDS = range(5)
FIELDS = ["relpath", "category", "class", "label", "subject", "rep", "cs", "cg", "cl", "cp"]


def assign(cl, cats):
    """{relpath: (category, class, signer, rep)} -> fold rows of one setting."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    rel = sorted(r for r, v in cl.items() if v[0] in cats)
    lab = [f"{cl[r][0]}/{cl[r][1]}" for r in rel]
    names = sorted(set(lab))
    y = np.array([names.index(x) for x in lab])
    subj = np.array([cl[r][2] for r in rel])
    cs, cg = np.empty(len(rel), int), np.empty(len(rel), int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y)):
        cs[te] = k
    grp = np.array([f"{x}|{s}" for x, s in zip(lab, subj)])
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, grp)):
        cg[te] = k
    session = {s: i for i, g in enumerate(groups()) for s in g}
    return [dict(relpath=r, category=cl[r][0], **{"class": cl[r][1]}, label=int(y[i]), subject=int(subj[i]),
                 rep=cl[r][3], cs=int(cs[i]), cg=int(cg[i]), cl=SUBJECTS.index(int(subj[i])), cp=session[int(subj[i])])
            for i, r in enumerate(rel)]


def cr_table():
    """(seed, s, r): r is drawn with the run seed from the 20 signers other than s and its partner."""
    partner = {a: b for p in PAIRS for a, b in (p, p[::-1])}
    out = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for s in sorted(partner):
            pool = [t for t in SUBJECTS if t not in (s, partner[s])]
            out.append(dict(seed=seed, test_subject=s, partner=partner[s], excluded=int(rng.choice(pool))))
    return out


def check(rows, cr):
    """Invariants of every protocol; checked before writing and in the demo."""
    by = lambda key: Counter(r[key] for r in rows)
    assert sorted(by("cs")) == sorted(by("cg")) == list(range(5))
    labels = {r["label"] for r in rows}
    for k in range(5):                                             # cs: every fold holds every class
        assert {r["label"] for r in rows if r["cs"] == k} == labels
    cell = {}
    for r in rows:                                                 # cg: one (signer, class) pair, one fold
        assert cell.setdefault((r["category"], r["class"], r["subject"]), r["cg"]) == r["cg"]
    assert all(r["cl"] == SUBJECTS.index(r["subject"]) for r in rows) and len(by("cl")) == len({r["subject"] for r in rows})
    assert len(by("cp")) == len({tuple(g) for g in groups() if set(g) & {r["subject"] for r in rows}})
    partner = {a: b for p in PAIRS for a, b in (p, p[::-1])}
    for x in cr:
        assert x["excluded"] not in (x["test_subject"], partner[x["test_subject"]])


def main(names):
    with open(f"{QC}/darmawan_files.csv", newline="") as fh:
        cl = clips(csv.DictReader(fh))
    with open(f"{QC}/kata_clips.csv", newline="") as fh:   # kata: rep = take of the sentence video, so several clips per (class, signer, rep)
        cl.update({r["relpath"]: ("kata", r["kelas"], int(r["subject"]), int(r["rep"])) for r in csv.DictReader(fh)})
    cr = cr_table()
    for name in names:
        rows = assign(cl, SETTINGS[name])
        check(rows, cr)
        n, k = Counter(r["category"] for r in rows), len({r["label"] for r in rows})
        assert name != "combined" or (n, k) == (dict(affix=3948, number=8538, alphabet=5678, kata=4507), 114), (n, k)  # benchmark size
        with open(f"{QC}/folds_{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"{name}: {len(rows)} clips {dict(n)}, {k} classes, cs {sorted(Counter(r['cs'] for r in rows).values())}, "
              f"cg {sorted(Counter(r['cg'] for r in rows).values())} -> {QC}/folds_{name}.csv")
    with open(f"{QC}/folds_cr.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cr[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(cr)
    print(f"cr: {len(cr)} rows (5 seeds x 14 signers) -> {QC}/folds_cr.csv")


def demo():
    cl = {}
    for cat, k in (("affix", "an"), ("affix", "ber"), ("number", "1"), ("alphabet", "a")):
        for s in SUBJECTS:
            for rep in range(1, 11):
                cl[f"{cat}/{k}/subject-{s}-{cat}-{k}-rep-{rep}"] = (cat, k, s, rep)
    for s in SUBJECTS:                                             # kata: ten sentences, same (class, signer, rep)
        for j in range(10):
            cl[f"kata/saya/subject-{s}-sentence-{j}-neutral-rep-1__0"] = ("kata", "saya", s, 1)
    cr = cr_table()
    for cats, n in ((("affix",), 2), (("affix", "number"), 3), (SETTINGS["combined"], 5)):
        rows = assign(cl, cats)
        check(rows, cr)
        assert len({r["label"] for r in rows}) == n and len(rows) == n * 22 * 10
    assert len(cr) == 5 * 14 and len({(x["seed"], x["test_subject"]) for x in cr}) == 70
    from train_main import SETTINGS as S, metrics                   # combined: label = class name, as in train_main.load
    assert S["combined"] == SETTINGS["combined"]
    names = sorted({f"{r['category']}/{r['class']}" for r in rows})
    y = np.array([r["label"] for r in rows])
    sc =np.eye(len(names))[y] + 2 * (np.arange(len(names)) == names.index("kata/saya"))   # argmax over all classes -> kata/saya
    m, _ = metrics(sc, y, {k: np.array([r[k] for r in rows]) for k in ("subject", "category")}, names)
    assert all(m[f"acc_within_{c}"] == m[f"f1_within_{c}"] == 1 and m[f"acc_{c}"] == (c == "kata") for c in S["combined"]), m
    print("demo ok: cs holds every class per fold, cg never splits a (signer, class) pair, cl/cp per signer/session, cr; "
          "combined: four categories, within-category metrics per category")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main(sys.argv[1:] or ["combined"])
