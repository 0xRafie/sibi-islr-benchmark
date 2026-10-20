"""Clip counts per protocol after the data rules (dropped duplicates and copies, labels from folder names).

Reads only the file inventory darmawan_files.csv (written by inventory_darmawan.py; its relpath column starts with
`data/darmawan/`). It opens no video or keypoint file and builds no fold list. The signer-based protocols C-L
(leave-one-signer-out), C-P (leave-one-session-out), and C-R (training-signer control) are fixed by the signers, so their
fold sizes are exact. The 5-fold splits (cs, cg in folds.py) also contain the word clips, so only their numbers of strata
and groups are reported here. folds.py and pack_mp.py reuse `clips()`, `groups()`, `PAIRS`, and `SUBJECTS`.

Paths: SIBI_QC (default `qc`): reads darmawan_files.csv, writes protocol_counts.csv.

    python scripts/protocol_counts.py          # -> qc/protocol_counts.csv + summary
    python scripts/protocol_counts.py --demo
"""
import csv
import os
import re
import sys
from collections import Counter, defaultdict

from gate import DROP
from gate_variants import DROP_ALPHABET, DUP_WRONG

# Signers recorded in pairs in one session (same background, identical frame counts per class and rep); fixed list.
PAIRS = [(4, 5), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]
SUBJECTS = list(range(1, 23))
FILE = re.compile(r"subject-(\d+)-(affix|number|alphabet)-.+-(rep|sep)-(-?)(\d+)\.mp4$")
QC = os.environ.get("SIBI_QC", "qc")
OUT = f"{QC}/protocol_counts.csv"
# Misnamed file, visually checked to be number 1: `number-10-rep-1` instead of `number-1-rep-10` (the real rep-1 is in
# the same folder)
FIX_REP = {"number/1/subject-7-number-10-rep-1": 10}


def clips(rows):
    """Inventory rows -> {relpath: (category, class, signer, rep)} for the clips that are used."""
    out = {}
    for r in rows:
        cat = r["category"]
        if cat not in ("affix", "number", "alphabet"):
            continue
        rel = r["relpath"].removeprefix("data/darmawan/").removesuffix(".mp4")
        cls = rel.split("/")[1]                                      # label from the folder name
        m = FILE.search(r["filename"])
        assert m, r["filename"]
        subj, rep = int(m.group(1)), FIX_REP.get(rel, int(m.group(5)))
        if rel in DROP or rel in DROP_ALPHABET or (cat, cls, subj) in DUP_WRONG:
            continue
        # affix rep--N files of signer 3 are copies and are all in DROP; number/alphabet rep--N and sep-N count as rep-N
        assert not (cat == "affix" and m.group(4)), rel
        out[rel] = (cat, cls, subj, rep)
    cells = Counter((c, k, s, p) for c, k, s, p in out.values())
    assert max(cells.values()) == 1, [x for x, v in cells.items() if v > 1]  # one clip per (class, signer, rep)
    assert max(Counter(x[:3] for x in cells).values()) <= 10                  # at most 10 reps per cell
    return out


def groups():
    """Session groups of C-P: the seven pairs, then the single signers."""
    paired = {s for p in PAIRS for s in p}
    return [list(p) for p in PAIRS] + [[s] for s in SUBJECTS if s not in paired]


def protocols(cl, cats=("affix", "number")):
    """One row per C-L fold (22), C-P fold (15), and C-R test signer (14), per category."""
    n, cls = defaultdict(Counter), defaultdict(lambda: defaultdict(set))
    for cat, k, s, _ in cl.values():
        n[cat][s] += 1
        cls[cat][s].add(k)
    rows = []
    for cat in cats:
        every = set().union(*cls[cat].values())

        def row(proto, fold, test, trains):
            te = set().union(*(cls[cat][s] for s in test))
            for tr in trains:
                assert set().union(*(cls[cat][s] for s in tr)) == every, (proto, fold)  # every class is in training
            size = [sum(n[cat][s] for s in tr) for tr in trains]
            rows.append(dict(protocol=proto, fold=fold, category=cat, n_test=sum(n[cat][s] for s in test),
                             n_train_min=min(size), n_train_max=max(size), classes_test=len(te)))

        for s in SUBJECTS:
            row("C-L", f"s{s}", [s], [[t for t in SUBJECTS if t != s]])
        for g in groups():
            row("C-P", "+".join(f"s{s}" for s in g), g, [[t for t in SUBJECTS if t not in g]])
        for p in PAIRS:  # C-R: train without s and r; r is any of the 20 signers other than s and its partner
            for s in p:
                row("C-R", f"s{s}", [s], [[t for t in SUBJECTS if t not in (s, r)] for r in SUBJECTS if r not in p])
    return rows


def summary(cl, rows):
    out = []
    for cat in ("affix", "number", "alphabet"):
        c = [v for v in cl.values() if v[0] == cat]
        gate = sum(v[2] <= 20 for v in c)
        if cat == "alphabet":
            out.append(f"alphabet (replication only): signers 1-20 {gate} clips")
            continue
        cells = {(k, s) for _, k, s, _ in c}
        out.append(f"{cat}: {len(c)} clips, {len({v[1] for v in c})} classes (cs strata), {len(cells)} "
                   f"(signer, class) cells = cg groups; replication signers 1-20 {gate} clips")
        for proto in ("C-L", "C-P", "C-R"):
            r = [x for x in rows if x["protocol"] == proto and x["category"] == cat]
            te, tr = [x["n_test"] for x in r], [x["n_train_min"] for x in r] + [x["n_train_max"] for x in r]
            k = [x["classes_test"] for x in r]
            out.append(f"  {proto}: {len(r)} folds, test {min(te)}-{max(te)} clips, train {min(tr)}-{max(tr)}, "
                       f"test classes {min(k)}-{max(k)}")
    return out


def main():
    with open(f"{QC}/darmawan_files.csv", newline="") as fh:
        cl = clips(csv.DictReader(fh))
    count = Counter(v[0] for v in cl.values())
    assert (count["affix"], count["number"]) == (3948, 8538), count       # number without `ribuan` of signer 17
    assert sum(v[2] <= 20 for v in cl.values() if v[0] == "affix") == 3588  # = affix clips of gate_variants.pack()
    assert sum(v[2] <= 20 for v in cl.values() if v[0] == "alphabet") == 5158  # rep--N and sep-N files included
    rows = protocols(cl)
    for cat in ("affix", "number"):  # every clip is tested exactly once in C-L and in C-P
        for proto in ("C-L", "C-P"):
            assert sum(x["n_test"] for x in rows if x["protocol"] == proto and x["category"] == cat) == count[cat]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(summary(cl, rows)))
    print(f"-> {OUT} ({len(rows)} rows)")


def demo():
    rows = [dict(category="affix", relpath=f"data/darmawan/affix/{k}/{f}", filename=f) for k, f in (
        ("di", "subject-3-affix-di-rep--9.mp4"),       # copy, in DROP
        ("di", "subject-3-affix-di-rep-9.mp4"),
        ("se", "subject-10-affix-se-rep-1.mp4"),       # wrong label, in DUP_WRONG
        ("pe", "subject-10-affix-pe-rep-1.mp4"))]
    rows += [dict(category="number", relpath=f"data/darmawan/number/{k}/{f}", filename=f) for k, f in (
        ("18", "subject-2-number-18-rep--2.mp4"),      # number rep--N = rep-N
        ("2", "subject-6-number-8-rep-1.mp4"),         # label from the folder
        ("8", "subject-6-number-8-rep-01.mp4"))]
    rows += [dict(category="alphabet", relpath="data/darmawan/alphabet/s/subject-11-alphabet-s-sep-3.mp4",
                  filename="subject-11-alphabet-s-sep-3.mp4"),
             dict(category="sentence", relpath="data/darmawan/sentence/x.mp4", filename="x.mp4")]
    cl = clips(rows)
    assert sorted(cl.values()) == [("affix", "di", 3, 9), ("affix", "pe", 10, 1), ("alphabet", "s", 11, 3),
                                   ("number", "18", 2, 2), ("number", "2", 6, 1), ("number", "8", 6, 1)]
    # grid of 22 signers x 2 classes x 2 reps; cell ("b", signer 10) is empty
    grid = {f"affix/{k}/s{s}r{p}": ("affix", k, s, p) for s in SUBJECTS for k in "ab" for p in (1, 2)
            if (k, s) != ("b", 10)}
    r = protocols(grid, cats=("affix",))
    by = {(x["protocol"], x["fold"]): x for x in r}
    assert len(groups()) == 15 and sum(x["protocol"] == "C-R" for x in r) == 14
    assert by["C-L", "s10"]["n_test"] == 2 and by["C-L", "s10"]["classes_test"] == 1
    assert by["C-L", "s1"]["n_train_min"] == len(grid) - 4
    assert by["C-P", "s4+s5"]["n_test"] == 8 and by["C-P", "s4+s5"]["n_train_max"] == len(grid) - 8
    assert (by["C-R", "s4"]["n_train_min"], by["C-R", "s4"]["n_train_max"]) == (len(grid) - 8, len(grid) - 6)
    assert sum(x["n_test"] for x in r if x["protocol"] == "C-P") == len(grid)
    print("demo ok: drop rules, folder labels, rep--N/sep-N, and C-L/C-P/C-R fold sizes")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
