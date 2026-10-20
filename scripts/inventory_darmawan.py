"""Inventory of the dataset videos (Darmawan et al.): one row per video, plus a completeness table per class folder.

Writes <SIBI_QC>/darmawan_files.csv (one row per video: category, sign, emotion, sentence_id, subject, rep, filename,
relpath, size_bytes, malformed_name) and <SIBI_QC>/darmawan_inventory.csv (one row per class folder x label in the file
name), then prints a summary per category. Only file names and sizes are written, never video content. Labels come from
the folder names, because some file names carry a wrong class. The relpath column is always written as
`data/darmawan/<category>/<class>/<file>`, the form that protocol_counts.clips() expects.

Paths: SIBI_DATA (default: `data` in the repository root) must contain darmawan/ with at least the affix, number, and
alphabet folders; SIBI_QC defaults to `qc` in the repository root. Rerun whenever videos are added.

    python scripts/inventory_darmawan.py
"""
import collections
import csv
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> repository root
DATA = Path(os.environ.get("SIBI_DATA", ROOT / "data")) / "darmawan"
OUT = Path(os.environ.get("SIBI_QC", ROOT / "qc"))
# Without these folders the script would overwrite the inventory with incomplete content.
assert all((DATA / c).is_dir() for c in ("affix", "number", "alphabet")), f"{DATA} needs the affix, number, and alphabet video folders"

# Full archive = 22 signers x 10 reps, minus the (class, signer) pairs that were never recorded (the same pattern as the
# public Mendeley release). Sentences: 5 reps.
TARGET = {"affix": 3960, "expression": 154, "alphabet": 5720, "number": 8580, "sentence": 7700}
NOT_RECORDED = {  # (number class) -> signers who did not record it
    **{str(n): {13, 14, 15, 16} for n in range(11, 20)},
    "jutaan": {15, 16},
    "sejuta": {4, 5, 6, 7, 8, 9},
}

SIGN_RE = re.compile(r"^subject-(\d+)-(affix|alphabet|number)-(.+)-(?:rep|sep)-+(\d+)\.mp4$")
SENT_RE = re.compile(r"^subject-(\d+)-sentence-(\d+)-(\w+)-rep-+(\d+)\.mp4$")
EXPR_RE = re.compile(r"^subject-(\d+)-expression-(\w+)\.mp4$")
CLEAN_RE = re.compile(r"^subject-\d+-[a-z]+-.+-rep-\d+\.mp4$|^subject-\d+-expression-\w+\.mp4$")


def parse(name):
    if m := SIGN_RE.match(name):
        return dict(category=m[2], sign=m[3], emotion="", sentence_id="", subject=m[1], rep=m[4])
    if m := SENT_RE.match(name):
        return dict(category="sentence", sign="", emotion=m[3].lower(), sentence_id=m[2],
                    subject=m[1], rep=m[4])
    if m := EXPR_RE.match(name):
        return dict(category="expression", sign="", emotion=m[2].lower(), sentence_id="",
                    subject=m[1], rep="")
    return None


rows, unparsed = [], []
for p in sorted(DATA.rglob("*.mp4")):
    if "__MACOSX" in p.parts or p.name.startswith("._"):
        continue
    info = parse(p.name)
    if info is None:
        unparsed.append(p)
        continue
    # Labels come from the folder name: some file names misspell the class
    # (e.g. number/2/subject-6-number-8-*, number/miliar/*-miliyar-*).
    folder = p.parent.name.lower()
    name_label = info["sign"] or info["emotion"]
    if info["category"] in ("affix", "alphabet", "number"):
        info["sign"] = p.parent.name
    elif info["category"] == "sentence":
        info["emotion"] = folder
    malformed = (not CLEAN_RE.match(p.name) or "--" in p.name or name_label.lower() != folder
                 or info["rep"].startswith("0"))
    info["rep"] = str(int(info["rep"])) if info["rep"] else ""
    rows.append({**info, "name_label": name_label.lower(), "filename": p.name, "relpath": f"data/darmawan/{p.relative_to(DATA)}",
                 "size_bytes": p.stat().st_size, "malformed_name": int(malformed)})

fields = ["category", "sign", "emotion", "sentence_id", "subject", "rep", "filename", "relpath",
          "size_bytes", "malformed_name"]
with open(OUT / "darmawan_files.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

# One row per (category, class folder, label in the file name), so files whose name label differs from their folder
# (miliyar, number/2 holding *-number-8-*) show up separately. "lengkap" (complete) is judged against the signers who
# recorded that class (see NOT_RECORDED).
groups = collections.defaultdict(lambda: collections.defaultdict(set))
per_class = collections.defaultdict(lambda: collections.defaultdict(set))
for r in rows:
    folder = r["sign"] or r["emotion"]
    groups[(r["category"], folder, r["name_label"])][int(r["subject"])].add(r["rep"] or r["filename"])
    per_class[(r["category"], folder)][int(r["subject"])].add(r["rep"] or r["filename"])
with open(OUT / "darmawan_inventory.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["folder", "folder_kelas", "label_nama_berkas", "n_video", "n_subjek",
                "sel_penuh_10rep", "harapan_22x10", "lengkap"])
    for (cat, folder, label), subj in sorted(groups.items()):
        n = sum(len(v) for v in subj.values())
        if cat not in ("affix", "alphabet", "number"):
            w.writerow([cat, folder, label, n, len(subj), "", "", ""])
            continue
        expected = set(range(1, 23)) - (NOT_RECORDED.get(folder, set()) if cat == "number" else set())
        whole = per_class[(cat, folder)]
        done = all(len(whole.get(s, ())) == 10 for s in expected)
        w.writerow([cat, folder, label, n, len(subj), sum(len(v) == 10 for v in subj.values()),
                    220, "ya" if done else "tidak"])

count = collections.Counter(r["category"] for r in rows)
total_disk = sum(count.values())
print(f"{'category':<11}{'on disk':>9}{'target':>9}")
for cat, tgt in TARGET.items():
    print(f"{cat:<11}{count[cat]:>9}{tgt:>9}")
print(f"{'TOTAL':<11}{total_disk:>9}{sum(TARGET.values()):>9}  "
      f"({100 * total_disk / sum(TARGET.values()):.1f}%)")
print("malformed names:", sum(r["malformed_name"] for r in rows), "| unparsed:", len(unparsed))
for p in unparsed:
    print("  ", f"data/darmawan/{p.relative_to(DATA)}")
