"""The released lists: file names and frame numbers only."""
import csv
from pathlib import Path

LISTS = Path(__file__).resolve().parents[1] / "lists"


def rows(name):
    with open(LISTS / name, newline="") as f:
        return list(csv.DictReader(f))


def test_clips():
    r = rows("clips.csv")
    assert len(r) == 22671
    assert set(r[0]) == {"clip", "category", "class", "signer", "take"}
    assert len({x["clip"] for x in r}) == len(r)
    assert {x["category"] for x in r} == {"affix", "number", "alphabet", "word"}
    per_cat = {c: len({x["class"] for x in r if x["category"] == c}) for c in ("affix", "number", "alphabet", "word")}
    assert per_cat == {"affix": 18, "number": 41, "alphabet": 26, "word": 29}
    assert len({x["signer"] for x in r}) == 22


def test_word_boundaries():
    r = rows("word_boundaries.csv")
    assert len(r) == 4507
    assert all(0 <= int(x["frame_start"]) < int(x["frame_end"]) for x in r)
    assert {x["source"] for x in r} <= {"manual", "forced_alignment"}
    assert len({x["video"] for x in r}) == 1099
    assert {x["clip"] for x in r} == {x["clip"] for x in rows("clips.csv") if x["category"] == "word"}
