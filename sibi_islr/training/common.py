"""Protocols, settings, train/test splits, metrics and fold/seed ranges shared by all training code
(moved from train_main.py)."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; set before torch is used
import argparse, csv, json, sys, time

import numpy as np


PROTOCOLS = {"cs": 5, "csn": 5, "cg": 5, "cl": 22, "cp": 15, "cr": 14, "all": 1,   # all: train on 22 signers, test on --external
             "rl": 10, "dr": 1, "rd": 1, "drr": 10}                         # cross-source protocols (setting dr)
CROSS = ("rl", "dr", "rd", "drr")
SETTINGS = {"affix": ("affix",), "an": ("affix", "number"), "number": ("number",), "alphabet": ("alphabet",),
            "dr": ("number", "ricky"),                                      # Darmawan numbers + external number set
            "combined": ("affix", "number", "alphabet", "kata")}             # all four categories, 114 classes


def split(meta, protocol, fold, seed, folds_dir):
    """Train and test indices. cr: without the test signer s and the removed signer r (folds_cr.csv, r drawn per seed)."""
    if protocol == "all":                                  # external test set (subject >= 100), train on all Darmawan clips
        te = meta["subject"] >= 100
        tr = ~te
    elif protocol == "csn":                                # StratifiedKFold without shuffling, file-name order per class
        f = np.array([int(r["fold"]) for r in csv.DictReader(open(f"{folds_dir}/folds_affix_csn.csv"))])
        assert len(f) == len(meta["label"]), "csn is for affix only"
        te = f == fold
        tr = ~te
    elif protocol in CROSS:                                # external number set = subjects 101-110; test on shared classes only
        r, shared = meta["subject"] > 100, meta["shared"] == 1
        if protocol == "dr":
            tr, te = ~r, r & shared
        elif protocol == "rd":
            tr, te = r, ~r & shared
        else:                                              # rl: train on the other 9 external signers; drr: + all Darmawan
            out = meta["subject"] == 101 + fold
            tr, te = (r if protocol == "rl" else np.ones_like(r)) & ~out, out & shared
    elif protocol == "cr":
        cr = [x for x in csv.DictReader(open(f"{folds_dir}/folds_cr.csv")) if int(x["seed"]) == seed]
        x = sorted(cr, key=lambda x: int(x["test_subject"]))[fold]
        s, r = int(x["test_subject"]), int(x["excluded"])
        te = meta["subject"] == s
        tr = ~te & (meta["subject"] != r)
    else:
        te = meta[protocol] == fold
        tr = ~te
    return np.flatnonzero(tr), np.flatnonzero(te)


def device_info():
    import torch
    cuda = torch.cuda.is_available()
    return dict(device="cuda" if cuda else "cpu", torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version() if cuda else None,
                gpu=torch.cuda.get_device_name(0) if cuda else "", deterministic=torch.are_deterministic_algorithms_enabled(),
                **versions())


def versions():
    """Numeric library versions, recorded per run (a cloud image cannot be pinned)."""
    import platform, scipy, sklearn
    return dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__)


def metrics(scores, y, meta_te, names):
    """Top-1 accuracy (main), top-5, macro-F1 over the classes present in the fold, F1 per class, accuracy per signer,
    and for the combined setting: within-category accuracy (argmax restricted to the classes of that category)."""
    from sklearn.metrics import f1_score
    pred = scores.argmax(1)
    present = np.unique(y)
    top5 = np.argsort(-scores, 1)[:, :5]
    m = dict(acc=float((pred == y).mean()), top5=float(np.mean([t in r for t, r in zip(y, top5)])),
             macro_f1=float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
             f1_per_class={names[c]: float(v) for c, v in zip(present, f1_score(y, pred, labels=present, average=None,
                                                                              zero_division=0))},
             acc_per_subject={str(s): float((pred[meta_te["subject"] == s] == y[meta_te["subject"] == s]).mean())
                              for s in np.unique(meta_te["subject"])})
    cats = sorted(set(meta_te["category"]))
    if len({n.split("/")[0] for n in names}) > 1:
        for c in cats:
            cls = np.array([n.startswith(c + "/") for n in names])
            sel = meta_te["category"] == c
            if sel.any():
                sc = np.where(cls, scores[sel], -np.inf)
                pc = np.unique(y[sel])                              # classes of this category present in the fold
                m[f"acc_within_{c}"] = float((sc.argmax(1) == y[sel]).mean())
                m[f"acc_{c}"] = float((pred[sel] == y[sel]).mean())
                m[f"f1_within_{c}"] = float(f1_score(y[sel], sc.argmax(1), labels=pc, average="macro", zero_division=0))
                m[f"f1_{c}"] = float(f1_score(y[sel], pred[sel], labels=pc, average="macro", zero_division=0))
    return m, pred


def rng(text, n):
    """'0-21' or '0 3 5' -> list of ints below n."""
    out = []
    for part in text:
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return [x for x in out if x < n]
