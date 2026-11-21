"""Paths and configuration files.

Paths are resolved once, when a module of the package is first imported, in this order: command-line options of the
entry points (--data-dir, --qc-dir, --paper-dir, --dict-dir, which set the environment variables below), then the
environment variables SIBI_DATA, SIBI_QC, SIBI_PAPER and SIBI_DICT, then the defaults `data`, `qc` and `paper`
(relative to the working directory) and the working directory for the dictionary.

- DATA:  the dataset and everything derived from it (darmawan/ videos, keypoints/, pilot/ keypoint packs, kata/ ...)
- QC:    inventories, fold tables, run outputs (final/) and analysis results (results/)
- PAPER: generated tables (*.tex) and Fig. 1-5 (figures/)
- DICT:  the SIBI dictionary videos (video/) and their keypoints (keypoints/)

The YAML files in configs/ record the settings of the paper runs; load_yaml() reads them and apply_overrides() writes
changed values into the recipe dictionaries of the modules (see train.py).
"""
import importlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA = os.environ.get("SIBI_DATA", "data")
QC = os.environ.get("SIBI_QC", "qc")
PAPER = os.environ.get("SIBI_PAPER", "paper")
DICT = os.environ.get("SIBI_DICT", "")

PATH_OPTIONS = {"data_dir": "SIBI_DATA", "qc_dir": "SIBI_QC", "paper_dir": "SIBI_PAPER", "dict_dir": "SIBI_DICT"}


def add_path_options(parser):
    """--data-dir, --qc-dir, --paper-dir, --dict-dir (default: the environment variable, else the default above)."""
    g = parser.add_argument_group("paths (override SIBI_DATA, SIBI_QC, SIBI_PAPER, SIBI_DICT)")
    for opt, env in PATH_OPTIONS.items():
        g.add_argument("--" + opt.replace("_", "-"), metavar="DIR", help=f"sets {env}")


def set_paths(args):
    """Write the path options into the environment before any module of the package reads them."""
    for opt, env in PATH_OPTIONS.items():
        value = getattr(args, opt, None)
        if value is not None:
            os.environ[env] = str(value)
    global DATA, QC, PAPER, DICT
    DATA, QC, PAPER, DICT = (os.environ.get("SIBI_DATA", "data"), os.environ.get("SIBI_QC", "qc"),
                             os.environ.get("SIBI_PAPER", "paper"), os.environ.get("SIBI_DICT", ""))


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _same(a, b):
    """YAML has lists where the code has tuples."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def _like(new, old):
    if isinstance(old, tuple) and isinstance(new, list):
        return tuple(_like(n, o) for n, o in zip(new, old)) if len(new) == len(old) else tuple(new)
    if isinstance(old, float) and isinstance(new, int):
        return float(new)
    return new


def recipe_dicts(cfg):
    """{"module.NAME": (dict object in the module, YAML mapping)} for the `recipes` section of a config."""
    out = {}
    for key, values in (cfg.get("recipes") or {}).items():
        mod, name = key.rsplit(".", 1)
        target = getattr(importlib.import_module(mod), name)
        unknown = set(values) - set(target)
        if unknown:
            raise ValueError(f"{key}: unknown keys {sorted(unknown)}")
        out[key] = (target, values)
    return out


def apply_overrides(cfg):
    """Write the YAML values that differ from the code into the recipe dictionaries (in place, so every module that
    imported the dictionary sees them). Returns the changed keys; with the shipped configs nothing changes."""
    changed = []
    for key, (target, values) in recipe_dicts(cfg).items():
        for k, v in values.items():
            if not _same(v, target[k]):
                target[k] = _like(v, target[k])
                changed.append(f"{key}[{k!r}]")
    return changed
