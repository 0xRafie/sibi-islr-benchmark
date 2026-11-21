"""configs/*.yaml hold the settings of the paper runs: every value equals the code, and loading a shipped config
changes nothing. With SIBI_REFERENCE_SCRIPTS set to the former scripts/ folder, the values are also compared with the
constants and argparse defaults of those scripts."""
import ast
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sibi_islr import config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_map import MAP  # noqa: E402

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))
REF = os.environ.get("SIBI_REFERENCE_SCRIPTS")
OLD_OF = {"sibi_islr." + new[:-3].replace("/", "."): old for old, news in MAP.items() for new in news}
OPTIONS_OF = {"bilstm": ("train_main.py", None), "unisign": ("unisign_finetune.py", None)}


def live(dotted):
    mod, name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def same(a, b):
    if isinstance(b, range):
        b = list(b)
    return config._same(a, b)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_values_equal_code(path):
    cfg = config.load_yaml(path)
    for key, values in (cfg.get("recipes") or {}).items():
        code = live(key)
        assert set(values) == set(code), key
        for k, v in values.items():
            assert same(v, code[k]), (key, k, v, code[k])
    for key, value in (cfg.get("reference") or {}).items():
        if key.startswith("sibi_islr."):
            assert same(value, live(key)), key
    assert config.apply_overrides(cfg) == []


def test_override_writes_into_the_shared_dict():
    from sibi_islr.models.bilstm import B1
    from sibi_islr.training import bilstm_inputs
    before = dict(B1)
    try:
        assert config.apply_overrides({"recipes": {"sibi_islr.models.bilstm.B1": {"epochs": 3}}}) == [
            "sibi_islr.models.bilstm.B1['epochs']"]
        assert bilstm_inputs.B1["epochs"] == 3
    finally:
        B1.clear(), B1.update(before)
    with pytest.raises(ValueError):
        config.recipe_dicts({"recipes": {"sibi_islr.models.bilstm.B1": {"not_a_key": 1}}})


def old_constants(path):
    """Module-level constants of a former script, evaluated without importing it (env lookups -> their defaults)."""
    env = SimpleNamespace(get=lambda k, d=None: d, setdefault=lambda *a: None)
    ns = {"os": SimpleNamespace(environ=env, path=os.path), "np": np, "dict": dict, "range": range,
          "SimpleNamespace": SimpleNamespace}
    for n in ast.parse(Path(path).read_text()).body:
        if isinstance(n, ast.Assign):
            try:
                exec(compile(ast.Module(body=[n], type_ignores=[]), str(path), "exec"), ns)
            except Exception:
                pass
    return ns


def old_argparse_defaults(path, ns):
    out = {}
    for n in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument" and n.args:
            flag = ast.literal_eval(n.args[0]) if isinstance(n.args[0], ast.Constant) else None
            kw = {k.arg: k.value for k in n.keywords}
            if flag and flag.startswith("--") and "default" in kw:
                try:
                    out[flag[2:].replace("-", "_")] = eval(compile(ast.Expression(kw["default"]), "", "eval"), ns)
                except Exception:
                    pass
    return out


@pytest.mark.skipif(not REF, reason="set SIBI_REFERENCE_SCRIPTS to the former scripts/ folder")
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_values_equal_former_scripts(path):
    cfg = config.load_yaml(path)
    checked = 0
    entries = dict(cfg.get("recipes") or {})
    entries.update({k: v for k, v in (cfg.get("reference") or {}).items() if k.startswith("sibi_islr.")})
    for key, value in entries.items():
        mod, name = key.rsplit(".", 1)
        old = old_constants(Path(REF, OLD_OF[mod]))
        assert name in old, key
        assert same(value, old[name]), (key, value, old[name])
        checked += 1
    model = cfg.get("model")
    opts = (cfg.get("reference") or {}).get("options")
    if opts:
        old_file = OPTIONS_OF[model][0]
        ns = old_constants(Path(REF, old_file))
        if model == "bilstm":
            ns["B1"] = old_constants(Path(REF, "gate.py"))["B1"]
        if model == "unisign":
            ns.update(old_constants(Path(REF, "unisign_features.py")))
        defaults = old_argparse_defaults(Path(REF, old_file), ns)
        for k, v in opts.items():
            assert same(v, defaults[k]), (model, k, v, defaults[k])
            checked += 1
    ref = cfg.get("reference") or {}
    if model == "unisign":
        old = old_constants(Path(REF, "unisign_features.py"))
        assert ref["max_len"] == old["MAX_LEN"] and ref["mt5"] == old["MT5"]
    if model == "shubert":
        ns = old_constants(Path(REF, "shubert_features.py"))
        d = old_argparse_defaults(Path(REF, "shubert_features.py"), ns)
        assert ref["features_options"] == {"stride": d["stride"], "shard": d["shard"]}
    assert checked
