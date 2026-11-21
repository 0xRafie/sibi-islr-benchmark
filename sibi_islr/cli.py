"""Shared plumbing of the entry points: path options, then hand the remaining arguments to a module's command line.

Every module keeps the command line it had as a script (its cli() function reads sys.argv), so a step accepts
exactly the arguments documented in the module's docstring, including --demo where the module has a self-check.
"""
import argparse
import importlib
import sys

from sibi_islr import config


def parser(doc, prog):
    p = argparse.ArgumentParser(prog=prog, description=doc, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--config", metavar="YAML", help="settings file (default: the one in configs/ for this step, if any)")
    config.add_path_options(p)
    return p


def run_module(module, argv, prog):
    """Import the module (after the paths are set) and run its command line with argv."""
    sys.argv = [prog, *argv]
    mod = importlib.import_module(module)       # sibi_islr.data.inventory runs on import (no cli function)
    if hasattr(mod, "cli"):
        mod.cli()


def apply_config(path):
    """Load a YAML file of configs/ and write values that differ from the code into the recipe dictionaries."""
    if not path:
        return
    cfg = config.load_yaml(path)
    changed = config.apply_overrides(cfg)
    if changed:
        print(f"{path}: {len(changed)} value(s) differ from the paper settings: {', '.join(changed)}", file=sys.stderr)


def dispatch(doc, prog, steps, argv=None, configs=None):
    """`prog [options] STEP [ARGS ...]`: ARGS go unchanged to the module of STEP."""
    p = parser(doc, prog)
    p.add_argument("step", choices=list(steps), help="see the list above")
    p.add_argument("args", nargs=argparse.REMAINDER, help="arguments of the step (see the module docstring)")
    a = p.parse_args(argv)
    config.set_paths(a)
    apply_config(a.config or (configs or {}).get(a.step))
    run_module(steps[a.step], a.args, f"{prog} {a.step}")
