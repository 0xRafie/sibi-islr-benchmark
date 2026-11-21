"""The package is a restructuring of the former flat scripts: every function, class and module-level constant has the
same AST after normalization (docstrings dropped, imports compared by the names they bind, the removed
`sys.path.insert` lines dropped). Set SIBI_REFERENCE_SCRIPTS to a copy of the former scripts/ folder to run it, or run
`python tests/test_equivalence.py <dir>` for the full report.
"""
import ast
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_map import MAP  # noqa: E402

PKG = Path(__file__).resolve().parents[1] / "sibi_islr"
REF = os.environ.get("SIBI_REFERENCE_SCRIPTS")

# Module-level path plumbing replaced by sibi_islr.config (same values: CLI > env SIBI_* > the same defaults)
PLUMBING = {
    'os.environ.get("SIBI_DATA", "data")': "config.DATA", 'os.environ.get("SIBI_QC", "qc")': "config.QC",
    'os.environ.get("SIBI_PAPER", "paper")': "config.PAPER", 'os.environ.get("SIBI_DICT", "")': "config.DICT",
    "Path(__file__).resolve().parent.parent": "config.REPO_ROOT", "Path(__file__).resolve().parents[1]": "config.REPO_ROOT",
    'os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qc")': 'str(config.REPO_ROOT / "qc")',
    'f"models.{a.model}"': 'f"sibi_islr.models.{a.model}"',
}


class Normalize(ast.NodeTransformer):
    def _body(self, body):
        out = []
        for i, s in enumerate(body):
            if i == 0 and isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                continue                                              # docstring
            if isinstance(s, ast.Expr) and "sys.path.insert" in ast.unparse(s) and (
                    "'scripts'" in ast.unparse(s) or "__file__" in ast.unparse(s)):
                continue                                              # former sys.path plumbing
            if isinstance(s, (ast.Import, ast.ImportFrom)):
                out += [ast.Import(names=[ast.alias(name=(a.asname or a.name).split(".")[0])]) for a in s.names]
                continue
            out.append(self.visit(s))
        return out

    def generic_visit(self, node):
        for field in ("body", "orelse", "finalbody"):
            if isinstance(getattr(node, field, None), list):
                setattr(node, field, self._body(getattr(node, field)))
        return super().generic_visit(node)


def norm(node):
    return ast.dump(Normalize().visit(ast.parse(ast.unparse(node))))


def units(tree):
    """name -> node for top-level functions, classes and assignments; '__main__' -> body of the main block or cli()."""
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out[n.name] = n
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            out["=" + ast.unparse(n.targets[0] if isinstance(n, ast.Assign) else n.target)] = n
        elif isinstance(n, ast.If) and "__main__" in ast.unparse(n.test):
            out["__main__"] = ast.Module(body=n.body, type_ignores=[])
        elif isinstance(n, ast.Expr) and not isinstance(n.value, ast.Constant):
            out[f"expr:{ast.unparse(n)[:60]}"] = n
    return out


def compare(ref_dir):
    """-> rows (old file, unit, new file, status) with status 'same', 'plumbing' or 'DIFFERENT'."""
    rows = []
    for old, news in MAP.items():
        old_units = units(ast.parse(Path(ref_dir, old).read_text()))
        new_units = {}
        for rel in news:
            u = units(ast.parse((PKG / rel).read_text()))
            if "cli" in u:
                u["__main__"] = ast.Module(body=u.pop("cli").body, type_ignores=[])
            for k, v in u.items():
                new_units.setdefault(k, (rel, v))
        for name, node in old_units.items():
            if name not in new_units:
                removed = name.startswith("expr:sys.path.insert")
                rows.append((old, name, "-", "removed: sys.path plumbing" if removed else "MISSING"))
                continue
            rel, new = new_units[name]
            a, b = norm(node), norm(new)
            raw = ast.unparse(node)
            status = "same" if "sys.path.insert(0, 'scripts')" not in raw and a == b else "plumbing"
            if a != b:
                src = ast.unparse(node)
                for k, v in PLUMBING.items():
                    src = src.replace(ast.unparse(ast.parse(k)), v)
                status = "plumbing" if norm(ast.parse(src)) == b else "DIFFERENT"
            rows.append((old, name, rel, status))
    return rows


@pytest.mark.skipif(not REF, reason="set SIBI_REFERENCE_SCRIPTS to the former scripts/ folder")
def test_every_unit_is_unchanged():
    bad = [r for r in compare(REF) if r[3] not in ("same", "plumbing", "removed: sys.path plumbing")]
    assert not bad, bad


def test_main_blocks_became_cli_without_leaking_names():
    """cli() holds the former `if __name__ == "__main__"` body; none of its local names is a module global read by a
    function (so moving the body into a function cannot change what the functions see)."""
    for rel in {r for v in MAP.values() for r in v}:
        tree = ast.parse((PKG / rel).read_text())
        cli = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cli"), None)
        if cli is None:
            continue
        assigned = {t.id for n in ast.walk(cli) for t in ast.walk(n) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
        module_level = {t.id for n in tree.body if not isinstance(n, ast.FunctionDef) or n.name != "cli"
                        for t in ast.walk(n) if isinstance(t, ast.Name)}
        globals_ = {t.id for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign)) for t in ast.walk(n)
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
        assert not (assigned & globals_), (rel, assigned & globals_)
        leaked = assigned & module_level - globals_
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != "cli"]
        free = {t.id for f in funcs for t in ast.walk(f) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Load)}
        local = {t.id for f in funcs for t in ast.walk(f) if isinstance(t, (ast.Name)) and isinstance(t.ctx, ast.Store)} | {
            a.arg for f in funcs for a in ast.walk(f) if isinstance(a, ast.arg)}
        assert not (leaked & (free - local)), (rel, leaked & (free - local))


def test_no_path_hacks_for_own_modules():
    for path in PKG.rglob("*.py"):
        src = path.read_text()
        assert 'sys.path.insert(0, "scripts")' not in src, path
        assert "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))" not in src, path


if __name__ == "__main__":
    rows = compare(sys.argv[1])
    for r in rows:
        print("\t".join(r))
    from collections import Counter
    print(Counter(r[3] for r in rows), file=sys.stderr)
