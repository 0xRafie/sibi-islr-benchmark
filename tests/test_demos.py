"""The self-checks (--demo) of the modules, run through the entry points on synthetic data (CPU, no dataset).

Not covered here because they need downloads or data: unisign-features and train --model unisign (Uni-Sign clone,
checkpoint and RTMW pack), shubert-features (pinned SHuBERT code and weights), explain occlusion (Uni-Sign clone),
train --model heads (a fold table), and the demos of models/signbart.py and models/dsta_slr.py (a MediaPipe pack).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMOS = [
    ("prepare_data.py", "extract"), ("prepare_data.py", "extract-rtmw"), ("prepare_data.py", "trim"),
    ("prepare_data.py", "pack"), ("prepare_data.py", "segment"), ("prepare_data.py", "align"),
    ("prepare_data.py", "fplus"), ("prepare_data.py", "cut"), ("prepare_data.py", "folds"),
    ("train.py", "--model", "bilstm"), ("train.py", "--model", "signbart"), ("train.py", "--model", "shubert"),
    ("train.py", "--model", "replication"), ("train.py", "--model", "input-variants"),
    ("train.py", "--model", "harness"),
    ("evaluate.py", "analyze"), ("evaluate.py", "tables"), ("evaluate.py", "protocol-counts"),
    ("make_figures.py",), ("explain.py", "run"),
]


@pytest.mark.parametrize("cmd", DEMOS, ids=lambda c: " ".join(c))
def test_demo(cmd, tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SIBI_")}
    r = subprocess.run([sys.executable, str(ROOT / cmd[0]), *cmd[1:], "--demo"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "demo ok" in r.stdout


def test_check_fx(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "train.py"), "--model", "bilstm", "--check-fx"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "fx ok" in r.stdout, r.stderr[-2000:]
