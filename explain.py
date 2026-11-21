"""Test-time occlusion of the fine-tuned Uni-Sign (pose) on one signer (sibi_islr/explain/).

    python explain.py [--qc-dir DIR] STEP [ARGS ...]

Steps:
    run        explain/run.py        <setting> <src> <state> <out> --fold K [--seed S] [--shard i/n] | --report ... | --demo
    occlusion  explain/occlusion.py  --demo (keypoint groups, coalitions, Shapley values; needs a Uni-Sign clone)

One GPU for `run` (Tesla T4 in the paper); --report and the figure (make_figures.py) run on a CPU.
"""
from sibi_islr.cli import dispatch

STEPS = {"run": "sibi_islr.explain.run", "occlusion": "sibi_islr.explain.occlusion"}

if __name__ == "__main__":
    dispatch(__doc__, "explain.py", STEPS)
