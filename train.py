"""Training of the five benchmark models and of the exploratory runs.

    python train.py --model MODEL [--protocol sd|si|all] [--config YAML] [--data-dir DIR] [--qc-dir DIR] [ARGS ...]

Paper models (one GPU each; all reported runs used a Tesla T4):
    bilstm    training/bilstm.py   BiLSTM of Ho and Santoso on MediaPipe hands (F0)
              ARGS: --setting combined --folds 0-21 --seeds 0-4 --data DIR --folds-dir DIR --out DIR [--shard i/n]
    signbart  training/slot2.py    SignBart with its official recipe       ARGS: <fold ...> batch=128 meanloss mp48 cat=combined
    dsta_slr  training/slot2.py    DSTA-SLR with its official recipe       ARGS: <fold ...> epochs=250 rtmw27 cat=combined [stop=E | resume]
    unisign   training/unisign.py  Uni-Sign (pose) fine-tuning on RTMW     ARGS: combined <rtmw_*.npz,...> <out> --folds K --seeds 0 [--save-dir DIR]
    shubert   training/shubert.py  SHuBERT + rank-1 LoRA (environment b)   ARGS: run <frames dir> <out dir> <fold ...>
Exploratory:
    replication     models/bilstm.py    replication check of the BiLSTM: build CATEGORY | folds | train [seed ...]
    input-variants  data/features.py    input variants of the BiLSTM: pack ... | screen | full
    heads           training/heads.py   linear heads on frozen features
    harness         training/harness.py common harness for signbart/dsta_slr (--how f0|native)

--protocol maps sd -> cs (stratified 5-fold over clips), si -> cl (leave-one-signer-out over 22 signers) and
all -> all (train on all 22 signers: the released Uni-Sign model); other values are passed unchanged. For signbart and
dsta_slr it becomes proto=..., and --mode chooses `full` (default) or `probe`. --demo in ARGS runs the self-check of
the module. --config defaults to configs/<model>.yaml; values that differ from the paper settings are written into the
recipe dictionaries of the code before training (the shipped files equal the code).
"""
import argparse
import sys
from pathlib import Path

from sibi_islr import config
from sibi_islr.cli import apply_config, parser, run_module

CONFIGS = Path(__file__).resolve().parent / "configs"
MODELS = {
    "bilstm": "sibi_islr.training.bilstm",
    "signbart": "sibi_islr.training.slot2",
    "dsta_slr": "sibi_islr.training.slot2",
    "unisign": "sibi_islr.training.unisign",
    "shubert": "sibi_islr.training.shubert",
    "replication": "sibi_islr.models.bilstm",
    "input-variants": "sibi_islr.data.features",
    "heads": "sibi_islr.training.heads",
    "harness": "sibi_islr.training.harness",
}
PROTOCOL = {"sd": "cs", "si": "cl", "all": "all"}


def model_args(model, protocol, mode, rest):
    """The command line of the model's module."""
    if "--demo" in rest or "--check-fx" in rest:
        return rest
    proto = PROTOCOL.get(protocol, protocol)
    if model in ("signbart", "dsta_slr"):
        return [mode, {"signbart": "signbart", "dsta_slr": "dsta"}[model], *rest] + ([f"proto={proto}"] if proto else [])
    if model in ("bilstm", "unisign") and proto:
        return [*rest, "--protocol", proto]
    if model == "shubert" and proto not in (None, "cl"):
        sys.exit("shubert: the LoRA runs are leave-one-signer-out only (--protocol si)")
    if proto and model not in ("shubert",):
        sys.exit(f"{model}: --protocol is not used; pass the options of the module instead")
    return rest


def main(argv=None):
    p = parser(__doc__, "train.py")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--protocol", help="sd | si | all (or an internal protocol name such as cg)")
    p.add_argument("--mode", default="full", choices=("full", "probe"), help="signbart and dsta_slr only")
    a, rest = p.parse_known_args(argv)
    config.set_paths(a)
    default = CONFIGS / f"{a.model}.yaml"
    apply_config(a.config or (default if default.exists() else None))
    run_module(MODELS[a.model], model_args(a.model, a.protocol, a.mode, rest), f"train.py --model {a.model}")


if __name__ == "__main__":
    main()
