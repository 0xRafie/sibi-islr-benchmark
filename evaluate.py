"""Evaluation: metrics, Wilcoxon signed-rank tests with Holm correction, protocol counts, LaTeX tables.

    python evaluate.py [--qc-dir DIR] [--data-dir DIR] [--paper-dir DIR] STEP [ARGS ...]

Steps:
    analyze          evaluation/analyze.py          final [DIR] | perclass [DIR] | ablasi [DIR] | <runs dir> | --demo
    tables           evaluation/tables.py           Table III and IV -> <paper>/table_results.tex, table_ablation.tex
    protocol-counts  evaluation/protocol_counts.py  clips per protocol -> <qc>/protocol_counts.csv

Example (CPU): python evaluate.py analyze final qc/final && python evaluate.py analyze perclass qc/final \
               && python evaluate.py tables
"""
from sibi_islr.cli import dispatch

STEPS = {
    "analyze": "sibi_islr.evaluation.analyze",
    "tables": "sibi_islr.evaluation.tables",
    "protocol-counts": "sibi_islr.evaluation.protocol_counts",
}

if __name__ == "__main__":
    dispatch(__doc__, "evaluate.py", STEPS)
