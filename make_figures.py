"""Paper figures (sibi_islr/figures/make.py).

    python make_figures.py [--qc-dir DIR] [--data-dir DIR] [--paper-dir DIR] [--dict-dir DIR] [final | --demo]

no argument  Fig. 1-5 -> <paper>/figures/ and Table II -> <paper>/table_keypoints.tex (needs the RTMW keypoint packs
             and the SIBI dictionary clip of panel 2c)
final        Fig. 6-7 from the final runs -> <qc>/figures/
--demo       self-check on synthetic data
"""
from sibi_islr import config
from sibi_islr.cli import parser, run_module

if __name__ == "__main__":
    p = parser(__doc__, "make_figures.py")
    a, rest = p.parse_known_args()
    config.set_paths(a)
    run_module("sibi_islr.figures.make", rest, "make_figures.py")
