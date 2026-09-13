"""Regenerate reported empirical figures and tables from saved outputs only.

No model training or reverse sampling is performed. The analytic figures
anchor_overview.pdf and theory_bounds.pdf have their own standalone scripts.
"""

import os
from pathlib import Path
import runpy
import sys

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
os.chdir(ROOT)

if __name__ == "__main__":
    for module in ("nab.figures", "nab.figures_ext"):
        sys.argv = [module]
        with matplotlib.rc_context():
            runpy.run_module(module, run_name="__main__")
    for script in ("covariance_anchor", "calibration", "rank_mi"):
        path = ROOT / "code" / "plots" / (script + ".py")
        sys.argv = [str(path)]
        with matplotlib.rc_context():
            runpy.run_path(str(path), run_name="__main__")
    print("Reported empirical figures and tables regenerated from cached outputs.")
