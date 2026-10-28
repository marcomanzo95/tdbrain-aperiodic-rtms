"""Paths and analysis constants.

Nothing here is hard-coded to one machine. Set TDBRAIN_ANALYSIS_DIR to the folder
produced by the pipeline scripts, or pass --analysis-dir on the command line.
"""
import os
from pathlib import Path

#: Folder holding cohort_*.csv and features_*.csv, written by pipeline/01 to 03.
ANALYSIS_DIR = Path(os.environ.get("TDBRAIN_ANALYSIS_DIR", "analysis_out"))

#: Folder for results written by the scripts in scripts/.
RESULTS_DIR = Path(os.environ.get("TDBRAIN_RESULTS_DIR", "results"))

#: Clinical covariates. BDI_pre is excluded: the blinded replication template does
#: not supply it, so a model that used it could not be scored externally.
CLINICAL = ["age", "gender"]

SEED = 42
K_FEATURES = 8          # EEG features retained by univariate selection
N_REPEATS = 20          # repeats of the whole nested cross-validation
N_PERMUTATIONS = 25     # permutations used to estimate the null of the procedure


def resolve(analysis_dir=None, results_dir=None):
    """Resolve the two directories, letting explicit arguments win over the env."""
    a = Path(analysis_dir) if analysis_dir else ANALYSIS_DIR
    r = Path(results_dir) if results_dir else RESULTS_DIR
    if not a.exists():
        raise SystemExit(
            f"Analysis directory not found: {a}\n"
            "Run the pipeline scripts first, or set TDBRAIN_ANALYSIS_DIR, "
            "or pass --analysis-dir."
        )
    r.mkdir(parents=True, exist_ok=True)
    return a, r
