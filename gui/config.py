# config.py
# Default parameters for the graph-theory analyses and for the test suite

import os
import tempfile

# --- Paths ---

# Directory used for the large temporary files written while preprocessing and
# while computing connectivity. On a machine with little free space, point
# TDBRAIN_SCRATCH_DIR at an external drive. It defaults to the system
# temporary directory and is created if it does not exist.
SCRATCH_DIR = os.environ.get("TDBRAIN_SCRATCH_DIR", tempfile.gettempdir())

# --- Parameters of the computation functions ---

# Small-worldness
N_NULL_MODELS_SW = 100  # number of null graphs used to estimate sigma
NSWAP_FACTOR_SW = 5     # multiplier for the edge swaps (n_edges * factor)

# Network-Based Statistics (NBS)
N_PERMUTATIONS_NBS = 5000  # standard number of permutations for a final analysis
P_VALUE_NBS_DEFAULT = 0.01  # p value behind the dynamic t threshold when none is given

# Multi-threshold analysis
THRESHOLD_RANGE_DEFAULT = (10, 35)  # density range, in per cent
N_STEPS_DEFAULT = 11                # number of steps across the range

# --- Parameters of the automated tests ---

N_PERMUTATIONS_NBS_TEST = 100  # kept low so the tests stay fast
N_NULL_MODELS_SW_TEST = 30     # kept low so the tests stay fast

# --- GUI defaults ---

DEFAULT_SEED = 42
DEFAULT_ALPHA_BACKBONE = 0.05
DEFAULT_T_THRESH_NBS = 3.0
