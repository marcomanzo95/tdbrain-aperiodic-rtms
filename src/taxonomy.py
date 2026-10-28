"""Feature taxonomy for the TDBRAIN aperiodic analysis.

Every EEG measure is assigned to exactly one family and one recording condition.
Ocular, myogenic and data-quality measures are deliberately kept out of the
predictor space: they are artefact proxies, not brain measures, and they are used
only to test whether a result depends on contamination.
"""
import re

ARTEFACT = re.compile(r"^(emg_|eog_|n_epochs_clean)")

FAMILIES = ("spectral", "aperiodic", "connectivity", "graph")
CONDITIONS = ("EC", "EO")


def family(column: str) -> str:
    """Return the feature family a column belongs to."""
    name = column.lower()
    if name.startswith("gt_"):
        return "graph"
    if re.match(r"^(dpli|imcoh|plv|coh)_", name):
        return "connectivity"
    if re.search(r"exponent|offset|knee|aperiodic_r2", name):
        return "aperiodic"
    return "spectral"


def condition(column: str):
    """Return 'EC' (eyes closed), 'EO' (eyes open) or None."""
    if "RESTEC" in column:
        return "EC"
    if "RESTEO" in column:
        return "EO"
    return None


def split_columns(all_columns):
    """Split raw feature columns into predictors and artefact proxies."""
    predictors = [c for c in all_columns if not ARTEFACT.match(c)]
    proxies = [c for c in all_columns if ARTEFACT.match(c)]
    return predictors, proxies


def family_blocks(predictors):
    """Return {('family', 'condition'): [columns]} for every non-empty block."""
    blocks = {}
    for fam in FAMILIES:
        for cond in CONDITIONS:
            cols = [c for c in predictors
                    if family(c) == fam and condition(c) == cond]
            if cols:
                blocks[(fam, cond)] = cols
    return blocks
