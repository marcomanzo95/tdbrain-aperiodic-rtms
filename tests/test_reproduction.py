"""Regression tests that pin the numbers reported in the paper.

They are skipped unless the analysis directory is present, so the suite still runs
for a reader without the TDBRAIN data. When the data are available they are the
check that the code in this repository still produces what the paper claims.

Run with:  TDBRAIN_ANALYSIS_DIR=/path/to/analysis_out pytest tests/ -m slow
"""
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src import config
from src.data import load_cohort, design_matrix
from src.evaluation import nested_cv
from src.taxonomy import condition, family, family_blocks

ANALYSIS = Path(os.environ.get("TDBRAIN_ANALYSIS_DIR", "analysis_out"))
needs_data = pytest.mark.skipif(
    not (ANALYSIS / "features_discovery.csv").exists(),
    reason="TDBRAIN_ANALYSIS_DIR not set or analysis outputs missing")

#: Values as printed in the paper. A failure here means the code and the
#: manuscript have drifted apart.
PAPER = {
    "cohort_n": 167,
    "responders": 98,
    "clinical_auc": 0.6062,
    "aperiodic_eo_auc": 0.6252,
    "all_eeg_auc": 0.4652,
    "graph_ec_auc": 0.4922,
    "feature_counts": {("spectral", "EC"): 21, ("aperiodic", "EO"): 5,
                       ("connectivity", "EC"): 64, ("graph", "EC"): 48},
}
TOLERANCE = 0.0005


@needs_data
class TestCohort:
    def test_cohort_size_matches_the_paper(self):
        frame, y, eeg, artefact = load_cohort(ANALYSIS, "Responder")
        assert len(y) == PAPER["cohort_n"]
        assert int(y.sum()) == PAPER["responders"]
        assert len(eeg) == 276
        assert len(artefact) == 14

    def test_one_session_per_subject(self):
        frame, _, _, _ = load_cohort(ANALYSIS, "Responder")
        assert frame.subject.is_unique

    def test_family_sizes(self):
        _, _, eeg, _ = load_cohort(ANALYSIS, "Responder")
        blocks = family_blocks(eeg)
        for key, expected in PAPER["feature_counts"].items():
            assert len(blocks[key]) == expected, f"{key}: {len(blocks[key])}"

    def test_no_artefact_proxy_reaches_the_predictors(self):
        _, _, eeg, _ = load_cohort(ANALYSIS, "Responder")
        assert not [c for c in eeg
                    if c.startswith(("emg_", "eog_", "n_epochs_clean"))]


@needs_data
@pytest.mark.slow
class TestPublishedNumbers:
    """Full 20-repeat reproduction. Slow: several minutes per model."""

    @staticmethod
    def _auc(columns, selection, target="Responder"):
        frame, y, eeg, _ = load_cohort(ANALYSIS, target)
        cols = columns(eeg) if callable(columns) else columns
        X = design_matrix(frame, cols)
        aucs = [roc_auc_score(y, nested_cv(X, y, seed=config.SEED + r,
                                           selection=selection)[0])
                for r in range(config.N_REPEATS)]
        return float(np.mean(aucs))

    def test_clinical_benchmark(self):
        assert self._auc([], "none") == pytest.approx(
            PAPER["clinical_auc"], abs=TOLERANCE)

    def test_eyes_open_aperiodic_model(self):
        cols = lambda eeg: [c for c in eeg if family(c) == "aperiodic"
                            and condition(c) == "EO"]
        assert self._auc(cols, "forced") == pytest.approx(
            PAPER["aperiodic_eo_auc"], abs=TOLERANCE)

    def test_the_whole_feature_space_collapses_to_the_null(self):
        assert self._auc(lambda eeg: eeg, "forced") == pytest.approx(
            PAPER["all_eeg_auc"], abs=TOLERANCE)

    def test_graph_features_do_not_beat_chance(self):
        cols = lambda eeg: [c for c in eeg if family(c) == "graph"
                            and condition(c) == "EC"]
        assert self._auc(cols, "forced") == pytest.approx(
            PAPER["graph_ec_auc"], abs=TOLERANCE)
