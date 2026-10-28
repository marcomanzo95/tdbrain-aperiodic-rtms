"""Unit tests for the pieces that encode the design decisions of the paper.

These run without the TDBRAIN data and are what a reader should run first.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import benjamini_hochberg, bootstrap_delta, nested_cv
from src.taxonomy import condition, family, family_blocks, split_columns
from src.transformers import ForcedKBest, Residualiser


class TestTaxonomy:
    def test_families_are_assigned_as_described_in_the_paper(self):
        assert family("gt_imcoh_alpha_C_global_RESTEC") == "graph"
        assert family("dpli_theta_global_RESTEC") == "connectivity"
        assert family("imcoh_beta_lr_RESTEO") == "connectivity"
        assert family("Right_exponent_RESTEO") == "aperiodic"
        assert family("Offset_RESTEO") == "aperiodic"
        assert family("pw_alpha_front_l_RESTEC") == "spectral"
        assert family("FAA_RESTEC") == "spectral"

    def test_conditions(self):
        assert condition("FAA_RESTEC") == "EC"
        assert condition("FAA_RESTEO") == "EO"
        assert condition("age") is None

    def test_artefact_proxies_never_enter_the_predictor_space(self):
        cols = ["emg_mass_rms_RESTEO", "eog_rms_RESTEC", "n_epochs_clean_RESTEO",
                "Exponent_RESTEO", "FAA_RESTEC"]
        predictors, proxies = split_columns(cols)
        assert set(proxies) == {"emg_mass_rms_RESTEO", "eog_rms_RESTEC",
                                "n_epochs_clean_RESTEO"}
        assert set(predictors) == {"Exponent_RESTEO", "FAA_RESTEC"}

    def test_blocks_are_disjoint_and_complete(self):
        cols = ["gt_a_RESTEC", "dpli_b_RESTEO", "Exponent_RESTEO", "pw_c_RESTEC"]
        blocks = family_blocks(cols)
        flat = [c for v in blocks.values() for c in v]
        assert sorted(flat) == sorted(cols)
        assert len(flat) == len(set(flat))


class TestForcedKBest:
    def test_forced_columns_are_always_retained(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 30))
        y = rng.integers(0, 2, 80)
        sel = ForcedKBest(k=5, n_forced=2).fit(X, y)
        assert 0 in sel.idx_ and 1 in sel.idx_
        assert len(sel.idx_) == 7
        assert sel.transform(X).shape == (80, 7)

    def test_passthrough_when_there_is_nothing_to_select(self):
        X = np.zeros((10, 4)); y = np.array([0, 1] * 5)
        sel = ForcedKBest(k=10, n_forced=2).fit(X, y)
        assert sel.transform(X).shape == (10, 4)

    def test_an_informative_feature_is_picked_up(self):
        rng = np.random.default_rng(1)
        y = np.array([0] * 50 + [1] * 50)
        X = rng.normal(size=(100, 12))
        X[:, 7] += y * 4.0                      # column 7 carries the signal
        sel = ForcedKBest(k=1, n_forced=2).fit(X, y)
        assert 7 in sel.idx_


class TestResidualiser:
    def test_a_feature_made_of_proxy_is_reduced_to_noise(self):
        rng = np.random.default_rng(2)
        proxy = rng.normal(size=(200, 1))
        keep = rng.normal(size=(200, 2))
        feature = 3.0 * proxy[:, [0]] + 0.01 * rng.normal(size=(200, 1))
        X = np.hstack([keep, proxy, feature])
        out = Residualiser(n_keep=2, n_proxy=1).fit(X).transform(X)
        assert out.shape == (200, 3)              # proxies consumed
        assert np.allclose(out[:, :2], keep)      # kept columns untouched
        assert out[:, 2].std() < 0.05             # feature reduced to residual noise

    def test_a_feature_independent_of_proxy_survives(self):
        rng = np.random.default_rng(3)
        proxy = rng.normal(size=(200, 1))
        feature = rng.normal(size=(200, 1))
        X = np.hstack([proxy, feature])
        out = Residualiser(n_keep=0, n_proxy=1).fit(X).transform(X)
        assert out[:, 0].std() > 0.8 * feature.std()

    def test_the_correction_is_fitted_on_training_data_only(self):
        rng = np.random.default_rng(4)
        train = np.hstack([rng.normal(size=(100, 1)), rng.normal(size=(100, 1))])
        r = Residualiser(n_keep=0, n_proxy=1).fit(train)
        coef_before = r.model_.coef_.copy()
        r.transform(np.hstack([rng.normal(size=(40, 1)), rng.normal(size=(40, 1))]))
        assert np.array_equal(coef_before, r.model_.coef_)


class TestInference:
    def test_benjamini_hochberg_matches_a_worked_example(self):
        q = benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.042])
        assert q[0] == pytest.approx(0.005, abs=1e-9)
        assert q[1] == pytest.approx(0.020, abs=1e-9)
        assert np.all(np.diff(q) >= -1e-12)          # monotone after sorting
        assert np.all(q <= 1.0)

    def test_bootstrap_delta_is_symmetric_under_swap(self):
        rng = np.random.default_rng(5)
        y = np.array([0] * 60 + [1] * 60)
        a = rng.normal(size=120) + y * 0.8
        b = rng.normal(size=120)
        fwd = bootstrap_delta(y, a, b, n_boot=600, seed=0)
        rev = bootstrap_delta(y, b, a, n_boot=600, seed=0)
        assert fwd["delta"] == pytest.approx(-rev["delta"], abs=1e-12)
        assert fwd["delta"] > 0
        assert fwd["ci"][0] < fwd["delta"] < fwd["ci"][1]

    def test_identical_scores_give_a_zero_difference(self):
        y = np.array([0, 1] * 40)
        s = np.linspace(0, 1, 80)
        out = bootstrap_delta(y, s, s, n_boot=300, seed=0)
        assert out["delta"] == pytest.approx(0.0, abs=1e-12)


class TestNestedCrossValidation:
    @staticmethod
    def _toy(n=120, p=6, signal=0.0, seed=0):
        rng = np.random.default_rng(seed)
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        X = rng.normal(size=(n, p))
        X[:, 0] += y * signal
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), y

    def test_output_is_a_probability_for_every_subject(self):
        X, y = self._toy()
        oof, _ = nested_cv(X, y, seed=0, selection="none")
        assert oof.shape == y.shape
        assert np.all((oof >= 0) & (oof <= 1))

    def test_it_is_deterministic_for_a_given_seed(self):
        X, y = self._toy(signal=1.5)
        a, _ = nested_cv(X, y, seed=7, selection="none")
        b, _ = nested_cv(X, y, seed=7, selection="none")
        assert np.array_equal(a, b)

    def test_real_signal_is_recovered_and_pure_noise_is_not(self):
        from sklearn.metrics import roc_auc_score
        X, y = self._toy(signal=2.0, seed=1)
        assert roc_auc_score(y, nested_cv(X, y, seed=0, selection="none")[0]) > 0.75
        X, y = self._toy(signal=0.0, seed=2)
        assert roc_auc_score(y, nested_cv(X, y, seed=0, selection="none")[0]) < 0.70
