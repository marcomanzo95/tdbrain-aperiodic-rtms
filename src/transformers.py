"""Two transformers that encode the design decisions of the analysis.

ForcedKBest keeps the clinical covariates in the model instead of letting them
compete with hundreds of EEG features inside univariate selection. The quantity
of interest is the incremental value of EEG over clinical information, not a
contest between the two.

Residualiser removes, within each training fold, the component of every EEG
feature that is linearly predictable from the ocular and myogenic proxies. It is
how the analysis asks whether a result survives the removal of measured artefact.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LinearRegression


class ForcedKBest(BaseEstimator, TransformerMixin):
    """Always keep the first ``n_forced`` columns, select ``k`` of the rest."""

    def __init__(self, k: int = 8, n_forced: int = 2):
        self.k = k
        self.n_forced = n_forced

    def fit(self, X, y=None):
        nf = self.n_forced
        rest = X[:, nf:]
        if rest.shape[1] <= self.k:
            self.idx_ = np.arange(X.shape[1])
        else:
            scores, _ = f_classif(rest, y)
            scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf)
            top = np.sort(np.argsort(scores)[::-1][: self.k])
            self.idx_ = np.concatenate([np.arange(nf), nf + top])
        return self

    def transform(self, X):
        return X[:, self.idx_]


class Residualiser(BaseEstimator, TransformerMixin):
    """Regress the artefact proxies out of the features that follow them.

    The input matrix is expected as ``[keep | proxies | features]``. The kept
    columns pass through untouched, the proxies are consumed, and the features
    are replaced by their residuals. The regression is fitted on training data
    only, so no information about a held-out fold leaks through the correction.
    """

    def __init__(self, n_keep: int = 0, n_proxy: int = 0):
        self.n_keep = n_keep
        self.n_proxy = n_proxy

    def fit(self, X, y=None):
        a, b = self.n_keep, self.n_keep + self.n_proxy
        self.model_ = LinearRegression().fit(X[:, a:b], X[:, b:])
        return self

    def transform(self, X):
        a, b = self.n_keep, self.n_keep + self.n_proxy
        residuals = X[:, b:] - self.model_.predict(X[:, a:b])
        return np.hstack([X[:, :a], residuals])
