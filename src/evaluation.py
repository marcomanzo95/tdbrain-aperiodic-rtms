"""Nested cross-validation, the empirical null, and inference.

The one methodological point worth stating plainly: because out-of-fold
calibrated probabilities are pooled across folds before a single AUC is computed,
the null of this procedure is not exactly 0.50. Every model is therefore compared
with a null estimated by permuting the outcome and rerunning the whole pipeline,
not with a nominal 0.50.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .transformers import ForcedKBest

GRID = {"clf__C": [0.02, 0.06, 0.2, 0.6], "clf__l1_ratio": [0.2, 0.5, 0.8]}


def nested_cv(X, y, seed, selection="forced", k=8, n_forced=2,
              n_outer=5, n_inner=3):
    """One pass of nested cross-validation. Returns pooled out-of-fold scores.

    ``selection`` is 'none' (no feature selection), 'free' (plain SelectKBest,
    clinical covariates compete with EEG features) or 'forced' (clinical
    covariates always retained).
    """
    outer = StratifiedKFold(n_outer, shuffle=True, random_state=seed)
    inner = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    selected = []

    for train, test in outer.split(X, y):
        # Imputation and scaling are fitted once per outer fold, on training
        # data only. Doing this inside the grid search instead would repeat the
        # same work for every hyperparameter combination at no statistical gain.
        pre = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())]).fit(X.iloc[train])
        X_tr, X_te = pre.transform(X.iloc[train]), pre.transform(X.iloc[test])

        steps = []
        if selection == "free" and X.shape[1] > k:
            steps.append(("select", SelectKBest(f_classif, k=k)))
        elif selection == "forced" and X.shape[1] > k + n_forced:
            steps.append(("select", ForcedKBest(k=k, n_forced=n_forced)))
        steps.append(("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=900,
            tol=1e-3, random_state=seed)))

        search = GridSearchCV(Pipeline(steps), GRID, cv=inner,
                              scoring="roc_auc", n_jobs=-1, refit=True)
        search.fit(X_tr, y[train])

        if "select" in dict(search.best_estimator_.steps):
            sel = search.best_estimator_.named_steps["select"]
            idx = sel.idx_ if hasattr(sel, "idx_") else np.where(sel.get_support())[0]
            selected.append([X.columns[i] for i in idx])

        calibrated = CalibratedClassifierCV(
            search.best_estimator_, method="sigmoid", cv=inner)
        calibrated.fit(X_tr, y[train])
        oof[test] = calibrated.predict_proba(X_te)[:, 1]

    return oof, selected


def permutation_null(X, y, selection, n_permutations=25, seed=0, **kwargs):
    """Estimate where the null of the whole procedure actually sits."""
    rng = np.random.default_rng(seed)
    null = []
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        oof, _ = nested_cv(X, y_perm, seed=seed + 500 + i,
                           selection=selection, **kwargs)
        null.append(roc_auc_score(y_perm, oof))
    return np.asarray(null)


def bootstrap_delta(y, scores_new, scores_base, n_boot=5000, seed=0):
    """Paired bootstrap of the difference in AUC between two sets of scores."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], scores_new[idx])
                      - roc_auc_score(y[idx], scores_base[idx]))
    deltas = np.asarray(deltas)
    observed = roc_auc_score(y, scores_new) - roc_auc_score(y, scores_base)
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {"delta": float(observed),
            "ci": [float(np.percentile(deltas, 2.5)),
                   float(np.percentile(deltas, 97.5))],
            "p": float(min(1.0, max(p, 1 / (len(deltas) + 1))))}


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg adjusted p values, in the order given."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(ranked, 1.0)
    return q
