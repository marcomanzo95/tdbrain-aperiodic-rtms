#!/usr/bin/env python3
"""Test whether the eyes-open aperiodic result depends on measured artefact.

Reproduces Table 3. The eyes-open condition carries roughly twice the myogenic load
of eyes closed, and the aperiodic exponent is the measure most sensitive to
broadband muscle activity, so the single positive finding of the paper has to be
shown not to be an artefact. Each aperiodic feature is residualised, within every
training fold, on the ocular and myogenic proxies, and the model refitted on the
residuals. A model built on the proxies alone is fitted for comparison.

Example:
    python scripts/run_artefact_control.py --analysis-dir analysis_out Responder
"""
import json

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import parser, setup
from src import config
from src.data import artefact_design, design_matrix, load_cohort
from src.evaluation import GRID, bootstrap_delta, nested_cv
from src.taxonomy import condition, family
from src.transformers import Residualiser


def residualised_cv(X, y, n_keep, n_proxy, seed):
    """Nested CV in which the artefact correction is refitted inside every fold."""
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for train, test in outer.split(X, y):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("residualise", Residualiser(n_keep=n_keep, n_proxy=n_proxy)),
            ("rescale", StandardScaler()),
            ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                       max_iter=900, tol=1e-3, random_state=seed))])
        search = GridSearchCV(pipe, GRID, cv=inner, scoring="roc_auc",
                              n_jobs=-1, refit=True).fit(X.iloc[train], y[train])
        calibrated = CalibratedClassifierCV(search.best_estimator_,
                                            method="sigmoid", cv=inner)
        calibrated.fit(X.iloc[train], y[train])
        oof[test] = calibrated.predict_proba(X.iloc[test])[:, 1]
    return oof


def run(target, analysis_dir, results_dir, args):
    frame, y, eeg, artefact = load_cohort(analysis_dir, target)
    aperiodic = [c for c in eeg
                 if family(c) == "aperiodic" and condition(c) == "EO"]
    proxies = [c for c in artefact if "RESTEO" in c]
    print(f"\n=== {target} ===")
    print(f"eyes-open aperiodic features ({len(aperiodic)}): {', '.join(aperiodic)}")
    print(f"artefact proxies ({len(proxies)}): {', '.join(proxies)}")

    # 1. do the proxies predict the outcome on their own?
    X_proxy = design_matrix(frame, proxies)
    proxy_auc = np.mean([roc_auc_score(y, nested_cv(X_proxy, y, seed=config.SEED + r,
                                                    selection="forced")[0])
                         for r in range(args.repeats)])

    # 2. raw aperiodic model
    X_raw = design_matrix(frame, aperiodic)
    oof_raw = np.mean([nested_cv(X_raw, y, seed=config.SEED + r,
                                 selection="forced")[0]
                       for r in range(args.repeats)], axis=0)

    # 3. the same features, with the artefact regressed out inside every fold
    X_adj, n_keep, n_proxy = artefact_design(frame, proxies, aperiodic)
    X_adj = __import__("pandas").DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_adj),
        columns=X_adj.columns)
    oof_adj = np.mean([residualised_cv(X_adj, y, n_keep, n_proxy, config.SEED + r)
                       for r in range(args.repeats)], axis=0)

    rng = np.random.default_rng(config.SEED)
    null = [roc_auc_score(p := rng.permutation(y),
                          residualised_cv(X_adj, p, n_keep, n_proxy,
                                          config.SEED + 700 + i))
            for i in range(args.permutations)]
    null = np.asarray(null)

    auc_raw, auc_adj = roc_auc_score(y, oof_raw), roc_auc_score(y, oof_adj)
    delta = bootstrap_delta(y, oof_adj, oof_raw, seed=config.SEED)
    print(f"\n{'model':34s} {'AUC':>8s} {'null':>8s} {'p':>8s}")
    print("-" * 62)
    print(f"{'artefact proxies alone':34s} {proxy_auc:8.4f} {'':>8s} {'':>8s}")
    print(f"{'aperiodic eyes open, raw':34s} {auc_raw:8.4f} {'':>8s} {'':>8s}")
    print(f"{'aperiodic eyes open, residualised':34s} {auc_adj:8.4f} "
          f"{null.mean():8.4f} {(null >= auc_adj).mean():8.3f}")
    print(f"\nchange on residualisation: {delta['delta']:+.4f} "
          f"[{delta['ci'][0]:+.4f}, {delta['ci'][1]:+.4f}], p = {delta['p']:.3f}")

    payload = {"target": target, "proxy_only_auc": float(proxy_auc),
               "auc_raw": float(auc_raw), "auc_residualised": float(auc_adj),
               "null_mean": float(null.mean()),
               "p_permutation": float((null >= auc_adj).mean()),
               "change_on_residualisation": delta}
    (results_dir / f"artefact_control_{target}.json").write_text(
        json.dumps(payload, indent=1))
    return payload


def main():
    args = parser(__doc__).parse_args()
    analysis_dir, results_dir = setup(args)
    for target in args.targets:
        run(target, analysis_dir, results_dir, args)


if __name__ == "__main__":
    main()
