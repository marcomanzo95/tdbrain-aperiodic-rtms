#!/usr/bin/env python3
"""Associations with the continuous outcome, the primary analysis of the paper.

Dichotomising a continuous outcome discards information and costs power, so the
percentage change in the Beck Depression Inventory is used here rather than the
50 per cent response flag. Reproduces Table 2 and the out-of-sample correlations.

Example:
    python scripts/run_continuous_outcome.py --analysis-dir analysis_out Responder
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import parser, participants_file, setup
from src import config
from src.data import continuous_outcome, load_cohort
from src.evaluation import benjamini_hochberg
from src.taxonomy import condition, family
from src.transformers import Residualiser

CONTINUOUS_GRID = {"model__alpha": [0.1, 0.3, 1, 3, 10],
                   "model__l1_ratio": [0.2, 0.5, 0.8]}


def partial_spearman(x, y, covariates):
    """Spearman correlation after linearly removing the covariates from both."""
    Z = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(covariates),
                     columns=covariates.columns)
    rx = x - LinearRegression().fit(Z, x).predict(Z)
    ry = y - LinearRegression().fit(Z, y).predict(Z)
    return stats.spearmanr(rx, ry)


def cross_validated_rho(frame, columns, y, residualise, n_keep, n_proxy,
                        repeats, seed):
    """Out-of-sample correlation between predicted and observed improvement."""
    X = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(frame[columns].astype(float)),
        columns=columns)
    rhos = []
    for r in range(repeats):
        folds = KFold(5, shuffle=True, random_state=seed + r)
        predicted = np.zeros(len(y))
        for train, test in folds.split(X):
            steps = [("scale", StandardScaler())]
            if residualise:
                steps += [("residualise", Residualiser(n_keep=n_keep,
                                                       n_proxy=n_proxy)),
                          ("rescale", StandardScaler())]
            steps.append(("model", ElasticNet(max_iter=5000, random_state=seed)))
            search = GridSearchCV(Pipeline(steps), CONTINUOUS_GRID,
                                  cv=KFold(3, shuffle=True, random_state=seed),
                                  scoring="r2", n_jobs=-1).fit(X.iloc[train], y[train])
            predicted[test] = search.predict(X.iloc[test])
        rhos.append(stats.spearmanr(predicted, y)[0])
    return float(np.mean(rhos)), float(np.std(rhos))


def run(target, analysis_dir, results_dir, args):
    frame, _, eeg, artefact = load_cohort(analysis_dir, target)
    y = continuous_outcome(frame, participants_file(args, analysis_dir))
    usable = np.isfinite(y)
    frame, y = frame[usable].reset_index(drop=True), y[usable]
    aperiodic = [c for c in eeg if family(c) == "aperiodic" and condition(c) == "EO"]
    proxies = [c for c in artefact if "RESTEO" in c]

    print(f"\n=== {target} ===")
    print(f"n with a continuous outcome: {len(y)}  "
          f"mean improvement {y.mean():.1f}%  SD {y.std():.1f}%")
    print(f"\n{'measure':32s} {'rho':>8s} {'p':>9s} {'rho adj':>9s} {'p adj':>9s} {'q':>7s}")
    print("-" * 80)

    rows, p_adjusted = [], []
    for col in aperiodic:
        v = frame[col].astype(float).to_numpy()
        ok = np.isfinite(v)
        rho, p = stats.spearmanr(v[ok], y[ok])
        rho_a, p_a = partial_spearman(v[ok], y[ok],
                                      frame.loc[ok, config.CLINICAL + proxies].astype(float))
        rows.append([col, float(rho), float(p), float(rho_a), float(p_a)])
        p_adjusted.append(p_a)
    for row, q in zip(rows, benjamini_hochberg(p_adjusted)):
        row.append(float(q))
        print(f"{row[0]:32s} {row[1]:+8.3f} {row[2]:9.4f} "
              f"{row[3]:+9.3f} {row[4]:9.4f} {q:7.3f}")

    print()
    models = {}
    for label, cols, resid in (
            ("clinical only", config.CLINICAL, False),
            ("clinical + aperiodic EO", config.CLINICAL + aperiodic, False),
            ("clinical + aperiodic EO, corrected",
             config.CLINICAL + proxies + aperiodic, True)):
        rho, sd = cross_validated_rho(frame, cols, y, resid,
                                      len(config.CLINICAL), len(proxies),
                                      args.repeats, config.SEED)
        models[label] = {"rho": rho, "sd": sd}
        print(f"{label:38s} out-of-sample rho = {rho:+.3f} (SD {sd:.3f})")

    payload = {"target": target, "n": int(len(y)),
               "mean_improvement": float(y.mean()),
               "associations": [dict(zip(
                   ["measure", "rho", "p", "rho_adjusted", "p_adjusted", "q_bh"], r))
                   for r in rows],
               "cross_validated": models}
    (results_dir / f"continuous_outcome_{target}.json").write_text(
        json.dumps(payload, indent=1))
    return payload


def main():
    args = parser(__doc__).parse_args()
    analysis_dir, results_dir = setup(args)
    for target in args.targets:
        run(target, analysis_dir, results_dir, args)


if __name__ == "__main__":
    main()
