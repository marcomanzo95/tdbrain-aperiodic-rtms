#!/usr/bin/env python3
"""Fit the locked model and fill in the TDBRAIN Challenge prediction template.

The challenge rules require predictions derived from the 26 scalp EEG channels
alone: no clinical or demographic variable may enter the model, and the EOG and EMG
channels may be used only for artefact correction. This script therefore drops age
and sex, keeps the eyes-open aperiodic features, and removes the artefact component
by within-fold residualisation, which is the use the rules permit.

The decision threshold is chosen on out-of-fold predictions as the value maximising
normalised positive predictive value subject to a sensitivity floor.

Example:
    python scripts/run_challenge_submission.py --analysis-dir analysis_out \
        --template TDBRAIN_Tx_Prediction_replicationV3.xlsx
"""
import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import parser, setup
from src import config
from src.data import load_cohort
from src.evaluation import GRID
from src.taxonomy import condition, family
from src.transformers import Residualiser

SENSITIVITY_FLOOR = 0.40
MIN_FLAGGED = 15


def build_pipeline(n_proxy, seed):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("residualise", Residualiser(n_keep=0, n_proxy=n_proxy)),
        ("rescale", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                   max_iter=1200, tol=1e-3, random_state=seed))])


def normalised_ppv(y, flagged):
    """PPV divided by the base rate: 100 per cent means no gain over prevalence."""
    tp = int(((flagged == 1) & (y == 1)).sum())
    fp = int(((flagged == 1) & (y == 0)).sum())
    if tp + fp == 0:
        return np.nan, np.nan, 0
    ppv = tp / (tp + fp)
    sensitivity = tp / max(int((y == 1).sum()), 1)
    return ppv / y.mean() * 100, sensitivity, tp + fp


def out_of_fold(X, y, n_proxy, repeats, seed):
    runs = []
    for r in range(repeats):
        outer = StratifiedKFold(5, shuffle=True, random_state=seed + r)
        inner = StratifiedKFold(3, shuffle=True, random_state=seed + r)
        oof = np.zeros(len(y))
        for train, test in outer.split(X, y):
            search = GridSearchCV(build_pipeline(n_proxy, seed + r), GRID, cv=inner,
                                  scoring="roc_auc", n_jobs=-1,
                                  refit=True).fit(X.iloc[train], y[train])
            calibrated = CalibratedClassifierCV(search.best_estimator_,
                                                method="sigmoid", cv=inner)
            calibrated.fit(X.iloc[train], y[train])
            oof[test] = calibrated.predict_proba(X.iloc[test])[:, 1]
        runs.append(oof)
    return np.mean(runs, axis=0)


def choose_threshold(y, oof):
    best = None
    for t in np.quantile(oof, np.linspace(0.05, 0.95, 91)):
        nppv, sensitivity, flagged = normalised_ppv(y, (oof >= t).astype(int))
        if (np.isfinite(nppv) and sensitivity >= SENSITIVITY_FLOOR
                and flagged >= MIN_FLAGGED and (best is None or nppv > best[1])):
            best = (float(t), float(nppv), float(sensitivity), int(flagged))
    if best is None:
        raise SystemExit("No threshold satisfies the sensitivity floor.")
    return best


def run(target, analysis_dir, results_dir, args, template):
    frame, y, eeg, artefact = load_cohort(analysis_dir, target)
    aperiodic = [c for c in eeg if family(c) == "aperiodic" and condition(c) == "EO"]
    proxies = [c for c in artefact if "RESTEO" in c]
    columns = proxies + aperiodic          # proxies first: Residualiser consumes them
    X = frame[columns].astype(float)

    print(f"\n=== {target} ===")
    print(f"n = {len(y)}, positives {y.sum()} ({y.mean():.3f}), "
          f"EEG features {len(aperiodic)}, proxies {len(proxies)}, "
          f"clinical covariates 0 (excluded by the challenge rules)")

    oof = out_of_fold(X, y, len(proxies), args.repeats, config.SEED)
    print(f"out-of-fold AUC, EEG only: {roc_auc_score(y, oof):.4f}")
    threshold, nppv, sensitivity, flagged = choose_threshold(y, oof)
    print(f"threshold {threshold:.4f} -> nPPV {nppv:.1f}%, "
          f"sensitivity {sensitivity:.3f}, {flagged}/{len(y)} flagged")

    inner = StratifiedKFold(3, shuffle=True, random_state=config.SEED)
    search = GridSearchCV(build_pipeline(len(proxies), config.SEED), GRID, cv=inner,
                          scoring="roc_auc", n_jobs=-1, refit=True).fit(X, y)
    final = CalibratedClassifierCV(search.best_estimator_, method="sigmoid",
                                   cv=inner).fit(X, y)

    replication = pd.read_csv(analysis_dir / "features_replication.csv")
    replication = (replication.sort_values(["subject", "session"])
                              .drop_duplicates("subject", keep="first").copy())
    replication["ID"] = (replication["subject"].astype(str)
                         .str.replace("sub-", "", regex=False).astype(int))
    aligned = replication.set_index("ID").reindex(template["ID"].to_numpy())
    missing = int(aligned[columns].isna().all(axis=1).sum())
    if missing:
        print(f"WARNING: {missing} replication rows have no features")
    predictions = (final.predict_proba(aligned[columns].astype(float))[:, 1]
                   >= threshold).astype(int)
    print(f"replication: {len(aligned)} rows aligned, "
          f"{predictions.sum()} predicted positive")
    return predictions, {"auc_out_of_fold": float(roc_auc_score(y, oof)),
                         "threshold": threshold, "nppv": nppv,
                         "sensitivity": sensitivity,
                         "predicted_positive": int(predictions.sum()),
                         "best_params": {k: float(v)
                                         for k, v in search.best_params_.items()}}


def main():
    p = parser(__doc__)
    p.add_argument("--template", required=True,
                   help="TDBRAIN_Tx_Prediction_replicationV3.xlsx")
    args = p.parse_args()
    analysis_dir, results_dir = setup(args)
    template = pd.read_excel(args.template)
    report = {}
    for target in args.targets:
        predictions, report[target] = run(target, analysis_dir, results_dir,
                                          args, template)
        template[target] = predictions
    out = results_dir / "TDBRAIN_Tx_Prediction_replication_filled.xlsx"
    template.to_excel(out, index=False)
    (results_dir / "submission_report.json").write_text(json.dumps(report, indent=1))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
