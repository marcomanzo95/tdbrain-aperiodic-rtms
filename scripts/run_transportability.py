#!/usr/bin/env python3
"""Leave-one-protocol-out validation: does the association become a predictor?

The model is trained on two of the three rTMS protocols and tested on the third,
rotating across all three. This is the question a deployed model actually faces,
and in this dataset neither the EEG model nor the clinical model survives it.

Example:
    python scripts/run_transportability.py --analysis-dir analysis_out Responder
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import parser, participants_file, setup
from src import config
from src.data import continuous_outcome, load_cohort
from src.taxonomy import condition, family

PROTOCOL = "rTMS PROTOCOL"
GRID = {"model__alpha": [0.1, 0.3, 1, 3, 10], "model__l1_ratio": [0.2, 0.5, 0.8]}


def held_out_predictions(frame, columns, y, protocol):
    """Train on every protocol but one, predict the one left out, pool the lot."""
    X = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(frame[columns].astype(float)),
        columns=columns)
    predicted = np.zeros(len(y))
    per_protocol = {}
    for held in np.unique(protocol):
        train, test = protocol != held, protocol == held
        search = GridSearchCV(
            Pipeline([("scale", StandardScaler()),
                      ("model", ElasticNet(max_iter=5000,
                                           random_state=config.SEED))]),
            GRID, cv=KFold(5, shuffle=True, random_state=config.SEED),
            scoring="r2", n_jobs=-1).fit(X[train], y[train])
        predicted[test] = search.predict(X[test])
        rho, p = stats.spearmanr(predicted[test], y[test])
        per_protocol[int(held)] = {"n": int(test.sum()), "rho": float(rho),
                                   "p": float(p)}
    return predicted, per_protocol


def bootstrap_rho(y, predicted, n_boot=4000, seed=config.SEED):
    rng = np.random.default_rng(seed)
    draws = [stats.spearmanr(predicted[i], y[i])[0]
             for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return (float(stats.spearmanr(predicted, y)[0]),
            float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def run(target, analysis_dir, results_dir, args):
    frame, _, eeg, artefact = load_cohort(analysis_dir, target)
    y = continuous_outcome(frame, participants_file(args, analysis_dir))
    keep = np.isfinite(y) & frame[PROTOCOL].notna().to_numpy()
    frame, y = frame[keep].reset_index(drop=True), y[keep]
    protocol = frame[PROTOCOL].astype(int).to_numpy()
    aperiodic = [c for c in eeg if family(c) == "aperiodic" and condition(c) == "EO"]
    proxies = [c for c in artefact if "RESTEO" in c]

    print(f"\n=== {target} ===")
    print(f"n = {len(y)}, protocols {dict(zip(*np.unique(protocol, return_counts=True)))}")
    table = pd.crosstab(frame[PROTOCOL], frame[target])
    chi2, p_chi, dof, _ = stats.chi2_contingency(table)
    groups = [y[protocol == k] for k in np.unique(protocol)]
    print(f"protocol vs outcome: chi-square = {chi2:.2f}, df = {dof}, p = {p_chi:.4f}")
    print(f"protocol vs improvement: Kruskal-Wallis H = {stats.kruskal(*groups)[0]:.2f}, "
          f"p = {stats.kruskal(*groups)[1]:.4f}")

    out = {"target": target, "n": int(len(y)),
           "protocol_vs_outcome": {"chi2": float(chi2), "df": int(dof),
                                   "p": float(p_chi)},
           "models": {}}
    pooled = {}
    for label, columns in (("clinical only", config.CLINICAL),
                           ("clinical + aperiodic EO", config.CLINICAL + aperiodic),
                           ("clinical + proxies + aperiodic EO",
                            config.CLINICAL + proxies + aperiodic)):
        predicted, per_protocol = held_out_predictions(frame, columns, y, protocol)
        rho, lo, hi = bootstrap_rho(y, predicted)
        pooled[label] = predicted
        out["models"][label] = {"pooled_rho": rho, "ci": [lo, hi],
                                "per_protocol": per_protocol}
        print(f"\n  {label}")
        for k, v in per_protocol.items():
            print(f"    held-out protocol {k} (n = {v['n']:3d}): "
                  f"rho = {v['rho']:+.3f}, p = {v['p']:.4f}")
        print(f"    pooled: rho = {rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    rng = np.random.default_rng(config.SEED)
    a = pooled["clinical + proxies + aperiodic EO"]
    b = pooled["clinical only"]
    draws = [stats.spearmanr(a[i], y[i])[0] - stats.spearmanr(b[i], y[i])[0]
             for i in (rng.integers(0, len(y), len(y)) for _ in range(4000))]
    draws = np.asarray(draws)
    observed = stats.spearmanr(a, y)[0] - stats.spearmanr(b, y)[0]
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    out["eeg_minus_clinical"] = {"delta": float(observed),
                                 "ci": [float(np.percentile(draws, 2.5)),
                                        float(np.percentile(draws, 97.5))],
                                 "p": float(max(p, 1 / (len(draws) + 1)))}
    print(f"\n  EEG minus clinical: {observed:+.3f} "
          f"[{np.percentile(draws, 2.5):+.3f}, {np.percentile(draws, 97.5):+.3f}], "
          f"p = {max(p, 1 / (len(draws) + 1)):.4f}")
    (results_dir / f"transportability_{target}.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    args = parser(__doc__).parse_args()
    analysis_dir, results_dir = setup(args)
    for target in args.targets:
        run(target, analysis_dir, results_dir, args)


if __name__ == "__main__":
    main()
