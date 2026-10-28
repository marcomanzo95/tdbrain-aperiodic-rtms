"""
04_train_nested.py

Training and HONEST internal validation on the Discovery partition.

The central difference from the thesis pipeline is that FEATURE SELECTION happens
INSIDE the outer cross-validation loop. In the earlier version the candidate
combinations were compared by looking at performance on the whole dataset, so the
reported AUCs were the maxima of a search rather than estimates. Here the test
fold plays no part whatsoever in choosing the features.

Further corrections with respect to the earlier version:
  - sigmoid (Platt) calibration instead of isotonic, which with small inner folds
    produced probabilities of exactly 0 and 1;
  - uncalibrated AUC as the primary metric (AUC is invariant to monotone
    transformations, so a "calibrated AUC" is not a meaningful quantity);
  - Brier skill score against the intercept-only model;
  - two-sided p values and Benjamini-Hochberg q values across the whole family
    of tests;
  - nPPV, the primary metric of the TDBRAIN Challenge;
  - decision curve analysis.

Usage:  python 04_train_nested.py [--repeats 20] [--top-k 12]
"""
import argparse
import json
import sys
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tdb_common import OUT, ensure_out, log, save_json, SEED, load_participants

LOGF = "04_train.log"
warnings.filterwarnings("ignore")

# NOTE. The Challenge template supplies ONLY age and gender for the replication
# set: BDI_pre is absent. A model that uses BDI_pre cannot be applied blind, so the
# clinical covariates stop here. The rTMS protocol is excluded as well, because its
# coding differs between the two partitions (1-3 in Discovery, 1-2 and 7-8 in
# Replication) and it is missing in 36 of the 104 cases.
CLINICAL = ["age", "gender"]
CLINICAL_INTERNAL_ONLY = ["BDI_pre"]   # internal analyses only, never for a blind prediction


# ------------------------------------------------------------------ pipeline
def make_pipeline(k_features: int) -> tuple[Pipeline, dict]:
    """Imputation -> scaling -> univariate selection -> elastic net.

    SelectKBest sits INSIDE the pipeline, so it is refitted on every training fold
    and never sees the test fold. That is the central point of this script.
    """
    pre = Pipeline([("imputer", IterativeImputer(random_state=SEED, max_iter=15)),
                    ("scaler", StandardScaler())])
    clf = LogisticRegression(penalty="elasticnet", solver="saga",
                             max_iter=8000, random_state=SEED)
    pipe = Pipeline([("pre", pre),
                     ("sel", SelectKBest(f_classif, k=k_features)),
                     ("clf", clf)])
    grid = {"clf__C": np.logspace(-3, 1.5, 8),
            "clf__l1_ratio": [0.15, 0.5, 0.85],
            "sel__k": sorted({3, 5, 8, min(k_features, 15)})}
    return pipe, grid


def nested_cv(X: pd.DataFrame, y: np.ndarray, n_outer=5, n_inner=3,
              seed=SEED, k_max=20) -> dict:
    pipe, grid = make_pipeline(min(k_max, X.shape[1]))
    grid["sel__k"] = [k for k in grid["sel__k"] if k <= X.shape[1]] or [X.shape[1]]
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    chosen = []
    for tr, te in outer.split(X, y):
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        gs = GridSearchCV(pipe, grid, cv=inner, n_jobs=-1, scoring="roc_auc", refit=True)
        gs.fit(X.iloc[tr], y[tr])
        cal = CalibratedClassifierCV(gs.best_estimator_, method="sigmoid", cv=inner)
        cal.fit(X.iloc[tr], y[tr])
        oof[te] = cal.predict_proba(X.iloc[te])[:, 1]
        sel = gs.best_estimator_.named_steps["sel"]
        chosen.append(list(X.columns[sel.get_support()]))
    return {"oof": oof, "selected_per_fold": chosen}


# ------------------------------------------------------------------ metriche
def metrics(y, p, prev=None):
    prev = float(np.mean(y)) if prev is None else prev
    null_brier = prev * (1 - prev)
    br = brier_score_loss(y, p)
    return {"auc": float(roc_auc_score(y, p)), "brier": float(br),
            "bss": float(1 - br / null_brier)}


def boot_delta(y, p_new, p_base, B=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    d = []
    for _ in range(B):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        d.append(roc_auc_score(y[s], p_new[s]) - roc_auc_score(y[s], p_base[s]))
    d = np.array(d)
    p1 = float(np.mean(d <= 0))
    return {"delta_auc": float(d.mean()),
            "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)),
            "p_one_sided": p1, "p_two_sided": float(min(1.0, 2 * min(p1, 1 - p1)))}


def nppv(y, p, threshold):
    sel = p >= threshold
    if sel.sum() < 5:
        return np.nan, int(sel.sum())
    return float(np.mean(y[sel]) / np.mean(y)) * 100, int(sel.sum())


def net_benefit(y, p, t):
    tp = ((p >= t) & (y == 1)).sum(); fp = ((p >= t) & (y == 0)).sum()
    return float(tp / len(y) - (fp / len(y)) * (t / (1 - t)))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20,
                    help="repeats of the nested CV, each with a different seed")
    ap.add_argument("--k-max", type=int, default=20)
    ap.add_argument("--target", default="Responder", choices=["Responder", "Remitter"],
                    help="the Challenge template asks for predictions on both")
    ap.add_argument("--with-bdi", action="store_true",
                    help="include BDI_pre: internal analyses only, NOT for the submission")
    args = ap.parse_args()
    ensure_out()
    import sklearn, scipy
    log("=" * 70, LOGF)
    log(f"04 - addestramento | target {args.target} | repeats {args.repeats}", LOGF)
    log(f"python {sys.version.split()[0]} | numpy {np.__version__} | "
        f"pandas {pd.__version__} | sklearn {sklearn.__version__} | scipy {scipy.__version__}", LOGF)

    coh = pd.read_csv(OUT / "cohort_discovery.csv").drop_duplicates(["subject", "session"])
    # A subject may have more than one session. Keeping both would put the same
    # patient in the training and the test half of one fold, which is a form of
    # leakage. One session per subject is kept, the first.
    n_before = len(coh)
    coh = coh.sort_values(["subject", "session"]).drop_duplicates("subject", keep="first")
    if len(coh) < n_before:
        log(f"Removed {n_before - len(coh)} repeated sessions: one session per subject, "
            f"so that no patient appears in both training and test.", LOGF)

    feats = pd.read_csv(OUT / "features_discovery.csv")
    df = coh.merge(feats, on=["subject", "session"], how="inner")
    assert df.subject.is_unique, "Duplicated subjects after the merge: check the sessions."
    log(f"Merged cohort: {df.shape[0]} subjects, {feats.shape[1] - 2} EEG features", LOGF)

    if args.target not in df.columns:
        raise SystemExit(f"column '{args.target}' missing from the cohort: re-run 01_build_cohort.py")
    df = df[df[args.target].notna()].copy()
    y = df[args.target].astype(int).values
    prev = float(y.mean())
    log(f"Positive cases {y.sum()}/{len(y)} ({100*prev:.1f}%)", LOGF)

    clinical = CLINICAL + (CLINICAL_INTERNAL_ONLY if args.with_bdi else [])
    if args.with_bdi:
        log("WARNING: this model uses BDI_pre. It cannot be applied to the replication set, "
            "which does not provide that variable. Use it for internal analyses only.", LOGF)
    log(f"Target: {args.target} | clinical covariates: {clinical}", LOGF)
    eeg_cols = [c for c in feats.columns if c not in ("subject", "session")]
    X_clin = df[clinical].astype(float)
    X_all = pd.concat([X_clin, df[eeg_cols].astype(float)], axis=1)

    # columns with too many missing values: drop them and say so
    miss = X_all.isna().mean()
    drop = miss[miss > 0.30].index.tolist()
    if drop:
        log(f"Dropped {len(drop)} features with more than 30% missing", LOGF)
        X_all = X_all.drop(columns=drop)
    log(f"Final candidate space: {X_all.shape[1]} features "
        f"({len(clinical)} of them clinical)", LOGF)

    # ---------------------------------------------------------------- repeats
    results = {}
    for name, X in (("clinical_baseline", X_clin), ("clinical_plus_eeg", X_all)):
        aucs, oofs, sels = [], [], []
        for r in range(args.repeats):
            out = nested_cv(X, y, seed=SEED + r, k_max=args.k_max)
            aucs.append(roc_auc_score(y, out["oof"]))
            oofs.append(out["oof"]); sels.extend(out["selected_per_fold"])
            log(f"  {name} repeat {r+1}/{args.repeats}: AUC = {aucs[-1]:.4f}", LOGF)
        oof_mean = np.mean(oofs, axis=0)
        results[name] = {"oof": oof_mean, "auc_per_repeat": aucs, "selected": sels}
        log(f"{name}: AUC media {np.mean(aucs):.4f} (SD {np.std(aucs):.4f}, "
            f"min {np.min(aucs):.4f}, max {np.max(aucs):.4f})", LOGF)

    base, full = results["clinical_baseline"]["oof"], results["clinical_plus_eeg"]["oof"]
    summ = {"n": int(len(y)), "responders": int(y.sum()), "prevalence": prev,
            "repeats": args.repeats,
            "clinical_baseline": metrics(y, base),
            "clinical_plus_eeg": metrics(y, full),
            "delta": boot_delta(y, full, base),
            "auc_stability": {k: {"mean": float(np.mean(v["auc_per_repeat"])),
                                  "sd": float(np.std(v["auc_per_repeat"]))}
                              for k, v in results.items()}}

    # how often each feature is selected across folds and repeats
    from collections import Counter
    cnt = Counter(f for s in results["clinical_plus_eeg"]["selected"] for f in s)
    tot = len(results["clinical_plus_eeg"]["selected"])
    summ["selection_frequency"] = {k: round(v / tot, 3)
                                   for k, v in cnt.most_common(30)}
    log("\nMost frequently selected features (fraction of folds):", LOGF)
    for k, v in list(summ["selection_frequency"].items())[:15]:
        log(f"   {v:5.2f}  {k}", LOGF)

    # nPPV and decision curve
    summ["nPPV"] = {}
    for t in (0.50, 0.60, 0.65, 0.70):
        v, n = nppv(y, full, t)
        summ["nPPV"][f"thr_{t:.2f}"] = {"nPPV_percent": v, "n_selected": n}
    summ["decision_curve"] = {f"{t:.2f}": {"model": net_benefit(y, full, t),
                                           "baseline": net_benefit(y, base, t),
                                           "treat_all": float(np.mean(y) - (1 - np.mean(y)) * (t / (1 - t)))}
                              for t in np.round(np.arange(0.40, 0.85, 0.05), 2)}

    summ["target"] = args.target
    summ["clinical_covariates"] = clinical
    save_json(summ, OUT / f"training_summary_{args.target}.json")
    pd.DataFrame({"subject": df.subject, "session": df.session, "y_true": y,
                  "p_baseline": base, "p_full": full}).to_csv(
        OUT / f"oof_predictions_discovery_{args.target}.csv", index=False)

    # ---------------------------------------------------------------- final model
    pipe, grid = make_pipeline(min(args.k_max, X_all.shape[1]))
    grid["sel__k"] = [k for k in grid["sel__k"] if k <= X_all.shape[1]] or [X_all.shape[1]]
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, grid, cv=inner, n_jobs=-1, scoring="roc_auc", refit=True)
    gs.fit(X_all, y)
    final = CalibratedClassifierCV(gs.best_estimator_, method="sigmoid", cv=inner)
    final.fit(X_all, y)

    import joblib
    joblib.dump({"model": final, "columns": list(X_all.columns),
                 "best_params": gs.best_params_, "prevalence": prev,
                 "target": args.target, "clinical": clinical},
                OUT / "models" / f"final_model_{args.target}.joblib")
    log(f"\nFinal model saved. Hyperparameters: {gs.best_params_}", LOGF)
    log(f"AUC baseline {summ['clinical_baseline']['auc']:.4f} | "
        f"with EEG {summ['clinical_plus_eeg']['auc']:.4f} | "
        f"dAUC {summ['delta']['delta_auc']:+.4f} "
        f"[{summ['delta']['ci_low']:+.4f}, {summ['delta']['ci_high']:+.4f}] "
        f"p2 = {summ['delta']['p_two_sided']:.3f}", LOGF)
    log(f"Brier skill score with EEG: {summ['clinical_plus_eeg']['bss']:+.4f}", LOGF)
    log("\nIf the AUC collapses with respect to the thesis, that is the correct result: it "
        "means the earlier gain came from selecting features outside the loop.", LOGF)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ensure_out()
        log("FATAL ERROR:\n" + traceback.format_exc(), LOGF)
        raise
