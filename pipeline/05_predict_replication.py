"""
05_predict_replication.py  (version 2, matched to the actual Challenge template)

The downloaded template has this verified structure:
  columns  ID | Diagnosis | Consent | Responder | Remitter | age | gender | rTMS PROTOCOL
  104 rows, with ID as a plain integer (100, 101, ...) rather than "sub-100"
  the cells to be filled carry the placeholder "REPLICATION"
  age and gender are supplied; BDI_pre is NOT
  rTMS PROTOCOL uses the codes 1, 2, 7, 8 and is missing in 36 of the 104 cases

The consequences honoured here:
  - the model's clinical covariates are age and gender only, taken from the template;
  - the Responder and Remitter columns are filled, not a generic "pred" column;
  - the identifiers have to be normalised between "sub-100" and 100.

Usage:
  python 05_predict_replication.py --target Responder --label modelA
  python 05_predict_replication.py --target Remitter  --label modelA
"""
import argparse
import shutil
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from tdb_common import OUT, SUBMISSION_TEMPLATE, ensure_out, log

LOGF = "05_predict.log"
PLACEHOLDER = "REPLICATION"


def norm_id(x):
    """Normalise 'sub-100', 'sub-0100' and 100 to the same integer key."""
    s = str(x).strip().lower().replace("sub-", "")
    try:
        return int(float(s))
    except ValueError:
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Responder", choices=["Responder", "Remitter"])
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--label", default="modelA")
    args = ap.parse_args()
    ensure_out()

    mpath = OUT / "models" / f"final_model_{args.target}.joblib"
    if not mpath.exists():
        raise SystemExit(f"Model not found: {mpath}\n"
                         f"Run this first: python 04_train_nested.py --target {args.target}")
    bundle = joblib.load(mpath)
    model, cols, prev = bundle["model"], bundle["columns"], bundle["prevalence"]
    log("=" * 70, LOGF)
    log(f"Target {args.target} | {len(cols)} input columns | "
        f"prevalence in the training set {prev:.3f}", LOGF)
    if "BDI_pre" in cols:
        raise SystemExit("This model uses BDI_pre, which the replication set does not supply. "
                         "Retrain without --with-bdi.")

    if not SUBMISSION_TEMPLATE.exists():
        raise SystemExit(f"Template not found: {SUBMISSION_TEMPLATE}")
    tpl = pd.read_excel(SUBMISSION_TEMPLATE)
    log(f"Template: {len(tpl)} rows, columns {list(tpl.columns)}", LOGF)
    tpl["_key"] = tpl["ID"].map(norm_id)

    # ------------------------------------------------------------ EEG features
    feats = pd.read_csv(OUT / "features_replication.csv")
    feats["_key"] = feats["subject"].map(norm_id)
    df = tpl.merge(feats, on="_key", how="left")
    n_missing_eeg = df[[c for c in feats.columns if c.startswith("imcoh_")][:1]].isna().sum().iloc[0] \
        if any(c.startswith("imcoh_") for c in feats.columns) else 0
    log(f"Template rows matched to EEG features: {len(df) - n_missing_eeg}/{len(df)}", LOGF)
    if n_missing_eeg:
        log("WARNING: some rows carry no EEG features. The model will impute them, "
            "but check that the identifiers line up.", LOGF)

    # ------------------------------------------------------------ clinical covariates
    for c in ("age", "gender"):
        if c not in df.columns:
            raise SystemExit(f"Covariate '{c}' missing from the template.")
    log(f"Age in the blinded set: {df.age.mean():.1f} +- {df.age.std():.1f} "
        f"(about 45 in Discovery). A distribution shift of this size has to be reported "
        f"in the paper: the model is being applied to a cohort younger than the one it "
        f"was trained on.", LOGF)

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        log(f"WARNING: {len(missing_cols)} model columns are absent and will be imputed: "
            f"{missing_cols[:8]}", LOGF)
        for c in missing_cols:
            df[c] = np.nan

    X = df[cols].astype(float)
    p = model.predict_proba(X)[:, 1]
    pred = (p >= args.threshold).astype(int)
    log(f"Predictions: mean probability {p.mean():.3f}, "
        f"positive fraction {pred.mean():.3f} (in the training set {prev:.3f})", LOGF)

    out_pred = pd.DataFrame({"ID": tpl["ID"], "pred_proba": p, "pred_binary": pred})
    out_pred.to_csv(OUT / f"predictions_{args.target}_{args.label}.csv", index=False)

    # ------------------------------------------------------------ fill the template
    dest = OUT / f"SUBMISSION_{args.label}_{datetime.now():%Y%m%d}.xlsx"
    if dest.exists():
        book = pd.read_excel(dest)          # finish a submission already started
    else:
        shutil.copy(SUBMISSION_TEMPLATE, dest)
        book = pd.read_excel(dest)
    book[args.target] = pred
    book.to_excel(dest, index=False)

    still = [c for c in ("Responder", "Remitter")
             if book[c].astype(str).eq(PLACEHOLDER).any()]
    log(f"Filled the '{args.target}' column in {dest.name}", LOGF)
    if still:
        log(f"Still to fill: {still}. Re-run with --target {still[0]}.", LOGF)
    else:
        log("Both columns are filled: the file is ready to be sent.", LOGF)
    if book["Diagnosis"].astype(str).eq(PLACEHOLDER).any():
        log("Note: 'Diagnosis' also carries the placeholder. It is not a Treatment Prediction "
            "target, so either leave it as it is or ask "
            "martijn@brainclinics.com to confirm before sending.", LOGF)

    # ------------------------------------------------------------ methodology report
    try:
        import json
        summ = json.load(open(OUT / f"training_summary_{args.target}.json"))
        auc = summ["clinical_plus_eeg"]["auc"]; bss = summ["clinical_plus_eeg"]["bss"]
        sd = summ["auc_stability"]["clinical_plus_eeg"]["sd"]
        nrep = summ.get("repeats", "n"); ntr = summ.get("n", "n")
        top = list(summ.get("selection_frequency", {}))[:6]
    except Exception:
        auc = bss = sd = float("nan"); nrep = ntr = "n"; top = []

    txt = f"""Methodology (one paragraph, for the submission email)

Target: {args.target}. Model development used only the Discovery partition of TDBRAIN
v3.1 (n = {ntr}), restricted to participants with a formal MDD diagnosis, a documented
course of at least ten sessions of DLPFC-targeted rTMS, and pre- and post-treatment
BDI-II. One session per participant was retained. Resting-state EEG in eyes-closed and
eyes-open conditions was cropped by 3 s at each edge to remove the amplifier
disconnection transient, band-pass filtered 1-100 Hz with notch filtering at 50 Hz and
harmonics, average-referenced, and cleaned by extended-Infomax ICA with automatic
component classification (ICLabel via mne-icalabel) supplemented by correlation with the
dedicated EOG and ECG channels; high-amplitude and muscle segments were annotated and
excluded. Features were computed on 2-s epochs excluding annotated segments and spanned
four domains: band power and classical indices (frontal alpha asymmetry, theta/beta
ratio, individual alpha frequency), spectral parameterisation of the aperiodic component
(exponent, offset, hemispheric asymmetry), sensor-space functional connectivity
(imaginary coherence, directed phase lag index, phase locking value, coherence)
summarised globally and at frontal regions of interest, and graph-theoretical metrics
(clustering, efficiency, path length, modularity, minimum spanning tree). Because the
replication template supplies only age and sex, the classifier was an elastic-net
logistic regression adjusted for those two covariates alone; baseline symptom severity
and stimulation protocol were deliberately excluded. Univariate feature selection,
multivariate imputation, standardisation, hyperparameter tuning and probability
calibration were performed strictly within training folds of a repeated nested
stratified cross-validation ({nrep} repeats, 5 outer x 3 inner folds), so no test
observation influenced its own prediction and no feature was chosen by inspecting
full-sample performance. Internal discrimination was AUC = {auc:.3f} (SD across repeats
{sd:.3f}) with a Brier skill score of {bss:+.3f} against an intercept-only model. Most
frequently selected features: {', '.join(top) if top else 'see training summary'}. No
participant from the replication partition was used at any stage. Binary labels were
thresholded at {args.threshold:.2f}.
"""
    (OUT / f"predictions_report_{args.target}_{args.label}.txt").write_text(txt)
    log(f"Methodology paragraph written to "
        f"predictions_report_{args.target}_{args.label}.txt", LOGF)
    log("\nSend to martijn@brainclinics.com: name, affiliation, country, the completed "
        "Excel file, the methodology paragraph and the pilot references.", LOGF)


if __name__ == "__main__":
    main()
