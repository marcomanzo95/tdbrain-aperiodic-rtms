"""Cohort assembly and design matrices.

One session per subject is retained. Keeping both would place the same patient on
either side of a cross-validation fold, which inflates every estimate downstream.
"""
import numpy as np
import pandas as pd

from .config import CLINICAL
from .taxonomy import split_columns

MISSINGNESS_LIMIT = 0.30


def load_cohort(analysis_dir, target, partition="discovery"):
    """Return (frame, outcome, eeg_columns, artefact_columns) for one target."""
    cohort = pd.read_csv(analysis_dir / f"cohort_{partition}.csv")
    cohort = cohort.drop_duplicates(["subject", "session"])
    cohort = (cohort.sort_values(["subject", "session"])
                    .drop_duplicates("subject", keep="first"))

    features = pd.read_csv(analysis_dir / f"features_{partition}.csv")
    frame = cohort.merge(features, on=["subject", "session"], how="inner")
    if not frame.subject.is_unique:
        raise ValueError("Duplicate subjects after merge: check the session filter.")

    if target not in frame.columns:
        raise SystemExit(f"Column '{target}' missing from the cohort file.")
    frame = frame[frame[target].notna()].copy().reset_index(drop=True)
    y = frame[target].astype(int).to_numpy()

    raw = [c for c in features.columns if c not in ("subject", "session")]
    eeg, artefact = split_columns(raw)
    return frame, y, eeg, artefact


def design_matrix(frame, feature_columns, clinical=None):
    """Clinical covariates first, then the EEG features, minus sparse columns."""
    clinical = CLINICAL if clinical is None else clinical
    X = pd.concat([frame[clinical].astype(float),
                   frame[list(feature_columns)].astype(float)], axis=1)
    sparse = X.columns[X.isna().mean() > MISSINGNESS_LIMIT]
    return X.drop(columns=list(sparse))


def artefact_design(frame, proxy_columns, feature_columns, clinical=None):
    """Matrix laid out as [clinical | proxies | features] for the Residualiser."""
    clinical = CLINICAL if clinical is None else clinical
    cols = list(clinical) + list(proxy_columns) + list(feature_columns)
    return frame[cols].astype(float), len(clinical), len(proxy_columns)


def continuous_outcome(frame, participants_file):
    """Percentage change in BDI, the primary outcome for association analyses."""
    table = pd.read_excel(participants_file)
    table.columns = [str(c).strip() for c in table.columns]
    table["_sid"] = table["TDBRAIN_ID"].astype(str).str.strip()
    table = table.drop_duplicates("_sid")
    merged = frame.assign(_sid=frame["subject"].astype(str)).merge(
        table[["_sid", "BDI_pre", "BDI_post"]], on="_sid",
        how="left", suffixes=("", "_ref"))
    pre = merged["BDI_pre_ref"].astype(float).to_numpy()
    post = merged["BDI_post"].astype(float).to_numpy()
    return (pre - post) / pre * 100.0
