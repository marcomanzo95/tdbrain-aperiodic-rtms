"""
01_build_cohort.py

Builds two lists:
  - cohort_discovery.csv   -> MDD-rTMS patients of the DISCOVERY partition, outcome known.
                              These are the training cases.
  - cohort_replication.csv -> recordings of the Treatment Prediction set, outcome blinded.
                              These are the cases to predict.

It also checks that the two lists do not overlap, which is the condition that
makes a Challenge submission valid.

Usage:  python 01_build_cohort.py
"""
import pandas as pd

from tdb_common import (DISCOVERY_DIR, REPLICATION_DIR, OUT, ensure_out,
                        find_recordings, load_participants, log, SEED)

LOGF = "01_build_cohort.log"


def main():
    ensure_out()
    log("=" * 70, LOGF)
    log("01 - Building the cohorts", LOGF)
    log("=" * 70, LOGF)

    # ---------------------------------------------------------------- datasheet
    part = load_participants()
    log(f"Datasheet: {part.shape[0]} rows, {part.shape[1]} columns", LOGF)

    needed = ["TDBRAIN_ID", "DISC/REP", "Dataset", "Responder", "Remitter", "age",
              "gender", "BDI_pre", "sessID"]
    missing = [c for c in needed if c not in part.columns]
    if missing:
        log(f"WARNING: columns missing from the datasheet: {missing}", LOGF)

    # ---------------------------------------------------------------- discovery
    disc = part[(part.get("Dataset") == "MDD-rTMS")].copy()
    if "DISC/REP" in disc.columns:
        disc = disc[disc["DISC/REP"].astype(str).str.upper().str.startswith("DISC")]
    disc = disc[disc["Responder"].notna()].copy()
    disc["Responder"] = disc["Responder"].astype(int)
    log(f"MDD-rTMS DISCOVERY with outcome: {disc.TDBRAIN_ID.nunique()} subjects, "
        f"{len(disc)} sessions", LOGF)
    log(f"  responders {int(disc.drop_duplicates('TDBRAIN_ID').Responder.sum())} / "
        f"{disc.TDBRAIN_ID.nunique()}", LOGF)

    # ---------------------------------------------- recordings found on disk
    rec_d = find_recordings(DISCOVERY_DIR)
    log(f".bdf recordings found in DISCOVERY: {len(rec_d)} "
        f"({rec_d.subject.nunique()} subjects)", LOGF)
    log(f"  tasks present: {rec_d.task.value_counts().to_dict()}", LOGF)

    # keep only the subjects that have BOTH restEC AND restEO
    both = (rec_d.groupby(["subject", "session"])["task"]
            .apply(lambda s: {"RESTEC", "RESTEO"}.issubset(set(s.str.upper())))
            .rename("has_both").reset_index())
    rec_d = rec_d.merge(both, on=["subject", "session"])
    rec_d = rec_d[rec_d.has_both]

    keep = ["TDBRAIN_ID", "sessID", "Responder", "age", "gender", "BDI_pre"]
    for extra in ("Remitter", "rTMS PROTOCOL"):
        if extra in disc.columns:
            keep.append(extra)
    coh = rec_d.merge(disc[keep],
                      left_on=["subject", "session"],
                      right_on=["TDBRAIN_ID", "sessID"], how="inner")
    coh = coh.drop(columns=["TDBRAIN_ID", "sessID", "has_both"])
    log(f"\nFinal DISCOVERY cohort: {coh.subject.nunique()} subjects "
        f"with EC+EO and a known outcome", LOGF)

    # ---------------------------------------------------------------- replication
    rec_r = find_recordings(REPLICATION_DIR)
    both_r = (rec_r.groupby(["subject", "session"])["task"]
              .apply(lambda s: {"RESTEC", "RESTEO"}.issubset(set(s.str.upper())))
              .rename("has_both").reset_index())
    rec_r = rec_r.merge(both_r, on=["subject", "session"])
    rec_r = rec_r[rec_r.has_both].drop(columns=["has_both"])
    log(f"REPLICATION cohort: {rec_r.subject.nunique()} subjects with EC+EO", LOGF)

    # ---------------------------------------------------------------- sanity check
    overlap = set(coh.subject) & set(rec_r.subject)
    if overlap:
        log(f"\n*** ERROR: {len(overlap)} subjects appear in BOTH partitions. "
            f"Remove them from the training set before going on: {sorted(overlap)[:10]}", LOGF)
    else:
        log("\nOK: no overlap between training and replication. "
            "The submission will be valid.", LOGF)

    coh.to_csv(OUT / "cohort_discovery.csv", index=False)
    rec_r.to_csv(OUT / "cohort_replication.csv", index=False)
    log(f"\nWritten:\n  {OUT / 'cohort_discovery.csv'}\n  {OUT / 'cohort_replication.csv'}", LOGF)


if __name__ == "__main__":
    main()
