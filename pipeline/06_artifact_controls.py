"""
06_artifact_controls.py

The six controls a reviewer asks for in order to decide whether the gamma
signal is neural or myogenic. Run it after script 03.

  1. EMG proxy by condition: 70-95 Hz power at the lateral sites, EC versus EO.
  2. Association between the EMG proxy and the gamma connectivity features.
  3. Does the gamma feature survive adjustment for the EMG proxy?
  4. Frequency profile: a myogenic effect grows monotonically towards the top.
  5. Variance explained by the components labelled as muscle, split by outcome.
  6. Absolute dPLI against the null hypothesis of 0.5.

Usage:  python 06_artifact_controls.py
"""
import numpy as np
import pandas as pd
from scipy import stats

from tdb_common import OUT, ensure_out, log

LOGF = "06_artifacts.log"


def main():
    ensure_out()
    coh = pd.read_csv(OUT / "cohort_discovery.csv").drop_duplicates(["subject", "session"])
    f = pd.read_csv(OUT / "features_discovery.csv")
    df = coh.merge(f, on=["subject", "session"])
    y = df["Responder"].astype(int).values
    log("=" * 70, LOGF); log("ARTEFACT CONTROLS", LOGF); log("=" * 70, LOGF)

    # ------------------------------------------------------------ 1
    log("\n[1] Peripheral proxies by condition (masseter EMG, spectral EMG, saccades)", LOGF)
    for side in ("emg_mass_log", "emg_ratio_lateral", "eog_saccade_proxy"):
        ec, eo = f"{side}_RESTEC", f"{side}_RESTEO"
        if ec in df and eo in df:
            a, b = df[ec].dropna(), df[eo].dropna()
            t = stats.ttest_rel(df[ec], df[eo], nan_policy="omit")
            log(f"    EC {a.mean():.3f} vs EO {b.mean():.3f}  t = {t.statistic:.2f}, "
                f"p = {t.pvalue:.4f}", LOGF)
            log("    If EO clearly exceeds EC, the predictive advantage of gamma in EO "
                "is compatible with an artefactual gradient.", LOGF)

    # ------------------------------------------------------------ 2 and 3
    gamma_cols = [c for c in df.columns if "gamma" in c and c.endswith("RESTEO")]
    # primary proxy: the masseter EMG channel. Fallback: the spectral ratio.
    emg = next((c for c in ("emg_mass_log_RESTEO", "emg_mass_rms_RESTEO",
                            "emg_ratio_lateral_RESTEO") if c in df), None)
    if emg and gamma_cols:
        log(f"\n[2] Correlation between the EMG proxy and {len(gamma_cols)} gamma features in EO", LOGF)
        rows = []
        for c in gamma_cols:
            ok = df[[c, emg]].dropna()
            if len(ok) < 20:
                continue
            r, p = stats.spearmanr(ok[c], ok[emg])
            rows.append({"feature": c, "rho_emg": r, "p": p})
        cor = pd.DataFrame(rows).sort_values("rho_emg", key=abs, ascending=False)
        for _, r in cor.head(10).iterrows():
            log(f"    {r.feature:44s} rho = {r.rho_emg:+.3f}  p = {r.p:.4f}", LOGF)
        cor.to_csv(OUT / "artifact_emg_correlations.csv", index=False)

        log("\n[3] Does the gamma feature still predict outcome once EMG is accounted for?", LOGF)
        import statsmodels.api as sm
        target = next((c for c in gamma_cols if "dpli" in c and "lr" in c), gamma_cols[0])
        sub = df[[target, emg]].copy(); sub["y"] = y
        sub = sub.dropna()
        z = (sub[[target, emg]] - sub[[target, emg]].mean()) / sub[[target, emg]].std()
        m1 = sm.Logit(sub.y, sm.add_constant(z[[target]])).fit(disp=0)
        m2 = sm.Logit(sub.y, sm.add_constant(z)).fit(disp=0)
        log(f"    {target}", LOGF)
        log(f"    on its own:         beta = {m1.params[target]:+.4f}  p = {m1.pvalues[target]:.4f}", LOGF)
        log(f"    adjusted for EMG:   beta = {m2.params[target]:+.4f}  p = {m2.pvalues[target]:.4f}", LOGF)
        log("    If the coefficient halves or loses significance, the myogenic account "
            "is the more parsimonious one and the paper should be framed as negative.", LOGF)

    # ------------------------------------------------------------ 4
    log("\n[4] Frequency profile of frontal power", LOGF)
    pw = {b: f"pw_{b}_front_r_RESTEO" for b in ("beta", "gamma", "emg_low", "emg_high")}
    have = {b: c for b, c in pw.items() if c in df}
    if len(have) >= 3:
        for b, c in have.items():
            r = stats.pointbiserialr(y, df[c].fillna(df[c].median()))
            log(f"    {b:9s} r with outcome = {r.statistic:+.3f}  p = {r.pvalue:.4f}", LOGF)
        log("    A neural effect should be band specific. If the relation grows "
            "monotonically towards the top, it is muscle.", LOGF)

    # ------------------------------------------------------------ 5
    prep = OUT / "preprocessing_report.csv"
    if prep.exists():
        log("\n[5] Muscle components removed, split by outcome", LOGF)
        rep = pd.read_csv(prep)
        rep = rep[rep.cohort == "discovery"]
        mm = rep.merge(coh[["subject", "session", "Responder"]], on=["subject", "session"])
        for task in mm.task.unique():
            s = mm[mm.task == task]
            if "muscle_seconds" in s and s.muscle_seconds.notna().sum() > 20:
                a = s[s.Responder == 1].muscle_seconds.dropna()
                b = s[s.Responder == 0].muscle_seconds.dropna()
                t = stats.ttest_ind(a, b, equal_var=False)
                log(f"    {task}: responder {a.mean():.1f}s vs non-responder {b.mean():.1f}s "
                    f"p = {t.pvalue:.3f}", LOGF)
        log("    A difference here would mean the biomarker measures muscle tension "
            "rather than network organisation.", LOGF)

    # ------------------------------------------------------------ 6
    log("\n[6] Absolute dPLI against the null of 0.5", LOGF)
    for c in [c for c in df.columns if c.startswith("dpli_") and "_lr" in c]:
        v = df[c].dropna()
        if len(v) < 20:
            continue
        t = stats.ttest_1samp(v, 0.5)
        grp = [df.loc[y == k, c].dropna().mean() for k in (0, 1)]
        log(f"    {c:40s} mean {v.mean():.4f}  vs 0.5: p = {t.pvalue:.4f}   "
            f"(NR {grp[0]:.4f} / R {grp[1]:.4f})", LOGF)
    log("\n    If both groups sit on the same side of 0.5, the claim that "
        "'responders show left-to-right phase leadership' is false as written "
        "and has to be rephrased.", LOGF)

    log(f"\nFull log in {OUT / 'logs' / LOGF}", LOGF)


if __name__ == "__main__":
    main()
