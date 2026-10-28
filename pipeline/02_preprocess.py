"""
02_preprocess.py  (version 2, corrected after the first run on the real data)

Two problems showed up on the real files and are solved here.

PROBLEM 1, the one that made every recording fail.
Channel typing depended on how the BDF reader interprets the channels, which
varies across MNE versions. The 26 cortical channels are now forced to 'eeg' by
name, and every call passes explicit channel lists rather than shortcuts such as
picks="eeg" or ch_type="eeg", which either do not exist or behave differently in
older versions.

PROBLEM 2, far more serious, raised no error but would have ruined the analysis.
Every recording carries a huge transient in its last seconds, typically between
1200 and 3000 microvolts, which is the amplifier disconnection artefact. That
single event dominates the covariance: the PCA preceding the ICA found that 2 to 5
components explained 99.9 per cent of the variance, so the ICA decomposed the
end-of-recording artefact and nothing else, in EVERY subject. Muscle and ocular
components would never have been separated and the features would have been
rubbish, with nothing to flag it.
Checked on five recordings: cropping 3 seconds at each end brings the rank back to
22-24 components and the median amplitude down from anomalous values to 3-8
microvolts.

Further corrections carried over from the previous version: a 1-100 Hz band-pass
before the ICA so that ICLabel can recognise muscle activity, average reference
before the ICA, EOG and ECG channels used for artefact identification, and
annotation of the muscle segments.

Usage:  python 02_preprocess.py [--limit N] [--overwrite] [--cohort ...] [--crop 3.0]
"""
import argparse
import traceback

import mne
import numpy as np
import pandas as pd

from tdb_common import (OUT, ensure_out, log, pick, EEG_CHANNELS, EOG_CHANNELS,
                        ECG_CHANNELS, EMG_CHANNELS, L_FREQ, H_FREQ, LINE_FREQ,
                        ICLABEL_REJECT, ICLABEL_MIN_PROB, SEED)

LOGF = "02_preprocess.log"
mne.set_log_level("ERROR")

CROP_EDGE_SEC = 3.0     # seconds cropped at the start and at the end
MAX_ICA_COMPONENTS = 25
AMP_REJECT_UV = 500.0   # segments above this amplitude are annotated and kept out of the ICA

try:
    from mne_icalabel import label_components
    HAVE_ICLABEL = True
except ImportError:
    HAVE_ICLABEL = False


def preprocess_one(path: str, out_fif, crop_sec: float = CROP_EDGE_SEC) -> dict:
    info_out = {}
    raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)

    # ---------------------------------------------------- 1. channels
    raw.drop_channels([c for c in raw.ch_names if c.lower() in ("status", "trigger")])
    eeg = pick(raw.ch_names, EEG_CHANNELS)
    eog = pick(raw.ch_names, EOG_CHANNELS)
    ecg = pick(raw.ch_names, ECG_CHANNELS)
    emg = pick(raw.ch_names, EMG_CHANNELS)
    if len(eeg) < 20:
        raise RuntimeError(f"only {len(eeg)} EEG channels recognised among {raw.ch_names}")
    mapping = ({c: "eeg" for c in eeg} | {c: "eog" for c in eog}
               | {c: "ecg" for c in ecg} | {c: "emg" for c in emg})
    extra = [c for c in raw.ch_names if c not in mapping]
    mapping |= {c: "misc" for c in extra}
    raw.set_channel_types(mapping)
    info_out["n_eeg"] = len(eeg)
    info_out["n_other"] = len(raw.ch_names) - len(eeg)

    try:
        raw.set_montage("standard_1020", match_case=False, on_missing="ignore")
    except TypeError:
        raw.set_montage("standard_1020", match_case=False)
    except Exception as exc:
        log(f"    montage not applied: {exc}", LOGF)

    # ---------------------------------------------------- 2. edge cropping
    dur = raw.times[-1]
    if crop_sec > 0 and dur > 4 * crop_sec:
        raw.crop(tmin=crop_sec, tmax=dur - crop_sec)
    info_out["duration_s"] = float(raw.times[-1])

    # ---------------------------------------------------- 3. filters
    sfreq = raw.info["sfreq"]
    h = min(H_FREQ, sfreq / 2 - 1)
    raw.filter(L_FREQ, h, picks=raw.ch_names, fir_design="firwin",
               phase="zero", verbose=False)
    notches = [float(f) for f in np.arange(LINE_FREQ, h, LINE_FREQ) if f < h]
    if notches:
        raw.notch_filter(notches, picks=raw.ch_names, verbose=False)

    # ---------------------------------------------------- 4. average reference
    raw.set_eeg_reference("average", projection=False, verbose=False)

    # ---------------------------------------------------- 5. raw EMG proxy
    if emg:
        d = (raw.copy().pick(emg)
             .filter(20, min(95, h - 1), picks="all", verbose=False).get_data())
        info_out["emg_mass_rms"] = float(np.sqrt(np.mean(d ** 2)))

    # ---------------------------------------------------- 6. annotations
    ann = raw.annotations
    musc_s = np.nan
    try:
        a_m, _ = mne.preprocessing.annotate_muscle_zscore(
            raw, ch_type="eeg", threshold=5.0, min_length_good=0.2,
            filter_freq=(60, min(95, h - 1)), verbose=False)
        ann = ann + a_m
        musc_s = float(sum(a_m.duration))
    except Exception as exc:
        log(f"    muscle annotation failed: {exc}", LOGF)
    info_out["muscle_seconds"] = musc_s

    # segments of extreme amplitude: annotated and excluded from the ICA fit
    try:
        d = raw.copy().pick(eeg).get_data() * 1e6
        bad = np.abs(d).max(axis=0) > AMP_REJECT_UV
        if bad.any():
            sf = int(sfreq)
            idx = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
            onsets, durs = [], []
            for s, e in zip(idx[::2], idx[1::2]):
                onsets.append(max(0, s / sf - 0.25))
                durs.append((e - s) / sf + 0.5)
            ann = ann + mne.Annotations(onsets, durs, ["BAD_amp"] * len(onsets),
                                        orig_time=raw.annotations.orig_time)
            info_out["amp_bad_seconds"] = float(np.sum(durs))
    except Exception:
        pass
    raw.set_annotations(ann)

    # ---------------------------------------------------- 7. ICA
    try:
        rank = mne.compute_rank(raw, rank=None, verbose=False).get("eeg", len(eeg))
    except Exception:
        rank = len(eeg) - 1
    n_comp = int(max(10, min(MAX_ICA_COMPONENTS, rank - 1, len(eeg) - 1)))
    info_out["rank"] = int(rank)

    ica = mne.preprocessing.ICA(n_components=n_comp, method="infomax",
                                fit_params=dict(extended=True),
                                random_state=SEED, max_iter="auto")
    ica.fit(raw, picks=eeg, reject_by_annotation=True, verbose=False)
    info_out["n_ica"] = int(ica.n_components_)

    counts, excl = {}, set()
    if HAVE_ICLABEL:
        try:
            lab = label_components(raw, ica, method="iclabel")
            labels = np.asarray(lab["labels"])
            probs = np.asarray(lab["y_pred_proba"]).ravel()
            for i, (lb, pr) in enumerate(zip(labels, probs)):
                counts[lb] = counts.get(lb, 0) + 1
                if lb in ICLABEL_REJECT and pr >= ICLABEL_MIN_PROB:
                    excl.add(int(i))
        except Exception as exc:
            log(f"    ICLabel failed: {exc}", LOGF)

    n_eog_extra = n_ecg_extra = 0
    if eog:
        try:
            idx, _ = ica.find_bads_eog(raw, ch_name=eog, verbose=False)
            n_eog_extra = len(set(idx) - excl); excl |= set(int(i) for i in idx)
        except Exception:
            pass
    if ecg:
        try:
            idx, _ = ica.find_bads_ecg(raw, ch_name=ecg[0], method="correlation", verbose=False)
            n_ecg_extra = len(set(idx) - excl); excl |= set(int(i) for i in idx)
        except Exception:
            pass

    ica.exclude = sorted(excl)
    ica.apply(raw, verbose=False)

    info_out |= {"n_excluded": len(excl), "n_extra_eog": n_eog_extra,
                 "n_extra_ecg": n_ecg_extra,
                 **{f"iclabel_{k.replace(' ', '_')}": v for k, v in counts.items()}}

    out_fif.parent.mkdir(parents=True, exist_ok=True)
    raw.save(out_fif, overwrite=True, verbose=False)
    return info_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--crop", type=float, default=CROP_EDGE_SEC)
    ap.add_argument("--debug", action="store_true", help="print the full traceback")
    ap.add_argument("--cohort", default="both",
                    choices=["discovery", "replication", "both"])
    args = ap.parse_args()

    ensure_out()
    log("=" * 70, LOGF)
    log(f"MNE {mne.__version__} | ICLabel {'present' if HAVE_ICLABEL else 'ABSENT'} | "
        f"edge crop {args.crop}s", LOGF)
    if not HAVE_ICLABEL:
        log("WARNING: run pip install mne-icalabel before going on.", LOGF)

    frames = []
    if args.cohort in ("discovery", "both"):
        frames.append(pd.read_csv(OUT / "cohort_discovery.csv").assign(cohort="discovery"))
    if args.cohort in ("replication", "both"):
        frames.append(pd.read_csv(OUT / "cohort_replication.csv").assign(cohort="replication"))
    todo = pd.concat(frames, ignore_index=True)
    if args.limit:
        todo = todo.head(args.limit)

    log(f"To process: {len(todo)} recordings", LOGF)
    reports, n_err = [], 0
    for i, r in todo.reset_index(drop=True).iterrows():
        tag = f"{r.subject}_ses-{r.session}_{r.task}"
        out_fif = OUT / "derivatives" / r.cohort / f"{tag}-raw.fif"
        if out_fif.exists() and not args.overwrite:
            continue
        log(f"[{i+1}/{len(todo)}] {tag}", LOGF)
        try:
            info = preprocess_one(r.path, out_fif, crop_sec=args.crop)
            info.update({"subject": r.subject, "session": r.session,
                         "task": r.task, "cohort": r.cohort})
            reports.append(info)
            log(f"    EEG {info['n_eeg']}ch rank {info['rank']} | ICA {info['n_ica']} comp | "
                f"removed {info['n_excluded']} (EOG+{info['n_extra_eog']}, "
                f"ECG+{info['n_extra_ecg']}) | muscle {info['muscle_seconds']:.1f}s", LOGF)
        except Exception as exc:
            n_err += 1
            log(f"    ERROR: {exc}", LOGF)
            if args.debug:
                log(traceback.format_exc(), LOGF)

    if reports:
        rep = pd.DataFrame(reports)
        f = OUT / "preprocessing_report.csv"
        if f.exists():
            rep = pd.concat([pd.read_csv(f), rep], ignore_index=True)
        rep.to_csv(f, index=False)
        log(f"\nCompleted {len(reports)}, errors {n_err}. Report in {f}", LOGF)
        if "n_ica" in rep and rep.n_ica.median() < 15:
            log("WARNING: the median ICA component count is below 15. The variance is still "
                "dominated by an artefact: raise --crop.", LOGF)
        if "n_excluded" in rep and rep.n_excluded.mean() < 1:
            log("WARNING: almost no component was removed: check ICLabel.", LOGF)
    else:
        log(f"\nNo recording completed. Errors: {n_err}. "
            f"Re-run with --debug --limit 1 to see the traceback.", LOGF)


if __name__ == "__main__":
    main()
