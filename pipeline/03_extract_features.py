"""
03_extract_features.py

Extracts the feature matrix from every -raw.fif file produced by script 02.
Domains: spectral, aperiodic (FOOOF), connectivity (iMCOH, dPLI, PLV, COH),
network topology, plus the EMG proxies the artefact controls rely on.

Every parameter used is recorded in features_params.json, so that the
specification requests raised in review can be answered directly.

Usage:  python 03_extract_features.py [--cohort discovery|replication|both]
"""
import argparse
import itertools

import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert

from tdb_common import (OUT, ensure_out, log, save_json, pick, BANDS, EMG_BANDS,
                        ANALYSIS_H_FREQ, ROI_LEFT, ROI_RIGHT, ROI_MIDLINE,
                        ROI_LATERAL, EMG_CHANNELS, EOG_CHANNELS,
                        GRAPH_DENSITY, GRAPH_BINARISE, SEED)

LOGF = "03_features.log"
mne.set_log_level("ERROR")

FOOOF_PARAMS = dict(freq_range=(2.0, 40.0), aperiodic_mode="knee",
                    peak_width_limits=(1.0, 8.0), max_n_peaks=6,
                    min_peak_height=0.10, peak_threshold=2.0)
EPOCH_SEC = 2.0


# ------------------------------------------------------------------ epochs
def make_epochs(raw, duration=None):
    """Fixed-length epochs that exclude the segments annotated BAD_.

    This serves two purposes: it keeps contaminated segments out of the spectral
    and connectivity estimates, and it yields windows of uniform length, which
    both Welch's method and the Hilbert phase estimate assume.
    """
    dur = duration or EPOCH_SEC
    ep = mne.make_fixed_length_epochs(raw, duration=dur, preload=True,
                                      reject_by_annotation=True, verbose=False)
    if len(ep) < 10:   # too few: fall back to the whole recording
        ep = mne.make_fixed_length_epochs(raw, duration=dur, preload=True,
                                          reject_by_annotation=False, verbose=False)
    return ep


# ------------------------------------------------------------------ spectrum
def band_power(psd, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return np.trapezoid(psd[:, m], freqs[m], axis=1)


def spectral_features(raw, epochs=None):
    ep = epochs if epochs is not None else make_epochs(raw)
    sf = ep.info["sfreq"]
    n_per_seg = int(min(sf * EPOCH_SEC, ep.get_data().shape[-1]))
    sp = ep.compute_psd(method="welch", fmin=1, fmax=min(100, sf / 2 - 1),
                        n_fft=n_per_seg, n_per_seg=n_per_seg,
                        n_overlap=n_per_seg // 2, verbose=False)
    psd, freqs = sp.get_data(return_freqs=True)
    psd = psd.mean(axis=0)          # averaged over epochs -> (n_channels, n_freq)
    ch = ep.ch_names
    out = {}
    for b, (lo, hi) in {**BANDS, **EMG_BANDS}.items():
        bp = band_power(psd, freqs, lo, hi)
        out[f"pw_{b}_global"] = float(np.mean(bp))
        for side, roi in (("l", ROI_LEFT), ("r", ROI_RIGHT)):
            idx = [ch.index(c) for c in pick(ch, roi)]
            if idx:
                out[f"pw_{b}_front_{side}"] = float(np.mean(bp[idx]))

    # FAA = ln(alpha F4) - ln(alpha F3)
    a = band_power(psd, freqs, *BANDS["alpha"])
    f3, f4 = pick(ch, ["F3"]), pick(ch, ["F4"])
    out["FAA"] = (float(np.log(a[ch.index(f4[0])]) - np.log(a[ch.index(f3[0])]))
                  if f3 and f4 else np.nan)

    # TBR midline
    mid = [ch.index(c) for c in pick(ch, ROI_MIDLINE)]
    if mid:
        th = band_power(psd, freqs, *BANDS["theta"])[mid].mean()
        be = band_power(psd, freqs, *BANDS["beta"])[mid].mean()
        out["TBR"] = float(th / be) if be > 0 else np.nan

    # IAF: 7-13 Hz centre of gravity over the posterior channels
    post = [i for i, c in enumerate(ch) if c.upper().startswith(("O", "P"))]
    if post:
        m = (freqs >= 7) & (freqs <= 13)
        p = psd[post][:, m].mean(axis=0)
        out["IAF"] = float(np.sum(freqs[m] * p) / np.sum(p))

    # proxies of muscle contamination, three independent versions
    lat = [ch.index(c) for c in pick(ch, ROI_LATERAL)]
    if lat:
        e = band_power(psd, freqs, *EMG_BANDS["emg_high"])[lat].mean()
        al = a[lat].mean()
        out["emg_ratio_lateral"] = float(e / al) if al > 0 else np.nan
    front = [ch.index(c) for c in pick(ch, ROI_LEFT + ROI_RIGHT)]
    if front:
        out["emg_high_front"] = float(band_power(psd, freqs, *EMG_BANDS["emg_high"])[front].mean())
    return out


def peripheral_features(raw):
    """Direct proxies from dedicated channels: masseter EMG and ocular amplitude.

    These are the strongest answer to the objection that gamma is myogenic in
    origin: they are not derived from the EEG itself, so they allow a clean
    adjustment.
    """
    out = {}
    emg = pick(raw.ch_names, EMG_CHANNELS)
    if emg:
        d = raw.copy().pick(emg).filter(20, 95, picks="all", verbose=False).get_data() * 1e6
        out["emg_mass_rms"] = float(np.sqrt(np.mean(d ** 2)))          # microvolt
        out["emg_mass_log"] = float(np.log(out["emg_mass_rms"] + 1e-9))
    eog = pick(raw.ch_names, EOG_CHANNELS)
    if eog:
        d = raw.copy().pick(eog).filter(1, 15, picks="all", verbose=False).get_data() * 1e6
        out["eog_rms"] = float(np.sqrt(np.mean(d ** 2)))               # microvolt
        # proxy of saccadic activity: 20-40 Hz power over the horizontal channels
        dh = raw.copy().pick([c for c in eog if c.startswith("H")] or eog)
        dh = dh.filter(20, 40, picks="all", verbose=False).get_data() * 1e6
        out["eog_saccade_proxy"] = float(np.sqrt(np.mean(dh ** 2)))    # microvolt
    return out


# ------------------------------------------------------------------ FOOOF
def _fit_specparam(freqs, psd):
    """Fit the spectral model, compatible with specparam 1.x/2.x and with fooof."""
    try:
        from specparam import SpectralGroupModel as SGM
        new_api = True
    except ImportError:
        try:
            from fooof import FOOOFGroup as SGM
            new_api = False
        except ImportError:
            return None, None, None, None
    fg = SGM(peak_width_limits=FOOOF_PARAMS["peak_width_limits"],
             max_n_peaks=FOOOF_PARAMS["max_n_peaks"],
             min_peak_height=FOOOF_PARAMS["min_peak_height"],
             peak_threshold=FOOOF_PARAMS["peak_threshold"],
             aperiodic_mode=FOOOF_PARAMS["aperiodic_mode"], verbose=False)
    fg.fit(freqs, psd, FOOOF_PARAMS["freq_range"])

    ap = r2 = err = None
    for key in ("aperiodic", "aperiodic_params"):
        try:
            ap = np.atleast_2d(fg.get_params(key)); break
        except Exception:
            continue
    if ap is None:
        return None, None, None, None
    # goodness-of-fit metrics: both the name and the location vary across versions
    try:
        met = fg.results.get_metrics() if hasattr(fg.results, "get_metrics") else None
        if isinstance(met, dict):
            r2 = np.mean([v for k, v in met.items() if "r_squared" in k or k == "gof"])
            err = np.mean([v for k, v in met.items() if "error" in k])
    except Exception:
        pass
    if r2 is None:
        for key in ("r_squared", "gof"):
            try:
                r2 = float(np.nanmean(fg.get_params(key))); break
            except Exception:
                continue
    if err is None:
        try:
            err = float(np.nanmean(fg.get_params("error")))
        except Exception:
            pass
    # columns: offset first, exponent last, with or without a knee
    return ap[:, 0], ap[:, -1], r2, err


def aperiodic_features(raw, epochs=None):
    ep = epochs if epochs is not None else make_epochs(raw)
    sf = ep.info["sfreq"]
    n_per_seg = int(min(sf * EPOCH_SEC, ep.get_data().shape[-1]))
    sp = ep.compute_psd(method="welch", fmin=1, fmax=45, n_fft=n_per_seg,
                        n_per_seg=n_per_seg, n_overlap=n_per_seg // 2, verbose=False)
    psd, freqs = sp.get_data(return_freqs=True)
    psd = psd.mean(axis=0)
    off, expo, r2, err = _fit_specparam(freqs, psd)
    if expo is None:
        return {}
    ch = ep.ch_names
    out = {"Exponent": float(np.nanmean(expo)), "Offset": float(np.nanmean(off))}
    if r2 is not None:
        out["R_squared"] = float(r2)
    if err is not None:
        out["Error"] = float(err)
    li = [i for i, c in enumerate(ch) if c[-1].isdigit() and int(c[-1]) % 2 == 1]
    ri = [i for i, c in enumerate(ch) if c[-1].isdigit() and int(c[-1]) % 2 == 0]
    if li and ri:
        out["Left_exponent"] = float(np.nanmean(expo[li]))
        out["Right_exponent"] = float(np.nanmean(expo[ri]))
        out["Exponent_asymmetry"] = out["Left_exponent"] - out["Right_exponent"]
    return out


# ------------------------------------------------------------------ connettivita'
def connectivity_matrices(epochs, lo, hi):
    """iMCOH, dPLI, PLV and coherence within a band, computed per epoch and averaged.

    Working epoch by epoch instead of concatenating removes the artificial
    discontinuities at the join points, which a phase estimate treats as
    broadband events.
    """
    ef = epochs.copy().filter(lo, hi, fir_design="firwin", picks="all", verbose=False)
    X = ef.get_data()                       # (n_epochs, n_channels, n_times)
    n_ep, n_ch, n_t = X.shape
    Z = hilbert(X, axis=-1)
    U = Z / (np.abs(Z) + 1e-30)             # unit-modulus signal

    imcoh = np.zeros((n_ch, n_ch)); coh = np.zeros((n_ch, n_ch))
    plv = np.zeros((n_ch, n_ch));   dpli = np.zeros((n_ch, n_ch))
    for e in range(n_ep):
        z, u = Z[e], U[e]
        S = (z @ z.conj().T) / n_t                     # cross spectrum
        d = np.sqrt(np.real(np.diag(S)))
        den = np.outer(d, d) + 1e-30
        C = S / den
        imcoh += np.abs(np.imag(C)); coh += np.abs(C)
        P = (u @ u.conj().T) / n_t
        plv += np.abs(P)
        for i in range(n_ch):                          # fraction of time during which i leads j
            dpli[i] += np.mean(np.imag(u[i] * np.conj(u)) > 0, axis=-1)
    imcoh /= n_ep; coh /= n_ep; plv /= n_ep; dpli /= n_ep
    for M in (imcoh, coh, plv):
        np.fill_diagonal(M, 0.0)
    np.fill_diagonal(dpli, 0.5)
    return {"imcoh": imcoh, "dpli": dpli, "plv": plv, "coh": coh}


def graph_metrics(mat, density=GRAPH_DENSITY, binarise=GRAPH_BINARISE):
    """Clustering, path length, efficiency, modularity and MST on a symmetric matrix."""
    import networkx as nx
    a = np.abs(np.array(mat, dtype=float)); np.fill_diagonal(a, 0)
    n = a.shape[0]
    k = int(round(density * n * (n - 1) / 2))
    iu = np.triu_indices(n, 1)
    vals = a[iu]
    if k < 1 or k >= len(vals):
        return {}
    thr = np.sort(vals)[-k]
    m = np.where(a >= thr, a, 0.0)
    if binarise:
        m = (m > 0).astype(float)
    G = nx.from_numpy_array(m)
    out = {"C_global": float(nx.average_clustering(G, weight=None if binarise else "weight"))}
    dist = {(u, v): (1.0 / d["weight"] if d["weight"] > 0 else np.inf)
            for u, v, d in G.edges(data=True)}
    nx.set_edge_attributes(G, dist, "distance")
    try:
        out["E_global"] = float(nx.global_efficiency(G))
    except Exception:
        pass
    if nx.is_connected(G):
        out["L_char"] = float(nx.average_shortest_path_length(G, weight="distance"))
    try:
        comm = nx.algorithms.community.louvain_communities(G, seed=SEED, weight="weight")
        out["Q_modularity"] = float(nx.algorithms.community.modularity(G, comm, weight="weight"))
    except Exception:
        pass
    # MST on the UNthresholded matrix, turned into distances, following Stam's method
    full = np.abs(np.array(mat, dtype=float)); np.fill_diagonal(full, 0)
    with np.errstate(divide="ignore"):
        dmat = np.where(full > 0, 1.0 / full, np.inf)
    Gf = nx.from_numpy_array(np.where(np.isfinite(dmat), dmat, 0))
    try:
        T = nx.minimum_spanning_tree(Gf)
        deg = np.array([d for _, d in T.degree()])
        out["MST_leaf_fraction"] = float((deg == 1).sum() / len(deg))
        out["MST_diameter"] = float(nx.diameter(T) / (len(deg) - 1))
    except Exception:
        pass
    return out


def connectivity_features(epochs):
    ch = epochs.ch_names
    L = [ch.index(c) for c in pick(ch, ROI_LEFT)]
    R = [ch.index(c) for c in pick(ch, ROI_RIGHT)]
    out = {}
    for band, (lo, hi) in BANDS.items():
        mats = connectivity_matrices(epochs, lo, hi)
        for name, M in mats.items():
            iu = np.triu_indices(M.shape[0], 1)
            out[f"{name}_{band}_global"] = float(np.mean(np.abs(M[iu])))
            if L:
                out[f"{name}_{band}_front_l"] = float(np.mean(np.abs(M[np.ix_(L, L)])))
            if R:
                out[f"{name}_{band}_front_r"] = float(np.mean(np.abs(M[np.ix_(R, R)])))
            if L and R:
                out[f"{name}_{band}_lr"] = float(np.mean(M[np.ix_(L, R)]))
        for src in ("imcoh", "plv"):
            for gk, gv in graph_metrics(mats[src]).items():
                out[f"gt_{src}_{band}_{gk}"] = gv
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="both", choices=["discovery", "replication", "both"])
    args = ap.parse_args()
    ensure_out()
    save_json({"bands": BANDS, "emg_bands": EMG_BANDS, "fooof": FOOOF_PARAMS,
               "epoch_sec": EPOCH_SEC, "graph_density": GRAPH_DENSITY,
               "graph_binarise": GRAPH_BINARISE, "roi_left": ROI_LEFT,
               "roi_right": ROI_RIGHT, "roi_midline": ROI_MIDLINE,
               "roi_lateral": ROI_LATERAL, "analysis_h_freq": ANALYSIS_H_FREQ,
               "epoching": "fixed-length 2 s epochs, segments annotated BAD_ excluded",
               "connectivity": ("Hilbert analytic signal per epoch, averaged across "
                                "epochs; dPLI = fraction of samples with positive "
                                "imaginary part of the cross-product"),
               "peripheral_channels": {"emg": "Mass (masseter)", "eog": "VPVA VNVB HPHL HNHR",
                                       "ecg": "Erbs"},
               "note": ("recordings cropped by 3 s at each edge in 02_preprocess to remove "
                        "the amplifier disconnection transient, which otherwise dominates "
                        "the covariance and collapses the ICA rank to 2-5 components")},
              OUT / "features_params.json")

    cohorts = ["discovery", "replication"] if args.cohort == "both" else [args.cohort]
    for coh in cohorts:
        files = sorted((OUT / "derivatives" / coh).glob("*-raw.fif"))
        log(f"\n{coh}: {len(files)} file", LOGF)
        rows = []
        for i, f in enumerate(files):
            sub, ses, task = f.stem.replace("-raw", "").rsplit("_", 2)
            log(f"[{i+1}/{len(files)}] {f.stem}", LOGF)
            try:
                raw = mne.io.read_raw_fif(f, preload=True, verbose=False)
                raw_eeg = raw.copy().pick("eeg")
                ep_eeg = make_epochs(raw_eeg)
                feats = spectral_features(raw_eeg, ep_eeg)
                feats.update(peripheral_features(raw))
                feats.update(aperiodic_features(raw_eeg, ep_eeg))
                ep45 = ep_eeg.copy().filter(None, ANALYSIS_H_FREQ, picks="all", verbose=False)
                feats.update(connectivity_features(ep45))
                feats["n_epochs_clean"] = len(ep_eeg)
                feats.update({"subject": sub, "session": int(ses.split("-")[1]),
                              "task": task})
                rows.append(feats)
            except Exception as exc:
                log(f"    ERROR: {exc}", LOGF)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        wide = df.pivot_table(index=["subject", "session"],
                              columns="task", values=[c for c in df.columns
                                                      if c not in ("subject", "session", "task")])
        wide.columns = [f"{a}_{b.replace('task-', '').upper()}" for a, b in wide.columns]
        wide = wide.reset_index()
        wide.to_csv(OUT / f"features_{coh}.csv", index=False)
        log(f"Wrote {OUT / f'features_{coh}.csv'}  shape {wide.shape}", LOGF)


if __name__ == "__main__":
    main()
