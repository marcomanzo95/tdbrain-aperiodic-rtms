"""
tdb_common.py
Shared configuration and helpers for the TDBRAIN Challenge pipeline.

The data location is read from the TDBRAIN_ROOT environment variable and must
contain the public TDBRAIN V3.1 release laid out as follows:

    $TDBRAIN_ROOT/
        TDBRAIN_Dataset_V3_1/                    Discovery recordings
        TDBRAIN_Tx_Response_ReplicationV3_1/     Replication recordings
        TDBRAIN_participants_V3.xlsx             clinical datasheet
        TDBRAIN_Tx_Prediction_replicationV3.xlsx submission template

Verified recording layout: sub-XXXX/ses-N/eeg/sub-XXXX_ses-N_task-restEC_eeg.bdf
Verified montage: 26 EEG channels plus 4 EOG, 1 ECG and 1 EMG (masseter),
sampled at 500 Hz, original linked-mastoid reference (A1A2), 50 Hz line noise.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================================
# CONFIG
# =====================================================================

# Root of the public TDBRAIN release. Set TDBRAIN_ROOT to point at it; the
# default assumes the data sit in ./tdbrain_data next to the repository.
ROOT = Path(os.environ.get("TDBRAIN_ROOT", "./tdbrain_data")).expanduser()

DISCOVERY_DIR = ROOT / "TDBRAIN_Dataset_V3_1"
REPLICATION_DIR = ROOT / "TDBRAIN_Tx_Response_ReplicationV3_1"
PARTICIPANTS_XLSX = ROOT / "TDBRAIN_participants_V3.xlsx"

# To be downloaded from:
# https://brainclinics.com/images/TDBRAINV3/TDBRAIN_Tx_Prediction_replicationV3.xlsx
SUBMISSION_TEMPLATE = ROOT / "TDBRAIN_Tx_Prediction_replicationV3.xlsx"

# Destination of every derivative. TDBRAIN_ANALYSIS_DIR is the same variable the
# analysis code in src/ reads, so both halves of the project share one tree.
OUT = Path(os.environ.get("TDBRAIN_ANALYSIS_DIR",
                          str(ROOT / "analysis_out"))).expanduser()

# --------------------------------------------------------------- montage
# The 26 EEG channels actually present in the dataset (read from channels.tsv)
EEG_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                "FC3", "FCz", "FC4", "T7", "C3", "Cz", "C4", "T8",
                "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8",
                "O1", "Oz", "O2"]

# Non-EEG channels present in the files. They must be typed correctly, otherwise
# they enter the ICA and the average reference and distort everything.
EOG_CHANNELS = ["VPVA", "VNVB", "HPHL", "HNHR"]   # vertical and horizontal
ECG_CHANNELS = ["Erbs"]
EMG_CHANNELS = ["Mass"]                            # masseter: direct EMG proxy

CHANNEL_TYPES = ({c: "eog" for c in EOG_CHANNELS}
                 | {c: "ecg" for c in ECG_CHANNELS}
                 | {c: "emg" for c in EMG_CHANNELS}
                 | {c: "misc" for c in ("Status", "STATUS", "Trigger")})

# --------------------------------------------------------------- filters
L_FREQ = 1.0
H_FREQ = 100.0          # wide, so that ICLabel can recognise muscle activity
LINE_FREQ = 50.0
ANALYSIS_H_FREQ = 45.0  # band for the classical features, applied AFTER ICA

ICLABEL_REJECT = ("muscle artifact", "eye blink", "heart beat",
                  "line noise", "channel noise")
ICLABEL_MIN_PROB = 0.80
N_ICA_COMPONENTS = 0.999

# --------------------------------------------------------------- bands
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 12.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
EMG_BANDS = {"emg_low": (55.0, 70.0), "emg_high": (70.0, 95.0)}

# --------------------------------------------------------------- ROIs
# Checked against channels.tsv: all of them exist.
ROI_LEFT = ["F3", "F7", "FC3"]
ROI_RIGHT = ["F4", "F8", "FC4"]
ROI_MIDLINE = ["FCz", "Cz"]
ROI_LATERAL = ["F7", "F8", "T7", "T8"]   # sites most exposed to cranial EMG

# --------------------------------------------------------------- graphs
GRAPH_DENSITY = 0.20
GRAPH_BINARISE = False

SEED = 42

# =====================================================================
# HELPERS
# =====================================================================

BIDS_RE = re.compile(r"(sub-[A-Za-z0-9]+)_ses-(\d+)_task-(restEC|restEO)", re.I)


def ensure_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("derivatives", "features", "models", "figures", "logs"):
        (OUT / sub).mkdir(exist_ok=True)
    return OUT


def find_recordings(root: Path) -> pd.DataFrame:
    """Find the resting-state .bdf files and read subject, session and task
    from the file name."""
    rows = []
    for f in sorted(Path(root).rglob("*_eeg.bdf")):
        m = BIDS_RE.search(f.name)
        if not m:
            continue  # this also drops the task-oddball files automatically
        rows.append({"path": str(f), "subject": m.group(1),
                     "session": int(m.group(2)), "task": m.group(3).upper()})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No restEC/restEO file found under {root}.")
    return df


def load_participants() -> pd.DataFrame:
    if not PARTICIPANTS_XLSX.exists():
        raise SystemExit(f"Datasheet not found: {PARTICIPANTS_XLSX}")
    d = pd.read_excel(PARTICIPANTS_XLSX)
    d.columns = [c.strip() for c in d.columns]
    return d


def log(msg: str, logfile: str | None = None) -> None:
    print(msg, flush=True)
    if logfile:
        p = OUT / "logs" / logfile
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as fh:
            fh.write(msg + "\n")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def pick(ch_names, wanted):
    """Those channels of `wanted` present in `ch_names`, matched
    case-insensitively."""
    low = {c.lower(): c for c in ch_names}
    return [low[w.lower()] for w in wanted if w.lower() in low]


def set_types(raw):
    """Type the EOG, ECG, EMG and misc channels and leave only the 26 cortical
    channels as EEG."""
    mapping = {c: t for c, t in CHANNEL_TYPES.items() if c in raw.ch_names}
    if mapping:
        raw.set_channel_types(mapping)
    unknown = [c for c in raw.ch_names
               if c not in CHANNEL_TYPES and c not in EEG_CHANNELS]
    if unknown:
        log(f"    unrecognised channels, left as EEG: {unknown}")
    return raw
