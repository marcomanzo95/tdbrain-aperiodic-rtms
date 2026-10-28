# Aperiodic EEG and rTMS outcome in major depressive disorder

Analysis code for Manzo, D'Urso and Santarnecchi, *A flatter eyes-open aperiodic
exponent is associated with rTMS outcome in major depressive disorder but does not
transport across treatment protocols*.

The study asks whether a multi-domain resting EEG feature set adds value over
clinical covariates for predicting response to repetitive transcranial magnetic
stimulation, whether any gain survives correction for multiplicity and explicit
artefact control, and whether it transports to patients treated with a protocol the
model has not seen. The short answers are: only the eyes-open aperiodic family
carries signal, that signal is not myogenic, and it does not transport.

## Why this repository exists

Three habits in the EEG biomarker literature make published gains hard to read, and
this code is organised around avoiding them.

**A clinical benchmark is always fitted alongside.** Every EEG model is compared
with a model built from age and sex. Without that comparison the quantity that
matters, incremental value, cannot be recovered from a reported AUC.

**The null is estimated, not assumed.** Pooling out-of-fold calibrated probabilities
across folds does not place the null of the procedure at 0.50. Here it sits near
0.49. `src/evaluation.py` estimates it by permuting the outcome and rerunning the
entire pipeline, and every comparison is referred to that empirical distribution.

**Artefact proxies are predictors of nothing.** Ocular, myogenic and data-quality
measures are excluded from the feature space by `src/taxonomy.py` and used only to
test whether a result survives their removal.

## Layout

```
src/          library code: feature taxonomy, transformers, nested CV and inference
scripts/      one runnable analysis per file, each reproducing a section of the paper
pipeline/     the upstream steps, from raw BIDS recordings to a feature table
gui/          a Tkinter application for interactive single-subject and group analysis
```

`src/transformers.py` holds the two design decisions worth naming. `ForcedKBest`
keeps the clinical covariates in the model rather than letting them compete with
hundreds of EEG features inside univariate selection. `Residualiser` removes, within
each training fold, the component of every EEG feature that is linearly predictable
from the artefact proxies.

## Reproducing the analysis

The data are not redistributed here. Obtain the TDBRAIN database from the
Brainclinics Foundation under its data use agreement, then:

```bash
pip install -e ".[pipeline]"

export TDBRAIN_ROOT=/path/to/tdbrain            # the release, laid out as below
export TDBRAIN_ANALYSIS_DIR=$TDBRAIN_ROOT/analysis_out

python pipeline/01_build_cohort.py
python pipeline/02_preprocess.py       --cohort both
python pipeline/03_extract_features.py --cohort both
python scripts/run_family_models.py    Responder Remitter
python scripts/run_artefact_control.py
python scripts/run_continuous_outcome.py
python scripts/run_transportability.py
python scripts/run_challenge_submission.py
```

A fixed random seed is used throughout, so the published numbers are reproducible
exactly.

## Configuration

Nothing is hardcoded to one machine. Four environment variables govern every path,
and each has a working default, so the code runs unmodified anywhere.

| Variable | Meaning | Default |
|---|---|---|
| `TDBRAIN_ROOT` | the TDBRAIN release | `./tdbrain_data` |
| `TDBRAIN_ANALYSIS_DIR` | derivatives and feature tables | `$TDBRAIN_ROOT/analysis_out` |
| `TDBRAIN_RESULTS_DIR` | tables and JSON written by `scripts/` | `./results` |
| `TDBRAIN_SCRATCH_DIR` | large temporary files written by the GUI | the system temp directory |

`TDBRAIN_ROOT` is expected to contain the release as Brainclinics ships it:

```
$TDBRAIN_ROOT/
    TDBRAIN_Dataset_V3_1/                     Discovery recordings
    TDBRAIN_Tx_Response_ReplicationV3_1/      Replication recordings
    TDBRAIN_participants_V3.xlsx              clinical datasheet
    TDBRAIN_Tx_Prediction_replicationV3.xlsx  challenge submission template
```

Preprocessing writes several gigabytes of intermediate `.fif` files. On a laptop
with little free space, point `TDBRAIN_SCRATCH_DIR` at an external drive.

## The GUI

`gui/` contains an interactive application built for this work: loading of BIDS
recordings, automated preprocessing with ICLabel component classification and visual
review, spectral parameterisation, connectivity and graph-theoretical analysis, and
nested cross-validation with calibrated probabilities. It is the tool the analyses
were developed in, and it is included so that the modelling choices in the paper can
be inspected and varied interactively rather than only read about.

The methodological corrections described in the paper are already applied: feature
selection sits inside the cross-validation loop, calibration is Platt rather than
isotonic, the bootstrap of the difference in AUC is two-sided, and the Brier skill
score, Benjamini-Hochberg correction and normalised positive predictive value are
available as utilities.

```bash
pip install -e ".[gui]"          # add ".[gui,extras]" for XGBoost, SHAP and bctpy
cd gui && python eeg_analysis_gui.py
```

XGBoost, SHAP, bctpy and the macOS Dock icon are each guarded at import time, so the
application starts without them and simply hides the feature concerned. The graph
module carries its own smoke test and unit tests:

```bash
cd gui && python graph_metrics.py        # smoke test, about 30 s
cd gui && pytest tests/ -q               # small-worldness, efficiency, NBS
```

## Data availability

TDBRAIN V3.1 is available from the Brainclinics Foundation at
<https://brainclinics.com/resources/tdbrain-dataset> under a data use agreement.
Thirty per cent of the clinical and treatment outcome data is withheld as a blinded
replication partition and scored centrally; the locked, EEG-only model used for our
submission is produced by `scripts/run_challenge_submission.py`.

## Verification

Every number in the paper is produced by the code in this repository, and the test
suite pins them. `tests/test_components.py` runs without the data and covers the
feature taxonomy, the two transformers, Benjamini-Hochberg and the bootstrap.
`tests/test_reproduction.py` checks the published values when the analysis outputs
are present, and is skipped otherwise.

```bash
pip install -e ".[dev]"
pytest tests/ -q                                   # components only
TDBRAIN_ANALYSIS_DIR=analysis_out pytest tests/ -q -m slow   # full reproduction
```

The reproduction tests assert the cohort size and family sizes, and the four AUC
values the paper leans on: the clinical benchmark at 0.6062, the eyes-open aperiodic
model at 0.6252, the whole feature space collapsing to 0.4652, and the graph-theory
model at 0.4922. A failure means the code and the manuscript have drifted apart.

A single hyperparameter grid, defined once in `src/evaluation.py`, is used by every
analysis including the challenge submission, so the predictions filed with
Brainclinics come from exactly the configuration described in the paper.

## Contributing

Comments, docstrings and messages are in English throughout, and the library code in
`src/` has no dependency on the GUI, on MNE or on a particular filesystem layout, so
it can be imported and tested on its own:

```python
from src.data import load_cohort, design_matrix
from src.evaluation import nested_cv, permutation_null, bootstrap_delta
from src.taxonomy import family, condition, family_blocks
```

`tests/test_components.py` needs no data and runs in seconds, which makes it the
right place to add a regression test for any change. If a change moves one of the
published numbers, `tests/test_reproduction.py` will fail: that is the intended
behaviour, and the manuscript has to move with it.

## Citation

If you use this code, please cite the paper and this repository. See `CITATION.cff`.

## Licence

MIT, see `LICENSE`. The licence covers the code only. The TDBRAIN data carry their
own data use agreement and are not redistributed here.
