#!/usr/bin/env python3
"""Evaluate every feature family against the clinical benchmark.

Reproduces Table 1 of the paper. For each outcome it fits a clinical model, a model
on the whole EEG feature space, and one model per feature family and recording
condition. Every model is compared with a null estimated by permuting the outcome
and rerunning the entire procedure, not with a nominal 0.50.

Example:
    python scripts/run_family_models.py --analysis-dir analysis_out Responder Remitter
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import design_matrix, load_cohort
from src.evaluation import (benjamini_hochberg, bootstrap_delta, nested_cv,
                            permutation_null)
from src.taxonomy import family_blocks


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def evaluate(X, y, selection, repeats, permutations, seed, k, n_forced):
    """Repeat the nested cross-validation, then estimate its null."""
    oofs, aucs = [], []
    for r in range(repeats):
        oof, _ = nested_cv(X, y, seed=seed + r, selection=selection,
                           k=k, n_forced=n_forced)
        oofs.append(oof)
        aucs.append(roc_auc_score(y, oof))
    oof = np.mean(oofs, axis=0)
    null = permutation_null(X, y, selection, n_permutations=permutations,
                            seed=seed, k=k, n_forced=n_forced)
    base_rate = np.full(len(y), y.mean())
    return {
        "n_features": int(X.shape[1]),
        "selection": selection,
        "auc": float(np.mean(aucs)),
        "auc_sd": float(np.std(aucs)),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "p_permutation": float((null >= np.mean(aucs)).mean()),
        "brier": float(brier_score_loss(y, oof)),
        "brier_skill_score": float(
            1 - brier_score_loss(y, oof) / brier_score_loss(y, base_rate)),
        "oof": oof.tolist(),
    }


def run(target, analysis_dir, results_dir, args):
    log(f"target {target}")
    frame, y, eeg, artefact = load_cohort(analysis_dir, target)
    log(f"n = {len(y)}, positives {y.sum()} ({y.mean():.3f}), "
        f"EEG features {len(eeg)}, artefact proxies {len(artefact)}")

    specs = {"clinical": ([], "none"),
             "all_eeg_free": (eeg, "free"),
             "all_eeg_forced": (eeg, "forced")}
    for (fam, cond), cols in family_blocks(eeg).items():
        specs[f"{fam}_{cond}"] = (cols, "forced")

    results = {}
    for name, (cols, selection) in specs.items():
        X = design_matrix(frame, cols)
        started = time.time()
        permutations = (args.permutations if name.startswith(("clinical", "all_"))
                        else max(args.permutations - 15, 10))
        results[name] = evaluate(X, y, selection, args.repeats, permutations,
                                 config.SEED, args.k, len(config.CLINICAL))
        r = results[name]
        log(f"  {name:22s} AUC {r['auc']:.4f} (SD {r['auc_sd']:.4f}) | "
            f"null {r['null_mean']:.4f} | p {r['p_permutation']:.3f} | "
            f"{r['n_features']:3d} features | {time.time() - started:5.0f}s")

    base = np.asarray(results["clinical"]["oof"])
    families = [k for k in results if k not in
                ("clinical", "all_eeg_free", "all_eeg_forced")]
    for name in [k for k in results if k != "clinical"]:
        results[name]["vs_clinical"] = bootstrap_delta(
            y, np.asarray(results[name]["oof"]), base, seed=config.SEED)
    for name, q in zip(families,
                       benjamini_hochberg([results[k]["vs_clinical"]["p"]
                                           for k in families])):
        results[name]["vs_clinical"]["q_bh"] = float(q)

    payload = {"target": target, "n": int(len(y)),
               "prevalence": float(y.mean()), "k": args.k,
               "repeats": args.repeats, "models": results}
    out = results_dir / f"family_models_{target}.json"
    out.write_text(json.dumps(payload, indent=1))
    log(f"written {out}")
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("targets", nargs="*", default=["Responder", "Remitter"])
    p.add_argument("--analysis-dir", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--repeats", type=int, default=config.N_REPEATS)
    p.add_argument("--permutations", type=int, default=config.N_PERMUTATIONS + 15)
    p.add_argument("--k", type=int, default=config.K_FEATURES)
    args = p.parse_args()
    analysis_dir, results_dir = config.resolve(args.analysis_dir, args.results_dir)
    for target in args.targets:
        run(target, analysis_dir, results_dir, args)


if __name__ == "__main__":
    main()
