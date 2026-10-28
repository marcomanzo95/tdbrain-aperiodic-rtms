"""Shared command-line plumbing for the analysis scripts."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402


def parser(description):
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("targets", nargs="*", default=["Responder", "Remitter"])
    p.add_argument("--analysis-dir", default=None,
                   help="folder with cohort_*.csv and features_*.csv")
    p.add_argument("--results-dir", default=None, help="where to write results")
    p.add_argument("--participants", default=None,
                   help="TDBRAIN_participants_V3.xlsx, needed for the BDI outcome")
    p.add_argument("--repeats", type=int, default=config.N_REPEATS)
    p.add_argument("--permutations", type=int, default=config.N_PERMUTATIONS)
    return p


def setup(args):
    return config.resolve(args.analysis_dir, args.results_dir)


def participants_file(args, analysis_dir):
    """Locate the participants table, which carries BDI_pre and BDI_post."""
    if args.participants:
        p = Path(args.participants)
    else:
        for candidate in (analysis_dir, analysis_dir.parent):
            p = candidate / "TDBRAIN_participants_V3.xlsx"
            if p.exists():
                break
    if not p.exists():
        raise SystemExit(
            "TDBRAIN_participants_V3.xlsx not found. It carries BDI_pre and "
            "BDI_post, which the continuous outcome needs. Pass --participants.")
    return p
