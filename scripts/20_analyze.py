# PURPOSE: Statistics over run CSVs.
# WHAT GOES HERE: Load results, pair conditions by page, run preregistered tests, write tables.
# FILLED IN AT STEP: W2.4 (pilot), W3.4 (full).
# DO NOT: Run analyses that are not in PREREGISTRATION.md without labelling them exploratory.
"""Steps W2.4 and W3.4: analyse results."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from mrr import stats
from mrr.manifest import new_run_id, write_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="configs/sweeps/<name>.yaml")
    p.add_argument("--runs-root", default="results/runs")
    p.add_argument("--out-dir", default="reports/tables")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sweep_name = Path(args.config).stem

    # TODO[W2.4]: find run folders for this sweep under --runs-root, concatenate results.csv,
    # apply exclusion rules from PREREGISTRATION.md, pair conventional vs patch_aware by page,
    # call stats.paired_wilcoxon, stats.median_paired_difference and stats.bootstrap_ci,
    # and write reports/tables/<sweep_name>_*.csv. Record inputs with write_manifest.
    raise NotImplementedError("TODO[W2.4]: implement the preregistered analysis")


if __name__ == "__main__":
    main()
