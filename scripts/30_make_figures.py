# PURPOSE: Generate the paper's figures from run CSVs.
# WHAT GOES HERE: Load results and call mrr.plotting for Figures 1 and 2.
# FILLED IN AT STEP: W2.5 (Figure 1), W3.4 (Figure 2).
# DO NOT: Hand-edit figures. Do not read data from anywhere but results/.
"""Steps W2.5 and W3.4: make figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from mrr import plotting


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="configs/sweeps/<name>.yaml")
    p.add_argument("--runs-root", default="results/runs")
    p.add_argument("--out-dir", default="reports/figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sweep_name = Path(args.config).stem

    # TODO[W2.5]: load results for this sweep into a DataFrame and call
    # plotting.plot_phase_curve(df, reports/figures/fig1_phase_<sweep_name>.pdf).
    # At W3.4 also call plotting.plot_tradeoff(df, reports/figures/fig2_tradeoff_<sweep_name>.pdf).
    raise NotImplementedError("TODO[W2.5]: load results and call the plotting functions")


if __name__ == "__main__":
    main()
