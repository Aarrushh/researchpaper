# PURPOSE: Generate the paper's figures from run CSVs.
# WHAT GOES HERE: Load results and call mrr.plotting for Figures 1 and 2.
# FILLED IN AT STEP: W2.5 (Figure 1), W3.4 (Figure 2).
# DO NOT: Hand-edit figures. Do not read data from anywhere but results/.
"""Steps W2.5 and W3.4: make figures.

    python scripts/30_make_figures.py --config configs/sweeps/pilot.yaml [--runs-root results/runs] [--out-dir reports/figures]

Reads the run folders of the sweep (results/runs/<run_id>/manifest.json and results.csv) through
``sweep.load_results`` and drops the pages section 8 of PREREGISTRATION.md excludes, exactly as scripts/20_analyze.py
does, so the figures show the rows the tables used. Writes ``fig1_phase_<sweep_name>.pdf`` (Figure 1) and, when the
rows hold block Z, ``fig2_tradeoff_<sweep_name>.pdf`` (Figure 2). Nothing is read from reports/tables/: the figures
recompute what they draw with mrr.stats, with the preregistered seed and 2,000 page resamples (section 9).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import yaml

from mrr import plotting
from mrr.sweep import RunFolder, complete_rows, load_results, sweep_run_dirs

RESAMPLES = 2000  # PREREGISTRATION.md section 9
SEED = 20261014

_log = logging.getLogger("figures")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="configs/sweeps/<name>.yaml")
    p.add_argument("--runs-root", default="results/runs", help="where the run folders are (default results/runs)")
    p.add_argument("--out-dir", default="reports/figures", help="where the PDFs go (default reports/figures)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    sweep_path, runs_root, out_dir = Path(args.config), Path(args.runs_root), Path(args.out_dir)
    sweep_cfg = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    sweep_name = sweep_path.stem
    run_dirs = sweep_run_dirs(sweep_path, runs_root)
    if not run_dirs:
        _log.error("no run folder of sweep %s under %s: run the sweep first (scripts/10_run_sweep.py)", sweep_name, runs_root)
        return 1
    folders = [RunFolder(d) for d in run_dirs]
    reader_cfgs = {}
    for folder in folders:
        reader_cfgs.setdefault(folder.reader, folder.reader_cfg)

    df_all = load_results(sweep_path, runs_root)  # raises on a repeated key or an edited sweep file
    df, excluded = complete_rows(df_all, folders)  # section 8: a page missing a planned row is not drawn
    _log.info("%s: %d rows in %d run folder(s); %d page(s) excluded under section 8; %d rows drawn",
              sweep_name, len(df_all), len(folders), sum(len(p) for p in excluded.values()), len(df))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig1 = out_dir / f"fig1_phase_{sweep_name}.pdf"
    plotting.plot_phase_curve(df, fig1, n_resamples=RESAMPLES, seed=SEED, reader_cfgs=reader_cfgs)
    _log.info("Figure 1: %s", fig1)
    if (df["block"] == "Z").any():
        fig2 = out_dir / f"fig2_tradeoff_{sweep_name}.pdf"
        plotting.plot_tradeoff(df, fig2)
        _log.info("Figure 2: %s", fig2)
    elif "Z" in (sweep_cfg.get("blocks") or {}):
        _log.info("block Z is in %s but no page of it is complete yet: no Figure 2", sweep_name)
    else:
        _log.info("no block Z in %s: no Figure 2 (the pilot has none; the full sweep has)", sweep_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
