# PURPOSE: Generate the paper's figures from results/.
# WHAT GOES HERE: Load results and call mrr.plotting for Figures 1, 2 and 3a.
# FILLED IN AT STEP: W2.5 (Figure 1), W3.4 (Figure 2), W3.6 (Figure 3a).
# DO NOT: Hand-edit figures. Do not read data from anywhere but results/.
"""Steps W2.5, W3.4 and W3.6: make figures.

    python scripts/30_make_figures.py [--config configs/sweeps/pilot.yaml] [--runs-root results/runs]
                                      [--mtf-dir results/mtf] [--out-dir reports/figures]

With --config, reads the run folders of the sweep (results/runs/<run_id>/manifest.json and results.csv) through
``sweep.load_results`` and drops the pages section 8 of PREREGISTRATION.md excludes, exactly as scripts/20_analyze.py
does, so the figures show the rows the tables used. Writes ``fig1_phase_<sweep_name>.pdf`` (Figure 1) and, when the
rows hold block Z, ``fig2_tradeoff_<sweep_name>.pdf`` (Figure 2). Nothing is read from reports/tables/: the figures
recompute what they draw with mrr.stats, with the preregistered seed and 2,000 page resamples (section 9).

Whenever --mtf-dir holds ``layer0_<reader>.csv`` files (scripts/04_mtf.py), writes ``fig3a_layer0_mtf.pdf``
(Figure 3a): each reader's ring profile against cycles per pixel, beside the Mannos-Sakrison contrast sensitivity
placed on the same axis at an assumed 40 pixels per degree (a 96 dpi screen seen from 60 cm gives 39.6; notes A8),
the assumption written in the axis label. Nothing to draw, neither a sweep nor MTF files, is an error.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

# One BLAS thread before NumPy loads, as scripts/20_analyze.py: the fitted lines come from the same least squares.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from mrr import plotting
from mrr.mtf import csf_mannos_sakrison
from mrr.sweep import RunFolder, complete_rows, load_results, sweep_run_dirs

RESAMPLES = 2000  # PREREGISTRATION.md section 9
SEED = 20261014
PX_PER_DEGREE = 40.0  # pixels per degree of visual angle: 96 dpi at 60 cm gives 39.6 (notes A8, equation (8)); an assumption

_log = logging.getLogger("figures")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="configs/sweeps/<name>.yaml for Figures 1 and 2")
    p.add_argument("--runs-root", default="results/runs", help="where the run folders are (default results/runs)")
    p.add_argument("--mtf-dir", default="results/mtf", help="where layer0_<reader>.csv are (default results/mtf)")
    p.add_argument("--out-dir", default="reports/figures", help="where the PDFs go (default reports/figures)")
    return p.parse_args(argv)


def layer0_curves(mtf_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """reader -> its ring profile from ``<mtf_dir>/layer0_<reader>.csv`` (the rows whose profile is ``ring``)."""
    curves: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(Path(mtf_dir).glob("layer0_*.csv")):
        with open(path, encoding="utf-8", newline="") as fh:
            rows = [row for row in csv.DictReader(fh) if row["profile"] == "ring"]
        if not rows:
            raise ValueError(f"{path}: no ring rows")
        curves[path.stem[len("layer0_"):]] = {
            "frequency": np.array([float(row["f_cyc_per_px"]) for row in rows]),
            "gain": np.array([float(row["value"]) for row in rows]),
        }
    return curves


def draw_figure_3a(curves: dict[str, dict[str, np.ndarray]], out_dir: Path) -> Path:
    """Figure 3a: the ring profiles beside the human CSF at PX_PER_DEGREE pixels per degree (build guide 3S.4)."""
    f_deg = np.linspace(0.0, 0.5 * PX_PER_DEGREE, 201)  # up to the pixel limit, half a cycle per pixel
    reference = {
        "name": f"human CSF (Mannos–Sakrison), ρ = {PX_PER_DEGREE:g} px/deg assumed",
        "frequency": f_deg / PX_PER_DEGREE,  # notes A8, equation (9)
        "gain": csf_mannos_sakrison(f_deg),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fig3a_layer0_mtf.pdf"
    plotting.plot_mtf(curves, out, px_per_degree=PX_PER_DEGREE, reference=reference)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)  # the PDF backend's font subsetting logs every glyph at INFO
    args = parse_args(argv)
    out_dir, mtf_dir = Path(args.out_dir), Path(args.mtf_dir)

    curves = layer0_curves(mtf_dir)
    if curves:
        fig3a = draw_figure_3a(curves, out_dir)
        _log.info("Figure 3a: %s (%s from %s)", fig3a, ", ".join(curves), mtf_dir)
    elif args.config is None:
        _log.error("nothing to draw: pass --config <sweep> for Figures 1 and 2, or write %s first (scripts/04_mtf.py) "
                   "for Figure 3a", mtf_dir)
        return 1
    if args.config is None:
        return 0

    sweep_path, runs_root = Path(args.config), Path(args.runs_root)
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
    if df["block"].isin(("V", "H", "C")).any():
        fig1 = out_dir / f"fig1_phase_{sweep_name}.pdf"
        plotting.plot_phase_curve(df, fig1, n_resamples=RESAMPLES, seed=SEED, reader_cfgs=reader_cfgs)
        _log.info("Figure 1: %s", fig1)
    else:  # the ladder: block Z only
        _log.info("no rows of blocks V, H or C in %s: no Figure 1", sweep_name)
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
