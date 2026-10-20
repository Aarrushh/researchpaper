# PURPOSE: Draw Study 3 stage 0's figures from its frozen tables (ANALYSIS_PLAN_3.md section 2.7).
# WHAT GOES HERE: Read cells.csv, acuity.csv and behaviour.csv of scripts/25_fit_stage0.py and call mrr.plotting for Figure 5 (each reader's primary and reading CER against size, with the fits and s01) and Figure 6 (behaviour rates against size).
# FILLED IN AT STEP: Study 3 stage 0, 20 October 2026, after the tag data-freeze-s3-stage0.
# DO NOT: Hand-edit figures. Refit, rescore or resample here: the figures draw the tables as they are, and scripts/40_freeze.py --check --fit s3_stage0 regenerates those tables from results/final/runs.
"""Study 3 stage 0: Figures 5 and 6 from the stage-0 tables.

    python scripts/31_stage0_figures.py [--tables-dir reports/tables/s3_stage0] [--out-dir reports/figures]

Unlike scripts/30_make_figures.py, which recomputes what it draws from results/, this script reads the tables of
scripts/25_fit_stage0.py: the curves are its fits and the intervals its passage bootstrap, so the figures show what
the tables report, and both are covered by ``scripts/40_freeze.py --check --fit s3_stage0`` (tag
data-freeze-s3-stage0). Writes, where <name> is the tables folder's name:

    fig5_acuity_<name>.pdf     each reader's primary and reading CER against size in pixels and in patches, the
                               cells with their intervals, the fitted curves and s01 with its interval
    fig6_behaviour_<name>.pdf  the behaviour rates against size: omitted (head, interior, tail), runaway inserted and
                               case normalised, stacked, with the behaviour rate's interval, and format on top
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Sequence

from mrr import plotting

TABLES = ("cells", "acuity", "behaviour")

_log = logging.getLogger("stage0_figures")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tables-dir", default="reports/tables/s3_stage0",
                   help="the tables of scripts/25_fit_stage0.py (default reports/tables/s3_stage0)")
    p.add_argument("--out-dir", default="reports/figures", help="where the PDFs go (default reports/figures)")
    return p.parse_args(argv)


def read_table(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)  # the PDF backend's font subsetting logs every glyph at INFO
    args = parse_args(argv)
    tables_dir, out_dir = Path(args.tables_dir), Path(args.out_dir)
    missing = [t for t in TABLES if not (tables_dir / f"{t}.csv").exists()]
    if missing:
        _log.error("no %s under %s: run scripts/25_fit_stage0.py first", ", ".join(f"{t}.csv" for t in missing),
                   tables_dir)
        return 1
    cells, fits, behaviour = (read_table(tables_dir / f"{t}.csv") for t in TABLES)
    name = tables_dir.name

    fig5 = out_dir / f"fig5_acuity_{name}.pdf"
    plotting.plot_stage0_curves(cells, fits, fig5)
    _log.info("Figure 5: %s (%d cells, %d fits from %s)", fig5, len(cells), len(fits), tables_dir)
    fig6 = out_dir / f"fig6_behaviour_{name}.pdf"
    plotting.plot_stage0_behaviour(behaviour, cells, fig6)
    _log.info("Figure 6: %s (%d cells from %s)", fig6, len(behaviour), tables_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
