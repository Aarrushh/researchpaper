# PURPOSE: Visual check that phase shifts move glyphs by the intended amount relative to the grid.
# WHAT GOES HERE: Render one line at four phases with the grid overlaid; save PNGs.
# FILLED IN AT STEP: W1.5.
# DO NOT: Treat these images as results; they are sanity checks only.
"""Step W1.5: render check."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from mrr import grid, render
from mrr.manifest import new_run_id, write_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="configs/render_defaults.yaml")
    p.add_argument("--cell-px", type=int, default=28, help="token cell size to overlay")
    p.add_argument("--out-dir", default="reports/figures")
    p.add_argument("--output-root", default="results/runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(args.output_root) / new_run_id("rendercheck", "none")
    write_manifest(run_dir, {"render": cfg, "cell_px": args.cell_px})

    phases = grid.phase_offsets(args.cell_px, 4)
    # TODO[W1.5]: for each phase, set phase_x_px, call render.render_page on one line of text,
    # apply render.overlay_grid, and save to reports/figures/render_check_<phase>.png.
    raise NotImplementedError("TODO[W1.5]: render one line at four phases and save overlays")


if __name__ == "__main__":
    main()
