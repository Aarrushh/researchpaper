# PURPOSE: Visual check that phase shifts move glyphs by the intended amount relative to the grid.
# WHAT GOES HERE: One page per reader at four cell phases and 0.5 px, grids overlaid; PNGs, 10x crops, hashes (D8; build guide 1A.10).
# FILLED IN AT STEP: W1.5.
# DO NOT: Treat these images as results; they are sanity checks only.
"""Step W1.5: render check.

Renders the first page of Persuasion on one reader's native page at the phases
grid.phase_offsets(cell, 4) and 0.5 px; each phase moves the text right and down
by the same amount. For every phase it saves the page with the reader's grids
overlaid, render_check_<reader>_<phase>.png, and the overlay around the first
word enlarged ten times by repeating pixels, render_check_<reader>_<phase>_crop.png.
The crop box is the same at every phase. It prints the SHA-1 of each raw page,
before the overlay, for LAB_NOTEBOOK.md.

Whole-pixel phases must show the same letters in different places against the
grid; a fractional phase must change the edge greys. Without --reader the page
is the render defaults' own, the cell is --cell-px and only cell lines are drawn.
Run it from the repository root: make render-check.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from mrr import grid, render, text_source
from mrr.manifest import git_commit, git_dirty

BOOK = Path("data/raw/texts/pg105-images-3.epub")  # Persuasion
MARKER, OCCURRENCE = "CHAPTER I.", 2  # where its front matter ends (data/raw/texts/README.md)
ZOOM = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/render_defaults.yaml", help="render defaults")
    p.add_argument("--readers-config", default="configs/readers.yaml")
    which = p.add_mutually_exclusive_group()
    which.add_argument("--reader", help="a reader in --readers-config: its cell, native page and grids")
    which.add_argument("--cell-px", type=int, default=28, help="cell size when no reader is given")
    p.add_argument("--out-dir", default="reports/figures/render_check")
    return p.parse_args()


def setup(args: argparse.Namespace) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Return the name used in file names, the render config and the sizes for overlay_grid."""
    defaults = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.reader is None:
        cfg = {**defaults, "blind_strip_px": 0}
        for key in ("max_phase_x_px", "max_phase_y_px"):
            if cfg.get(key) is None:
                cfg[key] = args.cell_px - 1  # one period less a pixel, as grid.resolve_render_cfg sets it
        return f"cell{args.cell_px}", cfg, {"cell_px": args.cell_px}
    readers = yaml.safe_load(Path(args.readers_config).read_text(encoding="utf-8"))["readers"]
    reader = next((r for r in readers if r["name"] == args.reader), None)
    if reader is None:
        raise SystemExit(f"unknown reader {args.reader!r}; {args.readers_config} has {[r['name'] for r in readers]}")
    grids = {
        "cell_px": grid.token_cell_px(reader),
        "patch_px": int(reader["patch_size"]),
        "window_px": reader.get("window_px"),
        "blind_from_px": reader.get("blind_from_px"),
    }
    return args.reader, grid.resolve_render_cfg(defaults, reader), grids


def at_phase(cfg: dict[str, Any], phase: float) -> dict[str, Any]:
    return {**cfg, "phase_x_px": phase, "phase_y_px": phase}


def crop_box(word: str, cfg: dict[str, Any], phases: list[float], cell_px: int) -> tuple[slice, slice]:
    """Return the whole cells that hold the word's ink at every phase, with the cell lines on all four sides."""
    ink = np.zeros((cfg["page_height_px"], cfg["page_width_px"]), dtype=bool)
    for phase in phases:
        ink |= render.render_coverage(word, at_phase(cfg, phase)) > 0

    def span(index: np.ndarray, size: int) -> slice:
        # One past the last cell, where the next cell's line is drawn, unless that is off the page.
        return slice(cell_px * (index.min() // cell_px), min(size, cell_px * (index.max() // cell_px + 1) + 1))

    rows, cols = np.nonzero(ink)
    return span(rows, ink.shape[0]), span(cols, ink.shape[1])


def main() -> None:
    args = parse_args()
    name, cfg, grids = setup(args)
    book = text_source.strip_gutenberg_boilerplate(text_source.load_book(BOOK))
    text = text_source.normalise_text(text_source.skip_front_matter(book, MARKER, OCCURRENCE))
    page = text_source.paginate(text, lambda p: render.fits_page(p, cfg), 1)[0]
    word = page.split(" ", 1)[0]
    phases = [*grid.phase_offsets(grids["cell_px"], 4), 0.5]
    rows, cols = crop_box(word, cfg, phases, grids["cell_px"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirty = " with uncommitted changes" if git_dirty() else ""
    print(
        f"{name}: {cfg['page_width_px']}x{cfg['page_height_px']} page, {grids['cell_px']} px cell, "
        f"{len(page)} characters from {word!r}; crop rows {rows.start}-{rows.stop - 1}, "
        f"columns {cols.start}-{cols.stop - 1}; commit {git_commit(short=True) or 'nogit'}{dirty}"
    )
    for phase in phases:
        img = render.render_page(page, at_phase(cfg, phase))
        overlay = render.overlay_grid(img, **grids)
        stem = f"render_check_{name}_{phase:g}"  # not Path.with_suffix, which would take ".5" for a suffix
        Image.fromarray(overlay).save(out_dir / f"{stem}.png")
        crop = np.repeat(np.repeat(overlay[rows, cols], ZOOM, axis=0), ZOOM, axis=1)  # nearest neighbour
        Image.fromarray(crop).save(out_dir / f"{stem}_crop.png")
        print(f"  phase {phase:g} px: sha1 {hashlib.sha1(img.tobytes()).hexdigest()}  {stem}.png")


if __name__ == "__main__":
    main()
