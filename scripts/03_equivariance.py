# PURPOSE: Measure the shift-equivariance error E(s) at several layers of each reader's vision tower, and draw paper Figure 0.
# WHAT GOES HERE: The E page per reader and text type, shifts from 0 to s_max in 0.25 px steps, E at the layers of section 2.1, six CSV files and a manifest in results/equivariance/, and reports/figures/fig0_equivariance.pdf drawn from those files only (build guide section 2.1).
# FILLED IN AT STEP: W1.9.
# DO NOT: Load a whole reader onto the GPU, or two towers at once. Do not make shifted pages with np.roll. Do not change configs from inside the script.
"""Step W1.9: shift-equivariance error per layer (paper Figure 0).

Each reader's vision tower is loaded alone, in float32 (readers.load_vision_tower), one reader at a
time. Two E pages per reader: the first page of Persuasion after the front-matter skip (natural) and
the first page of random_text(200_000, 0) (random62), in the conventional layout at the unaligned
pitch, with max_phase_x_px = s_max = 2 x the sweep period and max_phase_y_px = 0. Each page is
rendered at phases 0, 0.25, ..., s_max px, and E(s) of equation (1) is computed at the recorded
layers from one forward pass per page; the shift of 0 embeds the page a second time, so E(0) checks
that two forward passes agree. One CSV file per reader and text type goes to results/equivariance/
(columns layer, shift_px, e, k_used), then the manifest, and Figure 0 is drawn from the CSV files
alone. --figure-only redraws the figure without the models. The script prints the layer-0 numbers of
appendix C.3 for the notebook.
Run it from the repository root: python scripts/03_equivariance.py
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from mrr import equivariance, grid, plotting, readers, render, text_source
from mrr.manifest import write_manifest

BOOK = Path("data/raw/texts/pg105-images-3.epub")  # Persuasion, as in the render check
MARKER, OCCURRENCE = "CHAPTER I.", 2  # where its front matter ends (data/raw/texts/README.md)
RANDOM_CHARS, RANDOM_SEED = 200_000, 0  # the random62 source of appendix C
STEP_PX = 0.25
# Section 2.1: Qwen's 8, 16 and 32 are the outputs of its full-attention blocks 7, 15 and 31;
# SmolVLM's are thirds of 27 and 12 layers.
LAYERS = {"qwen": [0, 1, 8, 16, 32], "smolvlm": [0, 1, 9, 18, 27], "smolvlm500m": [0, 1, 4, 8, 12]}
TEXT_TYPES = ("natural", "random62")
COLUMNS = ["layer", "shift_px", "e", "k_used"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/readers.yaml")
    p.add_argument("--render-config", default="configs/render_defaults.yaml")
    p.add_argument("--out", default="results/equivariance")
    p.add_argument("--figure", default="reports/figures/fig0_equivariance.pdf")
    p.add_argument("--figure-only", action="store_true", help="redraw Figure 0 from the CSV files, without the models")
    return p.parse_args()


def e_page_cfg(defaults: Mapping[str, Any], reader_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The E page's render config: s_max = 2 x the sweep period as the horizontal allowance, none vertically."""
    s_max = 2 * grid.sweep_period_px(reader_cfg)
    return grid.resolve_render_cfg(defaults, reader_cfg, max_phase_x_px=s_max, max_phase_y_px=0)


def sources() -> dict[str, str]:
    """The text each E page starts from, by text type."""
    book = text_source.strip_gutenberg_boilerplate(text_source.load_book(BOOK))
    return {
        "natural": text_source.normalise_text(text_source.skip_front_matter(book, MARKER, OCCURRENCE)),
        "random62": text_source.random_text(RANDOM_CHARS, RANDOM_SEED),
    }


def measure(
    tower: readers.VisionTower, cfg: Mapping[str, Any], text: str, layers: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """E(s) and the move used at every shift and layer of one E page: (shifts, array (shifts, layers, 2))."""
    shifts = np.arange(0, cfg["max_phase_x_px"] + STEP_PX / 2, STEP_PX)
    blank = np.full((cfg["page_height_px"], cfg["page_width_px"], 3), cfg["background_color"], dtype=np.uint8)
    out = equivariance.e_curve(
        lambda s: render.render_page(text, {**cfg, "phase_x_px": s}),  # the renderer moves whole pixels exactly
        lambda page: readers.vision_hidden_states(tower, page, layers),
        shifts,
        tower.patch_px,
        blank,
    )
    return shifts, out


def write_csv(path: Path, shifts: np.ndarray, out: np.ndarray, layers: list[int]) -> None:
    rows = [
        (layer, float(s), float(out[i, j, 0]), int(out[i, j, 1]))
        for j, layer in enumerate(layers)
        for i, s in enumerate(shifts)
    ]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


def run_reader(
    reader_cfg: Mapping[str, Any], defaults: Mapping[str, Any], texts: Mapping[str, str], out_dir: Path
) -> dict[str, Any]:
    """Measure one reader's two E pages and free its tower; return the E page's render config for the manifest."""
    import torch

    name = reader_cfg["name"]
    cfg = e_page_cfg(defaults, reader_cfg)
    start = time.perf_counter()
    tower = readers.load_vision_tower(reader_cfg)
    print(f"{name}: tower loaded in {time.perf_counter() - start:.0f} s; "
          f"position-id fix added: {tower.position_ids_fixed}", flush=True)
    try:
        for text_type in TEXT_TYPES:
            text = text_source.paginate(texts[text_type], lambda candidate: render.fits_page(candidate, cfg), 1)[0]
            start = time.perf_counter()
            shifts, out = measure(tower, cfg, text, LAYERS[name])
            write_csv(out_dir / f"{name}_{text_type}.csv", shifts, out, LAYERS[name])
            numbers = equivariance.e_summary(shifts, out[:, 0, 0], tower.patch_px)
            seconds = time.perf_counter() - start
            print(f"  {text_type}: {len(text)} characters, {len(shifts)} shifts in {seconds:.0f} s; "
                  f"largest E(0) over the layers {out[0, :, 0].max():.1e}; layer 0: "
                  + ", ".join(f"{key} {value:.3g}" for key, value in numbers.items()), flush=True)
    finally:
        del tower
        gc.collect()
        torch.cuda.empty_cache()
    return cfg


def draw_figure(reader_cfgs: list[Mapping[str, Any]], out_dir: Path, figure: Path) -> None:
    """Draw Figure 0 from the six CSV files alone."""
    frames = [
        pd.read_csv(out_dir / f"{cfg['name']}_{text_type}.csv").assign(reader=cfg["name"], text_type=text_type)
        for cfg in reader_cfgs
        for text_type in TEXT_TYPES
    ]
    plotting.plot_equivariance(pd.concat(frames, ignore_index=True), reader_cfgs, figure)


def main() -> None:
    args = parse_args()
    reader_cfgs = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["readers"]
    out_dir = Path(args.out)
    if not args.figure_only:
        defaults = yaml.safe_load(Path(args.render_config).read_text(encoding="utf-8"))
        texts = sources()
        out_dir.mkdir(parents=True, exist_ok=True)
        e_pages = {reader_cfg["name"]: run_reader(reader_cfg, defaults, texts, out_dir) for reader_cfg in reader_cfgs}
        run_cfg = {
            "readers": reader_cfgs,
            "e_pages": e_pages,
            "layers": LAYERS,
            "shift_step_px": STEP_PX,
            "natural": {"book": str(BOOK), "marker": MARKER, "occurrence": OCCURRENCE},
            "random62": {"n_chars": RANDOM_CHARS, "seed": RANDOM_SEED},
        }
        print(f"manifest: {write_manifest(out_dir, run_cfg)}", flush=True)
    draw_figure(reader_cfgs, out_dir, Path(args.figure))
    print(f"Figure 0: {args.figure}", flush=True)


if __name__ == "__main__":
    main()
