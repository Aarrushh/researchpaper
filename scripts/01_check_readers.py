# PURPOSE: Verify each reader's resize behaviour and vision token count against the formula.
# WHAT GOES HERE: One native page per reader (D8), not the scaffold's three sizes and token_check.csv: Hub id, revision, dtype, placement, image grid, tokens against grid.token_count, one transcription with its latency, peak GPU memory; the manifest only after the reader's checks pass (build guide 1B.7).
# FILLED IN AT STEP: W1.6.
# DO NOT: Run sweeps from here. Do not change configs from inside the script.
"""Step W1.6: check readers.

Renders the first page of Persuasion on each reader's native page, as the render
check does, and loads the reader as configs/readers.yaml says. The checks: the
model is in the configured dtype, a reader configured for the GPU has every
parameter there, and the encoder gets the page itself with grid.token_count image
tokens. The script then transcribes the page once and prints the transcription,
its latency and the peak GPU memory. A reader's manifest is written to
results/runs/<run_id>/ only after its checks pass; a failed check stops the script.

The limit is max_new_tokens from configs/readers.yaml, set at W1.7. The latency is
the first generate call after loading, so it includes warm-up; W1.7 times pages
properly.
Readers are loaded one at a time: the laptop's 8 GiB holds one.
Run it from the repository root: make readers.
"""

from __future__ import annotations

import argparse
import gc
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from mrr import grid, readers, render, text_source
from mrr.manifest import new_run_id, write_manifest

BOOK = Path("data/raw/texts/pg105-images-3.epub")  # Persuasion, as in the render check
MARKER, OCCURRENCE = "CHAPTER I.", 2  # where its front matter ends (data/raw/texts/README.md)
GIB = 2**30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="configs/readers.yaml")
    p.add_argument("--render-config", default="configs/render_defaults.yaml")
    p.add_argument("--output-root", default="results/runs")
    return p.parse_args()


def native_page(text: str, render_cfg: Mapping[str, Any]) -> tuple[str, np.ndarray]:
    """Return the first page of text that fits the reader's native page, and the page rendered."""
    reference = text_source.paginate(text, lambda candidate: render.fits_page(candidate, render_cfg), 1)[0]
    return reference, render.render_page(reference, render_cfg)


def encoder_grid(reader_cfg: Mapping[str, Any], inputs: Mapping[str, Any], height: int, width: int) -> str:
    """Describe the encoder input and the patch and token grids of a height x width page."""
    patch, merge = int(reader_cfg["patch_size"]), int(reader_cfg["merge_size"])
    rows, cols = height // patch, width // patch
    if "image_grid_thw" in inputs:
        seen = f"image_grid_thw {inputs['image_grid_thw'].tolist()}"
    else:
        seen = f"pixel_values {tuple(inputs['pixel_values'].shape)}"
    return (
        f"{seen}: {rows} x {cols} patches of {patch} px, "
        f"{rows // merge} x {cols // merge} tokens of {merge} x {merge} patches"
    )


def placement(model: Any) -> str:
    """model.hf_device_map, summarised by device when it is long."""
    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        return "no hf_device_map"
    if len(device_map) <= 4:
        return f"hf_device_map {device_map}"
    counts = Counter(str(device) for device in device_map.values())
    return f"hf_device_map of {len(device_map)} entries, by device {dict(counts)}"


def check_reader(
    reader_cfg: Mapping[str, Any], text: str, defaults: Mapping[str, Any], output_root: Path
) -> None:
    """Run the checks for one reader, print what they found, and write its manifest if they pass."""
    import torch

    name = reader_cfg["name"]
    render_cfg = grid.resolve_render_cfg(defaults, reader_cfg)
    reference, page = native_page(text, render_cfg)
    height, width = grid.page_size_px(reader_cfg)
    if page.shape != (height, width, 3):
        raise SystemExit(f"{name}: the rendered page is {page.shape}; the native page is {(height, width, 3)}")

    torch.cuda.reset_peak_memory_stats()
    reader = readers.load_reader(reader_cfg)
    try:
        dtype = reader.model.dtype
        if dtype != getattr(torch, str(reader_cfg["dtype"])):
            raise SystemExit(f"{name}: loaded in {dtype}, configured {reader_cfg['dtype']}")
        devices = sorted({p.device.type for p in reader.model.parameters()})
        if reader_cfg["device_map"] == "cuda" and devices != ["cuda"]:
            raise SystemExit(f"{name}: device_map cuda, but parameters on {devices}")
        where = placement(reader.model)
        inputs = readers.processor_inputs(reader, page, reader_cfg["prompt"])  # raises unless the page is native
        tokens = readers.count_vision_tokens(reader, page)  # processor_inputs has checked it against the grid
        expected = grid.token_count(reader_cfg, height, width)

        result = readers.transcribe_full(reader, page, reader_cfg["prompt"])  # the limit in readers.yaml
        peak = torch.cuda.max_memory_allocated() / GIB
    finally:
        del reader
        gc.collect()
        torch.cuda.empty_cache()

    run_cfg = {
        "reader": dict(reader_cfg),
        "render": render_cfg,
        "generation": {**readers.GREEDY, "max_new_tokens": reader_cfg["max_new_tokens"]},  # build guide 1B.4
        "placement": {"parameter_devices": devices, "realised": where, "peak_allocated_gib": round(peak, 3)},  # D18
    }
    manifest = write_manifest(output_root / new_run_id("checkreaders", name), run_cfg)
    ending = "the limit" if result.hit_limit else "an end token"
    print(
        f"{name}: {reader_cfg['hf_id']} at {reader_cfg['revision']}, {dtype}\n"
        f"  placement: device_map {reader_cfg['device_map']!r}; {where}; parameters on {devices}\n"
        f"  page: {height} x {width} px, {len(reference)} characters from {reference.split(' ', 1)[0]!r}\n"
        f"  encoder input: {encoder_grid(reader_cfg, inputs, height, width)}\n"
        f"  vision tokens: {tokens}, grid.token_count {expected}; image token {reader_cfg['image_token']!r}\n"
        f"  max_new_tokens: {reader_cfg['max_new_tokens']}, from readers.yaml\n"
        f"  transcription: {result.new_tokens} new tokens, stopped by {ending}, {result.latency_s:.2f} s "
        f"(first call, with warm-up)\n"
        f"  peak GPU memory: {peak:.2f} GiB (torch.cuda.max_memory_allocated)\n"
        f"  manifest: {manifest.as_posix()}\n"
        f"{textwrap.indent(result.text, '    | ')}\n"
    )


def main() -> None:
    sys.stdout.reconfigure(errors="backslashreplace")  # a transcription may hold characters the console lacks
    args = parse_args()
    reader_cfgs = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["readers"]
    defaults = yaml.safe_load(Path(args.render_config).read_text(encoding="utf-8"))
    book = text_source.strip_gutenberg_boilerplate(text_source.load_book(BOOK))
    text = text_source.normalise_text(text_source.skip_front_matter(book, MARKER, OCCURRENCE))
    for reader_cfg in reader_cfgs:
        check_reader(reader_cfg, text, defaults, Path(args.output_root))


if __name__ == "__main__":
    main()
