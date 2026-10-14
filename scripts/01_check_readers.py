# PURPOSE: Verify each reader's resize behaviour and vision token count against the formula.
# WHAT GOES HERE: One native page per reader (D8), not the scaffold's three sizes and token_check.csv: Hub id, revision, dtype, placement, image grid, tokens against grid.token_count, one transcription with its latency, peak GPU memory; the manifest only after the reader's checks pass (build guide 1B.7). With --out, the reader facts table that R07 and R08 of spec/RENDERING_SPEC.md cite, with the blind-strip noise test (W4.1).
# FILLED IN AT STEP: W1.6; W4.1 (--out).
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

With --out PATH (make reader-facts) the script also writes the reader facts table,
one row per reader in FACTS_COLUMNS order, that R07 and R08 of spec/RENDERING_SPEC.md
cite, with a manifest beside it, once every reader has passed. For a reader with a
blind strip it replaces the strip's columns, then its rows, then the column before
the strip with seeded noise and records whether the encoder's output changed, as
tests/test_reader_facts.py does.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from mrr import grid, readers, render, text_source
from mrr.manifest import new_run_id, write_manifest

BOOK = Path("data/raw/texts/pg105-images-3.epub")  # Persuasion, as in the render check
MARKER, OCCURRENCE = "CHAPTER I.", 2  # where its front matter ends (data/raw/texts/README.md)
GIB = 2**30
FACTS_COLUMNS = (  # documented in docs/04_data_dictionary.md under reports/tables/readers/reader_facts.csv
    "reader", "hf_id", "revision", "dtype", "placement", "patch_px", "merge", "token_cell_px", "window_px",
    "page_height_px", "page_width_px", "encoder_input", "encoder_input_px", "vision_tokens", "grid_token_count",
    "native_frame", "blind_from_px", "blind_strip_px", "blind_strip_invisible", "control_column_visible",
    "max_new_tokens", "transcription_tokens", "hit_limit", "latency_s", "peak_gib",
)
NOISE_SEED = 2  # the blind-strip noise, the same generator as tests/test_reader_facts.py


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="configs/readers.yaml")
    p.add_argument("--render-config", default="configs/render_defaults.yaml")
    p.add_argument("--output-root", default="results/runs")
    p.add_argument("--out", default=None, help="also write the reader facts table (W4.1) here, with a manifest beside it")
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


def encoder_input_px(reader_cfg: Mapping[str, Any], inputs: Mapping[str, Any]) -> tuple[int, int, int]:
    """(frames, height, width) in pixels of what reached the encoder: Qwen's grid times its patch, or SmolVLM's frames."""
    if "image_grid_thw" in inputs:
        patch = int(reader_cfg["patch_size"])
        (frames, rows, cols), *_ = inputs["image_grid_thw"].tolist()
        return int(frames), int(rows) * patch, int(cols) * patch
    shape = tuple(int(v) for v in inputs["pixel_values"].shape)  # (batch, frames, channels, height, width)
    return shape[1], shape[-2], shape[-1]


def smolvlm_features(reader: readers.Reader, page: np.ndarray) -> tuple[Any, Any]:
    """SmolVLM's vision encoder output and the connector's output for one page (tests/test_reader_facts.py)."""
    import torch

    inputs = readers.processor_inputs(reader, page, reader.cfg["prompt"]).to(reader.model.device)
    with torch.inference_mode():
        out = reader.model.get_image_features(
            pixel_values=inputs["pixel_values"], pixel_attention_mask=inputs["pixel_attention_mask"]
        )
    return out.last_hidden_state, out.pooler_output


def region_changes_features(reader: readers.Reader, page: np.ndarray, region: tuple[slice, slice]) -> bool:
    """Whether seeded noise in region of the page changes the encoder's or the connector's output."""
    import torch

    changed = page.copy()
    changed[region] = np.random.default_rng(NOISE_SEED).integers(0, 256, size=changed[region].shape, dtype=np.uint8)
    pairs = zip(smolvlm_features(reader, page), smolvlm_features(reader, changed))
    return any(not torch.equal(before, after) for before, after in pairs)


def blind_strip_facts(reader: readers.Reader, reader_cfg: Mapping[str, Any], page: np.ndarray) -> dict[str, bool] | None:
    """R08: does noise in the strip's columns, in its rows, and in the column before it change the encoder's output?

    None for a reader whose patch grid covers the whole frame.
    """
    strip = grid.blind_strip_px(reader_cfg)
    if strip == 0:
        return None
    start = int(reader_cfg["blind_from_px"])
    if start + strip != page.shape[0] or page.shape[0] != page.shape[1]:
        raise SystemExit(f"{reader_cfg['name']}: blind_from_px {start} + strip {strip} px is not the frame {page.shape[:2]}")
    regions = {
        "columns": (slice(None), slice(start, None)),
        "rows": (slice(start, None), slice(None)),
        "control": (slice(None), slice(start - 1, start)),
    }
    return {name: region_changes_features(reader, page, region) for name, region in regions.items()}


def facts_row(
    reader_cfg: Mapping[str, Any],
    height: int,
    width: int,
    inputs: Mapping[str, Any],
    tokens: int,
    expected: int,
    result: Any,
    peak_gib: float,
    where: str,
    blind: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """One row of the reader facts table, FACTS_COLUMNS in order, from what check_reader measured."""
    frames, in_h, in_w = encoder_input_px(reader_cfg, inputs)
    row = {
        "reader": reader_cfg["name"],
        "hf_id": reader_cfg["hf_id"],
        "revision": reader_cfg["revision"],
        "dtype": reader_cfg["dtype"],
        "placement": where,
        "patch_px": int(reader_cfg["patch_size"]),
        "merge": int(reader_cfg["merge_size"]),
        "token_cell_px": grid.token_cell_px(reader_cfg),
        "window_px": reader_cfg.get("window_px", ""),
        "page_height_px": height,
        "page_width_px": width,
        "encoder_input": encoder_grid(reader_cfg, inputs, height, width),
        "encoder_input_px": f"{frames}x{in_h}x{in_w}",
        "vision_tokens": tokens,
        "grid_token_count": expected,
        "native_frame": frames == 1 and (in_h, in_w) == (height, width) and tokens == expected,  # R07
        "blind_from_px": reader_cfg.get("blind_from_px", ""),
        "blind_strip_px": grid.blind_strip_px(reader_cfg),
        "blind_strip_invisible": "" if blind is None else (not blind["columns"] and not blind["rows"]),  # R08
        "control_column_visible": "" if blind is None else blind["control"],
        "max_new_tokens": int(reader_cfg["max_new_tokens"]),
        "transcription_tokens": result.new_tokens,
        "hit_limit": result.hit_limit,
        "latency_s": f"{result.latency_s:.3f}",
        "peak_gib": f"{peak_gib:.3f}",
    }
    assert tuple(row) == FACTS_COLUMNS
    return row


def write_facts(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the reader facts table: a header of FACTS_COLUMNS, then one row per reader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FACTS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def check_reader(
    reader_cfg: Mapping[str, Any], text: str, defaults: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    """Run the checks for one reader, print what they found, write its manifest if they pass, return its facts row."""
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
        blind = blind_strip_facts(reader, reader_cfg, page)  # R08; None without a strip

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
        "blind_strip": blind,
    }
    manifest = write_manifest(output_root / new_run_id("checkreaders", name), run_cfg)
    ending = "the limit" if result.hit_limit else "an end token"
    if blind is None:
        strip_text = "none: the patch grid covers the frame"
    else:
        strip_text = (
            f"{grid.blind_strip_px(reader_cfg)} px from {reader_cfg['blind_from_px']}; noise changes the encoder "
            f"output in its columns: {blind['columns']}, in its rows: {blind['rows']}, "
            f"in the column before it: {blind['control']}"
        )
    print(
        f"{name}: {reader_cfg['hf_id']} at {reader_cfg['revision']}, {dtype}\n"
        f"  placement: device_map {reader_cfg['device_map']!r}; {where}; parameters on {devices}\n"
        f"  page: {height} x {width} px, {len(reference)} characters from {reference.split(' ', 1)[0]!r}\n"
        f"  encoder input: {encoder_grid(reader_cfg, inputs, height, width)}\n"
        f"  vision tokens: {tokens}, grid.token_count {expected}; image token {reader_cfg['image_token']!r}\n"
        f"  blind strip: {strip_text}\n"
        f"  max_new_tokens: {reader_cfg['max_new_tokens']}, from readers.yaml\n"
        f"  transcription: {result.new_tokens} new tokens, stopped by {ending}, {result.latency_s:.2f} s "
        f"(first call, with warm-up)\n"
        f"  peak GPU memory: {peak:.2f} GiB (torch.cuda.max_memory_allocated)\n"
        f"  manifest: {manifest.as_posix()}\n"
        f"{textwrap.indent(result.text, '    | ')}\n"
    )
    return facts_row(reader_cfg, height, width, inputs, tokens, expected, result, peak, where, blind)


def main() -> None:
    sys.stdout.reconfigure(errors="backslashreplace")  # a transcription may hold characters the console lacks
    args = parse_args()
    reader_cfgs = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["readers"]
    defaults = yaml.safe_load(Path(args.render_config).read_text(encoding="utf-8"))
    book = text_source.strip_gutenberg_boilerplate(text_source.load_book(BOOK))
    text = text_source.normalise_text(text_source.skip_front_matter(book, MARKER, OCCURRENCE))
    rows = [check_reader(reader_cfg, text, defaults, Path(args.output_root)) for reader_cfg in reader_cfgs]
    if args.out:  # W4.1: only reached when every reader passed
        out = Path(args.out)
        run_cfg = {
            "readers": reader_cfgs,
            "render_defaults": defaults,
            "columns": list(FACTS_COLUMNS),
            "noise_seed": NOISE_SEED,
        }
        manifest = write_manifest(out.parent, run_cfg, run_id=new_run_id("readerfacts", "all"))
        write_facts(out, rows)
        print(f"reader facts: {out.as_posix()} ({len(rows)} rows); manifest {manifest.as_posix()}")


if __name__ == "__main__":
    main()
