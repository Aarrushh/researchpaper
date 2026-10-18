# PURPOSE: Study 2 pre-run step 7.2: can each validation reader run here as its block in configs/readers_study2.yaml says?
# WHAT GOES HERE: Per reader: where its parameters are and how many linear layers are 4-bit (none in the vision tower), the vision-token count on a blank native page, one random-character page and one prose page in the contained monospace layout (M0) transcribed, peak GPU memory; feasibility.csv and a manifest under reports/tables/study2_prerun/.
# FILLED IN AT STEP: W5.1.
# DO NOT: Pool these pages with Study 2 (PREREGISTRATION_2.md section 8 excludes them). Do not change a reader block from here.
"""Study 2, pre-run step 7.2: validation-reader feasibility.

    python scripts/24_feasibility.py --readers qwen7b internvl3_2b --font <mono.ttf> --font-size <px>

A reader passes (section 7.2) when every parameter is on the GPU (no CPU offload of the language model), the vision
tower holds no 4-bit layer, a blank native page gets grid.token_count vision tokens, and both pages decode to text.
The pages: the first page of the monospace pilot's random text (seed 20261019, explore9.py) and the first page of
Persuasion after its front matter (the page every reader check reads), both in the M0 layout of section 3 (the mono
face at advance P, patch_aware, phase 0, unaligned pitch, the Qwen allowances); the limit per page is
ceil(1.5 x the reference in the reader's tokens), as in the sweep. Readers are loaded one at a time.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from mrr import grid, readers, render, score, text_source
from mrr.manifest import git_dirty, new_run_id, write_manifest

BOOK = Path("data/raw/texts/pg105-images-3.epub")  # Persuasion
MARKER, OCCURRENCE = "CHAPTER I.", 2
PILOT_SEED = 20261019  # the monospace pilot's random text (research/exploratory_2026-10-18/explore9.py)
ALLOWANCES = (105, 26)  # the Qwen phase allowances, which section 3 gives the validation readers (P = 14)
VISION_NAMES = ("visual", "vision_tower", "vision_model")
GIB = 2**30
COLUMNS = ("reader", "hf_id", "revision", "quantization", "parameter_devices", "linear4bit_layers",
           "vision_linear4bit_layers", "blank_tokens", "grid_token_count", "page", "chars", "max_new_tokens",
           "new_tokens", "hit_limit", "latency_s", "cer", "peak_gib", "passes")


def m0_cfg(defaults: Mapping[str, Any], reader_cfg: Mapping[str, Any], font: str, size: float) -> dict[str, Any]:
    """The render config of condition M0 for one reader."""
    return grid.resolve_render_cfg(
        defaults, reader_cfg, font_file=font, font_size_px=float(size), line_pitch_px=render.unaligned_pitch_px(size),
        render_mode="patch_aware", phase_x_px=0.0, phase_y_px=0.0,
        max_phase_x_px=ALLOWANCES[0], max_phase_y_px=ALLOWANCES[1])


def layer_counts(model: Any) -> tuple[int, int]:
    """(4-bit linear layers in the model, 4-bit linear layers inside its vision tower)."""
    try:
        from bitsandbytes.nn import Linear4bit
    except ImportError:
        return 0, 0
    names = [name for name, module in model.named_modules() if isinstance(module, Linear4bit)]
    return len(names), sum(any(part in name.split(".") for part in VISION_NAMES) for name in names)


def passes(devices: Sequence[str], vision_4bit: int, blank: int, expected: int, texts: Sequence[str]) -> bool:
    """Section 7.2: all parameters on the GPU, the tower unquantised, the token count right, both pages decoded."""
    return list(devices) == ["cuda"] and vision_4bit == 0 and blank == expected and all(t.strip() for t in texts)


def check(reader_cfg: Mapping[str, Any], pages: Mapping[str, str], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load one reader, run the checks and both pages, unload it; one row per page."""
    import torch

    torch.cuda.reset_peak_memory_stats()
    reader = readers.load_reader(reader_cfg)
    try:
        devices = sorted({p.device.type for p in reader.model.parameters()})
        four_bit, vision_four_bit = layer_counts(reader.model)
        height, width = grid.page_size_px(reader_cfg)
        blank = readers.count_vision_tokens(reader, np.full((height, width, 3), 255, np.uint8))
        expected = grid.token_count(reader_cfg, height, width)
        results = {}
        for kind, text in pages.items():
            limit = math.ceil(1.5 * readers.count_text_tokens(reader, text))
            results[kind] = (limit, readers.transcribe_full(reader, render.render_page(text, cfg), reader_cfg["prompt"],
                                                            max_new_tokens=limit))
        peak = torch.cuda.max_memory_allocated() / GIB
    finally:
        readers.unload_reader(reader)
    ok = passes(devices, vision_four_bit, blank, expected, [r.text for _, r in results.values()])
    rows = []
    for kind, (limit, r) in results.items():
        rows.append({
            "reader": reader_cfg["name"], "hf_id": reader_cfg["hf_id"], "revision": reader_cfg["revision"],
            "quantization": reader_cfg.get("quantization", ""), "parameter_devices": " ".join(devices),
            "linear4bit_layers": four_bit, "vision_linear4bit_layers": vision_four_bit, "blank_tokens": blank,
            "grid_token_count": expected, "page": kind, "chars": len(pages[kind]), "max_new_tokens": limit,
            "new_tokens": r.new_tokens, "hit_limit": r.hit_limit, "latency_s": f"{r.latency_s:.3f}",
            "cer": f"{score.cer(pages[kind], r.text):.4f}", "peak_gib": f"{peak:.3f}", "passes": ok,
        })
        print(f"{reader_cfg['name']:14s} {kind:7s} {len(pages[kind])} chars, {r.new_tokens} new tokens "
              f"(limit {limit}{', HIT' if r.hit_limit else ''}), {r.latency_s:.1f} s, CER {score.cer(pages[kind], r.text):.4f}\n"
              f"{'':15s}| {r.text[:160]!r}", flush=True)
    print(f"{reader_cfg['name']:14s} parameters on {devices}; 4-bit linear layers {four_bit} (vision tower {vision_four_bit}); "
          f"blank page {blank} tokens, grid {expected}; peak {peak:.2f} GiB; {'PASS' if ok else 'FAIL'}", flush=True)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/readers_study2.yaml")
    p.add_argument("--render-config", default="configs/render_defaults.yaml")
    p.add_argument("--readers", nargs="+", default=["qwen7b", "internvl3_2b"])
    p.add_argument("--font", required=True, help="the mono face chosen by step 7.1")
    p.add_argument("--font-size", type=float, required=True, help="its size at advance P (font_screen.csv, size_px)")
    p.add_argument("--out", default="reports/tables/study2_prerun")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_args(argv)
    dirty_at_start = git_dirty()  # the manifest's own git_dirty is taken when it is written, after the readers
    by_name = {r["name"]: r for r in yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["readers"]}
    defaults = yaml.safe_load(Path(args.render_config).read_text(encoding="utf-8"))
    book = text_source.strip_gutenberg_boilerplate(text_source.load_book(BOOK))
    prose = text_source.normalise_text(text_source.skip_front_matter(book, MARKER, OCCURRENCE))
    rows = []
    for name in args.readers:
        reader_cfg = by_name[name]
        cfg = m0_cfg(defaults, reader_cfg, args.font, args.font_size)
        fits = lambda t: render.fits_page(t, cfg)  # noqa: E731
        pages = {"random": text_source.paginate(text_source.random_text(100_000, PILOT_SEED), fits, 1)[0],
                 "prose": text_source.paginate(prose, fits, 1)[0]}
        rows.extend(check(reader_cfg, pages, cfg))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_cfg = {"readers": [by_name[n] for n in args.readers], "font": args.font, "font_size_px": args.font_size,
               "allowances_px": list(ALLOWANCES), "pilot_seed": PILOT_SEED, "git_dirty_at_start": dirty_at_start,
               "status": "pre-run step 7.2"}
    write_manifest(out / "feasibility", run_cfg, run_id=new_run_id("study2", "feasibility"))
    with open(out / "feasibility.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
