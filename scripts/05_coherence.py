# PURPOSE: Reproduce the coherence table of build guide section 2.2 (D10) from the repository's layout code, without a model.
# WHAT GOES HERE: The first 2,500 glyphs of Persuasion in Qwen's 308 px block at 16 px, in both render modes: glyphs per line, R_1 and R_2 at 14 px, R_1 at 28 and 112 px, the per-class R_1(14) and the straddle share over shifts 0 to 13, each checked against the table.
# FILLED IN AT STEP: W1.10.
# DO NOT: Compute coherence from a second implementation of the layout, or per phase: a common shift changes no |R|. Do not load a reader.
"""Step W1.10: the coherence of the two layouts (D10).

The page-level CER can follow the block phase only if the glyphs share a phase relative to the grid. This
script lays out the same 2,500 glyphs in the conventional and the patch_aware layout, reads each glyph's
ink box off the layout (render.glyph_ink_boxes) and prints, for each layout, the glyphs per line, the
pooled coherence R_k(T) at the periods of Qwen's grid, the per-class coherence and the straddle share at
every whole-pixel shift from 0 to 13, next to the values of the build guide's table. The patch_aware row
must match to three decimals and its per-class value must be 1 to 1e-12, because every pen position in
that layout is a whole pixel; the conventional row may differ by about 0.01, because the repository loads
each glyph at its own fractional phase while the table used phase-0 bitmaps. The script exits 1 if a
value is outside its tolerance.
Run it from the repository root: python scripts/05_coherence.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

from mrr import grid, render, text_source

BOOK = Path("data/raw/texts/pg105-images-3.epub")  # Persuasion, as in the equivariance run
MARKER, OCCURRENCE = "CHAPTER I.", 2  # where its front matter ends (data/raw/texts/README.md)
READER = "qwen"
GLYPHS = 2500
BLOCK = {"max_phase_x_px": 112, "max_phase_y_px": 0}  # 448 - 2 x 14 - 112 = 308 px wide
PATCH_PX = 14
PERIODS = ((14, 1), (14, 2), (28, 1), (112, 1))  # (period, harmonic) of the pooled coherences
SHIFTS = range(PATCH_PX)
# The table of section 2.2, per layout: glyphs per line, R_1(14), R_2(14), R_1(28), R_1(112), the per-class
# R_1(14), and the smallest and largest straddle share over the shifts, with the tolerance of the row.
NAMES = ("glyphs per line", "R1(14)", "R2(14)", "R1(28)", "R1(112)", "per-class R1(14)",
         "smallest straddle share", "largest straddle share")
EXPECTED = {
    "conventional": ((35.7, 0.022, 0.019, 0.013, 0.104, 0.081, 0.384, 0.430), 0.015),
    "patch_aware": ((16.4, 0.986, 0.947, 0.022, 0.088, 1.000, 0.000, 1.000), 0.0005),
}
GLYPHS_PER_LINE_TOL = 0.05  # the table gives one decimal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/readers.yaml")
    p.add_argument("--render-config", default="configs/render_defaults.yaml")
    return p.parse_args()


def first_glyphs(text: str, n: int) -> str:
    """The shortest prefix of text holding n glyphs, whitespace excluded."""
    count = 0
    for i, c in enumerate(text):
        if not c.isspace():
            count += 1
            if count == n:
                return text[: i + 1]
    raise ValueError(f"the text holds fewer than {n} glyphs")


def natural_text() -> str:
    """Persuasion after the front-matter skip, normalised as every page of the sweep is."""
    book = text_source.strip_gutenberg_boilerplate(text_source.load_book(BOOK))
    return text_source.normalise_text(text_source.skip_front_matter(book, MARKER, OCCURRENCE))


def measure(text: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """One layout's row: the ink boxes along x, their coherences and the straddle share at every shift."""
    boxes = render.glyph_ink_boxes(text, cfg)
    centres = [b.start_px + b.size_px / 2 for b in boxes]
    lines = render.layout_page(text, cfg)[-1].line + 1
    return {
        "glyphs": len(boxes),
        "lines": lines,
        "glyphs_per_line": len(boxes) / lines,
        "pooled": [grid.coherence(centres, period, k) for period, k in PERIODS],
        "per_class": grid.class_coherence(boxes, PATCH_PX),
        "straddle": [grid.straddle_share(boxes, PATCH_PX, s) for s in SHIFTS],
    }


def row_values(m: Mapping[str, Any]) -> tuple[float, ...]:
    """The row's cells in the order of NAMES."""
    return (m["glyphs_per_line"], *m["pooled"], m["per_class"], min(m["straddle"]), max(m["straddle"]))


def print_table(results: Mapping[str, Mapping[str, Any]]) -> None:
    print("| Layout | Glyphs per line | R1(14) | R2(14) | R1(28) | R1(112) | Per-class R1(14) "
          "| Straddle share over shifts 0 to 13 |")
    print("|---|---|---|---|---|---|---|---|")
    for mode, m in results.items():
        s = m["straddle"]
        low, high = min(range(len(s)), key=s.__getitem__), max(range(len(s)), key=s.__getitem__)
        pooled = " | ".join(f"{r:.3f}" for r in m["pooled"])
        print(f"| {mode} | {m['glyphs_per_line']:.1f} | {pooled} | {m['per_class']:.3f} "
              f"| {s[low]:.3f} at shift {low} to {s[high]:.3f} at shift {high} |")
    for mode, m in results.items():
        print(f"{mode}: {m['glyphs']} glyphs on {m['lines']} lines; straddle share by shift: "
              + " ".join(f"{v:.3f}" for v in m["straddle"]))


def checks(results: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, float, float, float]]:
    """(name, measured, expected, tolerance) for every cell of the table and the two exact patch_aware facts."""
    out = []
    for mode, (expected, tol) in EXPECTED.items():
        for name, got, exp in zip(NAMES, row_values(results[mode]), expected):
            out.append((f"{mode} {name}", got, exp, GLYPHS_PER_LINE_TOL if name == NAMES[0] else tol))
    pa = results["patch_aware"]
    out.append(("patch_aware per-class R1(14) is 1 to 1e-12", pa["per_class"], 1.0, 1e-12))
    out.append(("patch_aware straddle share at shift 0", pa["straddle"][0], 0.0, 0.0005))
    out.append(("patch_aware straddle share at shift 7", pa["straddle"][7], 1.0, 0.0005))
    return out


def main() -> None:
    args = parse_args()
    readers = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["readers"]
    reader_cfg = next(r for r in readers if r["name"] == READER)
    defaults = yaml.safe_load(Path(args.render_config).read_text(encoding="utf-8"))
    text = first_glyphs(natural_text(), GLYPHS)
    results = {}
    for mode in render.RENDER_MODES:
        cfg = grid.resolve_render_cfg(defaults, reader_cfg, render_mode=mode, **BLOCK)
        results[mode] = measure(text, cfg)
        if results[mode]["glyphs"] != GLYPHS:
            sys.exit(f"{mode}: {results[mode]['glyphs']} ink boxes for {GLYPHS} glyphs")
    width, _ = render.block_size_px(cfg)
    print(f"{READER}: {width:.0f} px block, {cfg['font_size_px']} px {Path(str(cfg['font_file'])).name}, "
          f"pitch {cfg['line_pitch_px']} px, phase 0; the first {GLYPHS} glyphs of Persuasion")
    print_table(results)
    failed = 0
    for name, got, exp, tol in checks(results):
        ok = abs(got - exp) <= tol
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {got:.6f}, table {exp:.3f}, tolerance {tol:g}")
    if failed:
        sys.exit(f"{failed} values outside their tolerance")
    print("every value is within its tolerance")


if __name__ == "__main__":
    main()
