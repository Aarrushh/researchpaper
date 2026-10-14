# PURPOSE: Check the reference renderer against rules R01 to R06 and the Conformance table of spec/RENDERING_SPEC.md, and write the table those rules cite as EVIDENCE.
# WHAT GOES HERE: One row per check (rule, check, expected, observed, passed) in reports/tables/contract/contract_checks.csv with the folder's manifest: every number the spec states for R01 to R06 and for the conformance tolerances, recomputed from the renderer and from the spec's equations.
# FILLED IN AT STEP: W4.1.
# DO NOT: Round with round(), np.round or torch.round: the spec rounds once, with floor(v + 0.5). Do not load a reader; R07 and R08 are the reader facts of scripts/01_check_readers.py. Do not change a number in the spec to make a row pass without a notebook entry saying why the earlier number was wrong.
"""Step W4.1: the contract checks of spec/RENDERING_SPEC.md (build guide section 4.1).

    python scripts/08_contract_checks.py [--config configs/readers.yaml] [--render-config configs/render_defaults.yaml]
                                         [--out-dir reports/tables/contract] [--root .]

Every number the specification states for rules R01 to R06 and for the Conformance table is recomputed here, from the
repository's renderer where the rule describes the renderer and from the spec's own equations where it states
arithmetic, and written as one row of contract_checks.csv: rule, check, expected, observed, passed. The page the
layout and rounding checks use is the first conventional Qwen page of Persuasion at 16 px, the page the reader tests
render (tests/conftest.py). The conformance facts (the 1/510 gap of coverage_srgb, single-precision agreement in
coverage_srgb, the triples single precision moves in coverage_linear) are computed over all 16,777,216 (S, T, B) byte
triples. The folder's manifest.json is written before the CSV, so its git_dirty flag describes the tree the checks
ran on; the script exits 1 if any row failed.
Run it from the repository root: make contract.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import freetype
import numpy as np
import yaml

from mrr import grid, render, text_source
from mrr.color import linear_to_srgb, srgb_to_linear
from mrr.manifest import new_run_id, write_manifest

BOOK = "pg105"  # Persuasion: the page of tests/conftest.py
READER = "qwen"
SIZE_PX = 16
COLUMNS = ("rule", "check", "expected", "observed", "passed")

# R01
FREETYPE_VERSION = (2, 13, 2)
ADVANCES_16PX = {"m": 13.264, "i": 3.936}  # linear advances in px, to ADVANCE_TOL
ADVANCE_TOL = 1e-3
# R02
PHASE_EXAMPLES = {3.25: (3, 16), 13.0078125: (13, 1)}  # phase in px: (whole pixels, 64ths)
# R03
PITCHES = {12: 15, 14: 18, 16: 20}  # unaligned line pitch per font size
DESCENDER_ROOM = {12: 4, 14: 5, 16: 6}
ASCENDER_16PX = 16
ALLOWANCES = {"qwen": (105, 26), "smolvlm": (40, 40), "smolvlm500m": (60, 60)}  # the paper's (A_x, A_y) in px
BLOCKS = {"qwen": (315, 394), "smolvlm": (310, 310), "smolvlm500m": (420, 420)}  # the text block (width, height) in px
FRACTIONAL = (3.25, 2.5)  # a fractional phase pair: (208, 160) in 1/64 px
# R04 and R05
SUMS = ((27, 100), (28, 100), (200, 100))  # pairs of glyph bytes at one pixel: S = 127, 128 and 255
# R05
BITMAP = np.array([[0, 64], [128, 255]])
BLACK, WHITE = (0, 0, 0), (255, 255, 255)
RED, CREAM = (200, 30, 60), (250, 245, 230)
EXAMPLES = (
    (BLACK, WHITE, {"binary": [[255, 255], [0, 0]], "coverage_srgb": [[255, 191], [127, 0]],
                    "coverage_linear": [[255, 224], [187, 0]]}),
    (WHITE, BLACK, {"binary": [[0, 0], [255, 255]], "coverage_srgb": [[0, 64], [128, 255]],
                    "coverage_linear": [[0, 137], [188, 255]]}),
)
RED_ON_CREAM = {"binary": [200, 30, 60], "coverage_srgb": [225, 137, 145], "coverage_linear": [227, 181, 173]}
GAMMA22_EXAMPLES = (11, 27, 2)  # (bytes changed, bytes compared, largest change) when a 2.2 power replaces the sRGB curve
GAMMA22_ALL = 9  # the largest change over all byte triples
# R06
ROUNDING = ((0.5, 128), (-0.1, 0), (1.2, 255))
# Conformance
HALF_INTEGER_GAP = 1 / 510  # equation (2): 255 * o is never closer than this to a rounding boundary in coverage_srgb
TRIPLES = 256**3
SINGLE_PRECISION_SHARE = 1e-4  # coverage_linear: fewer than one triple in 10,000 may move, and by one level


@dataclasses.dataclass(frozen=True)
class Check:
    """One row of the table."""

    rule: str
    check: str
    expected: Any
    observed: Any
    passed: bool

    def row(self) -> dict[str, str]:
        return {"rule": self.rule, "check": self.check, "expected": _text(self.expected),
                "observed": _text(self.observed), "passed": str(self.passed)}


def _text(value: Any) -> str:
    """One cell: arrays as lists, floats with their full repr, everything else as str."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def check(rule: str, name: str, expected: Any, observed: Any, passed: bool | None = None) -> Check:
    """A row; without passed, it passes when observed equals expected (arrays element for element)."""
    if passed is None:
        if isinstance(expected, (list, tuple, np.ndarray)):
            passed = np.asarray(expected).shape == np.asarray(observed).shape and bool(
                np.array_equal(np.asarray(expected), np.asarray(observed)))
        else:
            passed = expected == observed
    return Check(rule, name, expected, observed, bool(passed))


def to_bytes(values: Any) -> np.ndarray:
    """R06: floor(255 * clip(o, 0, 1) + 0.5), as integers."""
    return np.floor(255 * np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0) + 0.5).astype(np.int64)


# R01: glyph coverage


def r01_checks(font: str) -> list[Check]:
    """The rasteriser and its flags, the linear advances, and the grey levels of a bitmap."""
    out = [check("R01", "FreeType version", FREETYPE_VERSION, tuple(freetype.version()))]
    for name in ("FT_LOAD_RENDER", "FT_LOAD_NO_HINTING", "FT_LOAD_NO_BITMAP"):
        out.append(check("R01", f"load flags include {name}", True, bool(render.FLAGS & getattr(freetype, name))))
    out.append(check("R01", "load target normal (bits 16 to 19 of the flags are 0)", 0, (render.FLAGS >> 16) & 0xF))
    for char, expected in ADVANCES_16PX.items():
        observed = render.load_glyph(font, SIZE_PX, char).advance_px
        out.append(check("R01", f"linear advance of {char!r} at {SIZE_PX} px, to {ADVANCE_TOL:g} px", expected, observed,
                         abs(observed - expected) <= ADVANCE_TOL))
    glyph = render.load_glyph(font, SIZE_PX, "o")
    levels = to_bytes(glyph.coverage)
    grey = int(((levels >= 1) & (levels <= 254)).sum())
    out.append(check("R01", f"bitmap of 'o' at {SIZE_PX} px holds grey levels 1 to 254", "some pixels",
                     f"{grey} of {levels.size} pixels", grey > 0))
    out.append(check("R01", "coverage values are bytes / 255", True,
                     bool(np.all(np.abs(glyph.coverage * 255 - levels) < 1e-9))))
    out.append(check("R01", "the space has an empty bitmap", (0, 0), render.load_glyph(font, SIZE_PX, " ").coverage.shape))
    return out


# R02: phase


def r02_checks(font: str, text: str, cfg: Mapping[str, Any]) -> list[Check]:
    """The rounding to 1/64 px, the 64 fractional bitmaps, and a whole-patch phase as a pure shift of the page."""
    out = [check("R02", f"phase {phi} px splits into (whole pixels, 64ths)", expected, grid.split_phase_64(phi))
           for phi, expected in PHASE_EXAMPLES.items()]
    for axis in ("x", "y"):
        bitmaps = set()
        for sixty_fourths in range(64):
            glyph = render.load_glyph(font, SIZE_PX, "o", **{f"phase_{axis}_64": sixty_fourths})
            bitmaps.add((glyph.coverage.shape, glyph.coverage.tobytes()))
        out.append(check("R02", f"distinct bitmaps of 'o' at {SIZE_PX} px over the 64 fractional phases along {axis}",
                         64, len(bitmaps)))
    patch = int(cfg["margin_px"])
    base = render.render_page(text, {**cfg, "phase_x_px": 0.0, "phase_y_px": 0.0})
    background = np.asarray(cfg["background_color"], dtype=np.uint8)
    for axis in ("x", "y"):
        page = render.render_page(text, {**cfg, "phase_x_px": 0.0, "phase_y_px": 0.0, f"phase_{axis}_px": float(patch)})
        if axis == "x":
            moved, vacated, reference = page[:, patch:], page[:, :patch], base[:, :-patch]
        else:
            moved, vacated, reference = page[patch:], page[:patch], base[:-patch]
        differing = int((moved != reference).any(axis=2).sum()) + int((vacated != background).any(axis=2).sum())
        out.append(check("R02", f"a phase of one patch ({patch} px) along {axis} is the phase-zero page moved by {patch} px",
                         "0 differing pixels", f"{differing} differing pixels", differing == 0))
    return out


# R03: layout


def greedy_lines(words: Sequence[str], advance: Callable[[str], float], width: float) -> list[str]:
    """R03's word wrap, written from the spec: a word joins the line if line + space + word fits, else starts one."""
    lines: list[str] = []
    line, used = "", 0.0
    space = advance(" ")
    for word in words:
        word_width = sum(advance(c) for c in word)
        if line and used + space + word_width <= width:
            line, used = f"{line} {word}", used + space + word_width
        else:
            if line:
                lines.append(line)
            line, used = word, word_width
    if line:
        lines.append(line)
    return lines


def advance_sum_origins(text: str, cfg: Mapping[str, Any], font: str) -> list[tuple[str, int, int, int]]:
    """R03's formula for the conventional layout: each glyph's (char, x64, y64, line) from the linear advances."""
    layout_font = render.LayoutFont(font, cfg["font_size_px"])
    width, _ = render.block_size_px(cfg)
    lines = render.layout_lines(text, layout_font, width, cfg["line_pitch_px"])
    dx = math.floor(64 * cfg["margin_px"] + 0.5) + math.floor(64 * cfg["phase_x_px"] + 0.5)
    dy = math.floor(64 * cfg["margin_px"] + 0.5) + math.floor(64 * cfg["phase_y_px"] + 0.5)
    origins = []
    for k, (line, _) in enumerate(lines):
        baseline = layout_font.ascender_px + k * cfg["line_pitch_px"]
        pen = 0.0
        for char in line:
            if char != " ":
                origins.append((char, dx + math.floor(64 * pen + 0.5), dy + math.floor(64 * baseline + 0.5), k))
            pen += render.load_glyph(font, cfg["font_size_px"], char).advance_px
    return origins


def r03_checks(font: str, text: str, cfg: Mapping[str, Any], defaults: Mapping[str, Any],
               readers: Sequence[Mapping[str, Any]]) -> list[Check]:
    """Pitches, font metrics, block sizes, the word wrap, the origin formula, whole-pixel slots and the line bound."""
    out = [check("R03", f"unaligned line pitch at {size} px", pitch, render.unaligned_pitch_px(size))
           for size, pitch in PITCHES.items()]
    for size, room in DESCENDER_ROOM.items():
        out.append(check("R03", f"descender room at {size} px", room, render.LayoutFont(font, size).descender_room_px))
    out.append(check("R03", f"ascender at {SIZE_PX} px", ASCENDER_16PX, render.LayoutFont(font, SIZE_PX).ascender_px))
    for reader in readers:
        name = reader["name"]
        ax, ay = ALLOWANCES[name]
        resolved = grid.resolve_render_cfg(defaults, reader, font_file=font, max_phase_x_px=ax, max_phase_y_px=ay)
        width, height = render.block_size_px(resolved)
        out.append(check("R03", f"{name}: text block (width, height) with allowances ({ax}, {ay}) px", BLOCKS[name],
                         (int(width), int(height))))

    conventional = {**cfg, "render_mode": "conventional", "phase_x_px": 0.0, "phase_y_px": 0.0}
    layout_font = render.LayoutFont(font, cfg["font_size_px"])
    width, height = render.block_size_px(conventional)
    lines = [line for line, _ in render.layout_lines(text, layout_font, width, cfg["line_pitch_px"])]
    greedy = greedy_lines(text.split(), layout_font.advance_px, width)
    differ = sum(a != b for a, b in zip(lines, greedy)) + abs(len(lines) - len(greedy))
    out.append(check("R03", f"the {len(lines)} lines of the page are the greedy word wrap of the spec", "identical lines",
                     "identical lines" if differ == 0 else f"{differ} lines differ", differ == 0))
    for phase in ((0.0, 0.0), FRACTIONAL):
        at_phase = {**conventional, "phase_x_px": phase[0], "phase_y_px": phase[1]}
        placed = [(g.char, g.x_64, g.baseline_64, g.line) for g in render.layout_page(text, at_phase)]
        formula = advance_sum_origins(text, at_phase, font)
        mismatches = sum(a != b for a, b in zip(placed, formula)) + abs(len(placed) - len(formula))
        out.append(check("R03", f"conventional origins and baselines follow the advance-sum formula at phase {phase}",
                         f"0 of {len(formula)} glyphs differ", f"{mismatches} of {len(placed)} glyphs differ",
                         mismatches == 0))
    patch_aware = {**cfg, "render_mode": "patch_aware", "phase_y_px": 0.0}
    for phi in (0.0, 7.0):
        placed = render.layout_page(text, {**patch_aware, "phase_x_px": phi})
        fractional = sum(g.x_64 % 64 != 0 for g in placed)
        out.append(check("R03", f"patch-aware origin columns are whole pixels at phase {phi}", f"0 of {len(placed)}",
                         f"{fractional} of {len(placed)}", fractional == 0))
    at_zero = render.layout_page(text, conventional)
    at_fraction = render.layout_page(text, {**conventional, "phase_x_px": FRACTIONAL[0], "phase_y_px": FRACTIONAL[1]})
    same = [(g.char, g.line) for g in at_zero] == [(g.char, g.line) for g in at_fraction]
    shifts = {(b.x_64 - a.x_64, b.baseline_64 - a.baseline_64) for a, b in zip(at_zero, at_fraction)}
    expected_shift = (math.floor(64 * FRACTIONAL[0] + 0.5), math.floor(64 * FRACTIONAL[1] + 0.5))
    out.append(check("R03", f"the same text at phases (0, 0) and {FRACTIONAL}: the same glyphs on the same lines", True, same))
    out.append(check("R03", "every origin moved by the same (x, y) in 1/64 px between those phases", {expected_shift}, shifts))
    room = height - layout_font.ascender_px - layout_font.descender_room_px
    bound = math.floor(room / cfg["line_pitch_px"]) + 1
    out.append(check("R03", f"lines on the page at most floor((h - ascender - d) / pitch) + 1 = {bound}", True,
                     len(lines) <= bound and render.fits_page(text, conventional)))
    return out


# R04: accumulate


def integer_coverage(text: str, cfg: Mapping[str, Any], font: str) -> tuple[np.ndarray, np.ndarray]:
    """Equation (1) of R04 before the clip: the integer sum of the glyph bytes at every pixel, and how many bitmaps touch it."""
    height, width = cfg["page_height_px"], cfg["page_width_px"]
    total = np.zeros((height, width), dtype=np.int64)
    touched = np.zeros((height, width), dtype=np.int64)
    for g in render.layout_page(text, cfg):
        glyph = render.load_glyph(font, cfg["font_size_px"], g.char, g.x_64 & 63, g.baseline_64 & 63)
        k = to_bytes(glyph.coverage)
        r, q = (g.baseline_64 >> 6) - glyph.top, (g.x_64 >> 6) + glyph.left
        rows, cols = k.shape
        r0, r1, c0, c1 = max(0, r), min(height, r + rows), max(0, q), min(width, q + cols)
        if r0 < r1 and c0 < c1:
            total[r0:r1, c0:c1] += k[r0 - r : r1 - r, c0 - q : c1 - q]
            touched[r0:r1, c0:c1] += 1
    return total, touched


def r04_checks(text: str, cfg: Mapping[str, Any], font: str) -> list[Check]:
    """The sums the spec names, and the page's coverage as integers against the reference's fractions."""
    out = []
    for a, b in SUMS:
        fractional = np.minimum(1.0, a / 255.0 + b / 255.0)  # the reference accumulates fractions, clipped at 1
        out.append(check("R04", f"glyph bytes {a} and {b} at one pixel: S", min(255, a + b), int(to_bytes(fractional)[()])))
    total, touched = integer_coverage(text, cfg, font)
    clipped = np.minimum(255, total)
    reference = render.render_coverage(text, cfg)
    gap = float(np.abs(reference - clipped / 255.0).max())
    out.append(check("R04", "page coverage: integer sum / 255 against the reference's clipped fractional sums, largest difference",
                     "< 1e-12", gap, gap < 1e-12))
    out.append(check("R04", "page coverage never above 1", True, bool(reference.max() <= 1.0)))
    overlaps = int((touched >= 2).sum())
    out.append(check("R04", "pixels of the page inside two or more glyph boxes", "some pixels", f"{overlaps} pixels",
                     overlaps > 0))
    out.append(check("R04", "pixels of the page where the clip acts (unclipped sum above 255)", "reported",
                     f"{int((total > 255).sum())} pixels", True))
    return out


# R05: composite once


def composite(coverage: np.ndarray, text_rgb: Sequence[int], background_rgb: Sequence[int], mode: str) -> np.ndarray:
    """One composite of a coverage array on a page of the background colour, through the renderer, then R06."""
    page = np.empty((*coverage.shape, 3))
    page[...] = np.asarray(background_rgb, dtype=np.float64) / 255.0
    render.composite_coverage(page, coverage, 0, 0, text_rgb, mode)
    return to_bytes(page)


def power22(coverage: np.ndarray, text_rgb: Sequence[int], background_rgb: Sequence[int]) -> np.ndarray:
    """coverage_linear with a 2.2 power in place of the sRGB curve: the approximation D24 rules out."""
    t = np.asarray(text_rgb, dtype=np.float64) / 255.0
    b = np.asarray(background_rgb, dtype=np.float64) / 255.0
    c = coverage[..., None]
    return to_bytes((c * t**2.2 + (1 - c) * b**2.2) ** (1 / 2.2))


def r05_checks() -> list[Check]:
    """The worked examples, byte for byte, and the sRGB curve against a 2.2 power on them."""
    out = []
    coverage = BITMAP / 255.0
    for t, b, expected in EXAMPLES:
        for mode, want in expected.items():
            got = composite(coverage, t, b, mode)
            grey = bool(np.all(got == got[..., :1]))  # the three channels agree for grey colours
            out.append(check("R05", f"{mode}: bitmap {BITMAP.tolist()} of {t} on {b}", want, got[..., 0],
                             grey and np.array_equal(got[..., 0], np.asarray(want))))
    for mode, want in RED_ON_CREAM.items():
        out.append(check("R05", f"{mode}: byte 128 of {RED} on {CREAM}", want, composite(np.array([[128 / 255.0]]), RED, CREAM, mode)[0, 0]))
    for a, b in SUMS:
        s = min(255, a + b)
        for mode in render.AA_MODES:
            if mode == "binary":
                want = BLACK if s >= 128 else WHITE
            elif s == 255:
                want = BLACK
            else:
                continue
            got = composite(np.array([[s / 255.0]]), BLACK, WHITE, mode)[0, 0]
            out.append(check("R05", f"{mode}: glyph bytes {a} and {b} (S = {s}), black on white", list(want), got))
    compared = changed = largest = 0
    for t, b in ((BLACK, WHITE), (WHITE, BLACK), (RED, CREAM)):
        cov = coverage if (t, b) != (RED, CREAM) else np.array([[128 / 255.0]])
        diff = np.abs(composite(cov, t, b, "coverage_linear") - power22(cov, t, b))
        compared, changed, largest = compared + diff.size, changed + int((diff > 0).sum()), max(largest, int(diff.max()))
    out.append(check("R05", "a 2.2 power in place of the sRGB curve on the coverage_linear examples: "
                     "(bytes changed, bytes compared, largest change in levels)", GAMMA22_EXAMPLES, (changed, compared, largest)))
    return out


# R06: round once


def r06_checks(text: str, cfg: Mapping[str, Any]) -> list[Check]:
    """The rounding examples, the page as the rounded composite of its coverage, and truncation as the wrong rounding."""
    out = [check("R06", f"floor(255 * clip({o}, 0, 1) + 0.5)", want, int(to_bytes(o)[()])) for o, want in ROUNDING]
    for mode in render.AA_MODES:
        at_mode = {**cfg, "aa_mode": mode}
        coverage = render.render_coverage(text, at_mode)
        page = np.empty((*coverage.shape, 3))
        page[...] = np.asarray(at_mode["background_color"], dtype=np.float64) / 255.0
        render.composite_coverage(page, coverage, 0, 0, at_mode["text_color"], mode)
        once = to_bytes(page)
        rendered = render.render_page(text, at_mode).astype(np.int64)
        out.append(check("R06", f"{mode}: the page is the rounded composite of its coverage", "0 differing bytes",
                         f"{int((rendered != once).sum())} differing bytes", np.array_equal(rendered, once)))
        truncated = np.floor(255 * np.clip(page, 0.0, 1.0)).astype(np.int64)
        changed = int((truncated != once).sum())
        if mode == "binary":
            out.append(check("R06", f"{mode}: truncation in place of floor(v + 0.5) changes", "0 bytes (the values are integers)",
                             f"{changed} bytes", changed == 0))
        else:
            out.append(check("R06", f"{mode}: truncation in place of floor(v + 0.5) changes", "some bytes", f"{changed} bytes",
                             changed > 0))
    return out


# Conformance: the three facts behind the tolerance table, over every (S, T, B) byte triple


def decode(v: np.ndarray, dtype: type) -> np.ndarray:
    """D(v) of the spec in the given precision (the renderer's own functions work in double precision)."""
    v = np.asarray(v, dtype=dtype)
    return np.where(v <= dtype(0.04045), v / dtype(12.92), ((v + dtype(0.055)) / dtype(1.055)) ** dtype(2.4))


def encode(light: np.ndarray, dtype: type) -> np.ndarray:
    """E(l) of the spec in the given precision."""
    light = np.asarray(light, dtype=dtype)
    knee = dtype(0.04045) / dtype(12.92)
    return np.where(light <= knee, dtype(12.92) * light, dtype(1.055) * light ** (dtype(1) / dtype(2.4)) - dtype(0.055))


def bytes_in_precision(o: np.ndarray, dtype: type) -> np.ndarray:
    """R06 in the given precision."""
    return np.floor(dtype(255) * np.clip(np.asarray(o, dtype=dtype), dtype(0), dtype(1)) + dtype(0.5)).astype(np.int64)


def conformance_checks() -> list[Check]:
    """Equation (2)'s gap, single precision in both coverage modes, and the 2.2 power, over all 16,777,216 triples."""
    T = np.arange(256, dtype=np.float64)[:, None]
    B = np.arange(256, dtype=np.float64)[None, :]
    t64, b64 = T / 255.0, B / 255.0
    light_t64, light_b64 = srgb_to_linear(t64), srgb_to_linear(b64)
    t32, b32 = np.float32(T) / np.float32(255), np.float32(B) / np.float32(255)
    light_t32, light_b32 = decode(t32, np.float32), decode(b32, np.float32)
    power_t, power_b = t64**2.2, b64**2.2
    gap = 1.0
    srgb_moved = 0
    moved = {"a": 0, "b": 0}
    largest = {"a": 0, "b": 0}
    power_largest = 0
    for s in range(256):
        n = s * T + (255 - s) * B  # integers, exact in double precision
        v = n / 255.0
        gap = min(gap, float(np.abs(v - np.floor(v) - 0.5).min()))
        c64 = s / 255.0
        srgb_reference = to_bytes(c64 * t64 + (1 - c64) * b64)
        c32 = np.float32(s) / np.float32(255)
        srgb_moved += int((srgb_reference != bytes_in_precision(c32 * t32 + (1 - c32) * b32, np.float32)).sum())
        linear_reference = to_bytes(linear_to_srgb(c64 * light_t64 + (1 - c64) * light_b64))  # the renderer's arithmetic
        mixes = {"a": c32 * light_t32 + (1 - c32) * light_b32, "b": light_b32 + c32 * (light_t32 - light_b32)}
        for form, mix in mixes.items():
            diff = np.abs(linear_reference - bytes_in_precision(encode(mix, np.float32), np.float32))
            moved[form] += int((diff > 0).sum())
            largest[form] = max(largest[form], int(diff.max()))
        power = to_bytes((c64 * power_t + (1 - c64) * power_b) ** (1 / 2.2))
        power_largest = max(power_largest, int(np.abs(linear_reference - power).max()))
    out = [
        check("Conformance", "coverage_srgb: smallest distance of 255 * o from a half-integer over all byte triples (equation (2))",
              f"1/510 = {HALF_INTEGER_GAP!r}", gap, abs(gap - HALF_INTEGER_GAP) < 1e-12),
        check("Conformance", f"coverage_srgb: triples where single precision differs from double precision, of {TRIPLES:,}", 0,
              srgb_moved),
    ]
    limit = int(SINGLE_PRECISION_SHARE * TRIPLES)
    for form, name in (("a", "c * D(t) + (1 - c) * D(b)"), ("b", "D(b) + c * (D(t) - D(b))")):
        out.append(check("Conformance", f"coverage_linear in single precision, mix written {name}: "
                         f"(triples moved, largest change in levels) of {TRIPLES:,}", f"(fewer than {limit}, at most 1)",
                         (moved[form], largest[form]), moved[form] < limit and largest[form] <= 1))
    out.append(check("Conformance", "a 2.2 power in place of the sRGB curve over all byte triples: largest change in levels",
                     GAMMA22_ALL, power_largest))
    return out


# The script


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/readers.yaml")
    p.add_argument("--render-config", default="configs/render_defaults.yaml")
    p.add_argument("--out-dir", default="reports/tables/contract")
    p.add_argument("--root", default=".", help="repository root; the font and the book are resolved under it")
    return p.parse_args(argv)


def write_table(path: Path, checks: Sequence[Check]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(c.row() for c in checks)


def main(argv: Sequence[str] | None = None) -> int:
    sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_args(argv)
    root = Path(args.root)
    readers = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["readers"]
    defaults = yaml.safe_load(Path(args.render_config).read_text(encoding="utf-8"))
    font = str(root / defaults["font_file"])
    reader_cfg = next(r for r in readers if r["name"] == READER)
    cfg = grid.resolve_render_cfg(defaults, reader_cfg, font_file=font)
    text = text_source.paginate(text_source.book_text(BOOK, root), lambda candidate: render.fits_page(candidate, cfg), 1)[0]
    checks = [
        *r01_checks(font),
        *r02_checks(font, text, cfg),
        *r03_checks(font, text, cfg, defaults, readers),
        *r04_checks(text, cfg, font),
        *r05_checks(),
        *r06_checks(text, cfg),
        *conformance_checks(),
    ]
    for c in checks:
        print(f"{'PASS' if c.passed else 'FAIL'} {c.rule} {c.check}: expected {_text(c.expected)}, observed {_text(c.observed)}")
    failed = sum(not c.passed for c in checks)
    out_dir = Path(args.out_dir)
    run_cfg = {
        "spec": "spec/RENDERING_SPEC.md",
        "rules": ["R01", "R02", "R03", "R04", "R05", "R06", "Conformance"],
        "reader": READER,
        "book": BOOK,
        "page_characters": len(text),
        "render": {**cfg, "font_file": defaults["font_file"]},
        "freetype_version": ".".join(str(part) for part in freetype.version()),
        "checks": len(checks),
        "failed": failed,
    }
    manifest = write_manifest(out_dir, run_cfg, run_id=new_run_id("contract", "checks"))  # before the table: git_dirty
    table = out_dir / "contract_checks.csv"
    write_table(table, checks)
    print(f"{len(checks) - failed} of {len(checks)} checks passed; table {table.as_posix()}, manifest {manifest.as_posix()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
