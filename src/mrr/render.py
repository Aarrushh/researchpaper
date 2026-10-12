# PURPOSE: The reference renderer: text to page image with controlled phase, pitch and anti-aliasing.
# WHAT GOES HERE: Glyph coverage loading, compositing, line layout, full-page render, grid overlay.
# FILLED IN AT STEP: W1.5.
# DO NOT: Let Pillow draw text directly onto the page; composite coverage yourself so aa_mode and phase are controlled.
"""Reference text renderer.

Learn A3 (sampling), A4 (colour), A5 (compositing) and A6 (font rasterisation) before implementing.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import freetype
import numpy as np

from mrr.color import linear_to_srgb, srgb_to_linear

# Unhinted outlines, rendered to 8-bit coverage; NO_BITMAP skips any embedded bitmap strike (D11).
FLAGS = freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING | freetype.FT_LOAD_NO_BITMAP
IDENT = freetype.Matrix(0x10000, 0, 0, 0x10000)
AA_MODES = ("binary", "coverage_srgb", "coverage_linear")


@dataclass(frozen=True)
class Glyph:
    """One character rasterised at one phase.

    Attributes:
        coverage: float64 array (rows, width) in [0, 1], read-only; (0, 0) for a glyph without ink, such
            as the space.
        left: FreeType bitmap_left, columns from the pen to the bitmap's first column.
        top: FreeType bitmap_top, rows from the bitmap's first row up to the baseline.
        advance_px: linearHoriAdvance / 65536, the unhinted advance in pixels.
    """

    coverage: np.ndarray
    left: int
    top: int
    advance_px: float


@functools.lru_cache(maxsize=32)
def _face(font_path: str, size_64: int) -> freetype.Face:
    """Open a font once per (font file, size in 1/64 px).

    The file is read through a Python stream: freetype-py passes a path to FreeType as UTF-8 bytes,
    which FreeType cannot open on Windows when the path is not ASCII.
    """
    with open(font_path, "rb") as stream:
        face = freetype.Face(stream)
    face.set_char_size(0, size_64, 72, 72)  # at 72 dpi one point is one pixel; the size is in 1/64 px
    return face


@functools.lru_cache(maxsize=65536)
def load_glyph(font_path: str, size_px: float, char: str, phase_x_64: int = 0, phase_y_64: int = 0) -> Glyph:
    """Rasterise one character with its outline moved right and down by a fraction of a pixel.

    The phase moves the outline before FreeType computes coverage, so the coverage values change; whole
    pixels are added by the caller when it places the bitmap (D11). Results are cached per argument
    tuple; load_glyph.cache_clear() empties the cache.

    Args:
        font_path: path to a font file under data/external/fonts/.
        size_px: font size in pixels, rounded to 1/64 px.
        char: a single character.
        phase_x_64: rightward phase in 1/64 px, 0 to 63.
        phase_y_64: downward phase in 1/64 px, 0 to 63.

    Returns:
        The Glyph, whose coverage array is read-only because every later page reuses it.

    Raises:
        ValueError: if a phase is outside 0 to 63, or the font has no glyph for char.
    """
    if not (0 <= phase_x_64 < 64 and 0 <= phase_y_64 < 64):
        raise ValueError(f"phases are in 1/64 px, from 0 to 63; got ({phase_x_64}, {phase_y_64})")
    face = _face(font_path, math.floor(64 * size_px + 0.5))
    if face.get_char_index(char) == 0:
        raise ValueError(f"U+{ord(char):04X} not in {font_path}")
    face.set_transform(IDENT, freetype.Vector(phase_x_64, -phase_y_64))  # FreeType's y points up
    try:
        face.load_char(char, FLAGS)
        g = face.glyph
        bm = g.bitmap
        if bm.rows and bm.width:
            cov = np.frombuffer(bytes(bm.buffer), np.uint8).reshape(bm.rows, abs(bm.pitch))[:, : bm.width] / 255.0
        else:
            cov = np.zeros((0, 0))  # the space has no bitmap
        cov.flags.writeable = False
        return Glyph(cov, g.bitmap_left, g.bitmap_top, g.linearHoriAdvance / 65536.0)
    finally:
        face.set_transform(IDENT, freetype.Vector(0, 0))  # the transform is state on the face


def load_glyph_coverage(font_path: str, size_px: float, char: str) -> np.ndarray:
    """Load the coverage mask for one character at phase zero.

    Args:
        font_path: path to a font file under data/external/fonts/.
        size_px: font size in pixels.
        char: a single character.

    Returns:
        float64 array (h, w) of coverage in [0, 1], read-only.
    """
    return load_glyph(font_path, size_px, char).coverage


def composite_coverage(
    page: np.ndarray,
    coverage: np.ndarray,
    x: int,
    y: int,
    color: Sequence[int],
    aa_mode: str,
) -> np.ndarray:
    """Composite one coverage mask onto the page with its top-left pixel at (x, y).

    Source-over onto an opaque page is a straight mix of the text colour t and the page value b
    (section 1A.3 of the build guide, equations 4 and 5). alpha is 1 where c >= 0.5 and 0 elsewhere for
    "binary", and c itself for the coverage modes. "binary" and "coverage_srgb" mix the stored sRGB
    values; "coverage_linear" mixes in linear light, v(alpha L(t) + (1 - alpha) L(b)). Coverage that falls
    off the page is discarded. Nothing is rounded here: render_page rounds the whole page once (D11).

    Args:
        page: float64 array (H, W, 3), sRGB-encoded values in [0, 1]; written in place.
        coverage: float64 array (h, w) in [0, 1].
        x: column of the mask's left edge, a Python or NumPy integer; sub-pixel placement is
            load_glyph's phase.
        y: row of the mask's top edge, a Python or NumPy integer.
        color: text colour as an sRGB triple from 0 to 255.
        aa_mode: one of "binary", "coverage_srgb", "coverage_linear".

    Returns:
        page, updated in place.

    Raises:
        TypeError: if x or y is not an integer, or page or coverage is not a float64 array.
        ValueError: if aa_mode is unknown, or a shape, value or colour is outside the contract above.
    """
    for name, value in (("x", x), ("y", y)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__} {value!r}")
    if aa_mode not in AA_MODES:
        raise ValueError(f"aa_mode must be one of {AA_MODES}, got {aa_mode!r}")
    for name, array in (("page", page), ("coverage", coverage)):
        if not isinstance(array, np.ndarray) or array.dtype != np.float64:
            raise TypeError(f"{name} must be a float64 array, got {getattr(array, 'dtype', type(array).__name__)}")
        if not np.all((array >= 0) & (array <= 1)):
            raise ValueError(f"{name} values must lie in [0, 1]")
    if page.ndim != 3 or page.shape[2] != 3:
        raise ValueError(f"page must have shape (H, W, 3), got {page.shape}")
    if coverage.ndim != 2:
        raise ValueError(f"coverage must have shape (h, w), got {coverage.shape}")
    t = np.asarray(color, dtype=np.float64)
    if t.shape != (3,) or not np.all((t >= 0) & (t <= 255)):
        raise ValueError(f"color must be three values from 0 to 255, got {color!r}")
    t = t / 255.0

    x, y = int(x), int(y)  # NumPy integers would wrap in the sums below
    height, width = page.shape[:2]
    r0, r1 = max(0, y), min(height, y + coverage.shape[0])
    c0, c1 = max(0, x), min(width, x + coverage.shape[1])
    if r0 >= r1 or c0 >= c1:
        return page
    c = coverage[r0 - y : r1 - y, c0 - x : c1 - x, None]
    alpha = (c >= 0.5).astype(np.float64) if aa_mode == "binary" else c
    b = page[r0:r1, c0:c1]  # a view, so the writes below land in page
    if aa_mode == "coverage_linear":
        b[...] = linear_to_srgb(alpha * srgb_to_linear(t) + (1 - alpha) * srgb_to_linear(b))
    else:
        b[...] = alpha * t + (1 - alpha) * b
    return page


def layout_lines(text: str, font: Any, width_px: int, line_pitch_px: float) -> list[tuple[str, float]]:
    """Break text into lines that fit the page width.

    Args:
        text: page text.
        font: a loaded font object.
        width_px: usable width in pixels.
        line_pitch_px: vertical distance between baselines.

    Returns:
        List of (line_text, baseline_y) pairs.
    """
    raise NotImplementedError("TODO[W1.5]: greedy word wrap using glyph advances")


def render_page(text: str, cfg: Mapping[str, Any]) -> np.ndarray:
    """Render a full page of text.

    Args:
        text: page text.
        cfg: resolved render config (see configs/render_defaults.yaml).

    Returns:
        uint8 array (page_height_px, page_width_px, 3).
    """
    raise NotImplementedError("TODO[W1.5]: lay out lines, apply phase and render_mode, composite every glyph")


def overlay_grid(img: np.ndarray, cell_px: int) -> np.ndarray:
    """Draw the token grid over an image for visual checks.

    Args:
        img: uint8 array (H, W, 3).
        cell_px: cell side length in pixels.

    Returns:
        A copy of img with grid lines drawn every cell_px pixels.
    """
    raise NotImplementedError("TODO[W1.5]: draw thin coloured lines every cell_px pixels")
