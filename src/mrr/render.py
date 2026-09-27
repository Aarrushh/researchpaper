# PURPOSE: The reference renderer: text to page image with controlled phase, pitch and anti-aliasing.
# WHAT GOES HERE: Glyph coverage loading, compositing, line layout, full-page render, grid overlay.
# FILLED IN AT STEP: W1.5.
# DO NOT: Let Pillow draw text directly onto the page; composite coverage yourself so aa_mode and phase are controlled.
"""Reference text renderer.

Learn A3 (sampling), A4 (colour), A5 (compositing) and A6 (font rasterisation) before implementing.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def load_glyph_coverage(font_path: str, size_px: float, char: str) -> np.ndarray:
    """Load the coverage mask for one character.

    Args:
        font_path: path to a font file under data/external/fonts/.
        size_px: font size in pixels.
        char: a single character.

    Returns:
        float64 array (h, w) of coverage in [0, 1].
    """
    raise NotImplementedError("TODO[W1.5]: load glyph mask with Pillow ImageFont and scale to [0, 1]")


def composite_coverage(
    page: np.ndarray,
    coverage: np.ndarray,
    x: float,
    y: float,
    color: Sequence[float],
    aa_mode: str,
) -> np.ndarray:
    """Composite one coverage mask onto the page at (x, y).

    Args:
        page: float64 array (H, W, 3), sRGB-encoded values in [0, 1].
        coverage: float64 array (h, w) in [0, 1].
        x: left edge in pixels (may be fractional).
        y: top edge in pixels (may be fractional).
        color: text colour as sRGB values in [0, 1].
        aa_mode: one of "binary", "coverage_srgb", "coverage_linear".

    Returns:
        The updated page array.
    """
    raise NotImplementedError("TODO[W1.5]: implement over-compositing for each aa_mode")


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
