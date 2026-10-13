# PURPOSE: The reference renderer: text to page image with controlled phase, pitch and anti-aliasing.
# WHAT GOES HERE: Glyph coverage loading, compositing, line layout, full-page render, grid overlay.
# FILLED IN AT STEP: W1.5.
# DO NOT: Let Pillow draw text directly onto the page; composite coverage yourself so aa_mode and phase are controlled.
"""Reference text renderer.

Learn A3 (sampling), A4 (colour), A5 (compositing) and A6 (font rasterisation) before implementing.
"""

from __future__ import annotations

import functools
import logging
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
RENDER_MODES = ("conventional", "patch_aware")

_log = logging.getLogger(__name__)


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


def unaligned_pitch_px(font_size_px: float) -> int:
    """Return the unaligned line pitch of section 1A.4, floor(1.25 * size + 0.5).

    Every rounding in the layout is floor(v + 0.5); Python's round would turn 1.25 x 18 = 22.5 into 22.

    Args:
        font_size_px: font size in pixels.

    Returns:
        The pitch in whole pixels: 15, 18 and 20 at 12, 14 and 16 px.
    """
    return math.floor(1.25 * font_size_px + 0.5)


@dataclass(frozen=True)
class LayoutFont:
    """The advances of one layout, conventional or patch-aware (D10).

    Attributes:
        path: font file, as for load_glyph.
        size_px: font size in pixels.
        slot_px: None for conventional layout, where a character advances by its linear advance a. For
            patch_aware layout the slot width q: a character takes a' = q * ceil(a / q) and the space
            exactly one slot.
    """

    path: str
    size_px: float
    slot_px: int | None = None

    def advance_px(self, char: str) -> float:
        """Return how far the pen moves past char, in pixels."""
        a = load_glyph(self.path, self.size_px, char).advance_px
        if self.slot_px is None:
            return a
        if char == " ":
            return float(self.slot_px)
        return float(self.slot_px * math.ceil(a / self.slot_px))

    def offset_px(self, char: str) -> int:
        """Return the whole pixels from the pen to the glyph's origin.

        0 in conventional layout. In patch_aware layout the glyph is centred in its slots,
        floor((a' - a) / 2 + 0.5), a whole number so that every glyph keeps the same fractional phase.
        """
        if self.slot_px is None or char == " ":
            return 0
        a = load_glyph(self.path, self.size_px, char).advance_px
        return math.floor((self.advance_px(char) - a) / 2 + 0.5)

    @property
    def ascender_px(self) -> float:
        """face.size.ascender / 64, which FreeType rounds to whole pixels: 16 px at 16 px."""
        return _face(self.path, math.floor(64 * self.size_px + 0.5)).size.ascender / 64

    @property
    def descender_room_px(self) -> int:
        """ceil(-face.size.descender / 64), the room below the last baseline: 4, 5 and 6 px at 12, 14 and 16 px."""
        return math.ceil(-_face(self.path, math.floor(64 * self.size_px + 0.5)).size.descender / 64)


@dataclass(frozen=True)
class PlacedGlyph:
    """One character placed by the layout. Spaces move the pen but are not placed.

    Attributes:
        char: the character.
        x_64: column of the glyph's origin in 1/64 px: its pen position plus, in patch_aware layout, the
            centring offset. render_page draws the glyph here.
        baseline_64: row of the baseline in 1/64 px.
        line: line number, counting from 0.
    """

    char: str
    x_64: int
    baseline_64: int
    line: int


def layout_lines(text: str, font: LayoutFont, width_px: float, line_pitch_px: float) -> list[tuple[str, float]]:
    """Break text into lines greedily and place their baselines (section 1A.4 of the build guide).

    A word goes on the current line if it fits inside width_px; otherwise it starts a new line. A word wider
    than width_px starts a new line and is split where the line is full, with a warning, because that should
    not happen at the sizes used. There is no kerning, no ligatures, no justification and no hyphenation.

    Args:
        text: page text; its words are the runs between whitespace, and a line joins them with single spaces.
        font: the advances to use.
        width_px: block width in pixels.
        line_pitch_px: distance between baselines in pixels.

    Returns:
        (line_text, baseline_y) pairs, baseline k at ascender + k * line_pitch_px from the block top.

    Raises:
        ValueError: if line_pitch_px is not positive, or one character is wider than width_px.
    """
    if not line_pitch_px > 0:
        raise ValueError(f"line_pitch_px must be positive, got {line_pitch_px!r}")
    space = font.advance_px(" ")
    lines: list[str] = []
    line, used = "", 0.0
    for word in text.split():
        width = sum(font.advance_px(c) for c in word)
        if line and used + space + width <= width_px:
            line, used = f"{line} {word}", used + space + width
            continue
        if line:
            lines.append(line)
        if width <= width_px:
            line, used = word, width
            continue
        _log.warning("word %r is %.1f px wide, more than the %.1f px line; split where the line is full",
                     word, width, width_px)
        line, used = "", 0.0
        for c in word:
            advance = font.advance_px(c)
            if advance > width_px:
                raise ValueError(f"{c!r} is {advance:.1f} px wide, more than the {width_px} px line")
            if used + advance > width_px:
                lines.append(line)
                line, used = "", 0.0
            line, used = line + c, used + advance
    if line:
        lines.append(line)
    ascender = font.ascender_px
    return [(text_of_line, ascender + k * line_pitch_px) for k, text_of_line in enumerate(lines)]


def block_size_px(cfg: Mapping[str, Any]) -> tuple[float, float]:
    """Return the text block: the page less both margins, the phase allowance and the blind strip.

    The allowance keeps all ink on the page at every phase, and the blind strip keeps it out of the columns
    and rows the encoder never sees (sections 1A.4 and 2.4). With the allowances of section 2.4 the block
    is 315 x 394 px for Qwen, 310 x 310 for SmolVLM-2.2B and 420 x 420 for SmolVLM-500M.

    Args:
        cfg: render config resolved by grid.resolve_render_cfg.

    Returns:
        (width, height) in pixels.

    Raises:
        ValueError: if an allowance or the blind strip is not set.
    """
    for key in ("max_phase_x_px", "max_phase_y_px", "blind_strip_px"):
        if cfg.get(key) is None:
            raise ValueError(f"{key} is not set; resolve the config with grid.resolve_render_cfg")
    margin, blind = cfg["margin_px"], cfg["blind_strip_px"]
    return (
        cfg["page_width_px"] - 2 * margin - cfg["max_phase_x_px"] - blind,
        cfg["page_height_px"] - 2 * margin - cfg["max_phase_y_px"] - blind,
    )


def _layout_font(cfg: Mapping[str, Any]) -> LayoutFont:
    """Return the LayoutFont for cfg's render_mode: linear advances, or slots of patch_aware_slot_px."""
    mode = cfg["render_mode"]
    if mode not in RENDER_MODES:
        raise ValueError(f"render_mode must be one of {RENDER_MODES}, got {mode!r}")
    slot = None
    if mode == "patch_aware":
        slot = cfg.get("patch_aware_slot_px")
        if not isinstance(slot, (int, np.integer)) or isinstance(slot, bool) or slot < 1:
            raise ValueError(
                f"patch_aware layout needs patch_aware_slot_px in whole pixels, got {slot!r}; "
                "resolve the config with grid.resolve_render_cfg"
            )
        slot = int(slot)
    return LayoutFont(str(cfg["font_file"]), cfg["font_size_px"], slot)


def layout_block(text: str, cfg: Mapping[str, Any]) -> list[PlacedGlyph]:
    """Lay out one page in block coordinates, without margins or phase (section 1A.4).

    The layout is computed once per page; layout_page only translates it, so every phase reads the same
    lines. Pen positions and baselines stay in pixels until the end and are converted to 1/64 px once,
    with floor(64 * v + 0.5). Linear advances are 16.16 fixed-point numbers, so their sums are exact in
    float64 and that conversion is the only rounding.

    Args:
        text: normalised page text.
        cfg: render config resolved by grid.resolve_render_cfg.

    Returns:
        The placed glyphs in reading order, spaces left out.

    Raises:
        ValueError: if cfg is not resolved, render_mode is unknown, or a character is wider than the block.
    """
    font = _layout_font(cfg)
    width, _ = block_size_px(cfg)
    placed: list[PlacedGlyph] = []
    for number, (line, baseline) in enumerate(layout_lines(text, font, width, cfg["line_pitch_px"])):
        baseline_64 = math.floor(64 * baseline + 0.5)
        pen = 0.0
        for c in line:
            if c != " ":
                placed.append(PlacedGlyph(c, math.floor(64 * (pen + font.offset_px(c)) + 0.5), baseline_64, number))
            pen += font.advance_px(c)
    return placed


def layout_page(text: str, cfg: Mapping[str, Any]) -> list[PlacedGlyph]:
    """Lay out one page on the page: the block, moved by the margin and the phase.

    Every x_64 gains floor(64 * margin_px + 0.5) + floor(64 * phase_x_px + 0.5), and every baseline_64 the
    same with phase_y_px, so a whole-pixel phase s moves every glyph by exactly 64 * s.

    Args:
        text: normalised page text.
        cfg: render config resolved by grid.resolve_render_cfg.

    Returns:
        The placed glyphs in page coordinates, in reading order.

    Raises:
        ValueError: as layout_block, or if a phase lies outside [0, its allowance].
    """
    placed = layout_block(text, cfg)
    offsets = []
    for axis in ("x", "y"):
        phase, allowance = cfg[f"phase_{axis}_px"], cfg[f"max_phase_{axis}_px"]
        if not 0 <= phase <= allowance:
            raise ValueError(f"phase_{axis}_px must lie in [0, max_phase_{axis}_px = {allowance}], got {phase!r}")
        offsets.append(math.floor(64 * cfg["margin_px"] + 0.5) + math.floor(64 * phase + 0.5))
    dx, dy = offsets
    return [PlacedGlyph(g.char, g.x_64 + dx, g.baseline_64 + dy, g.line) for g in placed]


def fits_page(text: str, cfg: Mapping[str, Any]) -> bool:
    """Return whether text fits in the block of cfg.

    Every line fits the block width by construction, so the test is the line count: at most
    floor((h - a - d) / p) + 1 lines, with h the block height, a the ascender, d the descender room and p
    the line pitch (section 2.4). The first baseline sits one ascender below the block top, and the last
    leaves the descender room above its bottom. paginate calls this under every layout that will be
    compared on a page.

    Args:
        text: normalised page text.
        cfg: render config resolved by grid.resolve_render_cfg.

    Returns:
        True if the text fits.

    Raises:
        ValueError: as layout_block.
    """
    font = _layout_font(cfg)
    width, height = block_size_px(cfg)
    pitch = cfg["line_pitch_px"]
    n_lines = len(layout_lines(text, font, width, pitch))
    room = height - font.ascender_px - font.descender_room_px
    return n_lines <= (math.floor(room / pitch) + 1 if room >= 0 else 0)


def render_coverage(text: str, cfg: Mapping[str, Any]) -> np.ndarray:
    """Accumulate the coverage of one page on one page-sized canvas (section 1A.5 of the build guide).

    Each glyph from layout_page is rasterised at the fractional part of its position, f = v & 63, and placed
    at the whole part, n = v >> 6: the first row of its bitmap is n_y - top and the first column n_x + left.
    Coverage is added and clipped at 1, and whatever falls outside the page is discarded. One canvas gives
    the same result as one per line, because lines never overlap at the pitches used: at 16 px the ink spans
    12 px above the baseline and 4 px below it.

    Args:
        text: normalised page text.
        cfg: render config resolved by grid.resolve_render_cfg.

    Returns:
        float64 array (page_height_px, page_width_px) of coverage in [0, 1], before the anti-aliasing curve.

    Raises:
        ValueError: as layout_page, or if the font has no glyph for a character.
    """
    height, width = cfg["page_height_px"], cfg["page_width_px"]
    coverage = np.zeros((height, width))
    for g in layout_page(text, cfg):
        glyph = load_glyph(str(cfg["font_file"]), cfg["font_size_px"], g.char, g.x_64 & 63, g.baseline_64 & 63)
        top, left = (g.baseline_64 >> 6) - glyph.top, (g.x_64 >> 6) + glyph.left
        rows, cols = glyph.coverage.shape
        r0, r1 = max(0, top), min(height, top + rows)
        c0, c1 = max(0, left), min(width, left + cols)
        if r0 < r1 and c0 < c1:  # clip first: a slice with a negative start would wrap round the page
            target = coverage[r0:r1, c0:c1]
            np.minimum(1.0, target + glyph.coverage[r0 - top : r1 - top, c0 - left : c1 - left], out=target)
    return coverage


def render_page(text: str, cfg: Mapping[str, Any]) -> np.ndarray:
    """Render one page: its coverage, one composite and one rounding (section 1A.5; D11).

    The coverage from render_coverage is mixed into a float64 background page by a single call to
    composite_coverage, and equation (6), floor(255 * clip(o, 0, 1) + 0.5), converts the page to 8 bits
    once, at the end.

    Args:
        text: normalised page text.
        cfg: render config resolved by grid.resolve_render_cfg.

    Returns:
        uint8 array (page_height_px, page_width_px, 3).

    Raises:
        ValueError: as render_coverage and composite_coverage, or if background_color is not three values
            from 0 to 255.
    """
    background = np.asarray(cfg["background_color"], dtype=np.float64)
    if background.shape != (3,) or not np.all((background >= 0) & (background <= 255)):
        raise ValueError(f"background_color must be three values from 0 to 255, got {cfg['background_color']!r}")
    coverage = render_coverage(text, cfg)
    page = np.empty((*coverage.shape, 3))
    page[...] = background / 255.0
    composite_coverage(page, coverage, 0, 0, cfg["text_color"], cfg["aa_mode"])
    return np.floor(255 * np.clip(page, 0.0, 1.0) + 0.5).astype(np.uint8)


_GRID_RGB = (230, 0, 0)  # patch, token-cell and window lines
_BLIND_RGB = (255, 200, 0)  # the strip the encoder never sees


def overlay_grid(
    img: np.ndarray,
    cell_px: int,
    patch_px: int | None = None,
    window_px: int | None = None,
    blind_from_px: int | None = None,
) -> np.ndarray:
    """Draw a reader's grids over a page, for eyes only (section 1A.6).

    Each line marks the first column or row of a patch, a token cell or an attention window, counting from
    the page's top-left corner. Patch lines are faint, cell lines strong and window lines two pixels wide,
    and the strip from blind_from_px to the right and bottom edges is shaded. Lines are blended with the
    page, so the ink under them stays visible. The sweep never calls this function: no page with an overlay
    may reach a reader.

    Args:
        img: uint8 array (H, W, 3), a rendered page.
        cell_px: token cell side in pixels.
        patch_px: patch side in pixels, or None for no patch lines.
        window_px: attention window side in pixels, or None for no window lines.
        blind_from_px: the first row and column the encoder never sees, or None.

    Returns:
        A new uint8 array (H, W, 3); img is not changed.

    Raises:
        ValueError: if img is not a uint8 array (H, W, 3), or a size is not a positive whole number.
    """
    if not (isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3):
        raise ValueError(
            f"img must be a uint8 array (H, W, 3), got {getattr(img, 'dtype', type(img).__name__)} "
            f"{getattr(img, 'shape', '')}"
        )
    sizes = {"cell_px": cell_px, "patch_px": patch_px, "window_px": window_px, "blind_from_px": blind_from_px}
    for name, value in sizes.items():
        if value is not None and (not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1):
            raise ValueError(f"{name} must be a positive whole number of pixels, got {value!r}")
    out = img.astype(np.float64)
    height, width = img.shape[:2]

    def blend(region: np.ndarray, rgb: Sequence[int], alpha: float) -> None:
        region[...] = (1 - alpha) * region + alpha * np.asarray(rgb, dtype=np.float64)

    if blind_from_px is not None:
        blend(out[blind_from_px:], _BLIND_RGB, 0.35)
        blend(out[:blind_from_px, blind_from_px:], _BLIND_RGB, 0.35)
    for period, alpha, thickness in ((patch_px, 0.25, 1), (cell_px, 0.6, 1), (window_px, 0.6, 2)):
        if period is None:
            continue
        for k in range(0, width, period):
            blend(out[:, k : k + thickness], _GRID_RGB, alpha)
        for k in range(0, height, period):
            blend(out[k : k + thickness], _GRID_RGB, alpha)
    return np.floor(out + 0.5).astype(np.uint8)
