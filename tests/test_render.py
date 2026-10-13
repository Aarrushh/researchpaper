# PURPOSE: Correctness tests for the reference renderer.
# WHAT GOES HERE: Glyph loading at 1/64 px phases and compositing (D11); layout and page fit (D10); phase periodicity of the renderer itself; output shape.
# FILLED IN AT STEP: W1.5 (each todo mark goes when the function it tests is implemented).
# DO NOT: Depend on a specific font file being committed; skip if the font is missing.
"""Tests for mrr.render."""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import REPO_ROOT
from mrr.color import linear_to_srgb, srgb_to_linear
from mrr.grid import resolve_render_cfg
from mrr.render import (
    AA_MODES,
    LayoutFont,
    block_size_px,
    composite_coverage,
    fits_page,
    layout_block,
    layout_lines,
    layout_page,
    load_glyph,
    unaligned_pitch_px,
)
from mrr.text_source import load_book, normalise_text, paginate, skip_front_matter, strip_gutenberg_boilerplate

CELL = 28  # Qwen2.5-VL token cell; see configs/readers.yaml
MODES = ("binary", "coverage_srgb", "coverage_linear")
# A str, because freetype-py cannot take a pathlib.Path.
FONT = str(REPO_ROOT / "data" / "external" / "fonts" / "source-sans-3" / "SourceSans3-Regular.ttf")
needs_font = pytest.mark.skipif(not Path(FONT).exists(), reason="font file not downloaded yet (W1.4)")


@pytest.fixture
def cfg():
    c = yaml.safe_load((REPO_ROOT / "configs" / "render_defaults.yaml").read_text(encoding="utf-8"))
    if not Path(str(c["font_file"])).exists():
        pytest.skip("font file not downloaded yet (W1.4)")
    return c


@pytest.mark.todo  # TODO[W1.5]: remove once render_page is implemented
def test_phase_one_cell_equals_shift(cfg):
    from mrr.render import render_page

    text = "The quick brown fox"
    a = render_page(text, {**cfg, "phase_x_px": 0})
    b = render_page(text, {**cfg, "phase_x_px": CELL})
    # b is a shifted right by one cell; compare the overlapping region.
    np.testing.assert_array_equal(a[:, : a.shape[1] - CELL], b[:, CELL:])


@pytest.mark.todo  # TODO[W1.5]: remove once render_page is implemented
def test_output_shape_is_native(cfg):
    from mrr.render import render_page

    img = render_page("abc", cfg)
    assert img.shape == (int(cfg["page_height_px"]), int(cfg["page_width_px"]), 3)


@needs_font
def test_o_has_64_distinct_phases():
    seen = {(g.coverage.shape, g.coverage.tobytes())
            for g in (load_glyph(FONT, 16, "o", phase_x_64=k) for k in range(64))}
    assert len(seen) == 64


@needs_font
def test_o_coverage_total_stable():
    sums = [load_glyph(FONT, 16, "o", phase_x_64=8 * k).coverage.sum() for k in range(8)]
    assert (max(sums) - min(sums)) / min(sums) < 0.002
    assert 25.38 < min(sums) and max(sums) < 25.43


@needs_font
def test_linear_advances():
    assert load_glyph(FONT, 16, "m").advance_px == pytest.approx(13.264, abs=1e-3)
    assert load_glyph(FONT, 16, "i").advance_px == pytest.approx(3.936, abs=1e-3)


@needs_font
def test_half_pixel_down_moves_H_down():
    def centroid_row(g):
        r = np.arange(g.coverage.shape[0])[:, None] - g.top
        return float((r * g.coverage).sum() / g.coverage.sum())
    d = centroid_row(load_glyph(FONT, 16, "H", phase_y_64=32)) - centroid_row(load_glyph(FONT, 16, "H"))
    assert 0 < d < 1


@needs_font
def test_transform_reset_between_glyphs():
    load_glyph.cache_clear(); a = load_glyph(FONT, 16, "o").coverage.copy()
    load_glyph(FONT, 16, "o", phase_x_64=32)
    load_glyph.cache_clear(); b = load_glyph(FONT, 16, "o").coverage
    assert np.array_equal(a, b)


@needs_font
def test_cached_coverage_is_read_only():
    with pytest.raises(ValueError):
        load_glyph(FONT, 16, "o").coverage[0, 0] = 1.0


@needs_font
def test_load_glyph_rejects_missing_character():      # Review Focus 2
    with pytest.raises(ValueError, match="U\\+4E00"):
        load_glyph(FONT, 16, "一")


@needs_font
def test_load_glyph_rejects_a_phase_outside_one_pixel():
    # Phases are 64ths of a pixel; 64 or more would move the glyph by whole pixels inside its bitmap.
    with pytest.raises(ValueError):
        load_glyph(FONT, 16, "o", phase_x_64=64)
    with pytest.raises(ValueError):
        load_glyph(FONT, 16, "o", phase_y_64=-1)


@needs_font
def test_load_glyph_opens_a_font_under_a_non_ascii_path(tmp_path):
    # freetype-py hands FreeType the path as UTF-8 bytes, which FreeType cannot open on Windows
    # when the path is not ASCII ("cannot open resource").
    font = tmp_path / "fönts" / "SourceSans3-Regular.ttf"
    font.parent.mkdir()
    shutil.copyfile(FONT, font)
    assert np.array_equal(load_glyph(str(font), 16, "o").coverage, load_glyph(FONT, 16, "o").coverage)


def _bytes(page):
    # Equation (6) without the clip, so that a colour left in 0-255 shows up as a value above 255.
    return np.floor(255 * page + 0.5)


@pytest.mark.parametrize("mode,c,expected", [
    ("binary", .25, 255), ("coverage_srgb", .25, 191), ("coverage_linear", .25, 225),
    ("binary", .5, 0),    ("coverage_srgb", .5, 128),  ("coverage_linear", .5, 188),
    ("binary", .75, 0),   ("coverage_srgb", .75, 64),  ("coverage_linear", .75, 137)])
def test_black_on_white(mode, c, expected):
    page = np.ones((1, 1, 3))
    out = composite_coverage(page, np.full((1, 1), c), 0, 0, (0, 0, 0), mode)
    assert np.all(np.floor(255 * out + 0.5) == expected)


@pytest.mark.parametrize("mode", MODES)
def test_no_coverage_keeps_every_background_level(mode):
    page = np.repeat(np.arange(256.0)[:, None, None] / 255, 3, axis=2)  # one background level per row
    out = composite_coverage(page, np.zeros((256, 1)), 0, 0, (0, 0, 0), mode)
    assert out is page
    assert np.array_equal(_bytes(out)[:, 0, :], np.repeat(np.arange(256.0)[:, None], 3, axis=1))


@pytest.mark.parametrize("mode", MODES)
def test_full_coverage_gives_every_text_level(mode):
    for level in range(256):
        page = np.full((2, 2, 3), 0.5)
        composite_coverage(page, np.ones((2, 2)), 0, 0, (level, 0, 255), mode)
        assert np.all(_bytes(page) == [level, 0, 255]), level


@pytest.mark.parametrize("mode", ["coverage_srgb", "coverage_linear"])
def test_partial_coverage_mixes_each_channel(mode):
    page = np.full((1, 1, 3), 0.2)
    composite_coverage(page, np.full((1, 1), 0.3), 0, 0, (255, 51, 0), mode)
    t = np.array([255, 51, 0]) / 255
    if mode == "coverage_srgb":
        expected = 0.3 * t + 0.7 * 0.2
    else:
        expected = linear_to_srgb(0.3 * srgb_to_linear(t) + 0.7 * srgb_to_linear(np.full(3, 0.2)))
    np.testing.assert_allclose(page[0, 0], expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize(("x", "y"), [(1.5, 0), (0, 2.0), (True, 0), (np.float64(1), 0)])
def test_composite_rejects_a_coordinate_that_is_not_an_integer(x, y):
    with pytest.raises(TypeError):
        composite_coverage(np.ones((4, 4, 3)), np.ones((1, 1)), x, y, (0, 0, 0), "coverage_srgb")


def test_composite_accepts_numpy_integers():
    page = composite_coverage(np.ones((4, 4, 3)), np.ones((1, 1)), np.int64(1), np.uint8(2), (0, 0, 0), "binary")
    assert page[2, 1].tolist() == [0, 0, 0]
    assert page.sum() == 4 * 4 * 3 - 3


def test_composite_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        composite_coverage(np.ones((1, 1, 3)), np.ones((1, 1)), 0, 0, (0, 0, 0), "coverage_gamma")


def test_composite_rejects_a_page_or_coverage_outside_its_contract():
    cov = np.ones((1, 1))
    with pytest.raises(TypeError):  # uint8 arithmetic is not allowed anywhere before equation (6)
        composite_coverage(np.full((2, 2, 3), 255, np.uint8), cov, 0, 0, (0, 0, 0), "binary")
    with pytest.raises(ValueError):  # a page still in 0-255
        composite_coverage(np.full((2, 2, 3), 255.0), cov, 0, 0, (0, 0, 0), "binary")
    with pytest.raises(ValueError):  # not (H, W, 3)
        composite_coverage(np.ones((2, 2)), cov, 0, 0, (0, 0, 0), "binary")
    with pytest.raises(ValueError):  # coverage summed past 1 and not clipped
        composite_coverage(np.ones((2, 2, 3)), np.full((1, 1), 1.5), 0, 0, (0, 0, 0), "binary")
    with pytest.raises(ValueError):  # a colour outside 0-255
        composite_coverage(np.ones((2, 2, 3)), cov, 0, 0, (0, 0, 256), "binary")


def test_composite_clips_at_the_far_corner():
    page = np.ones((6, 8, 3))
    composite_coverage(page, np.ones((4, 4)), 8 - 2, 6 - 2, (0, 0, 0), "coverage_srgb")
    expected = np.ones((6, 8, 3))
    expected[4:, 6:] = 0
    assert np.array_equal(page, expected)


def test_composite_clips_negative_origin():      # Review Focus 5
    page = np.ones((8, 8, 3))
    composite_coverage(page, np.ones((4, 4)), -3, -2, (0, 0, 0), "coverage_srgb")
    expected = np.ones((8, 8, 3))
    expected[0:2, 0:1] = 0
    assert np.array_equal(page, expected)


def test_composite_ignores_a_block_wholly_off_the_page():
    page = np.ones((4, 4, 3))
    composite_coverage(page, np.ones((2, 2)), 10, -5, (0, 0, 0), "coverage_srgb")
    assert np.array_equal(page, np.ones((4, 4, 3)))


# Layout (section 1A.4 of the build guide; D10).

READERS = {
    reader["name"]: reader
    for reader in yaml.safe_load((REPO_ROOT / "configs" / "readers.yaml").read_text(encoding="utf-8"))["readers"]
}
DEFAULTS = yaml.safe_load((REPO_ROOT / "configs" / "render_defaults.yaml").read_text(encoding="utf-8"))
PERSUASION = REPO_ROOT / "data" / "raw" / "texts" / "pg105-images-3.epub"
# Section 2.4, step 1: each reader's phase allowances (x, y) in the full run.
ALLOWANCES = {"qwen": (105, 26), "smolvlm": (40, 40), "smolvlm500m": (60, 60)}


def _cfg(reader, **overrides):
    return resolve_render_cfg(DEFAULTS, READERS[reader], font_file=FONT, **overrides)


@pytest.fixture(scope="module")
def persuasion():
    if not (PERSUASION.exists() and Path(FONT).exists()):
        pytest.skip("book or font not present")
    return normalise_text(skip_front_matter(strip_gutenberg_boilerplate(load_book(PERSUASION)), "CHAPTER I.", 2))


def _first_page(text, *cfgs):
    return paginate(text, lambda page: all(fits_page(page, cfg) for cfg in cfgs), n_pages=1)[0]


@pytest.mark.parametrize(("size", "pitch"), [(12, 15), (14, 18), (16, 20), (18, 23)])
def test_unaligned_pitch(size, pitch):
    assert unaligned_pitch_px(size) == pitch  # 1.25 x 18 = 22.5 rounds up; Python's round would give 22


def test_render_defaults_are_filled_in():
    assert not [key for key, value in DEFAULTS.items() if isinstance(value, str) and "TODO" in value]
    assert (DEFAULTS["page_width_px"], DEFAULTS["page_height_px"]) == (448, 448)
    assert DEFAULTS["max_phase_x_px"] is None and DEFAULTS["max_phase_y_px"] is None
    assert DEFAULTS["line_pitch_px"] == unaligned_pitch_px(DEFAULTS["font_size_px"])
    assert DEFAULTS["aa_mode"] in AA_MODES and DEFAULTS["render_mode"] == "conventional"


@needs_font
def test_layout_needs_a_resolved_config():
    with pytest.raises(ValueError, match="resolve_render_cfg"):
        layout_block("abc", {**DEFAULTS, "font_file": FONT})
    with pytest.raises(ValueError, match="render_mode"):
        layout_block("abc", _cfg("qwen", render_mode="monospace"))


@needs_font
def test_layout_block_starts_at_the_block_origin():
    first = layout_block("Hm", _cfg("qwen"))[0]
    assert (first.char, first.x_64, first.baseline_64, first.line) == ("H", 0, 64 * 16, 0)  # ascender 16 px


@needs_font
def test_conventional_layout_uses_linear_advances():
    text, pen, expected = "Wim m", 0.0, []
    for char in text:
        if char != " ":  # a space advances the pen but is not placed
            expected.append(math.floor(64 * pen + 0.5))
        pen += load_glyph(FONT, 16, char).advance_px
    assert [g.x_64 for g in layout_block(text, _cfg("qwen"))] == expected
    assert load_glyph(FONT, 16, " ").advance_px == pytest.approx(3.2, abs=2**-16)  # 16.16 fixed point


@needs_font
def test_patch_aware_layout_centres_each_glyph_in_whole_slots():
    text, slot_start, expected = "Wim m", 0, []
    for char in text:
        a = load_glyph(FONT, 16, char).advance_px
        slots = 14 if char == " " else 14 * math.ceil(a / 14)  # a space takes one slot
        if char != " ":
            expected.append(64 * (slot_start + math.floor((slots - a) / 2 + 0.5)))
        slot_start += slots
    assert [g.x_64 for g in layout_block(text, _cfg("qwen", render_mode="patch_aware"))] == expected


@needs_font
@pytest.mark.parametrize("mode", ["conventional", "patch_aware"])
def test_layout_reads_the_same_lines_at_every_phase(persuasion, mode):
    cfg = _cfg("qwen", render_mode=mode)
    page = _first_page(persuasion, cfg)
    sequences = {
        tuple((g.char, g.line) for g in layout_page(page, {**cfg, "phase_x_px": phase, "phase_y_px": phase}))
        for phase in (0, 0.5, 111)
    }
    assert len(sequences) == 1


@needs_font
@pytest.mark.parametrize("s", [0.5, 1, 7, 13, 14, 28, 111])
def test_a_phase_moves_every_glyph_by_the_same_64ths(persuasion, s):
    cfg = _cfg("qwen")
    page = _first_page(persuasion, cfg)
    base = layout_page(page, cfg)
    right = layout_page(page, {**cfg, "phase_x_px": s})
    down = layout_page(page, {**cfg, "phase_y_px": s})
    shift = math.floor(64 * s + 0.5)  # 64 * s for a whole-pixel phase
    assert {r.x_64 - b.x_64 for r, b in zip(right, base)} == {shift}
    assert {r.baseline_64 - b.baseline_64 for r, b in zip(right, base)} == {0}
    assert {d.baseline_64 - b.baseline_64 for d, b in zip(down, base)} == {shift}
    assert {d.x_64 - b.x_64 for d, b in zip(down, base)} == {0}


@needs_font
def test_layout_page_adds_the_margin_and_the_phase():
    cfg = _cfg("smolvlm500m", phase_x_px=3.25, phase_y_px=2)  # margin 16 px
    block, page = layout_block("Hm", cfg)[0], layout_page("Hm", cfg)[0]
    assert (page.x_64 - block.x_64, page.baseline_64 - block.baseline_64) == (64 * 16 + 208, 64 * 16 + 128)


@needs_font
@pytest.mark.parametrize(("axis", "phase"), [("x", -1), ("x", 112), ("y", -0.5), ("y", 111.25)])
def test_layout_page_rejects_a_phase_outside_its_allowance(axis, phase):
    with pytest.raises(ValueError, match=f"phase_{axis}_px"):
        layout_page("Hm", _cfg("qwen", **{f"phase_{axis}_px": phase}))  # Qwen's allowance is 111 px


@needs_font
def test_an_over_wide_word_is_split_at_the_block_edge(caplog):
    cfg = _cfg("qwen")
    width, _ = block_size_px(cfg)
    token = "abcdefghij" * 6  # 60 characters, wider than the 309 px block
    with caplog.at_level(logging.WARNING, logger="mrr.render"):
        placed = layout_block(f"short {token} tail", cfg)
    assert "".join(g.char for g in placed) == f"short{token}tail"
    assert [g.line for g in placed if g.char == "s"] == [0] and max(g.line for g in placed) == 2
    assert {g.line for g in placed[5:65]} == {1, 2}  # the token starts a new line and is cut once
    assert max(g.x_64 / 64 + load_glyph(FONT, 16, g.char).advance_px for g in placed) <= width
    assert token in caplog.text


@needs_font
def test_layout_lines_rejects_a_character_wider_than_the_line():
    with pytest.raises(ValueError):
        layout_lines("m", LayoutFont(FONT, 16), 5, 20)


@needs_font
def test_fits_page_counts_lines_against_the_block_height():
    cfg = _cfg("qwen", render_mode="patch_aware")  # block 309 x 309 px; a line holds 22 slots of 14 px
    word = "x" * 22
    assert len(layout_lines(" ".join([word] * 3), LayoutFont(FONT, 16, 14), 309, 20)) == 3
    # floor((309 - 16 - 6) / 20) + 1 = 15 baselines fit between the ascender and the descender room.
    assert fits_page(" ".join([word] * 15), cfg)
    assert not fits_page(" ".join([word] * 16), cfg)


# Section 2.4: the block (width, height) under each reader's allowances, and the lines it holds at the
# unaligned and the aligned pitch, by font size and reader.
BLOCKS = {"qwen": (315, 394), "smolvlm": (310, 310), "smolvlm500m": (420, 420)}
PAGE_LINES = [
    (12, "qwen", 26, 14), (12, "smolvlm", 20, 8), (12, "smolvlm500m", 27, 7),
    (14, "qwen", 21, 14), (14, "smolvlm", 17, 7), (14, "smolvlm500m", 23, 7),
    (16, "qwen", 19, 14), (16, "smolvlm", 15, 7), (16, "smolvlm500m", 20, 7),
]


@needs_font
@pytest.mark.parametrize(("size", "reader", "unaligned", "aligned"), PAGE_LINES)
def test_fits_page_reproduces_the_line_counts(size, reader, unaligned, aligned):
    x, y = ALLOWANCES[reader]
    for lines, pitch in ((unaligned, unaligned_pitch_px(size)), (aligned, READERS[reader]["aligned_pitch_px"])):
        cfg = _cfg(reader, font_size_px=size, line_pitch_px=pitch, max_phase_x_px=x, max_phase_y_px=y)
        width, _ = block_size_px(cfg)
        assert block_size_px(cfg) == BLOCKS[reader]
        word = "x" * math.floor(width / load_glyph(FONT, size, "x").advance_px)  # one word fills a line
        assert fits_page(" ".join([word] * lines), cfg)
        assert not fits_page(" ".join([word] * (lines + 1)), cfg)


# Section 2.4: characters of Persuasion (spaces included, whole words) on the first page, by font size and
# reader, for conventional layout with unaligned pitch, conventional with aligned pitch (block V) and
# patch_aware with unaligned pitch ("mono"), at each reader's allowances [Measured 10 Oct].
PAGE_CHARS = [
    (12, "qwen", 1584, 840, 524), (12, "smolvlm", 1186, 479, 401), (12, "smolvlm500m", 2201, 566, 642),
    (14, "qwen", 1079, 713, 424), (14, "smolvlm", 847, 349, 339), (14, "smolvlm500m", 1618, 489, 546),
    (16, "qwen", 840, 616, 379), (16, "smolvlm", 651, 298, 298), (16, "smolvlm500m", 1213, 419, 471),
]


@needs_font
@pytest.mark.parametrize(("size", "reader", "conventional", "aligned", "mono"), PAGE_CHARS)
def test_fits_page_reproduces_the_page_capacity_table(persuasion, size, reader, conventional, aligned, mono):
    x, y = ALLOWANCES[reader]
    common = {"font_size_px": size, "max_phase_x_px": x, "max_phase_y_px": y}
    unaligned = unaligned_pitch_px(size)
    layouts = [
        (conventional, _cfg(reader, line_pitch_px=unaligned, **common)),
        (aligned, _cfg(reader, line_pitch_px=READERS[reader]["aligned_pitch_px"], **common)),
        (mono, _cfg(reader, render_mode="patch_aware", line_pitch_px=unaligned, **common)),
    ]
    for expected, cfg in layouts:
        got = len(_first_page(persuasion, cfg))
        assert got == expected, (cfg["render_mode"], cfg["line_pitch_px"], got)


def _ink_extent(placed, cfg):
    """The last row and column that any glyph's bitmap covers on the page."""
    rows, cols = [], []
    for g in placed:
        glyph = load_glyph(FONT, cfg["font_size_px"], g.char, g.x_64 & 63, g.baseline_64 & 63)
        height, width = glyph.coverage.shape
        if height and width:
            rows.append((g.baseline_64 >> 6) - glyph.top + height - 1)
            cols.append((g.x_64 >> 6) + glyph.left + width - 1)
    return max(rows), max(cols)


@needs_font
@pytest.mark.parametrize("mode", ["conventional", "patch_aware"])
def test_no_ink_reaches_smolvlm_blind_strip(persuasion, mode):
    cfg = _cfg("smolvlm", render_mode=mode)
    far = {**cfg, "phase_x_px": cfg["max_phase_x_px"], "phase_y_px": cfg["max_phase_y_px"]}
    last_row, last_col = _ink_extent(layout_page(_first_page(persuasion, cfg), far), far)
    assert max(last_row, last_col) < READERS["smolvlm"]["blind_from_px"]  # 378
