# PURPOSE: Correctness tests for the reference renderer.
# WHAT GOES HERE: Glyph loading at 1/64 px phases (D11); phase periodicity of the renderer itself; output shape.
# FILLED IN AT STEP: W1.5 (each todo mark goes when the function it tests is implemented).
# DO NOT: Depend on a specific font file being committed; skip if the font is missing.
"""Tests for mrr.render."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import REPO_ROOT
from mrr.color import linear_to_srgb, srgb_to_linear
from mrr.render import composite_coverage, load_glyph

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
