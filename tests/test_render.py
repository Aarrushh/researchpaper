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
from mrr.render import load_glyph

CELL = 28  # Qwen2.5-VL token cell; see configs/readers.yaml
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
