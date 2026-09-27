# PURPOSE: Correctness tests for the reference renderer.
# WHAT GOES HERE: Phase periodicity of the renderer itself; output shape.
# FILLED IN AT STEP: W1.5 (remove the todo mark once render.py is implemented).
# DO NOT: Depend on a specific font file being committed; skip if the font is missing.
"""Tests for mrr.render."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import REPO_ROOT

pytestmark = pytest.mark.todo  # TODO[W1.5]: remove once render.py is implemented

CELL = 28  # Qwen2.5-VL token cell; see configs/readers.yaml


@pytest.fixture
def cfg():
    c = yaml.safe_load((REPO_ROOT / "configs" / "render_defaults.yaml").read_text(encoding="utf-8"))
    if not Path(str(c["font_file"])).exists():
        pytest.skip("font file not downloaded yet (W1.4)")
    return c


def test_phase_one_cell_equals_shift(cfg):
    from mrr.render import render_page

    text = "The quick brown fox"
    a = render_page(text, {**cfg, "phase_x_px": 0})
    b = render_page(text, {**cfg, "phase_x_px": CELL})
    # b is a shifted right by one cell; compare the overlapping region.
    np.testing.assert_array_equal(a[:, : a.shape[1] - CELL], b[:, CELL:])


def test_output_shape_is_native(cfg):
    from mrr.render import render_page

    img = render_page("abc", cfg)
    assert img.shape == (int(cfg["page_height_px"]), int(cfg["page_width_px"]), 3)
