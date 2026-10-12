# PURPOSE: Tests for the reader geometry in mrr.grid.
# WHAT GOES HERE: One test per row of the table in section 1A.7 of the build guide, with the reader entries read from configs/readers.yaml; no model.
# FILLED IN AT STEP: W1.5/W1.6.
# DO NOT: Write reader entries by hand; read them from configs/readers.yaml so that a config change is caught here.
"""Tests for mrr.grid."""

from __future__ import annotations

import pathlib

import pytest
import yaml

from mrr import grid

ROOT = pathlib.Path(__file__).resolve().parents[1]
READERS = {
    reader["name"]: reader
    for reader in yaml.safe_load((ROOT / "configs" / "readers.yaml").read_text(encoding="utf-8"))["readers"]
}

# The keys of configs/render_defaults.yaml that resolve_render_cfg fills in, with the values W1.5 gives them.
DEFAULTS = {
    "page_width_px": 448,
    "page_height_px": 448,
    "margin_px": 14,
    "max_phase_x_px": None,
    "max_phase_y_px": None,
    "patch_aware_slot_px": None,
    "phase_x_px": 0,
    "phase_y_px": 0,
    "text_color": [0, 0, 0],
}


@pytest.mark.parametrize(("name", "expected"), [("qwen", 28), ("smolvlm", 42), ("smolvlm500m", 64)])
def test_token_cell_px(name, expected):
    assert grid.token_cell_px(READERS[name]) == expected


def test_token_cell_px_rejects_a_config_that_disagrees():
    with pytest.raises(ValueError):
        grid.token_cell_px({**READERS["qwen"], "token_cell_px": 30})


@pytest.mark.parametrize(("name", "expected"), [("qwen", 112), ("smolvlm", 42), ("smolvlm500m", 64)])
def test_sweep_period_px(name, expected):
    assert grid.sweep_period_px(READERS[name]) == expected


@pytest.mark.parametrize(("name", "expected"), [("qwen", 256), ("smolvlm", 81), ("smolvlm500m", 64)])
def test_token_count_on_the_native_page(name, expected):
    reader = READERS[name]
    assert grid.token_count(reader, *grid.page_size_px(reader)) == expected


def test_old_token_formula_overcounts_smolvlm():
    # The scaffold's (h / cell) * (w / cell) ignores the patches that do not fit in 384 px.
    reader = READERS["smolvlm"]
    cell = grid.token_cell_px(reader)
    assert (384 / cell) * (384 / cell) == pytest.approx(83.6, abs=0.05)
    assert grid.token_count(reader, 384, 384) == 81


@pytest.mark.parametrize(("name", "expected"), [("qwen", 0), ("smolvlm", 6), ("smolvlm500m", 0)])
def test_blind_strip_px(name, expected):
    assert grid.blind_strip_px(READERS[name]) == expected


def test_blind_strip_px_rejects_a_config_that_disagrees():
    with pytest.raises(ValueError):
        grid.blind_strip_px({**READERS["smolvlm"], "blind_from_px": 384})


@pytest.mark.parametrize(
    ("name", "expected"), [("qwen", (448, 448)), ("smolvlm", (384, 384)), ("smolvlm500m", (512, 512))]
)
def test_page_size_px(name, expected):
    assert grid.page_size_px(READERS[name]) == expected


def test_page_size_px_rejects_a_qwen_page_the_processor_would_resize():
    with pytest.raises(ValueError):
        grid.page_size_px({**READERS["qwen"], "page_px": [450, 450]})


def test_page_size_px_rejects_a_smolvlm_page_that_is_not_square():
    with pytest.raises(ValueError):
        grid.page_size_px({**READERS["smolvlm"], "page_px": [384, 378]})


@pytest.mark.parametrize(("side", "expected"), [(448, 448), (384, 392), (512, 504), (1000, 1008), (70, 56)])
def test_qwen_smart_resize(side, expected):
    # 70 / 28 = 2.5, and Python's round takes it to 2, so 70 px becomes 56, not 84.
    assert grid.qwen_smart_resize(side, side) == (expected, expected)


@pytest.mark.parametrize(("phase", "expected"), [(3.25, (3, 16)), (13.0078125, (13, 1)), (0.0, (0, 0))])
def test_split_phase_64(phase, expected):
    # 64 * 13.0078125 = 832.5, which rounds up to 833 = 13 * 64 + 1.
    assert grid.split_phase_64(phase) == expected


def test_phase_offsets():
    assert grid.phase_offsets(28, 4) == [0, 7, 14, 21]


def test_phase_offsets_rejects_zero_steps():
    with pytest.raises(ValueError):
        grid.phase_offsets(28, 0)


@pytest.mark.parametrize(("x", "expected"), [(30, 56), (28, 28), (0, 0), (27.5, 28)])
def test_snap_to_grid_rounds_up(x, expected):
    assert grid.snap_to_grid(x, 28) == expected


def test_resolve_render_cfg_for_qwen():
    cfg = grid.resolve_render_cfg(DEFAULTS, READERS["qwen"])
    assert (cfg["page_height_px"], cfg["page_width_px"]) == (448, 448)
    assert cfg["margin_px"] == 14
    assert (cfg["max_phase_x_px"], cfg["max_phase_y_px"]) == (111, 111)
    assert cfg["patch_aware_slot_px"] == 14
    assert cfg["blind_strip_px"] == 0


def test_resolve_render_cfg_for_smolvlm500m():
    cfg = grid.resolve_render_cfg(DEFAULTS, READERS["smolvlm500m"])
    assert (cfg["page_height_px"], cfg["page_width_px"]) == (512, 512)
    assert cfg["margin_px"] == 16
    assert (cfg["max_phase_x_px"], cfg["max_phase_y_px"]) == (63, 63)
    assert cfg["patch_aware_slot_px"] == 16


def test_resolve_render_cfg_for_smolvlm_has_the_blind_strip():
    cfg = grid.resolve_render_cfg(DEFAULTS, READERS["smolvlm"])
    assert (cfg["page_height_px"], cfg["max_phase_x_px"], cfg["blind_strip_px"]) == (384, 41, 6)


def test_resolve_render_cfg_keeps_a_value_that_is_not_null():
    cfg = grid.resolve_render_cfg({**DEFAULTS, "max_phase_y_px": 26, "patch_aware_slot_px": 28}, READERS["qwen"])
    assert (cfg["max_phase_x_px"], cfg["max_phase_y_px"], cfg["patch_aware_slot_px"]) == (111, 26, 28)


def test_resolve_render_cfg_applies_overrides_last():
    cfg = grid.resolve_render_cfg(DEFAULTS, READERS["qwen"], max_phase_x_px=105, phase_x_px=3.25)
    assert (cfg["max_phase_x_px"], cfg["phase_x_px"]) == (105, 3.25)


def test_resolve_render_cfg_rejects_an_unknown_override():
    with pytest.raises(KeyError):
        grid.resolve_render_cfg(DEFAULTS, READERS["qwen"], max_phase_px=105)


def test_resolve_render_cfg_leaves_the_defaults_unchanged():
    before = {key: (list(value) if isinstance(value, list) else value) for key, value in DEFAULTS.items()}
    cfg = grid.resolve_render_cfg(DEFAULTS, READERS["qwen"])
    cfg["text_color"][0] = 255
    assert DEFAULTS == before
