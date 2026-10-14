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
from mrr.render import InkBox

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


# Coherence of the layouts (section 2.2, D10): R_k(T) of ink centres, its per-class mean and the straddle share.


def test_coherence_is_one_for_centres_a_period_apart():
    assert grid.coherence([0, 14, 28], 14) == pytest.approx(1.0)


def test_coherence_is_zero_for_centres_half_a_period_apart():
    assert grid.coherence([0, 7], 14) == pytest.approx(0.0, abs=1e-12)


def test_coherence_ignores_a_common_shift():
    centres = [1.5, 4.0, 9.25, 13.0, 20.5]
    assert grid.coherence([c + 5 for c in centres], 14) == pytest.approx(grid.coherence(centres, 14), abs=1e-12)


def test_coherence_at_the_second_harmonic_sees_half_the_period():
    assert grid.coherence([0, 7, 14, 21], 14, k=2) == pytest.approx(1.0)
    assert grid.coherence([0, 7, 14, 21], 14) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("centres", "period", "k"), [([], 14, 1), ([0.0], 0, 1), ([0.0], -14, 1), ([0.0], 14, 0), ([0.0], 14, 1.5)]
)
def test_coherence_refuses_no_centres_a_bad_period_or_a_bad_harmonic(centres, period, k):
    with pytest.raises(ValueError):
        grid.coherence(centres, period, k)


def _boxes(char, starts, size=1):
    return [InkBox(char, start, size) for start in starts]


def test_class_coherence_uses_only_characters_with_enough_glyphs():
    """19 'a' glyphs alternate between two phases, so their coherence is 1/19; 20 'b' glyphs share one phase."""
    a = _boxes("a", [14 * i + 7 * (i % 2) for i in range(19)])
    b = _boxes("b", [14 * i for i in range(20)])
    assert grid.class_coherence(a + b, 14) == pytest.approx(1.0, abs=1e-12)
    assert grid.class_coherence(a + b, 14, min_count=19) == pytest.approx((19 * (1 / 19) + 20 * 1.0) / 39)


def test_class_coherence_refuses_boxes_without_a_class_large_enough():
    with pytest.raises(ValueError, match="20"):
        grid.class_coherence(_boxes("a", range(19)), 14)


@pytest.mark.parametrize(("shift", "expected"), [(0, 1.0), (4, 0.0)])
def test_straddle_share_counts_a_glyph_cut_by_a_patch_edge(shift, expected):
    assert grid.straddle_share([InkBox("x", 10, 5)], 14, shift) == expected


def test_straddle_share_is_the_share_of_cut_glyphs():
    boxes = [InkBox("a", 0, 14), InkBox("b", 13, 2), InkBox("c", 14, 1), InkBox("d", 20, 10)]  # b and d are cut
    assert grid.straddle_share(boxes, 14) == 0.5


def test_straddle_share_refuses_no_boxes_or_a_bad_patch():
    with pytest.raises(ValueError):
        grid.straddle_share([], 14)
    with pytest.raises(ValueError):
        grid.straddle_share([InkBox("x", 0, 1)], 0)


# Phase levels of a block (section 2.3, D10): the range, the step and the aliasing rule T_app = lcm(sigma, T).

V = {
    "render_mode": "conventional",
    "line_pitch_mode": "aligned",
    "axis": "y",
    "range": "cell",
    "step_px": {"qwen": 2, "smolvlm": 2, "smolvlm500m": 4},
}
C = {
    "render_mode": "conventional",
    "line_pitch_mode": "unaligned",
    "axis": "x",
    "range": "sweep",
    "step_px": {"qwen": 7, "smolvlm": 2, "smolvlm500m": 4},
}


def test_phase_levels_qwen_v_steps_two_pixels_over_the_cell():
    assert grid.phase_levels(V, READERS["qwen"], block="V") == [float(k) for k in range(0, 28, 2)]


def test_phase_levels_qwen_c_steps_seven_pixels_over_the_sweep_period():
    levels = grid.phase_levels(C, READERS["qwen"], block="C")
    assert levels == [float(k) for k in range(0, 112, 7)]
    assert len(levels) == 16


def test_phase_levels_smolvlm500m_c_steps_four_pixels():
    levels = grid.phase_levels(C, READERS["smolvlm500m"], block="C")
    assert levels == [float(k) for k in range(0, 64, 4)]
    assert len(levels) == 16


@pytest.mark.parametrize(("reader", "sigma", "expected"), [("qwen", 2, 14), ("smolvlm", 2, 21), ("smolvlm500m", 4, 16)])
def test_phase_levels_counts_of_block_v(reader, sigma, expected):
    assert len(grid.phase_levels({"range": "cell", "step_px": sigma}, READERS[reader])) == expected


@pytest.mark.parametrize(("reader", "expected"), [("qwen", 14), ("smolvlm", 14), ("smolvlm500m", 16)])
def test_phase_levels_block_h_has_one_level_per_pixel_of_the_patch(reader, expected):
    assert len(grid.phase_levels({"range": "patch", "step_px": 1}, READERS[reader])) == expected


@pytest.mark.parametrize(("reader", "sigma", "apparent"), [("qwen", 4, 28), ("smolvlm", 3, 42)])
def test_phase_levels_refuses_a_step_that_aliases_the_patch_period(reader, sigma, apparent):
    # A 14 px curve sampled every 4 px repeats every 28 px, and every 3 px every 42 px: T_app = lcm(sigma, T).
    with pytest.raises(ValueError, match=f"block C, reader {reader}.*{apparent} px"):
        grid.phase_levels({**C, "step_px": sigma}, READERS[reader], block="C")


def test_phase_levels_returns_an_explicit_list_as_floats():
    assert grid.phase_levels({"phases_px": [0, 0.25, 0.5, 0.75]}, READERS["qwen"]) == [0.0, 0.25, 0.5, 0.75]


def test_phase_levels_refuses_an_explicit_phase_outside_the_patch():
    with pytest.raises(ValueError, match=r"\[0, 14\)"):
        grid.phase_levels({"phases_px": [0.0, 14]}, READERS["qwen"])


@pytest.mark.parametrize(
    ("spec", "key"),
    [
        ({"phases_px": ["TODO"]}, "phases_px"),
        ({"range": "cell", "step_px": "from the ladder"}, "step_px"),
        ({"range": "cell", "step_px": {"qwen": "2"}}, "step_px"),
    ],
)
def test_phase_levels_names_the_key_when_a_string_stands_for_a_number(spec, key):
    with pytest.raises(ValueError, match=key):
        grid.phase_levels(spec, READERS["qwen"])


@pytest.mark.parametrize(
    "spec",
    [
        {"range": "window", "step_px": 1},
        {"range": "cell"},
        {"step_px": 2},
        {"phases_px": [0.0], "range": "cell", "step_px": 2},
        {"range": "cell", "step_px": {"smolvlm": 2}},
        {"range": "cell", "step_px": 0},
        {"range": "cell", "step_px": True},
        {"phases_px": []},
        {},
    ],
)
def test_phase_levels_refuses_a_malformed_block(spec):
    with pytest.raises(ValueError):
        grid.phase_levels(spec, READERS["qwen"])
