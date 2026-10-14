# PURPOSE: Tests for every sweep config in configs/sweeps/ (build guide section 2.3).
# WHAT GOES HERE: CPU checks of the blocks: phase levels, condition counts, the header's run count, the determinism pair, placeholders.
# FILLED IN AT STEP: W2.2.
# DO NOT: Load a model. Edit a config to make a test pass after its run has started; copy it to a new file instead.
"""Tests for the sweep configs."""

from __future__ import annotations

import copy
import dataclasses
import math
import pathlib
import re

import pytest
import yaml

from mrr import grid
from mrr.sweep import HASH_KEYS, expand_conditions

ROOT = pathlib.Path(__file__).resolve().parents[1]
SWEEPS = sorted((ROOT / "configs" / "sweeps").glob("*.yaml"))
PILOT = ROOT / "configs" / "sweeps" / "pilot.yaml"
LADDER = ROOT / "configs" / "sweeps" / "ladder.yaml"
READERS = {
    reader["name"]: reader
    for reader in yaml.safe_load((ROOT / "configs" / "readers.yaml").read_text(encoding="utf-8"))["readers"]
}
PILOT_CONDITIONS = {"qwen": 50, "smolvlm": 62, "smolvlm500m": 54}  # per page per text type, section 2.3


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _concrete(sweep, size=16):
    """Return a copy whose placeholder font sizes (strings such as "from the ladder") are replaced by ``size``."""
    sweep = copy.deepcopy(sweep)
    sizes = sweep.get("font_size_px")
    if isinstance(sizes, str):
        sweep["font_size_px"] = size
    elif isinstance(sizes, dict):
        sweep["font_size_px"] = {name: size if isinstance(value, str) else value for name, value in sizes.items()}
    return sweep


def _pages_per_reader(sweep):
    """Return the pages per reader from the config's counts, or None while a count is still a placeholder."""
    counts = [sweep["pages"][kind]["per_text_type"] for kind in ("natural", "random62")]
    if any(isinstance(count, str) for count in counts):
        return None
    return sum(counts)


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_every_block_passes_phase_levels_for_every_reader(path):
    sweep = _load(path)
    for reader in sweep["readers"]:
        for name, block in sweep["blocks"].items():
            levels = grid.phase_levels(block, READERS[reader], block=name)
            assert levels and levels[0] == 0.0 and len(set(levels)) == len(levels)


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_every_block_expands_once_the_sizes_are_numbers(path):
    sweep = _concrete(_load(path))
    for reader in sweep["readers"]:
        conditions = expand_conditions(sweep, READERS[reader])
        assert len({dataclasses.astuple(c) for c in conditions}) == len(conditions)
        assert {c.block for c in conditions} == set(sweep["blocks"])


@pytest.mark.parametrize(("reader", "expected"), sorted(PILOT_CONDITIONS.items()))
def test_pilot_conditions_per_page_per_text_type(reader, expected):
    assert len(expand_conditions(_concrete(_load(PILOT)), READERS[reader])) == expected


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_header_run_count_equals_the_computed_count(path):
    # "total runs = 3320" on its own line, or "...; total runs = 216." at the end of the ladder's one-line formula.
    header = re.search(r"total runs = (TODO\S*|\d+)", path.read_text(encoding="utf-8"))
    assert header, "the header comment must state the total run count"
    sweep = _load(path)
    pages = _pages_per_reader(sweep)
    if header.group(1).startswith("TODO"):
        assert pages is None, "a config whose page counts are set must state its run count"
        pytest.skip(f"{path.name}: the run count is not set yet")
    sweep = _concrete(sweep)
    total = sum(len(expand_conditions(sweep, READERS[r])) * pages * sweep["repeats"] for r in sweep["readers"])
    assert int(header.group(1)) == total


def test_h0_and_s0_differ_only_in_their_block_label():
    sweep = _concrete(_load(PILOT))
    for reader in sweep["readers"]:
        conditions = expand_conditions(sweep, READERS[reader])
        at_zero = {c.block: c for c in conditions if c.block in "HS" and (c.phase_x_px, c.phase_y_px) == (0.0, 0.0)}
        assert set(at_zero) == {"H", "S"}
        assert dataclasses.replace(at_zero["H"], block="S") == at_zero["S"]


def test_block_v_uses_the_aligned_pitch_and_block_h_the_patch_slot():
    sweep = _concrete(_load(PILOT))
    for reader, cell in (("qwen", 28), ("smolvlm", 42), ("smolvlm500m", 64)):
        conditions = expand_conditions(sweep, READERS[reader])
        v = [c for c in conditions if c.block == "V"]
        h = [c for c in conditions if c.block == "H"]
        assert {c.line_pitch_px for c in v} == {cell}
        assert all(c.patch_aware_slot_px is None and c.phase_x_px == 0.0 for c in v)
        assert {c.patch_aware_slot_px for c in h} == {READERS[reader]["patch_size"]}
        assert all(c.phase_y_px == 0.0 and c.line_pitch_px == 20 and c.paginate == "shared" for c in h)


def test_conditions_carry_the_hash_keys_as_floats_and_the_file_name_only():
    conditions = expand_conditions(_concrete(_load(PILOT)), READERS["qwen"])
    cond = next(c for c in conditions if c.block == "C" and c.phase_x_px == 7.0)
    assert tuple(cond.hash_keys()) == HASH_KEYS
    assert cond.hash_keys() == {  # the test vector of section 2.5
        "aa_mode": "coverage_srgb",
        "block": "C",
        "font_file": "SourceSans3-Regular.ttf",
        "font_size_px": 16.0,
        "line_pitch_mode": "unaligned",
        "line_pitch_px": 20.0,
        "patch_aware_slot_px": None,
        "phase_x_px": 7.0,
        "phase_y_px": 0.0,
        "render_mode": "conventional",
    }


def test_ladder_block_z_gives_twelve_conditions_per_reader_each_on_its_own_pages():
    sweep = _load(LADDER)
    for reader in sweep["readers"]:
        conditions = expand_conditions(sweep, READERS[reader])
        assert len(conditions) == 12
        pairs = {(mode, size) for mode in ("conventional", "patch_aware") for size in (10, 12, 14, 16, 20, 24)}
        assert {(c.render_mode, c.font_size_px) for c in conditions} == pairs
        assert all(c.paginate == "per_condition" and (c.phase_x_px, c.phase_y_px) == (0.0, 0.0) for c in conditions)
        assert all(c.line_pitch_px == math.floor(1.25 * c.font_size_px + 0.5) for c in conditions)


def test_pilot_placeholders_stop_the_expansion_naming_the_key():
    # The pilot's sizes read "from the ladder" until the ladder has run; a string where a number belongs must raise.
    sweep = _concrete(_load(PILOT))
    sweep["font_size_px"] = {reader: 16 for reader in sweep["readers"]}
    sweep["font_size_px"]["qwen"] = "from the ladder"
    with pytest.raises(ValueError, match="font_size_px"):
        expand_conditions(sweep, READERS["qwen"])
    assert len(expand_conditions(sweep, READERS["smolvlm"])) == PILOT_CONDITIONS["smolvlm"]


@pytest.mark.parametrize(
    ("block", "key", "value"),
    [
        ("H", "step_px", "one"),
        ("S", "phases_px", ["TODO"]),
        ("A", "aa_mode", ["binary", "coverage"]),
        ("V", "line_pitch_mode", "snapped"),
        ("H", "render_mode", "mono"),
        ("C", "paginate", "per_page"),
        ("H", "step", 1),
    ],
)
def test_a_bad_block_value_stops_the_expansion_naming_the_key(block, key, value):
    sweep = _concrete(_load(PILOT))
    sweep["blocks"][block][key] = value
    with pytest.raises(ValueError, match=key):
        expand_conditions(sweep, READERS["qwen"])


def test_a_sweep_without_blocks_is_refused():
    with pytest.raises(ValueError, match="blocks"):
        expand_conditions({"factors": {"phase_x_px": [0, 7]}}, READERS["qwen"])
