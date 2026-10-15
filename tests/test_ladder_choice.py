# PURPOSE: Tests for scripts/09_ladder_choice.py, the ladder's size choice (W1.10, build guide section 2.9).
# WHAT GOES HERE: The closest-to-target rule, the tie rule, the refusal of an empty ladder.
# FILLED IN AT STEP: W1.10, under D26.
# DO NOT: Load a model.
"""Tests for the ladder's size choice."""

from __future__ import annotations

import importlib.util
import sys

import pytest

from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "09_ladder_choice.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("ladder_choice", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ladder_choice"] = module
    spec.loader.exec_module(module)
    return module


lc = _load_script()


def test_the_size_closest_to_the_target_is_chosen():
    assert lc.choose({10: 0.9, 12: 0.4, 14: 0.16, 16: 0.05, 20: 0.01}, 0.15) == 14


def test_a_tie_goes_to_the_larger_size_even_when_the_floats_differ_in_the_last_bit():
    assert abs(0.10 - 0.15) != abs(0.20 - 0.15)  # the floats alone would split this tie
    assert lc.choose({12: 0.20, 14: 0.10}, 0.15) == 14
    assert lc.choose({12: 0.10, 14: 0.20}, 0.15) == 14


def test_a_ladder_that_never_reaches_the_target_takes_the_nearest_end():
    assert lc.choose({10: 1.0, 12: 0.9, 24: 0.4}, 0.15) == 24
    assert lc.choose({10: 0.05, 12: 0.01, 24: 0.0}, 0.15) == 10


def test_an_empty_ladder_is_refused():
    with pytest.raises(ValueError):
        lc.choose({}, 0.15)
