# PURPOSE: Correctness tests for CER and WER.
# WHAT GOES HERE: Known-value checks.
# FILLED IN AT STEP: W2.4 (remove the todo mark once score.py is implemented).
# DO NOT: Change expected values to match an implementation.
"""Tests for mrr.score."""

from __future__ import annotations

import pytest

from mrr.score import cer

pytestmark = pytest.mark.todo  # TODO[W2.4]: remove once score.py is implemented


def test_cer_transposition():
    assert abs(cer("the cat", "teh cat") - 2 / 7) < 1e-9


@pytest.mark.parametrize("x", ["a", "the cat", "Rendering for Machine Readers"])
def test_cer_identity(x):
    assert cer(x, x) == 0
