# PURPOSE: Correctness tests for sRGB <-> linear conversion.
# WHAT GOES HERE: Round-trip, continuity and a known blend value.
# FILLED IN AT STEP: W1.5 (remove the todo mark once color.py is implemented).
# DO NOT: Loosen tolerances to make an implementation pass.
"""Tests for mrr.color."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mrr.color import linear_to_srgb, srgb_to_linear


def test_round_trip():
    x = np.linspace(0.0, 1.0, 100001, dtype=np.float64)
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-12, rtol=0)


def test_continuity_at_threshold():
    t = 0.04045
    eps = 1e-12
    below = srgb_to_linear(np.array([t - eps]))[0]
    above = srgb_to_linear(np.array([t + eps]))[0]
    assert abs(above - below) < 1e-8


def test_linear_blend_black_white():
    black = srgb_to_linear(np.array([0.0]))
    white = srgb_to_linear(np.array([1.0]))
    mixed = 0.5 * black + 0.5 * white
    encoded = int(np.round(linear_to_srgb(mixed)[0] * 255))
    assert encoded == 188


def test_decode_mid_grey():
    assert float(srgb_to_linear(np.array(128 / 255))) == pytest.approx(0.21586050011389926, rel=1e-12)


def test_encode_half():
    v = float(linear_to_srgb(np.array(0.5))) * 255
    assert v == pytest.approx(187.516, abs=1e-3)
    assert math.floor(v + 0.5) == 188
