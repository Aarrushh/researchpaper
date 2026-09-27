# PURPOSE: Correctness tests for paired statistics.
# WHAT GOES HERE: Reproducibility and degenerate-case checks.
# FILLED IN AT STEP: W2.4 (remove the todo mark once stats.py is implemented).
# DO NOT: Use unseeded randomness in tests.
"""Tests for mrr.stats."""

from __future__ import annotations

import numpy as np
import pytest

from mrr.stats import bootstrap_ci

pytestmark = pytest.mark.todo  # TODO[W2.4]: remove once stats.py is implemented


def test_bootstrap_reproducible():
    values = np.random.default_rng(0).normal(size=50)
    a = bootstrap_ci(values, np.median, n_resamples=2000, alpha=0.05, seed=42)
    b = bootstrap_ci(values, np.median, n_resamples=2000, alpha=0.05, seed=42)
    assert a == b


def test_bootstrap_constant_zero_width():
    low, high = bootstrap_ci(np.full(30, 0.1), np.median, n_resamples=1000, alpha=0.05, seed=0)
    assert low == high
