# PURPOSE: Statistics for the paired design.
# WHAT GOES HERE: Wilcoxon signed-rank test, bootstrap CIs, median paired difference.
# FILLED IN AT STEP: W2.4.
# DO NOT: Use an unpaired test. Do not use an unseeded random generator.
"""Paired statistics.

Learn B4 (statistics) before implementing.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test on paired samples.

    Args:
        a: CER per page under condition A.
        b: CER per page under condition B, same page order.

    Returns:
        Dict with keys "statistic", "pvalue", "n".
    """
    raise NotImplementedError("TODO[W2.4]: call scipy.stats.wilcoxon with the preregistered alternative")


def bootstrap_ci(
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_resamples: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval.

    Args:
        values: 1D array of per-page values (for example paired differences).
        stat_fn: statistic to compute on each resample (for example np.median).
        n_resamples: number of bootstrap resamples.
        alpha: 1 minus confidence level (0.05 for a 95% interval).
        seed: seed for numpy.random.default_rng.

    Returns:
        (low, high) interval bounds.
    """
    raise NotImplementedError("TODO[W2.4]: resample with replacement using a seeded generator")


def median_paired_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Median of (b - a) over paired pages.

    Args:
        a: values under condition A.
        b: values under condition B, same page order.

    Returns:
        Median paired difference.
    """
    raise NotImplementedError("TODO[W2.4]: return the median of b - a")
