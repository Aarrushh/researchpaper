# PURPOSE: The paper's figures.
# WHAT GOES HERE: Figure 1 (phase curve) and Figure 2 (accuracy versus tokens trade-off).
# FILLED IN AT STEP: W2.5 (Figure 1), W3.4 (Figure 2).
# DO NOT: Hand-edit output images. Do not import matplotlib at module top level.
"""Figures.

Learn B5 (plotting) before implementing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_phase_curve(df: Any, out_path: str | Path) -> None:
    """Figure 1: CER versus glyph phase, one line per reader, bootstrap band.

    Args:
        df: pandas DataFrame of results rows.
        out_path: output image path under reports/figures/.
    """
    raise NotImplementedError("TODO[W2.5]: plot CER against phase per reader with a bootstrap band")


def plot_tradeoff(df: Any, out_path: str | Path) -> None:
    """Figure 2: CER versus vision tokens, conventional versus patch-aware.

    Args:
        df: pandas DataFrame of results rows.
        out_path: output image path under reports/figures/.
    """
    raise NotImplementedError("TODO[W3.4]: plot CER against vision tokens for each render mode")
