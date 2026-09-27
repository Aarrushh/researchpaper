# PURPOSE: sRGB <-> linear-light conversion used by coverage compositing.
# WHAT GOES HERE: The two transfer functions, vectorised over NumPy arrays.
# FILLED IN AT STEP: W1.5.
# DO NOT: Use approximations such as a plain 2.2 power; the tests expect the exact piecewise curve.
"""Colour transfer functions.

Learn A4 (Colour and light) before implementing.

The sRGB transfer function (IEC 61966-2-1) is piecewise:

    decode (sRGB value v -> linear l):
        l = v / 12.92                        if v <= 0.04045
        l = ((v + 0.055) / 1.055) ** 2.4     otherwise

    encode (linear l -> sRGB value v):
        v = 12.92 * l                        if l <= 0.0031308
        v = 1.055 * l ** (1 / 2.4) - 0.055   otherwise

TODO[W1.5]: verify constants against a primary or reliable secondary source and log it in citations.md
"""

from __future__ import annotations

import numpy as np


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """Decode sRGB-encoded values to linear light.

    Args:
        v: float64 array of sRGB-encoded values in [0, 1].

    Returns:
        float64 array of the same shape with linear-light values in [0, 1].
    """
    raise NotImplementedError("TODO[W1.5]: implement the piecewise sRGB decode in the module docstring")


def linear_to_srgb(l: np.ndarray) -> np.ndarray:
    """Encode linear-light values to sRGB.

    Args:
        l: float64 array of linear-light values in [0, 1].

    Returns:
        float64 array of the same shape with sRGB-encoded values in [0, 1].
    """
    raise NotImplementedError("TODO[W1.5]: implement the piecewise sRGB encode in the module docstring")
