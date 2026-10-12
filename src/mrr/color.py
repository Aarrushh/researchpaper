# PURPOSE: sRGB <-> linear-light conversion used by coverage compositing.
# WHAT GOES HERE: The two transfer functions, vectorised over NumPy arrays.
# FILLED IN AT STEP: W1.5.
# DO NOT: Approximate the transfer curve with one power law (D24).
"""Colour transfer functions.

Learn A4 (Colour and light) before implementing.

The sRGB transfer function (IEC 61966-2-1) is piecewise:

    decode (sRGB value v -> linear l):
        l = v / 12.92                        if v <= 0.04045
        l = ((v + 0.055) / 1.055) ** 2.4     otherwise

    encode (linear l -> sRGB value v):
        v = 12.92 * l                        if l <= 0.04045 / 12.92
        v = 1.055 * l ** (1 / 2.4) - 0.055   otherwise

The constants match the sample code in W3C CSS Color 4, section 19 (references/citations.md).
That code encodes with a threshold rounded to five significant figures. We use 0.04045 / 12.92,
where the decode's straight piece ends, so that encode inverts decode to within 2e-16 and
tests/test_color.py::test_round_trip holds; the rounded threshold misses by 3e-8 (D24).
"""

from __future__ import annotations

import numpy as np

_DECODE_KNEE = 0.04045
_ENCODE_KNEE = _DECODE_KNEE / 12.92   # 0.0031308049535603713, the knee that round-trips (D24)


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """Decode sRGB-encoded values to linear light.

    Args:
        v: array-like of sRGB-encoded values in [0, 1]; converted to float64.

    Returns:
        New float64 array of the same shape with linear-light values in [0, 1].
    """
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= _DECODE_KNEE, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(l: np.ndarray) -> np.ndarray:
    """Encode linear-light values to sRGB.

    Args:
        l: array-like of linear-light values in [0, 1]; converted to float64.

    Returns:
        New float64 array of the same shape with sRGB-encoded values in [0, 1].
    """
    l = np.asarray(l, dtype=np.float64)
    return np.where(l <= _ENCODE_KNEE, 12.92 * l, 1.055 * np.power(l, 1 / 2.4) - 0.055)
