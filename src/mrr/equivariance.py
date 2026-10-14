# PURPOSE: The shift-equivariance error E(s) of a reader's vision tokens, for paper Figure 0.
# WHAT GOES HERE: Moving token columns, equation (1) of build guide section 2.1 for one shift, the curve over shifts, and the four numbers of appendix C.3 for one curve.
# FILLED IN AT STEP: W1.9.
# DO NOT: Import torch or transformers; this module is NumPy only and runs in CI. Never pass Qwen block outputs that are still in window order (readers.qwen_to_raster undoes it).
"""Shift-equivariance error.

Learn A7 (shift variance of strided convolution) before implementing.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def shift_right(z: np.ndarray, k: int, fill: float = 0.0) -> np.ndarray:
    """Return z with its columns moved k to the right; the k new columns on the left hold fill.

    Columns are axis 1, as in the (rows, cols, D) arrays of readers.vision_hidden_states, and columns moved
    past the right edge are dropped. A page moved by k whole patches should give these tokens, apart from
    the page edges, which is what the tests use it for.

    Args:
        z: array with at least two axes.
        k: columns to move, 0 or more.
        fill: value of the new columns.

    Returns:
        A new array of z's shape and dtype.
    """
    if z.ndim < 2:
        raise ValueError(f"z needs a column axis (axis 1), got shape {z.shape}")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError(f"k must be a whole number of columns, 0 or more, got {k!r}")
    n = z.shape[1]
    out = np.full_like(z, fill)
    if k < n:
        out[:, k:] = z[:, : n - k]
    return out


def e_value(z0: np.ndarray, zs: np.ndarray, zw: np.ndarray, s_px: float, patch_px: int) -> tuple[float, int]:
    """Equation (1) of build guide section 2.1: E(s) after undoing the best whole-patch move.

        E(s) = min over k in {floor(s/P), ceil(s/P)} of ||zs[:, k:] - z0[:, :n-k]|| / ||z0[:, :n-k] - zw[:, :n-k]||

    n is the number of token columns and the norm is the Frobenius norm over the compared columns. The
    denominator is how far the text page is from a blank page in those columns, so E = 1 means the shift
    moved the tokens as far as taking the text away would. With k = 0 the slices hold every column.

    Args:
        z0: tokens of the page at shift 0, (rows, cols, D) in raster order.
        zs: tokens of the page moved s px to the right, the same shape.
        zw: tokens of a blank page of the background colour, the same shape.
        s_px: the shift in pixels, 0 or more.
        patch_px: the reader's patch size P.

    Returns:
        (E, k): the smaller of the two values and the move that gave it; on a tie, the smaller move.

    Raises:
        ValueError: if the shapes differ, s or P is out of range, a move leaves no column to compare, or the
            text page equals the blank page in the compared columns.
    """
    if not (z0.shape == zs.shape == zw.shape) or z0.ndim < 2:
        raise ValueError(
            f"z0, zs and zw must share one shape with a column axis, got {z0.shape}, {zs.shape}, {zw.shape}"
        )
    if isinstance(patch_px, bool) or not isinstance(patch_px, (int, np.integer)) or patch_px < 1:
        raise ValueError(f"patch_px must be a positive whole number, got {patch_px!r}")
    if not (math.isfinite(s_px) and s_px >= 0):
        raise ValueError(f"s_px must be finite and 0 or more, got {s_px!r}")
    n = z0.shape[1]
    best: tuple[float, int] | None = None
    for k in sorted({math.floor(s_px / patch_px), math.ceil(s_px / patch_px)}):
        if k >= n:
            raise ValueError(f"a move of {k} columns leaves none of the {n} to compare (s = {s_px} px, P = {patch_px})")
        signal = float(np.linalg.norm(z0[:, : n - k] - zw[:, : n - k]))
        if signal == 0.0:
            raise ValueError("the text page equals the blank page in the compared columns")
        e = float(np.linalg.norm(zs[:, k:] - z0[:, : n - k])) / signal
        if best is None or e < best[0]:
            best = (e, int(k))
    assert best is not None
    return best


def _as_layers(z: np.ndarray | Mapping[Any, np.ndarray]) -> tuple[list[Any], list[np.ndarray]]:
    """Return (keys, arrays): a mapping's items in order, or one unnamed layer for a plain array."""
    if isinstance(z, Mapping):
        return list(z.keys()), [np.asarray(v) for v in z.values()]
    return [None], [np.asarray(z)]


def e_curve(
    render_fn: Callable[[float], np.ndarray],
    embed_fn: Callable[[np.ndarray], np.ndarray | Mapping[Any, np.ndarray]],
    shifts_px: Sequence[float],
    patch_px: int,
    blank: np.ndarray,
) -> np.ndarray:
    """E(s) and the move it used, for every shift and every layer embed_fn returns.

    Copied from e_curve in the notes' scripts/make_figures.py:1108. z0 = embed_fn(render_fn(0.0)) and
    zw = embed_fn(blank) are computed once, then each shift costs one render and one embedding, so a
    forward pass that records several layers serves all of them. A shift of 0 embeds the page a second
    time, which checks that two forward passes agree.

    Args:
        render_fn: shift in px -> the page moved that far to the right.
        embed_fn: page -> tokens (rows, cols, D), or a mapping layer -> tokens such as
            readers.vision_hidden_states returns; every call must return the same layers in the same order.
        shifts_px: the shifts, each 0 or more.
        patch_px: the reader's patch size P.
        blank: a page of the background colour, the size of the rendered pages.

    Returns:
        float64 array (len(shifts_px), n_layers, 2): [..., 0] is E and [..., 1] the move k, with the layers in
        embed_fn's order; a plain array from embed_fn is one layer.
    """
    keys, z0 = _as_layers(embed_fn(render_fn(0.0)))
    keys_w, zw = _as_layers(embed_fn(blank))
    if keys_w != keys:
        raise ValueError(f"the blank page gave layers {keys_w}, the page gave {keys}")
    out = np.empty((len(shifts_px), len(z0), 2), dtype=np.float64)
    for i, s in enumerate(shifts_px):
        keys_s, zs = _as_layers(embed_fn(render_fn(float(s))))
        if keys_s != keys:
            raise ValueError(f"shift {s} gave layers {keys_s}, the page gave {keys}")
        for j, (a, b, w) in enumerate(zip(z0, zs, zw)):
            out[i, j] = e_value(a, b, w, float(s), patch_px)
    return out


def e_summary(shifts_px: Sequence[float], e: Sequence[float], patch_px: int) -> dict[str, float]:
    """The four numbers of build guide appendix C.3 for one curve.

    - at_multiples_of_p: the largest E at s = P, 2P and so on;
    - at_quarter_px: E(0.25);
    - plateau: the mean E over the whole-pixel shifts from 1 px up that are not multiples of P;
    - largest: the largest E over every shift.

    Args:
        shifts_px: the shifts of the curve, which must include 0.25 px, a positive multiple of P and a
            whole-pixel shift that is not one.
        e: E at each shift.
        patch_px: the reader's patch size P.

    Returns:
        The four numbers, keyed as above.
    """
    s = np.asarray(shifts_px, dtype=np.float64)
    v = np.asarray(e, dtype=np.float64)
    if s.shape != v.shape or s.ndim != 1:
        raise ValueError(f"shifts_px and e must be 1-D and the same length, got {s.shape} and {v.shape}")
    whole = np.isclose(s, np.round(s), rtol=0, atol=1e-9) & (s >= 1)
    multiple = whole & np.isclose(np.mod(s, patch_px), 0, rtol=0, atol=1e-9)
    quarter = np.isclose(s, 0.25, rtol=0, atol=1e-9)
    if not (multiple.any() and (whole & ~multiple).any() and quarter.sum() == 1):
        raise ValueError("the shifts need 0.25 px, a positive multiple of P and a whole-pixel shift that is not one")
    return {
        "at_multiples_of_p": float(v[multiple].max()),
        "at_quarter_px": float(v[quarter][0]),
        "plateau": float(v[whole & ~multiple].mean()),
        "largest": float(v.max()),
    }
