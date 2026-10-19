# PURPOSE: Study 3 stage 0's computations: each reader's acuity curve (CER against font size) and what it implies, the sizes for a target CER and the characters a vision token carries there (design note R7, sections 2 and 3).
# WHAT GOES HERE: The curve and its inverse; the curve's fit to pooled cells; the characters-per-page model; the cells of one reader and text type; the passage bootstrap of every fitted quantity.
# FILLED IN AT STEP: Study 3 stage 0 (19 October 2026), with scripts/25_fit_stage0.py.
# DO NOT: Average row CERs: pool the counts (D13). Resample rows or sizes instead of passages: page i of every size is one passage, and the passage is the unit.
"""Study 3 stage 0: a reader's acuity curve and the sizes and capacity it implies.

The curve (R7, formula F2) is

    CER(s) = e0 + (1 - e0) / (1 + (s / a) ** beta)

with s the font size in pixels, a the size at which CER is halfway between e0 and 1, beta the slope of the cliff and
e0 the floor that size cannot remove. Its inverse gives the size for a target CER eps,

    s_eps = a * ((1 - eps) / (eps - e0)) ** (1 / beta),

infinite when eps <= e0: no size reaches the target. A cell is one size of one reader and text type; its CER pools the
errors (S + D + I) and the reference characters (H + S + D) of its pages, and is capped at 1 for the fit (a reader
that writes garbage past the reference has CER above 1, and the curve's ceiling is 1). The fit minimises the binomial
deviance of the capped cell CERs, each cell weighted by its number of pages (the page is the unit, D13).

The reported fit has two parameters, a and beta, with e0 = 0. A free floor is not identified when the top of a ladder
still sits on the falling tail: on Study 1's block Z it took the value of the last cell (qwen random e0 = 0.037, its
24 px CER), which made every smaller target unreachable. The free-floor fit is kept as a check (``free_floor=True``):
a floor well above zero on a ladder that reaches past the reader's 1% size is a real floor.

Characters a vision token carries at size s: the characters of a full page at s over the page's vision tokens. A full
page's characters follow chars(s) = C * s ** -gamma (gamma near 2), fitted on the cells' mean reference characters, so
that the capacity can be read between the measured sizes.

Intervals resample passages: in a stage-0 sweep page i of every size starts at the same passage (block Z,
sweep.plan_pages), so one resample draws passages with replacement and keeps each drawn passage's pages at every size.
"""

from __future__ import annotations

import dataclasses
import math
import re
import zlib
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

EPSILONS = (0.05, 0.01)  # the target CERs whose sizes are reported (R7, Tier B)
BETA_BOUNDS = (0.5, 40.0)
E0_BOUNDS = (0.0, 0.5)
A_RANGE_FACTOR = 4.0  # a may lie up to this factor outside the measured sizes
P_CLIP = 1e-9  # the curve is kept inside (P_CLIP, 1 - P_CLIP) in the deviance
ALPHA = 0.05  # 95% intervals
_PAGE_INDEX = re.compile(r"^(?P<book>.+?)_z[0-9.]+_(?:conv|pa)_p(?P<index>\d{4})$")


def cer_curve(size: Any, a: float, beta: float, e0: float = 0.0) -> np.ndarray:
    """CER at ``size`` (a number or an array) on the curve with half-error size ``a``, slope ``beta`` and floor ``e0``."""
    s = np.asarray(size, dtype=np.float64)
    return e0 + (1.0 - e0) / (1.0 + (s / a) ** beta)


def size_at(eps: float, a: float, beta: float, e0: float = 0.0) -> float:
    """The size at which the curve reaches CER ``eps``: inf when ``eps <= e0`` (never reached), nan when eps >= 1."""
    if not 0.0 < eps < 1.0:
        return math.nan
    if eps <= e0:
        return math.inf
    return float(a * ((1.0 - eps) / (eps - e0)) ** (1.0 / beta))


def _deviance(theta: np.ndarray, sizes: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted binomial deviance of proportions ``y`` against the curve at ``theta`` = (log a, log beta, e0)."""
    p = np.clip(cer_curve(sizes, math.exp(theta[0]), math.exp(theta[1]), theta[2]), P_CLIP, 1.0 - P_CLIP)
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(y > 0, y * np.log(y / p), 0.0)
        term2 = np.where(y < 1, (1.0 - y) * np.log((1.0 - y) / (1.0 - p)), 0.0)
    return float(2.0 * np.sum(w * (term1 + term2)))


@dataclasses.dataclass(frozen=True)
class CurveFit:
    """A fitted curve: half-error size ``a`` (px), slope ``beta``, floor ``e0``, its deviance and whether it converged."""

    a: float
    beta: float
    e0: float
    deviance: float
    converged: bool

    def size_at(self, eps: float) -> float:
        return size_at(eps, self.a, self.beta, self.e0)


def fit_curve(sizes: Sequence[float], cer: Sequence[float], weights: Sequence[float] | None = None,
              starts: Sequence[tuple[float, float, float]] | None = None, free_floor: bool = False) -> CurveFit:
    """Fit the curve to cells: ``sizes`` (px), pooled ``cer`` (capped at 1 here) and ``weights`` (pages a cell).

    Minimises ``_deviance`` with L-BFGS-B from each start (a, beta, e0) and keeps the lowest. Without ``starts``, a grid
    of every measured size for a, slopes 2, 5 and 10, and floors 0.001 and 0.02. Bounds: a within ``A_RANGE_FACTOR``
    of the measured sizes, ``BETA_BOUNDS``, and e0 = 0 unless ``free_floor`` (then ``E0_BOUNDS``). Raises ValueError
    with fewer than three cells.
    """
    s = np.asarray(sizes, dtype=np.float64)
    y = np.minimum(np.asarray(cer, dtype=np.float64), 1.0)
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=np.float64)
    if s.ndim != 1 or s.size < 3 or y.shape != s.shape or w.shape != s.shape:
        raise ValueError("fit_curve needs three or more cells, with one CER and one weight for each size")
    if np.any(s <= 0) or np.any(y < 0) or np.any(w < 0):
        raise ValueError("sizes must be positive, CERs and weights non-negative")
    bounds = [(math.log(s.min() / A_RANGE_FACTOR), math.log(s.max() * A_RANGE_FACTOR)),
              (math.log(BETA_BOUNDS[0]), math.log(BETA_BOUNDS[1])), E0_BOUNDS if free_floor else (0.0, 0.0)]
    if starts is None:
        floors = (0.001, 0.02) if free_floor else (0.0,)
        starts = [(float(a0), b0, e00) for a0 in np.unique(s) for b0 in (2.0, 5.0, 10.0) for e00 in floors]
    best = None
    for a0, b0, e00 in starts:
        theta0 = np.array([math.log(a0), math.log(b0), e00])
        theta0 = np.clip(theta0, [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(_deviance, theta0, args=(s, y, w), method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    return CurveFit(a=math.exp(best.x[0]), beta=math.exp(best.x[1]), e0=float(best.x[2]), deviance=float(best.fun),
                    converged=bool(best.success))


def chars_model(sizes: Sequence[float], chars_per_page: Sequence[float]) -> tuple[float, float]:
    """``(C, gamma)`` of chars(s) = C * s ** -gamma, least squares on log chars against log size."""
    x = np.log(np.asarray(sizes, dtype=np.float64))
    z = np.log(np.asarray(chars_per_page, dtype=np.float64))
    if x.size < 2:
        raise ValueError("chars_model needs two or more sizes")
    slope, intercept = np.polyfit(x, z, 1)
    return float(math.exp(intercept)), float(-slope)


def chars_at(size: float, c: float, gamma: float) -> float:
    """Characters of a full page at ``size`` on the model of ``chars_model``: 0 at an infinite size."""
    return 0.0 if math.isinf(size) else float(c * size ** -gamma)


def passage_key(page_id: str) -> tuple[str, int]:
    """``(book_id, index)`` of a block Z page id ``<book>_z<size>_<conv|pa>_p<NNNN>``: the passage it starts at."""
    m = _PAGE_INDEX.match(str(page_id))
    if m is None:
        raise ValueError(f"{page_id!r} is not a block Z page id (<book>_z<size>_<conv|pa>_p<NNNN>)")
    return m["book"], int(m["index"])


def group_seed(seed: int, *names: str) -> np.random.SeedSequence:
    """A seed for one group from the analysis seed and its names, stable across runs and machines (CRC-32)."""
    return np.random.SeedSequence([int(seed), *(zlib.crc32(n.encode("utf-8")) for n in names)])


def _quantiles(values: np.ndarray) -> tuple[float, float]:
    """The 95% percentile interval of ``values``; inf values count as larger than any number, nan values are dropped."""
    v = values[~np.isnan(values)]
    if v.size == 0:
        return math.nan, math.nan
    big = np.finfo(np.float64).max
    q = np.quantile(np.where(np.isinf(v), big, v), [ALPHA / 2, 1 - ALPHA / 2], method="linear")
    return tuple(math.inf if x > 1e300 else float(x) for x in q)  # a bound beside an infinite resample is infinite


def fit_group(rows: Any, *, resamples: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cells and fit of one reader and text type: ``rows`` are its results rows (a DataFrame), one per page and size.

    A ``sweep`` column, when present, names each row's sweep; a passage is then (sweep, book, index), so that rows of
    two sweeps (Study 1's ladder and full block Z) never share a passage. Returns the cell rows (one per size) and the
    fit row: a and beta of the two-parameter curve, s_eps and the characters a vision token carries at s_eps for each
    of ``EPSILONS``, each with its passage-bootstrap interval; the characters-per-page model; the free-floor check
    (``floor_*``, no interval); and flags. ``status`` is ``ok`` when the measured cells straddle CER one half and a
    lies inside the measured sizes, ``a_below_range`` or ``a_above_range`` when a is extrapolated.
    ``s<eps>_in_range`` says whether s_eps lies inside the measured sizes, so that it is read off the cells and not
    extrapolated.
    """
    reader = str(rows["reader"].iloc[0])
    text_type = str(rows["text_type"].iloc[0])
    sweep = ";".join(sorted({str(v) for v in rows["sweep"]})) if "sweep" in rows else ""

    def key(r: Any) -> tuple[str, str, int]:
        return (str(getattr(r, "sweep", "")),) + passage_key(r.page_id)

    sizes = np.array(sorted({float(v) for v in rows["font_size_px"]}), dtype=np.float64)
    passages = sorted({key(r) for r in rows.itertuples(index=False)})
    p_index = {p: i for i, p in enumerate(passages)}
    s_index = {s: k for k, s in enumerate(sizes)}
    errors = np.zeros((len(passages), sizes.size))
    chars = np.zeros_like(errors)
    present = np.zeros_like(errors)
    for r in rows.itertuples(index=False):
        i, k = p_index[key(r)], s_index[float(r.font_size_px)]
        errors[i, k] += float(r.substitutions) + float(r.deletions) + float(r.insertions)
        chars[i, k] += float(r.hits) + float(r.substitutions) + float(r.deletions)
        present[i, k] += 1

    def pooled(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        e, n, w = errors[idx].sum(axis=0), chars[idx].sum(axis=0), present[idx].sum(axis=0)
        keep = n > 0
        return sizes[keep], e[keep] / n[keep], w[keep]

    cells: list[dict[str, Any]] = []
    vision_tokens = int(np.median(rows["vision_tokens"].to_numpy(dtype=np.float64)))
    mean_chars = []
    for k, s in enumerate(sizes):
        g = rows[rows["font_size_px"].astype(float) == s]
        ref = g["reference_chars"].to_numpy(dtype=np.float64)
        mean_chars.append(ref.mean())
        cells.append({
            "sweep": sweep, "reader": reader, "text_type": text_type, "font_size_px": float(s), "pages": int(len(g)),
            "reference_chars": int(chars[:, k].sum()), "errors": int(errors[:, k].sum()),
            "cer": float(errors[:, k].sum() / chars[:, k].sum()) if chars[:, k].sum() else math.nan,
            "chars_per_page": float(ref.mean()), "vision_tokens": vision_tokens,
            "chars_per_vision_token": float(ref.mean() / vision_tokens),
            "hit_limit_share": float(np.mean(g["hit_limit"].astype(str).str.lower() == "true")),
            "output_tokens_mean": float(g["output_tokens"].to_numpy(dtype=np.float64).mean()),
            "latency_s_mean": float(g["latency_s"].to_numpy(dtype=np.float64).mean()),
        })
    c_chars, gamma = chars_model(sizes, mean_chars)

    all_idx = np.arange(len(passages))
    s_fit, y_fit, w_fit = pooled(all_idx)
    point = fit_curve(s_fit, y_fit, w_fit)
    floor = fit_curve(s_fit, y_fit, w_fit, free_floor=True)  # the check of the module docstring, no interval

    def derived(fit: CurveFit) -> list[float]:
        out = [fit.a, fit.beta]
        for eps in EPSILONS:
            s_eps = fit.size_at(eps)
            out += [s_eps, chars_at(s_eps, c_chars, gamma) / vision_tokens]
        return out

    names = ["a", "beta"] + [f"{q}{round(eps * 100):02d}" for eps in EPSILONS for q in ("s", "kappa")]
    rng = np.random.default_rng(group_seed(seed, reader, text_type))
    starts = [(point.a, point.beta, 0.0), (float(np.median(sizes)), 5.0, 0.0)]
    boot = np.full((resamples, len(names)), np.nan)
    for b in range(resamples):
        idx = rng.integers(0, len(passages), size=len(passages))
        sb, yb, wb = pooled(idx)
        if sb.size >= 3:
            boot[b] = derived(fit_curve(sb, yb, wb, starts=starts))

    capped = np.minimum(y_fit, 1.0)
    if capped.min() >= 0.5:
        status = "a_above_range"
    elif capped.max() < 0.5:
        status = "a_below_range"
    else:
        status = "ok" if sizes.min() <= point.a <= sizes.max() else ("a_below_range" if point.a < sizes.min()
                                                                     else "a_above_range")
    fit_row: dict[str, Any] = {
        "sweep": sweep, "reader": reader, "text_type": text_type, "status": status,
        "sizes_px": ";".join(f"{s:g}" for s in sizes), "passages": len(passages),
        "pages_min": int(present.sum(axis=0).min()), "vision_tokens": vision_tokens,
    }
    for j, (name, value) in enumerate(zip(names, derived(point))):
        lo, hi = _quantiles(boot[:, j])
        fit_row.update({name: value, f"{name}_lo": lo, f"{name}_hi": hi})
    for eps in EPSILONS:
        tag = f"{round(eps * 100):02d}"
        fit_row[f"s{tag}_in_range"] = bool(sizes.min() <= fit_row[f"s{tag}"] <= sizes.max())
    fit_row.update({"chars_c": c_chars, "chars_gamma": gamma, "deviance": point.deviance,
                    "converged": point.converged, "resamples": int(resamples),
                    "resamples_fitted": int(np.sum(~np.isnan(boot[:, 0]))),
                    "floor_e0": floor.e0, "floor_a": floor.a, "floor_beta": floor.beta,
                    "floor_s01": floor.size_at(0.01), "floor_deviance": floor.deviance})
    return cells, fit_row
