# PURPOSE: Statistics for the sweep: the harmonic model with page fixed effects, Holm's correction, the page bootstrap, the paired Wilcoxon test, power and the Fano floor.
# WHAT GOES HERE: holm, bin_groups, harmonic_design, group_f_test, group_rms, harmonic_fit, harmonic_component, page_bootstrap_rms, pooled_cer_by_level, paired_wilcoxon, bootstrap_ci, median_paired_difference, fano_floor_bits, power_noncentrality, pages_needed, paired_mde.
# FILLED IN AT STEP: W2.4.
# DO NOT: Use an unpaired test. Do not use an unseeded random generator. Do not resample rows: resample pages. Do not fit two blocks or two readers in one model. Do not report a Wilcoxon p-value without its method. Do not compute an amplitude over rows: compute it at the levels.
"""Statistics for the phase sweep (build guide section 3.1; D13, D14; PREREGISTRATION.md sections 6 and 9).

Learn B6 (statistics) before implementing.

Five jobs. (i) The saturated harmonic model of PREREGISTRATION.md section 6: for one reader, one block and
one text type, row CER = page fixed effect + sum over the identifiable bins k of
b_k cos(2 pi k phi / L) + c_k sin(2 pi k phi / L). Each period group (the bins that `bin_groups` assigns to
one period) is tested with an F-test of the full model against the model without the group's columns, with
page dummies in both, and gets an RMS amplitude at the sampled levels and a share of the fitted harmonic
variance. (ii) Holm's correction within a family. (iii) The paired Wilcoxon signed-rank test over pages,
with the method that SciPy's ``method="auto"`` would choose resolved here and reported. (iv) The power
functions that turn the pilot's residual standard deviation into a page count. (v) The Fano floor that turns
a symbol-error probability into a lower bound on bits per symbol, for Figure 2.

Every resampling is by page, never by row: the rows of one page share its text and its fixed effect, so
rows are not exchangeable and the page is the unit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy import optimize
from scipy import stats as sps

__all__ = [
    "holm",
    "bin_groups",
    "harmonic_design",
    "group_f_test",
    "group_rms",
    "harmonic_fit",
    "harmonic_component",
    "page_bootstrap_rms",
    "pooled_cer_by_level",
    "paired_wilcoxon",
    "bootstrap_ci",
    "median_paired_difference",
    "fano_floor_bits",
    "power_noncentrality",
    "pages_needed",
    "paired_mde",
    "WITHIN_SS_FLOOR",
    "UndefinedTest",
]

# group_f_test: a within-page sum of squares at or below this fraction of y·y is rounding error, not variation.
# One row of a page off by one character in 500 gives about 4e-6 of a CER-squared, far above it.
WITHIN_SS_FLOOR = 1e-12


class UndefinedTest(ValueError):
    """An F-test that cannot be formed; ``cause`` names why in words that do not depend on the arithmetic."""

    def __init__(self, cause: str, detail: str):
        super().__init__(f"F-test undefined, {cause}: {detail}")
        self.cause = cause


# --------------------------------------------------------------------------- Holm


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm's step-down adjustment of one family of p-values, returned in the input order.

    Sort the m p-values, multiply the i-th smallest by m - i + 1, take the running maximum and clip at 1
    (PREREGISTRATION.md section 6). Raises ValueError on an empty family, a NaN or a value outside [0, 1].
    """
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("holm needs a non-empty one-dimensional sequence of p-values")
    if np.any(np.isnan(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p-values must lie in [0, 1] and must not be NaN")
    m = p.size
    order = np.argsort(p, kind="stable")
    scaled = np.minimum(1.0, p[order] * (m - np.arange(m)))
    adjusted = np.maximum.accumulate(scaled)
    out = np.empty(m, dtype=np.float64)
    out[order] = adjusted
    return out


# --------------------------------------------------------------------------- the design


def bin_groups(sweep_px: int, periods_px: Sequence[int], n_levels: int) -> dict[int, list[int]]:
    """Assign every identifiable bin k (1 <= k <= n_levels // 2) to the shortest tested period it belongs to.

    Bin k of a sweep of length L belongs to period T when k is a multiple of L / T (build guide section 2.3).
    Periods are tried shortest first, so a bin that fits several periods goes to the shortest. Every period
    must divide the sweep.
    """
    sweep_px = int(sweep_px)
    periods = sorted(int(T) for T in periods_px)
    if sweep_px <= 0 or n_levels <= 0 or not periods:
        raise ValueError("bin_groups needs a positive sweep, at least one level and at least one period")
    for T in periods:
        if T <= 0 or sweep_px % T != 0:
            raise ValueError(f"period {T} px does not divide the sweep of {sweep_px} px")
    groups: dict[int, list[int]] = {}
    for k in range(1, n_levels // 2 + 1):
        for T in periods:
            if k % (sweep_px // T) == 0:
                groups.setdefault(T, []).append(k)
                break
    return groups


def harmonic_design(phase_px, sweep_px: float, bins: Sequence[int]) -> tuple[np.ndarray, list[str]]:
    """Cosine and sine columns of the given bins at the given phases, dropping columns that are zero everywhere.

    Returns the design matrix (one row per phase) and the column names, ``cos<k>`` and ``sin<k>`` in bin
    order. A sine that vanishes at every sampled phase (the Nyquist bin of an even number of levels) is
    dropped, and a dropped column counts as a coefficient of 0 everywhere else in this module.
    """
    phase = np.asarray(phase_px, dtype=np.float64)
    bins = [int(k) for k in bins]
    if phase.ndim != 1 or phase.size == 0 or not bins:
        raise ValueError("harmonic_design needs a one-dimensional phase array and at least one bin")
    cols, names = [], []
    for k in bins:
        for name, fn in (("cos", np.cos), ("sin", np.sin)):
            c = fn(2 * np.pi * k * phase / sweep_px)
            if np.max(np.abs(c)) > 1e-9:
                cols.append(c)
                names.append(f"{name}{k}")
    return np.column_stack(cols), names


def _columns(names: Sequence[str], phase_px, sweep_px: float) -> np.ndarray:
    """The columns named in `names` at the given phases, without dropping any: the fitted curve's basis."""
    phase = np.asarray(phase_px, dtype=np.float64)
    cols = []
    for name in names:
        k = int(name[3:])
        fn = np.cos if name.startswith("cos") else np.sin
        cols.append(fn(2 * np.pi * k * phase / sweep_px))
    return np.column_stack(cols) if cols else np.zeros((phase.size, 0))


def _page_dummies(page_ids) -> tuple[np.ndarray, int]:
    pages, index = np.unique(np.asarray(page_ids), return_inverse=True)
    index = np.asarray(index).reshape(-1)
    return np.eye(len(pages))[index], len(pages)


# --------------------------------------------------------------------------- the F-test and the fit


def group_f_test(y, page_ids, X_harm, names: Sequence[str], group_cols: Sequence[str]) -> dict:
    """F-test of one period group: the full model against the model without the group's columns.

    Both models hold the page dummies (PREREGISTRATION.md section 6, equation (7)). Returns F, df1 (the
    rank the group adds), df2 (rows minus the full model's rank) and p. Raises ValueError when the test is
    undefined: no column added, no residual degree of freedom, or no variation within any page. The last
    case (every page's CER the same at every phase, as a reader at its floor can give) leaves both residual
    sums of squares at rounding level, about 1e-33, where their ratio is noise that can read as any F; it is
    caught on the within-page sum of squares, relative to y·y, before the ratio is formed.
    """
    y = np.asarray(y, dtype=np.float64)
    names = list(names)
    group = set(group_cols)
    missing = group - set(names)
    if missing:
        raise ValueError(f"group columns not in the design: {sorted(missing)}")
    D, _ = _page_dummies(page_ids)
    X_harm = np.asarray(X_harm, dtype=np.float64)
    X1 = np.column_stack([D, X_harm])
    X0 = np.column_stack([D, X_harm[:, [i for i, n in enumerate(names) if n not in group]]])

    def rss(X: np.ndarray) -> float:
        return float(np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2))

    rss1, rss0 = rss(X1), rss(X0)
    within = rss(D)  # the within-page sum of squares: zero when no page varies across phases
    p1, p0 = np.linalg.matrix_rank(X1), np.linalg.matrix_rank(X0)
    q, dfd = int(p1 - p0), int(len(y) - p1)
    floor = within <= WITHIN_SS_FLOOR * max(1.0, float(y @ y))
    if q <= 0 or dfd <= 0 or rss1 <= 0 or floor:
        cause = ("the group adds no column" if q <= 0 else "no residual degree of freedom" if dfd <= 0
                 else "no variation within any page" if floor else "no residual variation")
        raise UndefinedTest(cause, f"q={q}, dfd={dfd}, rss1={rss1}, within-page SS={within}")
    F = (max(rss0 - rss1, 0.0) / q) / (rss1 / dfd)  # rss0 >= rss1 in exact arithmetic
    return {"F": float(F), "df1": q, "df2": dfd, "p": float(sps.f.sf(F, q, dfd))}


def group_rms(beta_harm, names: Sequence[str], group_cols: Sequence[str], levels, sweep_px: float) -> float:
    """RMS amplitude of one group's fitted component over the block's sampled levels (equation (1) of 3.1).

    One value per level, not per row: an amplitude is a property of the curve, not of the number of pages.
    """
    beta = np.asarray(beta_harm, dtype=np.float64)
    names = list(names)
    if beta.shape != (len(names),):
        raise ValueError(f"{beta.shape[0] if beta.ndim else 0} coefficients for {len(names)} columns")
    group = [n for n in names if n in set(group_cols)]
    if not group:
        raise ValueError("the group has no column in the design")
    coef = beta[[names.index(n) for n in group]]
    component = _columns(group, levels, sweep_px) @ coef
    return float(np.sqrt(np.mean(component**2)))


def harmonic_fit(y, page_ids, phase_px, sweep_px: float, groups: Mapping[int, Sequence[int]], levels) -> dict:
    """Fit the saturated harmonic model with page fixed effects and test every period group.

    Returns ``groups`` (per period: bins, columns, q, F, df1, df2, p, rms and share of the fitted harmonic
    variance), ``sigma`` (the residual standard deviation, sqrt(RSS / df2), which the page-count rule
    needs), ``n_rows``, ``n_pages``, ``names`` and ``beta_harm`` (the harmonic coefficients of the full fit).
    Fit one reader, one block and one text type at a time.
    """
    y = np.asarray(y, dtype=np.float64)
    phase = np.asarray(phase_px, dtype=np.float64)
    if y.ndim != 1 or y.shape != phase.shape or len(np.asarray(page_ids)) != y.size:
        raise ValueError("y, page_ids and phase_px must be one-dimensional and of one length")
    bins = sorted({int(k) for ks in groups.values() for k in ks})
    if not bins:
        raise ValueError("no bins to fit")
    X, names = harmonic_design(phase, sweep_px, bins)

    result: dict[int, dict] = {}
    for T, ks in groups.items():
        wanted = {int(k) for k in ks}
        cols = [n for n in names if int(n[3:]) in wanted]
        if not cols:
            raise ValueError(f"period {T} px has no identifiable column at these phases")
        test = group_f_test(y, page_ids, X, names, cols)
        result[int(T)] = {"bins": sorted(wanted), "columns": cols, "q": test["df1"], **test}

    D, n_pages = _page_dummies(page_ids)
    X1 = np.column_stack([D, X])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    residual = y - X1 @ beta
    df2 = int(y.size - np.linalg.matrix_rank(X1))
    sigma = math.sqrt(float(residual @ residual) / df2) if df2 > 0 else math.nan
    beta_harm = beta[n_pages:]

    for g in result.values():
        g["rms"] = group_rms(beta_harm, names, g["columns"], levels, sweep_px)
    total = sum(g["rms"] ** 2 for g in result.values())
    for g in result.values():
        g["share"] = g["rms"] ** 2 / total if total > 0 else math.nan

    return {
        "groups": result,
        "sigma": sigma,
        "n_rows": int(y.size),
        "n_pages": int(n_pages),
        "names": names,
        "beta_harm": beta_harm,
    }


def harmonic_component(beta_harm, names: Sequence[str], sweep_px: float, phase_grid) -> np.ndarray:
    """The fitted harmonic sum on any grid of phases: the line of Figure 1."""
    beta = np.asarray(beta_harm, dtype=np.float64)
    names = list(names)
    if beta.shape != (len(names),):
        raise ValueError(f"{beta.shape[0] if beta.ndim else 0} coefficients for {len(names)} columns")
    return _columns(names, phase_grid, sweep_px) @ beta


# --------------------------------------------------------------------------- the page bootstrap


def page_bootstrap_rms(
    y,
    page_ids,
    phase_px,
    sweep_px: float,
    groups: Mapping[int, Sequence[int]],
    levels,
    period_px: int,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile page-bootstrap interval for the RMS amplitude of one period group.

    Pages are numbered in the sorted order of their ids and drawn with replacement; the i-th drawn page
    enters the refit as page i, so a page drawn twice is two pages with two fixed effects. Every resample
    refits the full model and recomputes the group's RMS at the sampled levels.
    """
    y = np.asarray(y, dtype=np.float64)
    phase = np.asarray(phase_px, dtype=np.float64)
    if int(period_px) not in {int(T) for T in groups}:
        raise ValueError(f"period {period_px} px is not one of the groups")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    pages, index = np.unique(np.asarray(page_ids), return_inverse=True)
    index = np.asarray(index).reshape(-1)
    rows_of = [np.flatnonzero(index == i) for i in range(len(pages))]
    bins = sorted({int(k) for ks in groups.values() for k in ks})
    X, names = harmonic_design(phase, sweep_px, bins)
    wanted = {int(k) for T, ks in groups.items() if int(T) == int(period_px) for k in ks}
    group_cols = [n for n in names if int(n[3:]) in wanted]
    if not group_cols:
        raise ValueError(f"period {period_px} px has no identifiable column at these phases")

    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=np.float64)
    n_pages = len(pages)
    for r in range(n_resamples):
        draw = rng.integers(0, n_pages, size=n_pages)
        rows = np.concatenate([rows_of[i] for i in draw])
        labels = np.repeat(np.arange(n_pages), [rows_of[i].size for i in draw])
        D = np.eye(n_pages)[labels]
        Xb = np.column_stack([D, X[rows]])
        beta = np.linalg.lstsq(Xb, y[rows], rcond=None)[0]
        values[r] = group_rms(beta[n_pages:], names, group_cols, levels, sweep_px)
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def pooled_cer_by_level(
    errors, ref_chars, page_ids, phase_px, n_resamples: int, seed: int, alpha: float = 0.05
) -> dict[str, np.ndarray]:
    """Pooled CER at each level with a page-bootstrap band shared by every level.

    errors = substitutions + deletions + insertions and ref_chars = hits + substitutions + deletions per
    row (D13); the CER of a level is the sum of errors over the sum of reference characters across pages.
    One draw of pages serves every level, so the band follows the shape of the curve. Returns ``levels``
    (sorted), ``cer``, ``low`` and ``high``.
    """
    errors = np.asarray(errors, dtype=np.float64)
    ref = np.asarray(ref_chars, dtype=np.float64)
    phase = np.asarray(phase_px, dtype=np.float64)
    ids = np.asarray(page_ids)
    if not (errors.shape == ref.shape == phase.shape == ids.shape) or errors.ndim != 1 or errors.size == 0:
        raise ValueError("errors, ref_chars, page_ids and phase_px must be one-dimensional and of one length")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    pages, page_index = np.unique(ids, return_inverse=True)
    levels, level_index = np.unique(phase, return_inverse=True)
    page_index = np.asarray(page_index).reshape(-1)
    level_index = np.asarray(level_index).reshape(-1)
    E = np.zeros((len(pages), len(levels)))
    R = np.zeros((len(pages), len(levels)))
    np.add.at(E, (page_index, level_index), errors)
    np.add.at(R, (page_index, level_index), ref)
    totals = R.sum(axis=0)
    if np.any(totals <= 0):
        raise ValueError("a level has no reference characters")
    cer = E.sum(axis=0) / totals

    rng = np.random.default_rng(seed)
    boots = np.empty((n_resamples, len(levels)), dtype=np.float64)
    n_pages = len(pages)
    for r in range(n_resamples):
        draw = rng.integers(0, n_pages, size=n_pages)
        with np.errstate(divide="ignore", invalid="ignore"):
            boots[r] = E[draw].sum(axis=0) / R[draw].sum(axis=0)
    low, high = np.nanquantile(boots, [alpha / 2, 1 - alpha / 2], axis=0)
    return {"levels": levels, "cer": cer, "low": low, "high": high}


# --------------------------------------------------------------------------- the paired tests


def paired_wilcoxon(a, b, alternative: str = "two-sided") -> dict:
    """Wilcoxon signed-rank test on paired page values, with the method SciPy's auto rule would choose.

    d = a - b as SciPy defines it. The method is resolved exactly as ``method="auto"`` does in SciPy 1.17.1
    (``scipy/stats/_wilcoxon.py``): more than 50 pairs, ``asymptotic``; otherwise no zero difference and no
    tie among the non-zero absolute differences, ``exact``; otherwise at most 13 pairs, a permutation test
    over all 2**n sign assignments; otherwise ``asymptotic``. Zeros are dropped (``zero_method="wilcox"``)
    and no continuity correction is applied. Returns statistic, pvalue, n (pairs), n_zero (zero differences)
    and method as ``"exact"``, ``"permutation"`` or ``"asymptotic"``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("a and b must be one-dimensional, non-empty and of one length")
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        raise ValueError("a and b must not contain NaN")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative {alternative!r}")
    d = a - b
    n = int(d.size)
    nonzero = np.abs(d[d != 0])
    n_zero = n - int(nonzero.size)
    if n > 50:
        method = "asymptotic"
    elif n_zero == 0 and np.unique(nonzero).size == nonzero.size:
        method = "exact"
    elif n <= 13:
        method = "permutation"
    else:
        method = "asymptotic"
    scipy_method = sps.PermutationMethod() if method == "permutation" else method
    res = sps.wilcoxon(
        a, b, zero_method="wilcox", correction=False, alternative=alternative, method=scipy_method
    )
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "n": n,
        "n_zero": n_zero,
        "method": method,
    }


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
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(seed)
    n = values.size
    out = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        out[i] = stat_fn(values[rng.integers(0, n, size=n)])
    low, high = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def median_paired_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Median of (b - a) over paired pages.

    Pass the baseline as a and the treatment as b, so that a negative value means the treatment lowered
    CER.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("a and b must be one-dimensional, non-empty and of one length")
    return float(np.median(b - a))


# --------------------------------------------------------------------------- Fano and power


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def fano_floor_bits(pe: float, M: int = 62, k: float = 1.0, erasure: bool = False) -> float:
    """Fano's lower bound on the bits needed to name k symbols out of M at symbol-error probability pe.

    R_min = k [log2 M - h(pe) - pe log2(M - 1)], with h the binary entropy, h(0) = h(1) = 0, clipped at 0
    against rounding (equation (4) of build guide 3.1; learning notes A9). With ``erasure=True`` the last
    term uses log2 M, the bound for a reader that may also output nothing.
    """
    if not 0.0 <= pe <= 1.0:
        raise ValueError("pe must lie in [0, 1]")
    if M < 2:
        raise ValueError("M must be at least 2")
    last = pe * (math.log2(M) if erasure else math.log2(M - 1))
    return float(k * max(0.0, math.log2(M) - _binary_entropy(pe) - last))


def power_noncentrality(q: int, dfd: int, alpha: float, power: float = 0.8) -> float:
    """The noncentrality an F-test with q and dfd degrees of freedom needs for the given power at alpha."""
    if q < 1 or dfd < 1 or not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("power_noncentrality needs q >= 1, dfd >= 1, 0 < alpha < 1 and 0 < power < 1")
    f_crit = sps.f.isf(alpha, q, dfd)
    return float(optimize.brentq(lambda lam: sps.ncf.sf(f_crit, q, dfd, lam) - power, 1e-6, 400))


def pages_needed(lam: float, sigma: float, n_levels: int, s_target: float, n_min: int) -> int:
    """Pages per text type for an RMS amplitude s_target: max(n_min, ceil(lam sigma^2 / (n s_target^2)))."""
    if sigma <= 0 or s_target <= 0 or n_levels < 1 or lam <= 0:
        raise ValueError("pages_needed needs positive lam, sigma, s_target and at least one level")
    return int(max(int(n_min), math.ceil(lam * sigma**2 / (n_levels * s_target**2))))


def paired_mde(
    sigma_d: float, n_pages: int, alpha: float = 0.05, power: float = 0.8, are: float = 3 / math.pi
) -> float:
    """Smallest detectable median paired difference: (z_{1-alpha/2} + z_{1-beta}) sigma_d / sqrt(are N).

    ``are`` is the asymptotic relative efficiency of the signed-rank test against the t-test, 3/pi.
    """
    if sigma_d <= 0 or n_pages < 1 or not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0 or are <= 0:
        raise ValueError("paired_mde needs positive sigma_d, n_pages >= 1, 0 < alpha, power < 1 and are > 0")
    z = sps.norm.isf(alpha / 2) + sps.norm.isf(1 - power)
    return float(z * sigma_d / math.sqrt(are * n_pages))
