# PURPOSE: Correctness tests for the harmonic model, Holm's correction, the page bootstrap, the paired Wilcoxon test, power and the Fano floor.
# WHAT GOES HERE: The measured values of build guide section 3.1, reproducibility and degenerate-case checks.
# FILLED IN AT STEP: W2.4.
# DO NOT: Use unseeded randomness in tests. Do not compare p-values tighter than a relative 1e-3 or F statistics tighter than three decimals.
"""Tests for mrr.stats (build guide section 3.1; D14).

The harmonic data come from the two synthetic curves of the learning notes' ``a7_harmonics``: a bump every
14 px (patch-driven) and a bump every 28 px (merge-driven), on ten pages at every whole-pixel phase of a
112 px sweep, with a page offset of standard deviation 0.02 and row noise of 0.01. Every expected value was
measured on 10 October 2026 with NumPy 2.4.6 and SciPy 1.17.1 (build guide section 3.1).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mrr import stats
from mrr.stats import bootstrap_ci

L, N_PAGES = 112, 10
LEVELS = np.arange(L, dtype=float)
GROUPS = stats.bin_groups(112, [14, 28, 112], 112)


def make_data(curve, seed=0):
    rng = np.random.default_rng(seed)
    phi = np.tile(np.arange(L, dtype=float), N_PAGES)
    page = np.repeat(np.arange(N_PAGES), L)
    y = curve(phi) + rng.normal(0, 0.02, N_PAGES)[page] + rng.normal(0, 0.01, N_PAGES * L)
    return y, page, phi


def patch_curve(phi):
    return 0.12 + 0.05 * np.exp(-((phi % 14) - 7.0) ** 2 / 3.0)


def merge_curve(phi):
    return 0.12 + 0.05 * np.exp(-((phi % 28) - 14.0) ** 2 / 3.0)


@pytest.fixture(scope="module")
def patch_fit():
    y, page, phi = make_data(patch_curve)
    return stats.harmonic_fit(y, page, phi, 112, GROUPS, LEVELS)


@pytest.fixture(scope="module")
def merge_fit():
    y, page, phi = make_data(merge_curve)
    return stats.harmonic_fit(y, page, phi, 112, GROUPS, LEVELS)


# --------------------------------------------------------------------------- Holm


@pytest.mark.parametrize(
    "pvalues, expected",
    [
        ([0.01, 0.04, 0.03], [0.03, 0.06, 0.06]),
        ([0.02, 0.5, 0.01, 0.04], [0.06, 0.5, 0.04, 0.08]),
        ([0.04, 0.04], [0.08, 0.08]),
    ],
)
def test_holm(pvalues, expected):
    np.testing.assert_allclose(stats.holm(pvalues), expected, atol=1e-12)


def test_holm_single_value_and_clip():
    assert stats.holm([0.2]) == pytest.approx([0.2])
    assert stats.holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("bad", [[0.1, float("nan")], [-0.1, 0.2], [0.5, 1.5], []])
def test_holm_refuses_bad_input(bad):
    with pytest.raises(ValueError):
        stats.holm(bad)


# --------------------------------------------------------------------------- bin groups and the design


def test_bin_groups_qwen_block_c():
    assert stats.bin_groups(112, [14, 28, 112], 16) == {14: [8], 28: [4], 112: [1, 2, 3, 5, 6, 7]}


def test_bin_groups_qwen_block_v():
    assert stats.bin_groups(28, [14, 28], 14) == {28: [1, 3, 5, 7], 14: [2, 4, 6]}


def test_bin_groups_full_sweep_sizes():
    assert [len(GROUPS[T]) for T in (14, 28, 112)] == [7, 7, 42]


def test_harmonic_design_drops_the_nyquist_sine():
    levels = np.arange(0, 112, 7, dtype=float)
    X, names = stats.harmonic_design(levels, 112, range(1, 9))
    assert X.shape == (16, 15)
    assert len(names) == 15
    assert "sin8" not in names
    assert "cos8" in names
    assert names[:2] == ["cos1", "sin1"]


# --------------------------------------------------------------------------- the harmonic fit


def test_patch_curve_groups(patch_fit):
    g14, g28, g112 = patch_fit["groups"][14], patch_fit["groups"][28], patch_fit["groups"][112]
    assert (g14["q"], g14["df1"], g14["df2"]) == (13, 13, 999)
    assert g14["F"] == pytest.approx(252.508, abs=5e-4)
    assert g14["p"] == pytest.approx(1.15e-304, rel=1e-3)
    assert g14["rms"] == pytest.approx(0.01676, abs=5e-6)
    assert (g28["q"], g112["q"]) == (14, 84)
    assert g28["F"] == pytest.approx(0.867, abs=5e-4)
    assert g28["p"] == pytest.approx(0.595, rel=1e-3)
    assert g112["F"] == pytest.approx(1.109, abs=5e-4)
    assert g112["p"] == pytest.approx(0.2419, rel=1e-3)
    assert patch_fit["n_rows"] == 1120 and patch_fit["n_pages"] == 10
    assert len(patch_fit["names"]) == 111 and patch_fit["beta_harm"].shape == (111,)
    assert 0.009 < patch_fit["sigma"] < 0.011
    assert g14["share"] > 0.95


def test_merge_curve_groups(merge_fit):
    g14, g28 = merge_fit["groups"][14], merge_fit["groups"][28]
    assert g14["F"] == pytest.approx(57.066, abs=5e-4)
    assert g14["p"] == pytest.approx(7.73e-111, rel=1e-3)
    assert g14["rms"] == pytest.approx(0.00797, abs=5e-6)
    assert g28["F"] == pytest.approx(81.512, abs=5e-4)
    assert g28["p"] == pytest.approx(2.77e-154, rel=1e-3)
    assert g28["rms"] == pytest.approx(0.00988, abs=5e-6)
    assert g14["share"] + g28["share"] > 0.95


def test_variance_shares_of_the_noiseless_curves():
    # One page, every level: the saturated model reproduces the curve exactly, so the shares are those of
    # the curve itself. With 112 rows and 112 parameters there are no residual degrees of freedom, so this
    # goes through the design and group_rms directly rather than through the F-tests of harmonic_fit.
    bins = sorted(k for ks in GROUPS.values() for k in ks)
    X, names = stats.harmonic_design(LEVELS, 112, bins)
    for curve, expected in ((patch_curve, {14: 1.0, 28: 0.0, 112: 0.0}), (merge_curve, {14: 0.4082, 28: 0.5918, 112: 0.0})):
        y = curve(LEVELS)
        beta = np.linalg.lstsq(np.column_stack([np.ones(L), X]), y, rcond=None)[0][1:]
        rms = {
            T: stats.group_rms(beta, names, [n for n in names if int(n[3:]) in ks], LEVELS, 112)
            for T, ks in GROUPS.items()
        }
        total = sum(v**2 for v in rms.values())
        shares = {T: v**2 / total for T, v in rms.items()}
        for T, s in expected.items():
            assert shares[T] == pytest.approx(s, abs=5e-5), (curve.__name__, T)


def test_group_rms_is_an_amplitude_at_the_levels():
    # A pure cosine of amplitude 0.03 at bin 8 has RMS 0.03 / sqrt(2) over the 112 whole-pixel levels, whatever
    # the number of pages. At Qwen C's 16 levels (0, 7, ..., 105) bin 8 is the Nyquist bin: the cosine is +1 or
    # -1 at every level, so its RMS there is the whole amplitude.
    names = ["cos8", "sin8"]
    beta = np.array([0.03, 0.0])
    assert stats.group_rms(beta, names, names, LEVELS, 112) == pytest.approx(0.03 / math.sqrt(2), abs=1e-12)
    assert stats.group_rms(beta, names, ["cos8"], np.arange(0, 112, 7, dtype=float), 112) == pytest.approx(
        0.03, abs=1e-12
    )


def test_harmonic_component_matches_the_design(patch_fit):
    bins = sorted(k for ks in GROUPS.values() for k in ks)
    X, names = stats.harmonic_design(LEVELS, 112, bins)
    assert names == patch_fit["names"]
    expected = X @ patch_fit["beta_harm"]
    got = stats.harmonic_component(patch_fit["beta_harm"], names, 112, LEVELS)
    np.testing.assert_allclose(got, expected, atol=1e-12)
    fine = stats.harmonic_component(patch_fit["beta_harm"], names, 112, np.linspace(0, 112, 1121))
    assert fine.shape == (1121,)
    assert fine[0] == pytest.approx(fine[-1], abs=1e-9)


def test_group_f_test_refuses_an_undefined_test():
    y = patch_curve(LEVELS)
    bins = sorted(k for ks in GROUPS.values() for k in ks)
    X, names = stats.harmonic_design(LEVELS, 112, bins)
    with pytest.raises(ValueError, match="undefined"):
        stats.group_f_test(y, np.zeros(L, dtype=int), X, names, [n for n in names if int(n[3:]) in GROUPS[14]])


# --------------------------------------------------------------------------- the page bootstrap


def test_page_bootstrap_rms_patch_curve():
    y, page, phi = make_data(patch_curve)
    low, high = stats.page_bootstrap_rms(y, page, phi, 112, GROUPS, LEVELS, 14, n_resamples=500, seed=42)
    assert (low, high) == pytest.approx((0.01626, 0.01739), abs=5e-6)


def test_page_bootstrap_rms_merge_curve():
    y, page, phi = make_data(merge_curve)
    low, high = stats.page_bootstrap_rms(y, page, phi, 112, GROUPS, LEVELS, 28, n_resamples=500, seed=42)
    assert (low, high) == pytest.approx((0.00935, 0.01053), abs=5e-6)


def test_page_bootstrap_rms_is_seeded_and_resamples_pages():
    y, page, phi = make_data(patch_curve)
    a = stats.page_bootstrap_rms(y, page, phi, 112, GROUPS, LEVELS, 14, n_resamples=50, seed=7)
    b = stats.page_bootstrap_rms(y, page, phi, 112, GROUPS, LEVELS, 14, n_resamples=50, seed=7)
    c = stats.page_bootstrap_rms(y, page, phi, 112, GROUPS, LEVELS, 14, n_resamples=50, seed=8)
    assert a == b and a != c
    # String page ids are numbered in sorted order, so the draw is the same as with integers 0 to 9.
    ids = np.array([f"pg_{i:02d}" for i in page])
    assert stats.page_bootstrap_rms(y, ids, phi, 112, GROUPS, LEVELS, 14, n_resamples=50, seed=7) == a


def test_pooled_cer_by_level():
    # Three pages at four levels: errors and reference characters per row; the pooled CER of a level is the
    # ratio of the sums over pages, not the mean of the row CERs.
    page_ids = np.repeat(["a", "b", "c"], 4)
    phase = np.tile([0.0, 7.0, 14.0, 21.0], 3)
    errors = np.array([2, 4, 6, 8, 1, 1, 1, 1, 0, 5, 10, 15], dtype=float)
    ref = np.array([100, 100, 100, 100, 50, 50, 50, 50, 200, 200, 200, 200], dtype=float)
    out = stats.pooled_cer_by_level(errors, ref, page_ids, phase, n_resamples=200, seed=1)
    np.testing.assert_array_equal(out["levels"], [0.0, 7.0, 14.0, 21.0])
    np.testing.assert_allclose(out["cer"], [3 / 350, 10 / 350, 17 / 350, 24 / 350])
    assert np.all(out["low"] <= out["cer"] + 1e-12) and np.all(out["cer"] <= out["high"] + 1e-12)
    assert out["low"].shape == out["high"].shape == (4,)
    same = stats.pooled_cer_by_level(errors, ref, page_ids, phase, n_resamples=200, seed=1)
    np.testing.assert_array_equal(out["low"], same["low"])


def test_pooled_cer_by_level_identical_pages_zero_width():
    page_ids = np.repeat(["a", "b"], 2)
    phase = np.tile([0.0, 7.0], 2)
    out = stats.pooled_cer_by_level([1, 2, 1, 2], [10, 10, 10, 10], page_ids, phase, n_resamples=100, seed=0)
    np.testing.assert_allclose(out["low"], out["cer"])
    np.testing.assert_allclose(out["high"], out["cer"])


# --------------------------------------------------------------------------- the paired tests

A = np.array([12, 30, 25, 8, 40, 22, 18, 35, 27, 15], dtype=float)
B = np.array([10, 21, 26, 5, 31, 20, 12, 30, 22, 15], dtype=float)


def test_paired_wilcoxon_permutation():
    out = stats.paired_wilcoxon(A, B)
    assert out["method"] == "permutation"
    assert out["statistic"] == pytest.approx(1.0)
    assert out["pvalue"] == pytest.approx(0.0078125, rel=1e-3)
    assert (out["n"], out["n_zero"]) == (10, 1)


def test_paired_wilcoxon_exact():
    d = np.array([1, 2, 3, 4, 5, 6, 7, 8, -9, 10], dtype=float)
    out = stats.paired_wilcoxon(d, np.zeros(10))
    assert out["method"] == "exact"
    assert out["statistic"] == pytest.approx(9.0)
    assert out["pvalue"] == pytest.approx(0.0644531, rel=1e-3)
    assert (out["n"], out["n_zero"]) == (10, 0)


def test_paired_wilcoxon_asymptotic():
    d = np.array([1, 2, 3, 4, 5, 6, 7, 8, -9, 10, 11, 12, 13, 13], dtype=float)
    out = stats.paired_wilcoxon(d, np.zeros(14))
    assert out["method"] == "asymptotic"
    assert out["statistic"] == pytest.approx(9.0)
    assert out["pvalue"] == pytest.approx(0.0063056, rel=1e-3)


def test_paired_wilcoxon_alternative_and_refusals():
    two = stats.paired_wilcoxon(A, B)
    greater = stats.paired_wilcoxon(A, B, alternative="greater")
    assert greater["pvalue"] == pytest.approx(two["pvalue"] / 2, rel=1e-3)
    with pytest.raises(ValueError):
        stats.paired_wilcoxon(A, B[:-1])
    with pytest.raises(ValueError):
        stats.paired_wilcoxon(np.array([1.0, np.nan, 2.0]), np.zeros(3))


def test_median_paired_difference():
    assert stats.median_paired_difference(A, B) == -4.0


def test_bootstrap_reproducible():
    values = np.random.default_rng(0).normal(size=50)
    a = bootstrap_ci(values, np.median, n_resamples=2000, alpha=0.05, seed=42)
    b = bootstrap_ci(values, np.median, n_resamples=2000, alpha=0.05, seed=42)
    assert a == b


def test_bootstrap_constant_zero_width():
    low, high = bootstrap_ci(np.full(30, 0.1), np.median, n_resamples=1000, alpha=0.05, seed=0)
    assert low == high


def test_bootstrap_ci_brackets_the_median():
    values = np.random.default_rng(3).normal(size=80)
    low, high = bootstrap_ci(values, np.median, n_resamples=1000, alpha=0.05, seed=1)
    assert isinstance(low, float) and isinstance(high, float)
    assert low < float(np.median(values)) < high


# --------------------------------------------------------------------------- Fano and power


@pytest.mark.parametrize(
    "pe, expected", [(0.0, 5.9542), (0.01, 5.8141), (0.03, 5.5819), (0.1, 4.8921), (0.4, 2.6110)]
)
def test_fano_floor_bits(pe, expected):
    assert stats.fano_floor_bits(pe) == pytest.approx(expected, abs=5e-5)


@pytest.mark.parametrize("pe, expected", [(0.01, 5.8139), (0.03, 5.5812), (0.1, 4.8898), (0.4, 2.6016)])
def test_fano_floor_bits_erasure(pe, expected):
    assert stats.fano_floor_bits(pe, erasure=True) == pytest.approx(expected, abs=5e-5)


def test_fano_floor_bits_symbols_and_refusals():
    assert stats.fano_floor_bits(0.03, k=3.95) == pytest.approx(22.05, abs=5e-3)
    # The bound is 0 at pe = (M - 1) / M, uniform guessing, and the clip holds it at 0 against rounding there;
    # at pe = 1 it is log2(M / (M - 1)) again, as the notes' fano_min_bits gives (guide 3.1, equation (4)).
    assert stats.fano_floor_bits(61 / 62) == 0.0
    assert stats.fano_floor_bits(1.0) == pytest.approx(math.log2(62 / 61), abs=1e-12)
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError):
            stats.fano_floor_bits(bad)


def test_power_noncentrality():
    assert stats.power_noncentrality(13, 1000, 0.05 / 3) == pytest.approx(22.58, abs=5e-3)


def test_pages_needed():
    assert stats.pages_needed(22.58, 0.05, 14, 0.01, 10) == 41
    assert stats.pages_needed(22.58, 0.05, 14, 0.005, 10) == 162
    assert stats.pages_needed(22.58, 0.001, 14, 0.01, 10) == 10


def test_paired_mde():
    assert stats.paired_mde(0.05, 41) == pytest.approx(0.0224, abs=5e-5)
    assert stats.paired_mde(0.05, 41, alpha=0.05 / 3) == pytest.approx(0.0259, abs=5e-5)


def test_a_group_f_test_with_no_variation_within_any_page_is_undefined_not_a_ratio_of_rounding_errors():
    # Every page reads the same at every phase (a reader at its floor): both residual sums of squares are rounding
    # error, about 1e-33, and their ratio could read as any F. The test must refuse instead.
    levels = np.arange(0.0, 28.0, 2.0)
    pages = np.repeat(np.arange(16), levels.size)
    phase = np.tile(levels, 16)
    y = np.repeat(np.where(np.arange(16) % 5 == 0, 1 / 312, 0.0), levels.size)
    groups = stats.bin_groups(28, [14, 28], levels.size)
    bins = sorted(k for ks in groups.values() for k in ks)
    X, names = stats.harmonic_design(phase, 28, bins)
    cols = [n for n in names if int(n[3:]) in set(groups[14])]
    with pytest.raises(ValueError, match="within-page"):
        stats.group_f_test(y, pages, X, names, cols)
    with pytest.raises(ValueError, match="within-page"):
        stats.harmonic_fit(y, pages, phase, 28, groups, levels)
    y2 = y.copy()
    y2[3] += 1 / 500  # one row of one page off by one character in 500: real variation, a defined test
    test = stats.group_f_test(y2, pages, X, names, cols)
    assert test["F"] >= 0 and 0 <= test["p"] <= 1 and test["df2"] > 0


def test_an_undefined_f_test_names_its_cause():
    # The table writes the cause (review of 15 October, stats-2): one page that varies is not a reader at its floor.
    levels = np.arange(0.0, 28.0, 2.0)
    groups = stats.bin_groups(28, [14, 28], levels.size)
    with pytest.raises(stats.UndefinedTest) as one_page:
        stats.harmonic_fit(np.linspace(0.3, 0.6, levels.size), np.zeros(levels.size, dtype=int), levels, 28, groups,
                           levels)
    assert one_page.value.cause == "no residual degree of freedom"
    pages = np.repeat(np.arange(4), levels.size)
    with pytest.raises(stats.UndefinedTest) as flat:
        stats.harmonic_fit(np.zeros(pages.size), pages, np.tile(levels, 4), 28, groups, levels)
    assert flat.value.cause == "no variation within any page" and isinstance(flat.value, ValueError)
