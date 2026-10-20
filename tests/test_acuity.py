"""Tests for src/mrr/acuity.py and scripts/25_fit_stage0.py (Study 3 stage 0, design note R7 section 3).

Synthetic cells drawn from a known curve check the fit, its inverse and the capacity; synthetic rows check the
passage bootstrap; the committed run folders of Study 1's ladder check the script end to end.
"""

from __future__ import annotations

import csv
import importlib.util
import math

import numpy as np
import pandas as pd
import pytest

from conftest import REPO_ROOT
from mrr import acuity

SIZES = (10.0, 12.0, 15.0, 19.0, 24.0, 29.0, 36.0)


def _script():
    spec = importlib.util.spec_from_file_location("fit_stage0", REPO_ROOT / "scripts" / "25_fit_stage0.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(a: float, beta: float, *, passages: int = 8, sizes=SIZES, chars_c: float = 2.75e5, noise: float = 0.0,
          seed: int = 0, sweep: str | None = None, reader: str = "qwen", text_type: str = "random62") -> pd.DataFrame:
    """One results row per passage and size: errors from the curve (times a page factor when ``noise``)."""
    rng = np.random.default_rng(seed)
    factors = np.exp(rng.normal(0.0, noise, size=passages)) if noise else np.ones(passages)
    out = []
    for i in range(passages):
        for s in sizes:
            n = int(round(chars_c / s ** 2))
            p = float(min(1.0, acuity.cer_curve(s, a, beta) * factors[i]))
            errors = int(round(p * n))
            row = {"reader": reader, "text_type": text_type, "page_id": f"rnd1_z{s:g}_conv_p{i:04d}",
                   "font_size_px": s, "hits": n - errors, "substitutions": errors, "deletions": 0, "insertions": 0,
                   "reference_chars": n, "vision_tokens": 256, "hit_limit": "False", "output_tokens": n,
                   "latency_s": 1.0}
            if sweep is not None:
                row["sweep"] = sweep
            out.append(row)
    return pd.DataFrame(out)


@pytest.mark.parametrize("a,beta,e0", [(13.4, 6.1, 0.0), (21.2, 7.5, 0.0), (8.0, 12.0, 0.003)])
@pytest.mark.parametrize("eps", [0.05, 0.01])
def test_size_at_inverts_the_curve(a, beta, e0, eps):
    s = acuity.size_at(eps, a, beta, e0)
    assert float(acuity.cer_curve(s, a, beta, e0)) == pytest.approx(eps, rel=1e-12)


def test_size_at_edges():
    assert acuity.size_at(0.01, 13.0, 6.0, e0=0.01) == math.inf  # the floor is the target: never reached
    assert acuity.size_at(0.01, 13.0, 6.0, e0=0.05) == math.inf
    assert math.isnan(acuity.size_at(1.0, 13.0, 6.0))
    assert acuity.size_at(0.5, 13.0, 6.0) == pytest.approx(13.0)  # a is the half-error size


def test_fit_recovers_a_known_curve():
    cer = acuity.cer_curve(np.array(SIZES), 13.4, 6.1)
    fit = acuity.fit_curve(SIZES, cer, np.full(len(SIZES), 8.0))
    assert fit.converged
    assert fit.e0 == 0.0
    assert fit.a == pytest.approx(13.4, rel=1e-4)
    assert fit.beta == pytest.approx(6.1, rel=1e-3)


def test_fit_caps_cer_above_one():
    cer = acuity.cer_curve(np.array(SIZES), 15.0, 7.0)
    garbage = cer.copy()
    garbage[0] = 1.04  # a reader writing past the reference
    capped = cer.copy()
    capped[0] = 1.0
    a, b = acuity.fit_curve(SIZES, garbage), acuity.fit_curve(SIZES, capped)
    assert (a.a, a.beta) == pytest.approx((b.a, b.beta))


def test_free_floor_finds_a_real_floor():
    sizes = np.geomspace(8.0, 60.0, 9)  # reaching far past the cliff, where the floor shows
    cer = acuity.cer_curve(sizes, 14.0, 6.0, e0=0.02)
    fit = acuity.fit_curve(sizes, cer, free_floor=True)
    assert fit.e0 == pytest.approx(0.02, abs=1e-3)
    assert fit.size_at(0.01) == math.inf


def test_fit_needs_three_cells():
    with pytest.raises(ValueError, match="three or more"):
        acuity.fit_curve([10.0, 20.0], [0.8, 0.1])


def test_chars_model_recovers_inverse_square():
    sizes = np.array(SIZES)
    c, gamma = acuity.chars_model(sizes, 2.75e5 / sizes ** 2)
    assert (c, gamma) == pytest.approx((2.75e5, 2.0))
    assert acuity.chars_at(20.0, c, gamma) == pytest.approx(2.75e5 / 400)
    assert acuity.chars_at(math.inf, c, gamma) == 0.0


@pytest.mark.parametrize("page_id,key", [("rnd20261023_z10_conv_p0003", ("rnd20261023", 3)),
                                         ("pg105_z7.5_pa_p0012", ("pg105", 12)),
                                         ("pg1661_z6_conv_p0000", ("pg1661", 0))])
def test_passage_key(page_id, key):
    assert acuity.passage_key(page_id) == key


def test_passage_key_refuses_a_shared_page():
    with pytest.raises(ValueError, match="block Z"):
        acuity.passage_key("pg105_p0003")


def test_fit_group_cells_pool_the_counts():
    rows = _rows(13.4, 6.1)
    cells, fit = acuity.fit_group(rows, resamples=20, seed=1)
    assert [c["font_size_px"] for c in cells] == list(SIZES)
    for c in cells:
        g = rows[rows["font_size_px"] == c["font_size_px"]]
        assert c["pages"] == 8
        assert c["errors"] == int(g["substitutions"].sum())
        assert c["cer"] == pytest.approx(g["substitutions"].sum() / g["reference_chars"].sum())
        assert c["chars_per_vision_token"] == pytest.approx(g["reference_chars"].mean() / 256)
    assert fit["status"] == "ok"
    assert fit["passages"] == 8 and fit["pages_min"] == 8
    assert fit["a"] == pytest.approx(13.4, rel=0.01)
    assert fit["s01"] == pytest.approx(acuity.size_at(0.01, 13.4, 6.1), rel=0.02)
    assert fit["kappa01"] == pytest.approx(2.75e5 / fit["s01"] ** 2 / 256, rel=0.02)
    assert fit["s01_in_range"] and fit["s05_in_range"]


def test_identical_passages_give_a_zero_width_interval():
    _, fit = acuity.fit_group(_rows(13.4, 6.1), resamples=50, seed=1)
    for name in ("a", "beta", "s05", "s01", "kappa01"):
        assert fit[f"{name}_lo"] == pytest.approx(fit[name], rel=1e-6)
        assert fit[f"{name}_hi"] == pytest.approx(fit[name], rel=1e-6)


def test_noisy_passages_give_an_interval_around_the_estimate():
    _, fit = acuity.fit_group(_rows(13.4, 6.1, noise=0.3, seed=4), resamples=200, seed=1)
    for name in ("a", "beta", "s01"):
        assert fit[f"{name}_lo"] < fit[f"{name}_hi"]
        assert fit[f"{name}_lo"] <= fit[name] <= fit[f"{name}_hi"]
    assert fit["resamples_fitted"] == 200


def test_fit_group_is_deterministic():
    rows = _rows(13.4, 6.1, noise=0.3, seed=4)
    assert acuity.fit_group(rows, resamples=50, seed=7) == acuity.fit_group(rows, resamples=50, seed=7)


def test_two_sweeps_never_share_a_passage():
    rows = pd.concat([_rows(13.4, 6.1, sweep="ladder"), _rows(13.4, 6.1, sweep="full")], ignore_index=True)
    cells, fit = acuity.fit_group(rows, resamples=10, seed=1)
    assert fit["passages"] == 16
    assert fit["sweep"] == "full;ladder"
    assert cells[0]["pages"] == 16


@pytest.mark.parametrize("a,status", [(80.0, "a_above_range"), (5.0, "a_below_range"), (16.0, "ok")])
def test_status_flags_an_extrapolated_a(a, status):
    _, fit = acuity.fit_group(_rows(a, 6.0), resamples=5, seed=1)
    assert fit["status"] == status


def test_fit_group_on_columns_equal_to_the_default_is_unchanged():
    rows = _rows(13.4, 6.1, noise=0.3, seed=4)
    rows["e"] = rows["substitutions"] + rows["deletions"] + rows["insertions"]
    rows["n"] = rows["hits"] + rows["substitutions"] + rows["deletions"]
    assert (acuity.fit_group(rows, resamples=30, seed=7, errors_column="e", chars_column="n")
            == acuity.fit_group(rows, resamples=30, seed=7))


def test_fit_group_needs_both_columns():
    with pytest.raises(ValueError, match="both"):
        acuity.fit_group(_rows(13.4, 6.1), resamples=5, seed=1, errors_column="hits")


def test_passage_matrices_sum_per_passage_and_size():
    rows = pd.concat([_rows(13.4, 6.1, passages=3), _rows(13.4, 6.1, passages=1)], ignore_index=True)
    passages, sizes, mats, present = acuity.passage_matrices(rows, ["substitutions", "reference_chars"])
    assert passages == [("", "rnd1", i) for i in range(3)]
    assert sizes.tolist() == list(SIZES)
    assert present[:, 0].tolist() == [2, 1, 1]  # passage 0 twice
    assert mats["reference_chars"][:, 0].tolist() == [2 * round(2.75e5 / 100), round(2.75e5 / 100),
                                                      round(2.75e5 / 100)]


def test_ratio_intervals_hold_the_pooled_ratio():
    for noise, width in ((0.0, False), (0.3, True)):
        _, _, mats, _ = acuity.passage_matrices(_rows(13.4, 6.1, noise=noise, seed=4),
                                                ["substitutions", "reference_chars"])
        bands = acuity.ratio_intervals(mats, {"cer": ("substitutions", "reference_chars")}, resamples=100,
                                       rng=np.random.default_rng(0))
        point = mats["substitutions"].sum(axis=0) / mats["reference_chars"].sum(axis=0)
        lo, hi = bands["cer"]
        assert np.all(lo <= point + 1e-12) and np.all(point <= hi + 1e-12)
        assert bool(hi[0] > lo[0]) is width  # identical passages: a zero-width interval


def test_readoff_size_reads_the_first_crossing():
    s, how = acuity.readoff_size([10, 20, 40], [0.5, 0.1, 0.001], 0.01)
    assert how == "interpolated" and s == pytest.approx(math.sqrt(20 * 40))  # log CER linear in log size
    s, how = acuity.readoff_size([40, 10, 20], [0.0, 0.5, 0.1], 0.05)  # unsorted; CER linear to a zero cell
    assert how == "interpolated" and s == pytest.approx(math.sqrt(20 * 40))
    assert acuity.readoff_size([10, 20, 40], [0.004, 0.002, 0.001], 0.01) == (10.0, "at_or_below_first")
    s, how = acuity.readoff_size([10, 20, 40], [0.9, 0.5, 0.2], 0.01)
    assert math.isnan(s) and how == "not_reached"
    s, how = acuity.readoff_size([10, 20, 40, 80], [0.5, 0.005, 0.05, 0.002], 0.01)
    assert 10 < s < 20 and how == "interpolated"  # a later rise does not move the first crossing


def test_readoff_size_on_the_curve_is_near_the_fitted_size():
    sizes = np.geomspace(8.0, 48.0, 15)
    s, how = acuity.readoff_size(sizes, acuity.cer_curve(sizes, 13.4, 6.1), 0.01)
    assert how == "interpolated" and s == pytest.approx(acuity.size_at(0.01, 13.4, 6.1), rel=0.01)


def test_interp_log():
    assert acuity.interp_log([10, 40], [0.0, 1.0], 20.0) == pytest.approx(0.5)  # halfway in log size
    for size in (50.0, 5.0, math.nan, math.inf):
        assert math.isnan(acuity.interp_log([10, 40], [0.0, 1.0], size))


def test_leave_one_size_out_finds_a_cell_off_the_curve():
    rows = _rows(13.4, 6.1, chars_c=2.75e7)  # many characters a page, so that rounding stays off the curve's tail
    _, sizes, mats, present = acuity.passage_matrices(rows, ["substitutions", "reference_chars"])
    errors, chars = mats["substitutions"], mats["reference_chars"]
    out = acuity.leave_one_size_out(sizes, errors, chars, present, resamples=10, rng=np.random.default_rng(0))
    assert [r["font_size_px"] for r in out] == list(SIZES)
    assert max(abs(r["difference"]) for r in out) < 0.01
    k = list(SIZES).index(24.0)
    bumped = errors.copy()
    bumped[:, k] = np.round(0.2 * chars[:, k])  # CER 0.2 where the curve gives 0.03
    out = acuity.leave_one_size_out(sizes, bumped, chars, present, resamples=10, rng=np.random.default_rng(0))
    assert out[k]["difference"] > 0.15 and out[k]["difference_lo"] > 0 and not out[k]["inside"]
    assert all(math.isfinite(r["predicted"]) for r in out)


def test_leave_one_size_out_needs_three_other_cells():
    rows = _rows(13.4, 6.1, sizes=(10.0, 15.0, 24.0))
    _, sizes, mats, present = acuity.passage_matrices(rows, ["substitutions", "reference_chars"])
    out = acuity.leave_one_size_out(sizes, mats["substitutions"], mats["reference_chars"], present, resamples=5,
                                    rng=np.random.default_rng(0))
    assert all(math.isnan(r["predicted"]) and not r["inside"] for r in out)


def test_subsample_fits():
    rows = _rows(13.4, 6.1, chars_c=2.75e7)
    _, sizes, mats, present = acuity.passage_matrices(rows, ["substitutions", "reference_chars"])
    errors, chars = mats["substitutions"], mats["reference_chars"]
    true = acuity.size_at(0.01, 13.4, 6.1)
    every = np.ones(sizes.size, dtype=bool)
    full = acuity.subsample_fits(sizes, errors, chars, present, size_keep=every, n_passages=None, subsets=10,
                                 rng=np.random.default_rng(0))
    assert full.shape == (1,) and full[0] == pytest.approx(true, rel=0.01)
    four = acuity.subsample_fits(sizes, errors, chars, present, size_keep=every, n_passages=4, subsets=10,
                                 rng=np.random.default_rng(0))
    assert four.shape == (10,) and np.allclose(four, full[0], rtol=1e-6)  # identical passages
    every_other = np.zeros(sizes.size, dtype=bool)
    every_other[::2] = True
    half = acuity.subsample_fits(sizes, errors, chars, present, size_keep=every_other, n_passages=None, subsets=10,
                                 rng=np.random.default_rng(0))
    assert half[0] == pytest.approx(true, rel=0.02)
    two = np.zeros(sizes.size, dtype=bool)
    two[:2] = True
    assert np.isnan(acuity.subsample_fits(sizes, errors, chars, present, size_keep=two, n_passages=None,
                                          subsets=10, rng=np.random.default_rng(0))).all()


def test_script_on_study1_ladder(tmp_path):
    script = _script()
    code = script.main(["--configs", str(REPO_ROOT / "configs" / "sweeps" / "ladder.yaml"),
                        "--runs-root", str(REPO_ROOT / "results" / "runs"), "--out-dir", str(tmp_path),
                        "--name", "ladder_check", "--resamples", "20", "--loso-resamples", "3", "--subsets", "3",
                        "--no-tokens"])
    assert code == 0  # every recomputed primary count equals results.csv's
    out = tmp_path / "ladder_check"
    with open(out / "acuity.csv", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == script.ACUITY_COLUMNS
        fits = {(r["reader"], r["text_type"]): r for r in reader}
    qwen = fits[("qwen", "random62")]
    assert qwen["status"] == "ok"
    assert 11.0 < float(qwen["a"]) < 15.0  # Study 1's ladder alone, 3 passages: about 13 px
    assert qwen["patch_size"] == "14" and float(qwen["a_over_patch"]) == pytest.approx(float(qwen["a"]) / 14)
    assert float(qwen["reading_a"]) <= float(qwen["a"]) * 1.05  # behaviour removed: no worse a reader
    with open(out / "cells.csv", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == script.CELL_COLUMNS
        cells = list(reader)
    assert {float(c["font_size_px"]) for c in cells if c["reader"] == "qwen"} == {10, 12, 14, 16, 20, 24}
    with open(out / "pages.csv", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == script.PAGE_COLUMNS
        pages = list(reader)
    assert len(pages) == sum(int(c["pages"]) for c in cells)
    assert all(p["errors"] == p["results_errors"] for p in pages)
    for p in pages:
        omitted = int(p["omitted_head"]) + int(p["omitted_interior"]) + int(p["omitted_tail"])
        assert int(p["omitted"]) == omitted and int(p["read_chars"]) == int(p["reference_chars"]) - omitted
    tables = {"behaviour": script.BEHAVIOUR_COLUMNS, "sensitivity": script.SENSITIVITY_COLUMNS,
              "sensitivity_fits": script.SENSITIVITY_FIT_COLUMNS, "form": script.FORM_COLUMNS,
              "subsample": script.SUBSAMPLE_COLUMNS, "cost": script.COST_COLUMNS, "latency": script.LATENCY_COLUMNS}
    for name, columns in tables.items():
        with open(out / f"{name}.csv", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            assert tuple(reader.fieldnames) == columns, name
            assert list(reader), name
    assert not (out / "tokens.csv").exists()
    assert (out / "manifest.json").exists()
