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


def test_script_on_study1_ladder(tmp_path):
    script = _script()
    code = script.main(["--configs", str(REPO_ROOT / "configs" / "sweeps" / "ladder.yaml"),
                        "--runs-root", str(REPO_ROOT / "results" / "runs"), "--out-dir", str(tmp_path),
                        "--name", "ladder_check", "--resamples", "20"])
    assert code == 0
    out = tmp_path / "ladder_check"
    with open(out / "acuity.csv", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == script.ACUITY_COLUMNS
        fits = {(r["reader"], r["text_type"]): r for r in reader}
    qwen = fits[("qwen", "random62")]
    assert qwen["status"] == "ok"
    assert 11.0 < float(qwen["a"]) < 15.0  # Study 1's ladder alone, 3 passages: about 13 px
    assert qwen["patch_size"] == "14" and float(qwen["a_over_patch"]) == pytest.approx(float(qwen["a"]) / 14)
    with open(out / "cells.csv", encoding="utf-8", newline="") as fh:
        cells = list(csv.DictReader(fh))
    assert {float(c["font_size_px"]) for c in cells if c["reader"] == "qwen"} == {10, 12, 14, 16, 20, 24}
    assert (out / "manifest.json").exists()
