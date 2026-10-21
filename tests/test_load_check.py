"""Tests for scripts/27_load_check.py (ANALYSIS_PLAN_3.md, Amendment 2), on synthetic rows and tables. CPU only.

Every page of a cell carries the same counts, so each contrast's bootstrap interval collapses onto its ratio and the
ratios are known exactly. No font is rasterised: the region test gives each character its position directly.
"""

from __future__ import annotations

import importlib.util

import pytest

from conftest import REPO_ROOT
from mrr import score

SCRIPT = REPO_ROOT / "scripts" / "27_load_check.py"


def _script():
    spec = importlib.util.spec_from_file_location("load_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rec(source: str, reader: str, size: float, page: str, counts: tuple[int, int, int, int]) -> dict:
    return {"source": source, "reader": reader, "text_type": "random62", "size": size, "page_id": page,
            "counts": counts}


def test_align_charges_each_error_to_its_reference_position_and_agrees_with_score():
    m = _script()
    reference, hypothesis = "abcdefgh", "QabXdegh"  # Q inserted first, c read as X, f dropped
    sd, ins, counts = m.align(reference, hypothesis)
    c = score.char_counts(reference, hypothesis)
    assert counts == (c.hits, c.substitutions, c.deletions, c.insertions) == (6, 1, 1, 1)
    assert sd == [0, 0, 1, 0, 0, 1, 0, 0] and ins == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    sd, ins, _ = m.align("abc", "abcd")
    assert sd == [0, 0, 0] and ins == [0, 0, 0, 1]  # the last slot holds what follows the page's end


def test_the_contrasts_set_each_ratio_beside_the_curve():
    m = _script()
    records = []
    for reader in m.CONTRAST_READERS:
        load, equal = m.FULL_SIZES.get(reader, m.FULL_SIZES_DEFAULT)
        for p in range(4):
            records.append(_rec(m.S0, reader, 25.0, f"p{p}", (90, 10, 0, 0)))  # 10 errors per 100 characters
            records.append(_rec(m.S28, reader, 28.0, f"p{p}", (95, 3, 2, 0)))  # 5
        for p in range(3):
            records.append(_rec(m.STAGE0, reader, load, f"z{p}", (80, 10, 10, 0)))  # 20
            records.append(_rec(m.STAGE0, reader, equal, f"y{p}", (95, 5, 0, 5)))  # 10, insertions included
    fits = {(r, "random62"): {"a": "20", "beta": "4"} for r in m.CONTRAST_READERS}
    rows = m.contrasts_table(records, fits, resamples=50)
    assert [(r["reader"], r["contrast"]) for r in rows] == [(r, c) for r in m.CONTRAST_READERS
                                                            for c in ("size", "load", "equal")]
    by = {(r["reader"], r["contrast"]): r for r in rows}
    for reader in m.CONTRAST_READERS:
        load, equal = m.FULL_SIZES.get(reader, m.FULL_SIZES_DEFAULT)
        for contrast, size, ratio, paired, n in (("size", 28.0, 0.5, True, 4), ("load", load, 2.0, False, 3),
                                                 ("equal", equal, 1.0, False, 3)):
            r = by[(reader, contrast)]
            assert r["ratio"] == pytest.approx(ratio) and r["ratio_low"] == pytest.approx(ratio)
            assert r["ratio_high"] == pytest.approx(ratio)
            assert r["treatment_size_px"] == size and r["baseline_size_px"] == 25.0 and r["paired"] is paired
            assert r["treatment_pages"] == n and r["baseline_pages"] == 4 and r["analysis_status"] == "exploratory"
            assert r["curve_ratio"] == pytest.approx((1 + (25 / 20) ** 4) / (1 + (size / 20) ** 4))
    assert by[("smolvlm", "load")]["treatment_size_px"] == 23.0


def test_a_missing_stage0_size_is_named():
    m = _script()
    records = [_rec(src, r, size, "p0", (90, 10, 0, 0)) for r in m.CONTRAST_READERS
               for src, size in ((m.S0, 25.0), (m.S28, 28.0))]
    with pytest.raises(ValueError, match="qwen: no stage-0 random-text row at 24 px"):
        m.contrasts_table(records, {(r, "random62"): {"a": "20", "beta": "4"} for r in m.CONTRAST_READERS})


def test_the_readoff_ratio_is_the_capacity_curve_at_the_readoff_size_or_a_bound_at_the_largest():
    m = _script()
    curve = {"chars_c": "1000", "chars_gamma": "2", "vision_tokens": "10", "s01": "10", "kappa01": "1.0",
             "s01_disagrees": "True"}  # kappa(s) = 100 / s^2
    acuity = [{**curve, "reader": "r", "text_type": "natural", "readoff_s01_how": "interpolated",
               "readoff_s01": "5", "sizes_px": "4;8"},
              {**curve, "reader": "r", "text_type": "random62", "readoff_s01_how": "not_reached",
               "readoff_s01": "", "sizes_px": "10;20"}]
    tokens = {("r", t): {"chars_per_text_token": "2", "token_ratio01": "0.5"} for t in ("natural", "random62")}
    rows, worst = m.readoff_table(acuity, tokens)
    assert worst < 1e-12
    reached, bound = rows
    assert reached["size_px"] == 5.0 and reached["kappa"] == pytest.approx(4.0)
    assert reached["token_ratio_readoff"] == pytest.approx(2.0) and reached["upper_bound"] is False
    assert reached["s01_disagrees"] is True and reached["token_ratio01"] == 0.5
    assert bound["size_px"] == 20.0 and bound["token_ratio_readoff"] == pytest.approx(0.125)
    assert bound["upper_bound"] is True and bound["readoff_s01"] != bound["readoff_s01"]  # NaN: not reached
    _, worst = m.readoff_table([{**acuity[0], "kappa01": "1.1"}], tokens)
    assert worst == pytest.approx(0.1)  # kappa01 over characters per text token is 1.1, against a ratio of 1.0


def test_position_reports_the_quarters_and_the_first_k_characters_of_a_study2_page():
    m = _script()
    groups = {
        (m.S0, "r", "natural", 25.0): [{"sd": [0] * 8, "ins": [0] * 9}],
        (m.STAGE0, "r", "natural", 20.0): [{"sd": [1] * 4 + [0] * 12, "ins": [0] * 16 + [2]}],
    }
    s0, full = sorted(m.position_table(groups), key=lambda r: r["sweep"] != m.S0)
    assert s0["k"] == 8 and s0["cer"] == 0.0 and s0["sd_first_k"] == 0.0
    assert full["k"] == 8 and full["chars_per_page"] == 16 and full["sd_first_k"] == 0.5
    assert [full[f"sd_q{q}"] for q in range(1, 5)] == [1.0, 0.0, 0.0, 0.0]
    assert full["cer"] == 6 / 16 and full["ins_after_end_rate"] == 2 / 16 and full["ins_inside_rate"] == 0.0


def test_region_splits_errors_inside_and_outside_a_study2_page():
    m = _script()
    cfg = {"page_width_px": 448, "page_height_px": 448, "margin_px": 14, "max_phase_x_px": 105,
           "max_phase_y_px": 26, "blind_strip_px": 0}  # Qwen's Study 2 block: 315 px wide, so x below 329
    groups = {
        (m.S0, "qwen", "random62", 25.0): [{"sd": [0, 0], "pos": [(20.0, 30.0, 0), (300.0, 60.0, 1)], "cfg": cfg}],
        (m.STAGE0, "qwen", "random62", 24.0): [{"sd": [0, 1, 1], "cfg": cfg,
                                                "pos": [(20.0, 30.0, 0), (400.0, 30.0, 0), (20.0, 300.0, 5)]}],
    }
    s0, full = sorted(m.region_table(groups), key=lambda r: r["sweep"] != m.S0)
    assert s0["x_edge"] == 329 and s0["s2_lines"] == 2 and s0["sd_outside"] != s0["sd_outside"]  # NaN: none
    assert full["chars_inside"] == 1 and full["sd_inside"] == 0.0
    assert full["chars_outside"] == 2 and full["sd_outside"] == 1.0
    assert full["sd_x1"] == 0.5 and full["sd_x4"] == 1.0 and full["sd_y1"] == 0.5 and full["sd_y3"] == 1.0
    assert full["lines_per_page"] == 6 and full["analysis_status"] == "exploratory"


def test_the_check_never_writes_into_another_analysis_folder(tmp_path):
    m = _script()
    assert m.main(["--name", "s3_stage0", "--out-dir", str(tmp_path)]) == 1
    assert m.main(["--name", "study2", "--out-dir", str(tmp_path)]) == 1
    assert not any(tmp_path.iterdir())
