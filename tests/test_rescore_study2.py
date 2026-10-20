"""Tests for scripts/26_rescore_study2.py (ANALYSIS_PLAN_3.md section 3.2), on synthetic rows.

One reader (P = 14, so the cut is M7) reads six random pages and six prose pages in Study 2's conditions. Errors of
known class are planted: on random text a misread letter (S0) and a shape-alike case swap (u in the sans, S5) in the
conventional layout, a shape-distinct case swap (u in the mono, M3) in the contained one, and an O/0 swap, a
shape-alike swap (x) and a shape-distinct swap (p) in the cut; on prose an anchored line skipped in the contained
layout. The shape-alike sets are the measured ones (ANALYSIS_PLAN_3.md R4), given by name, so no font is rasterised.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import REPO_ROOT
from mrr import score, study2
from mrr.text_source import normalise_text

SCRIPT = REPO_ROOT / "scripts" / "26_rescore_study2.py"
SANS, MONO = "sans.ttf", "mono.ttf"
ALIKE = {SANS: frozenset("cosuvxz"), MONO: frozenset("coqsvxz")}
CONDITIONS = (("S", 0.0, SANS), ("S", 5.0, SANS), ("S", 10.0, SANS), ("M", 0.0, MONO), ("M", 3.0, MONO),
              ("M", 7.0, MONO))
PAGES = 6
SKIPPED = "fell on the quiet street "  # 25 characters inside correct prose: an interior omission


def _script():
    spec = importlib.util.spec_from_file_location("rescore_study2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _random(i: int) -> str:
    return f"Ola{i} xAb3 Qrt{i} mnP9 Zef{i} kLw2 yuR{i} hgT5"


def _prose(i: int) -> str:
    return f"It was the {i}th of May and the morning light {SKIPPED}where nobody walked yet today."


SKIPPED_SHARE = len(SKIPPED) / len(normalise_text(_prose(0)))  # every prose page has the same length


def _planted(text_type: str, block: str, phase: float, reference: str) -> str:
    if text_type == "random62":
        return {("S", 0.0): reference.replace("Qrt", "Prt"),  # a misreading: reading error
                ("S", 5.0): reference.replace("yuR", "yUR"),  # u is shape-alike in the sans: reading error
                ("M", 3.0): reference.replace("yuR", "yUR"),  # u is shape-distinct in the mono: case normalised
                ("M", 7.0): reference.replace("Ola", "0la").replace("xAb", "XAb").replace("mnP", "mnp"),
                }.get((block, phase), reference)
    if block == "M" and phase in (0.0, 3.0):
        return reference.replace(SKIPPED, "")
    return reference


def _row(run_dir: Path, text_type: str, page_id: str, reference: str, block: str, phase: float, font: str,
         hypothesis: str, reader: str = "qwen") -> dict:
    (run_dir / "references").mkdir(parents=True, exist_ok=True)
    (run_dir / "transcriptions").mkdir(exist_ok=True)
    (run_dir / "references" / f"{page_id}.txt").write_text(reference, encoding="utf-8")
    name = f"transcriptions/{page_id}_{block}{phase:g}.txt"
    (run_dir / name).write_text(hypothesis, encoding="utf-8")
    c, n = score.char_counts(reference, hypothesis), score.char_counts_letters_only(reference, hypothesis)
    return {"run_dir": str(run_dir), "reader": reader, "text_type": text_type, "page_id": page_id, "block": block,
            "phase_x_px": phase, "font_size_px": 25.0, "font_file": font,
            "render_mode": "conventional" if block == "S" else "patch_aware", "transcription_file": name,
            "hits": c.hits, "substitutions": c.substitutions, "deletions": c.deletions, "insertions": c.insertions,
            "hits_ns": n.hits, "substitutions_ns": n.substitutions, "deletions_ns": n.deletions,
            "insertions_ns": n.insertions, "spaces_inserted": 0, "wer": 0.0, "hit_limit": "False"}


def _rows(run_dir: Path) -> pd.DataFrame:
    rows = []
    for i in range(PAGES):
        for text_type, page_id, reference in (("random62", f"rnd_s1_p{i}", _random(i)),
                                              ("natural", f"pg1_c1_p{i}", _prose(i))):
            for block, phase, font in CONDITIONS:
                rows.append(_row(run_dir, text_type, page_id, reference, block, phase, font,
                                 _planted(text_type, block, phase, reference)))
    return study2.with_codes(pd.DataFrame(rows), {"qwen": 14})


@pytest.fixture(scope="module")
def m():
    return _script()


@pytest.fixture(scope="module")
def scored(m, tmp_path_factory):
    rows = _rows(tmp_path_factory.mktemp("study2") / "run")
    return m.score_rows(rows, alike_for=ALIKE.__getitem__)


def _one(df: pd.DataFrame, **where) -> pd.Series:
    mask = np.ones(len(df), dtype=bool)
    for k, v in where.items():
        mask &= (df[k] == v).to_numpy()
    assert mask.sum() == 1, where
    return df[mask].iloc[0]


def _contrast(rows: list[dict], **where) -> dict:
    found = [r for r in rows if all(r[k] == v for k, v in where.items())]
    assert len(found) == 1, where
    return found[0]


# --------------------------------------------------------------------------- scoring every row


def test_each_planted_error_lands_in_its_class_on_both_strings(m, scored):
    frame, records = scored
    assert len(frame) == len(records) == PAGES * 2 * len(CONDITIONS)
    s0 = _one(frame, page_id="rnd_s1_p0", condition="S0")
    assert (s0.reading_errors, s0.case_alike, s0.case_normalised, s0.omitted) == (1, 0, 0, 0)
    s5 = _one(frame, page_id="rnd_s1_p0", condition="S5")
    assert (s5.reading_errors, s5.case_alike, s5.case_normalised) == (1, 1, 0)  # u is alike in the sans
    m3 = _one(frame, page_id="rnd_s1_p0", condition="M3")
    assert (m3.reading_errors, m3.case_alike, m3.case_normalised) == (0, 0, 1)  # and distinct in the mono
    cut = _one(frame, page_id="rnd_s1_p0", condition="M7")
    assert (cut.letters_o0, cut.letters_il1, cut.letters_case_alike, cut.letters_case_normalised) == (1, 0, 1, 1)
    assert cut.letters_reading_errors == 2 and cut.letters_reading_noclass_errors == 0
    assert cut.letters_errors == cut.letters_results_errors == 3
    skipped = _one(frame, page_id="pg1_c1_p0", condition="M0")
    assert (skipped.omitted, skipped.omitted_interior, skipped.reading_errors) == (len(SKIPPED), len(SKIPPED), 0)
    assert skipped.read_chars == skipped.ref_chars - len(SKIPPED)
    assert frame["parts_add_up"].all() and frame["letters_parts_add_up"].all()
    assert {r["analysis_status"] for r in records} == {"exploratory"}
    assert {r["shape_alike"] for r in records} == {"cosuvxz", "coqsvxz"}
    assert m.score_checks(frame) == {"primary_mismatches": [], "letters_mismatches": [], "format_touched": [],
                                     "letters_format_touched": []}


def test_the_checks_compare_with_results_csv_and_spare_rows_r1_touched(m, tmp_path):
    run_dir = tmp_path / "run"
    reference = _prose(1)
    rows = study2.with_codes(pd.DataFrame([
        _row(run_dir, "natural", "pg1_c1_p1", reference, "S", 0.0, SANS, "Sure, here is the text:\n" + reference),
        _row(run_dir, "natural", "pg1_c1_p2", reference, "S", 5.0, SANS, reference.replace("May", "Mav")),
    ]), {"qwen": 14})
    rows.loc[1, "substitutions"] += 1  # results.csv disagrees with the transcription on the second row
    frame, _ = m.score_rows(rows, alike_for=ALIKE.__getitem__)
    checks = m.score_checks(frame)
    assert checks["format_touched"] == ["run/pg1_c1_p1"] and checks["letters_format_touched"] == []
    assert checks["primary_mismatches"] == ["run/pg1_c1_p2"]
    # R1 removed the preamble before the letters-only score, so its count differs from the stored one by design
    touched = _one(frame, page_id="pg1_c1_p1")
    assert touched.letters_errors == 0 and touched.letters_results_errors > 0
    assert checks["letters_mismatches"] == []


# --------------------------------------------------------------------------- pooling through mrr.study2


def test_a_counts_view_makes_study2_pool_the_chosen_counts(m, scored):
    frame, _ = scored
    sub = frame[frame["text_type"] == "natural"]
    values = study2.page_values(m.counts_view(sub, "omitted_rate"), "cer")
    for page, g in sub.groupby("page_id"):
        assert values[page] == pytest.approx(g["omitted"].sum() / g["ref_chars"].sum())
    assert m.counts_view(sub, "cer") is sub and m.counts_view(sub, "letters") is sub
    assert m.pooled(sub, "coverage") == pytest.approx(1.0 - sub["omitted"].sum() / sub["ref_chars"].sum())
    pages, a, b = m.paired_values(sub, "conventional", "contained", "coverage")
    _, oa, ob = m.paired_values(sub, "conventional", "contained", "omitted_rate")
    assert len(pages) == PAGES and np.allclose(a, 1.0 - oa) and np.allclose(b, 1.0 - ob)


def test_families_2_to_4_are_recomputed_on_reading_cer_with_study2s_method(m, scored):
    frame, _ = scored
    rows = m.paired_table(frame, m.FAMILIES)
    assert [(r["family"], r["measure"]) for r in rows] == [
        ("containment", "letters_reading_cer"), ("advantage", "reading_cer"), ("prose", "reading_cer")]
    assert all(r["status"] == r["analysis_status"] == "exploratory" for r in rows)
    advantage = _contrast(rows, family="advantage")
    assert advantage["contrast"] == "contained_vs_conventional" and advantage["n"] == PAGES
    assert advantage["median_diff"] < 0 and advantage["p_holm"] == advantage["p"]  # one reader: Holm leaves p
    assert advantage["treatment_pooled"] == 0.0 and advantage["baseline_pooled"] > 0
    containment = _contrast(rows, family="containment")
    assert containment["contrast"] == "contained_vs_cut" and containment["median_diff"] < 0
    prose = _contrast(rows, family="prose")  # the skipped line is behaviour: reading CER is 0 on both sides
    assert prose["bound"] == m.DELTA and prose["passes"] is True
    assert prose["method"] == "undefined: every difference is zero" and prose["median_diff"] == 0.0


def test_q1_separates_an_omission_habit_from_misreading(m, scored):
    frame, _ = scored
    rows = m.paired_table(frame, m.Q1)
    assert {r["measure"] for r in rows} == {"cer", "coverage", "reading_cer", "omitted_rate", "runaway_rate",
                                            "case_normalised_rate", "behaviour_rate"}
    cer, coverage, reading = (_contrast(rows, measure=k) for k in ("cer", "coverage", "reading_cer"))
    assert cer["median_diff"] == pytest.approx(SKIPPED_SHARE)  # the primary CER charges the skipped line
    assert coverage["median_diff"] == pytest.approx(-SKIPPED_SHARE)  # M0 and M3 both skip it on every page
    assert coverage["baseline_pooled"] == 1.0 and coverage["treatment_pooled"] == pytest.approx(1 - SKIPPED_SHARE)
    assert reading["median_diff"] == 0.0 and reading["method"] == "undefined: every difference is zero"
    assert all(math.isnan(r["p_holm"]) and r["significant"] == "" for r in rows)  # exploratory: no Holm


def test_q2_removes_the_o0_il1_and_case_classes_before_comparing_contained_with_cut(m, scored):
    frame, records = scored
    rows = m.paired_table(frame, m.Q2)
    letters, reading, noclass = (_contrast(rows, measure=k)
                                 for k in ("letters", "letters_reading_cer", "letters_reading_cer_noclass"))
    assert letters["contrast"] == "contained_vs_cut"
    assert letters["median_diff"] < reading["median_diff"] < 0  # x and 0 stay reading; the p swap is behaviour
    assert noclass["median_diff"] == 0.0 and noclass["baseline_pooled"] == noclass["treatment_pooled"] == 0.0
    cut = _contrast(m.classes_table(frame, m._alike_of(records)), text_type="random62", group="cut")
    assert (cut["letters_o0"], cut["letters_case_alike"], cut["letters_case_normalised"]) == (PAGES, PAGES, PAGES)
    assert cut["letters_reading_other"] == 0 and cut["letters_reading_cer_noclass"] == 0.0
    assert cut["shape_alike"] == "coqsvxz" and cut["letters_errors"] == 3 * PAGES


def test_q3_splits_case_errors_by_the_font_of_each_condition(m, scored):
    frame, records = scored
    rows = m.case_table(frame, m._alike_of(records))
    s5 = _contrast(rows, text_type="random62", level="condition", condition="S5")
    m3 = _contrast(rows, text_type="random62", level="condition", condition="M3")
    m7 = _contrast(rows, text_type="random62", level="condition", condition="M7")
    assert (s5["case_alike"], s5["case_normalised"], s5["case_share_of_errors"]) == (PAGES, 0, 1.0)
    assert (m3["case_alike"], m3["case_normalised"], m3["case_share_of_errors"]) == (0, PAGES, 1.0)
    assert (m7["case_alike"], m7["case_normalised"]) == (PAGES, PAGES)
    assert m7["case_share_of_errors"] == pytest.approx(2 / 3)
    contained = _contrast(rows, text_type="random62", level="group", group="contained")
    assert contained["condition"] == "" and contained["n_rows"] == 2 * PAGES and contained["case_normalised"] == PAGES
    assert [r["condition"] for r in rows if r["text_type"] == "random62" and r["level"] == "condition"] == [
        "S0", "S5", "S10", "M0", "M3", "M7"]


# --------------------------------------------------------------------------- the tables


def test_every_table_is_exploratory_and_two_builds_give_the_same_bytes(m, scored, tmp_path):
    frame, records = scored
    written = []
    for k in range(2):
        tables = m.build_tables(frame, records)
        assert set(tables) == {"pages", "conditions", "paired", "q1_prose", "q2_letterform", "q2_classes", "q3_case"}
        out = tmp_path / str(k)
        out.mkdir()
        for name, (rows, columns) in tables.items():
            assert rows and {r["analysis_status"] for r in rows} == {"exploratory"}, name
            m.analyze1.write_table(out / f"{name}.csv", rows, columns)
        written.append({p.name: p.read_bytes() for p in sorted(out.glob("*.csv"))})
    assert written[0] == written[1]
    m0 = _contrast(m.build_tables(frame, records)["conditions"][0], text_type="natural", condition="M0")
    assert m0["coverage"] == pytest.approx(1 - SKIPPED_SHARE) and m0["cer"] == pytest.approx(m0["omitted_rate"])
    assert m0["reading_cer"] == 0.0 and m0["pages_with_omission"] == 1.0 and m0["rows_not_adding_up"] == 0
    assert m0["coverage_low"] <= m0["coverage"] <= m0["coverage_high"]


def test_the_script_never_writes_into_the_confirmatory_folder(m, tmp_path):
    out = tmp_path / "tables"
    code = m.main(["--config", str(REPO_ROOT / "configs" / "sweeps" / "study2.yaml"), "--runs-root",
                   str(tmp_path / "runs"), "--out-dir", str(out), "--name", "study2"])
    assert code == 1 and not out.exists()
