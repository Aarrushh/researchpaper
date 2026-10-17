"""Tests for scripts/21_levels.py: the exploratory per-level table behind R09 to R11 (W4.1)."""

from __future__ import annotations

import importlib.util
import re
import sys

import pytest

from conftest import REPO_ROOT
from mrr.score import char_counts

SCRIPT = REPO_ROOT / "scripts" / "21_levels.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("levels", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["levels"] = module
    spec.loader.exec_module(module)
    return module


levels = _load_script()


def _record(reference: str, transcription: str, page_id: str = "rnd1_p0000", phase_x: float = 0.0,
            hit_limit: bool = False, **stored) -> dict:
    counts = char_counts(reference, transcription)
    return {
        "reader": "qwen", "text_type": "random62", "block": "H", "render_mode": "patch_aware",
        "line_pitch_mode": "unaligned", "aa_mode": "coverage_srgb", "font_size_px": 24.0, "phase_x_px": phase_x,
        "phase_y_px": 0.0, "page_id": page_id, "hits": counts.hits, "substitutions": counts.substitutions,
        "deletions": counts.deletions, "insertions": counts.insertions, "hit_limit": hit_limit,
        "reference": reference, "transcription": transcription, **stored,
    }


def test_an_inserted_space_is_an_insertion_and_an_added_space_and_no_error_without_spaces():
    (row,) = levels.level_rows([_record("ab cd", "a b cd")])
    assert row["cer"] == pytest.approx(1 / 5)
    assert row["insertion_rate"] == pytest.approx(1 / 5)
    assert row["substitution_rate"] == row["deletion_rate"] == 0
    assert row["added_spaces_per_char"] == pytest.approx(1 / 5)
    assert row["cer_without_spaces"] == 0
    assert row["status"] == "exploratory"


def test_the_parts_add_up_to_the_cer_and_are_pooled_over_rows_not_averaged():
    rows = [_record("ab cd", "ab cd", page_id="rnd1_p0000"), _record("abcdefghij", "abcdefghiX", page_id="rnd1_p0001")]
    (row,) = levels.level_rows(rows)
    assert row["cer"] == pytest.approx(1 / 15)  # the mean of the page CERs would be 0.05
    assert row["substitution_rate"] + row["deletion_rate"] + row["insertion_rate"] == pytest.approx(row["cer"])
    assert (row["rows"], row["pages"], row["ref_chars_per_row"]) == (2, 2, 7.5)


def test_a_dropped_space_counts_negative_and_the_letters_alone_decide_the_cer_without_spaces():
    (row,) = levels.level_rows([_record("ab cd", "abcX")])
    assert row["added_spaces_per_char"] == pytest.approx(-1 / 5)
    assert row["cer_without_spaces"] == pytest.approx(1 / 4)  # abcd against abcX: one substitution in four


def test_levels_are_kept_apart_sorted_and_carry_their_hit_limit_share():
    rows = levels.level_rows([
        _record("ab cd", "ab cd", page_id="rnd1_p0000", phase_x=3.0, hit_limit=True),
        _record("ab cd", "ab cd", page_id="rnd1_p0001", phase_x=3.0),
        _record("ab cd", "ab cd", page_id="rnd1_p0000", phase_x=1.0),
    ])
    assert [r["phase_x_px"] for r in rows] == [1.0, 3.0]
    assert [r["hit_limit_share"] for r in rows] == [0.0, 0.5]


def test_counts_that_differ_from_the_texts_stop_the_table():
    with pytest.raises(ValueError, match="results.csv holds"):
        levels.level_rows([_record("ab cd", "ab cd", hits=4)])


def test_the_columns_start_with_the_level_and_end_with_status_and_family():
    (row,) = levels.level_rows([_record("ab cd", "ab cd")])
    assert tuple(row) == levels.COLUMNS


def test_the_data_dictionary_documents_exactly_the_columns():
    text = (REPO_ROOT / "docs" / "04_data_dictionary.md").read_text(encoding="utf-8")
    start = text.index("\n## reports/tables/exploratory/levels.csv")
    section = text[start : text.index("\n## ", start + 1)]
    assert tuple(re.findall(r"^\| `([a-z_]+)` \|", section, flags=re.MULTILINE)) == levels.COLUMNS
