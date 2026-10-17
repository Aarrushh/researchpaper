"""Tests for scripts/21_levels.py: the exploratory per-level table behind R09 to R11 (W4.1)."""

from __future__ import annotations

import importlib.util
import math
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
        "phase_y_px": 0.0, "page_id": page_id, "patch_px": 14, "hits": counts.hits,
        "substitutions": counts.substitutions, "deletions": counts.deletions, "insertions": counts.insertions,
        "hit_limit": hit_limit, "reference": reference, "transcription": transcription, **stored,
    }


def _wide(char: str, size: float, patch: int) -> bool:
    return char in "MW"


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


def test_spaces_inserted_inside_a_word_are_counted_with_those_beside_a_wide_character():
    (row,) = levels.level_rows([_record("aW bc", "a W b c")], wide=_wide)
    assert row["spaces_inserted_in_words_per_char"] == pytest.approx(2 / 5)
    assert row["inserted_beside_wide_share"] == pytest.approx(1 / 2)  # a|W is beside W, b|c is not
    assert row["gaps_beside_wide_share"] == pytest.approx(1 / 2)  # the gaps inside words: a-W and b-c


def test_a_space_inserted_beside_a_word_space_or_at_the_end_is_not_inside_a_word():
    (row,) = levels.level_rows([_record("aW bc", "aW x bc x")], wide=_wide)
    assert row["insertion_rate"] == pytest.approx(4 / 5)
    assert row["spaces_inserted_in_words_per_char"] == 0


def test_without_a_font_the_wide_shares_are_empty():
    (row,) = levels.level_rows([_record("aW bc", "a W bc")])
    assert math.isnan(row["inserted_beside_wide_share"]) and math.isnan(row["gaps_beside_wide_share"])


def test_the_wide_characters_of_the_font_are_those_the_preregistration_amendment_names():
    font = REPO_ROOT / "data" / "external" / "fonts" / "source-sans-3" / "SourceSans3-Regular.ttf"
    if not font.exists():
        pytest.skip("the font is not downloaded (data/external/fonts/README.md)")
    wide = levels.font_wide(str(font))
    random62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    assert {c for c in random62 if wide(c, 24.0, 14)} == set("BDGHMNOQUWmw")  # amendment of 15 October, item 1
    assert {c for c in random62 if wide(c, 20.0, 16)} == {"m"}


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
