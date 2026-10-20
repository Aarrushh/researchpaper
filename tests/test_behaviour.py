# PURPOSE: Tests for the scoring rules of ANALYSIS_PLAN_3.md section 1 (mrr.behaviour), on planted strings. CPU only.
# WHAT GOES HERE: One test per rule: an omission inside correct text, the same deletion inside misreading, head and tail omissions, a runaway insertion, a preamble with fences, case errors split by shape, and that the classes add up to the primary count.
# FILLED IN AT STEP: Analysis plan 3 (20 October 2026), before any stage-0 or Study 2 row is scored.
# DO NOT: Change an expected value to match the implementation; change the rule in ANALYSIS_PLAN_3.md first.
"""Tests for mrr.behaviour."""

from __future__ import annotations

import random

import numpy as np
import pytest

from conftest import REPO_ROOT
from mrr import behaviour
from mrr.text_source import normalise_text

LINES = (
    "The harbour lay quiet under a pale grey morning sky.",
    "Fishing boats rocked at their moorings near the old stone wall.",
    "A woman in a blue coat walked slowly along the quay with her dog.",
    "Gulls circled above the market stalls, calling to one another.",
    "By noon the tide had turned and the boats began to leave.",
)
REF = " ".join(LINES)
ALIKE = frozenset("csouvxz")
SANS = REPO_ROOT / "data" / "external" / "fonts" / "source-sans-3" / "SourceSans3-Regular.ttf"
MONO = REPO_ROOT / "data" / "external" / "fonts" / "fira-mono" / "FiraMono-Regular.ttf"


def _score(hyp: str, reference: str = REF, **kw) -> behaviour.RowScore:
    return behaviour.score_row(reference, hyp, ALIKE, **kw)


def _without(i: int) -> str:
    return "\n".join(line for k, line in enumerate(LINES) if k != i)


def _garbled(line: str) -> str:
    return "".join("#" if k % 3 == 1 else ch for k, ch in enumerate(line))


def test_a_line_skipped_inside_text_read_right_is_an_interior_omission():
    s = _score(_without(2))
    assert (s.omitted_interior, s.omitted_head, s.omitted_tail) == (len(LINES[2]) + 1, 0, 0)
    assert s.omissions == 1 and s.reading_errors == 0 and s.errors == s.omitted and s.parts_add_up
    assert s.coverage == pytest.approx(1 - (len(LINES[2]) + 1) / len(REF))
    assert s.reading_cer == 0.0


def test_the_same_deletion_inside_misread_text_stays_a_reading_error():
    hyp = "\n".join([LINES[0], _garbled(LINES[1]), _garbled(LINES[3]), LINES[4]])
    s = _score(hyp)
    assert s.omitted == 0 and s.omissions == 0
    assert s.reading_errors == s.errors > len(LINES[2]) and s.parts_add_up


def test_a_dropped_first_or_last_line_is_a_head_or_tail_omission():
    head, tail = _score(_without(0)), _score(_without(4))
    assert (head.omitted_head, head.omitted_interior, head.omitted_tail) == (len(LINES[0]) + 1, 0, 0)
    assert (tail.omitted_head, tail.omitted_interior, tail.omitted_tail) == (0, 0, len(LINES[4]) + 1)
    assert head.reading_errors == tail.reading_errors == 0


def test_the_anchor_allows_three_errors_beside_a_run_and_not_four():
    def hyp(n_errors: int) -> str:
        line = list(LINES[1])
        for k in range(n_errors):
            line[-3 - 3 * k] = "#"  # inside the 30 reference characters before the skipped line
        return "\n".join([LINES[0], "".join(line), LINES[3], LINES[4]])

    three, four = _score(hyp(3)), _score(hyp(4))
    assert three.omissions == 1 and three.reading_errors == 3
    assert four.omissions == 0 and four.reading_errors == four.errors
    assert three.parts_add_up and four.parts_add_up


def test_a_short_deletion_is_a_reading_error_and_the_run_length_is_a_parameter():
    hyp = REF.replace("walked slowly along", "along")  # 14 characters: "walked slowly "
    s = _score(hyp)
    assert s.omissions == 0 and s.reading_errors == s.errors == 14
    assert _score(hyp, min_run=10).omitted_interior == 14


def test_an_empty_transcription_is_not_an_omission():
    s = _score("")
    assert s.omitted == 0 and s.reading_errors == s.errors == len(REF) and s.coverage == 1.0


def test_a_repeated_block_at_the_end_is_a_runaway_insertion():
    s = _score(REF + "\n" + LINES[4])
    assert s.runaways == 1 and s.runaway_inserted == len(LINES[4]) + 1
    assert s.reading_errors == 0 and s.omitted == 0 and s.parts_add_up


def test_a_preamble_and_its_fences_are_format_and_counted_apart():
    preamble = "Sure, here is the transcribed text from the image:"
    s = _score(preamble + "\n```text\n" + "\n".join(LINES) + "\n```")
    assert s.format_touched and s.format_chars == len(normalise_text(f"```text ``` {preamble}"))
    assert s.reading_errors == 0 and s.behaviour_errors == 0
    assert s.errors == len(preamble) + len(" ```text ") + len(" ```")  # the primary count keeps them


def test_strip_format_touches_only_fences_and_a_preamble_first_line():
    assert behaviour.strip_format("The text was dark.\nNext line") == ("The text was dark.\nNext line", [])
    assert behaviour.strip_format("abc\nHere is:\ndef") == ("abc\nHere is:\ndef", [])
    assert behaviour.strip_format("\n Here’s the text:\nabc") == ("\nabc", [" Here’s the text:"])
    assert behaviour.strip_format("```\nabc\n```") == ("abc", ["```", "```"])


def test_case_errors_split_by_letter_shape_and_o_zero_stays_reading():
    s = behaviour.score_row("Coach Kim saw Otto", "coach kim saw otto", ALIKE)
    assert (s.case_alike, s.case_normalised) == (2, 1)  # C and O alike, K not
    assert s.reading_errors == 2 and s.behaviour_errors == 1 and s.parts_add_up
    z = behaviour.score_row("Coach Kim saw Otto", "Coach Kim saw 0tto", ALIKE)
    assert (z.case_alike, z.case_normalised, z.reading_errors) == (0, 0, 1)


def test_the_classes_add_up_to_the_primary_count_on_random_edits():
    rng = random.Random(20261027)
    for _ in range(200):
        chars = list(REF)
        for _ in range(rng.randint(0, 6)):
            i = rng.randrange(len(chars))
            op = rng.choice(("sub", "case", "del", "long_del", "ins", "long_ins"))
            if op == "sub":
                chars[i] = rng.choice("#%&0123")
            elif op == "case":
                chars[i] = chars[i].swapcase()
            elif op == "del":
                del chars[i:i + rng.randint(1, 5)]
            elif op == "long_del":
                del chars[i:i + rng.randint(15, 60)]
            elif op == "ins":
                chars[i:i] = list("~" * rng.randint(1, 5))
            else:
                chars[i:i] = list(REF[: rng.randint(15, 60)])
            if not chars:
                chars = ["x"]
        s = _score("".join(chars))
        assert s.parts_add_up and not s.format_touched
        assert min(s.reading_errors, s.omitted, s.runaway_inserted, s.case_alike, s.case_normalised) >= 0
        assert 0.0 <= s.coverage <= 1.0


@pytest.mark.skipif(not (SANS.exists() and MONO.exists()), reason="fonts not present (W1.4)")
def test_the_shape_alike_letters_of_the_two_study_fonts():
    # Measured on 20 October 2026 with the study's renderer (ANALYSIS_PLAN_3.md section 1.1, R4, and Amendment 1).
    assert behaviour.shape_alike_letters(str(SANS)) == frozenset("csouvxz")
    assert behaviour.shape_alike_letters(str(MONO)) == frozenset("coqsvxz")
    overlaps = behaviour.letter_overlaps(SANS)
    assert overlaps["l"] < 0.2 and overlaps["o"] > overlaps["a"]


def test_overlap_of_a_mask_with_itself_and_with_a_wider_one():
    bar = np.full((6, 2), 255, dtype=np.uint8)
    assert behaviour.overlap(bar, bar) == 1.0
    assert behaviour.overlap(bar, np.full((6, 4), 255, dtype=np.uint8)) == pytest.approx(12 / 24)
    assert behaviour.overlap(bar, np.zeros((6, 2), dtype=np.uint8)) == 0.0
