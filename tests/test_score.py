# PURPOSE: Correctness tests for CER, WER and the character counts.
# WHAT GOES HERE: Known-value checks; every expected value is from build guide section 2.6 [Measured with jiwer 4.0.0].
# FILLED IN AT STEP: W2.4.
# DO NOT: Change expected values to match an implementation.
"""Tests for mrr.score."""

from __future__ import annotations

import dataclasses

import jiwer
import pytest

from mrr.score import CharCounts, cer, char_counts, pooled_cer, wer


def test_cer_transposition():
    assert abs(cer("the cat", "teh cat") - 2 / 7) < 1e-9


@pytest.mark.parametrize("x", ["a", "the cat", "Rendering for Machine Readers"])
def test_cer_identity(x):
    assert cer(x, x) == 0


def test_the_transposition_counts_two_errors_whatever_the_split():
    counts = char_counts("the cat", "teh cat")
    assert counts.substitutions + counts.deletions + counts.insertions == 2  # jiwer reports D = 1, I = 1
    assert counts.hits + counts.substitutions + counts.deletions == len("the cat")


@pytest.mark.parametrize(
    ("hypothesis", "expected", "field"),
    [("tha cat", 1 / 7, "substitutions"), ("thee cat", 1 / 7, "insertions")],
)
def test_one_wrong_or_one_extra_character(hypothesis, expected, field):
    counts = char_counts("the cat", hypothesis)
    assert abs(cer("the cat", hypothesis) - expected) < 1e-9
    assert getattr(counts, field) == 1
    assert sum(dataclasses.astuple(counts)) - counts.hits == 1


def test_no_case_folding():
    assert abs(cer("the", "The") - 1 / 3) < 1e-9


def test_line_breaks_are_normalised_before_scoring():
    assert cer("the cat sat", "the cat\nsat") == 0
    # Without normalisation jiwer charges the line break as one substitution: 1/11.
    assert abs(jiwer.process_characters("the cat sat", "the cat\nsat").cer - 1 / 11) < 1e-9


def test_two_insertions():
    assert abs(cer("abc", "abc d") - 2 / 3) < 1e-9
    assert char_counts("abc", "abc d").insertions == 2


def test_an_empty_transcription_is_three_deletions_not_an_error():
    assert cer("abc", "") == 1.0
    assert char_counts("abc", "") == CharCounts(hits=0, substitutions=0, deletions=3, insertions=0)
    assert wer("the cat sat", "") == 1.0


@pytest.mark.parametrize("reference", ["", "  \n "])
def test_an_empty_reference_is_refused(reference):
    # jiwer.cer("", "abc") returns 3 instead of failing (build guide section 0.3).
    for fn in (cer, wer, char_counts):
        with pytest.raises(ValueError):
            fn(reference, "abc")


def test_pooled_cer_pools_the_counts_not_the_page_rates():
    pages = [("ab", "xb"), ("abcd", "abcd")]
    counts = [char_counts(reference, hypothesis) for reference, hypothesis in pages]
    assert abs(pooled_cer(counts) - 1 / 6) < 1e-9
    mean_of_pages = sum(cer(r, h) for r, h in pages) / len(pages)
    assert abs(mean_of_pages - 1 / 4) < 1e-9  # the number D13 forbids


def test_pooled_cer_refuses_no_counts():
    with pytest.raises(ValueError):
        pooled_cer([])


def test_wer_one_wrong_word_in_three():
    assert abs(wer("the cat sat", "the dog sat") - 1 / 3) < 1e-9


def test_char_counts_is_frozen():
    counts = char_counts("abc", "abc")
    with pytest.raises(dataclasses.FrozenInstanceError):
        counts.hits = 0  # type: ignore[misc]
