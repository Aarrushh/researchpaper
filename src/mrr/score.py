# PURPOSE: Error metrics for transcriptions.
# WHAT GOES HERE: CER and WER, both computed with jiwer; the four character counts per row and the pooled CER (D13).
# FILLED IN AT STEP: W2.4.
# DO NOT: Write your own edit distance; use jiwer so results match the cited library. Lower-case or strip punctuation. Average page CERs (pool the counts, D13).
"""Transcription error metrics.

Learn B5 (error metrics) before implementing. Both strings pass through ``text_source.normalise_text`` first: the
reference already has, the transcription has not, and the readers write a line break at the end of every rendered
line. CER = (S + D + I) / (H + S + D) over the characters jiwer aligns (build guide section 2.6, D13).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jiwer

from mrr.text_source import normalise_text


@dataclass(frozen=True)
class CharCounts:
    """jiwer's four character counts for one reference and one transcription (section 2.6)."""

    hits: int
    substitutions: int
    deletions: int
    insertions: int


def _normalised(reference: str, hypothesis: str) -> tuple[str, str]:
    """Normalise both strings; refuse an empty reference, for which jiwer returns a count instead of failing."""
    ref = normalise_text(reference)
    if not ref:
        raise ValueError("score: the normalised reference is empty (jiwer.cer('', 'abc') would return 3, not fail)")
    return ref, normalise_text(hypothesis)


def char_counts(reference: str, hypothesis: str) -> CharCounts:
    """Return jiwer's hits, substitutions, deletions and insertions, both strings normalised first.

    Args:
        reference: ground-truth page text.
        hypothesis: model transcription.
    """
    ref, hyp = _normalised(reference, hypothesis)
    out = jiwer.process_characters(ref, hyp)
    return CharCounts(out.hits, out.substitutions, out.deletions, out.insertions)


def char_counts_letters_only(reference: str, hypothesis: str) -> CharCounts:
    """Return jiwer's four counts with every space removed from both normalised strings (Study 2's letters-only CER).

    PREREGISTRATION_2.md section 4; the same strings as ``cer_without_spaces`` of scripts/21_levels.py.

    Args:
        reference: ground-truth page text.
        hypothesis: model transcription.
    """
    ref, hyp = _normalised(reference, hypothesis)
    out = jiwer.process_characters(ref.replace(" ", ""), hyp.replace(" ", ""))
    return CharCounts(out.hits, out.substitutions, out.deletions, out.insertions)


def spaces_inserted(reference: str, hypothesis: str) -> int:
    """Return the number of spaces in the insert chunks of jiwer's character alignment, both strings normalised first.

    Study 2's gap response (PREREGISTRATION_2.md section 4, H3): divided by the reference length it is the inserted
    spaces per reference character. A space that replaces a character (a substitute chunk) is not counted.

    Args:
        reference: ground-truth page text.
        hypothesis: model transcription.
    """
    ref, hyp = _normalised(reference, hypothesis)
    chunks = jiwer.process_characters(ref, hyp).alignments[0]
    return sum(hyp[c.hyp_start_idx:c.hyp_end_idx].count(" ") for c in chunks if c.type == "insert")


def pooled_cer(counts: Iterable[CharCounts]) -> float:
    """Return (S + D + I) / (H + S + D) over every row in ``counts`` (D13): never the mean of per-page CERs."""
    hits = substitutions = deletions = insertions = 0
    for c in counts:
        hits += c.hits
        substitutions += c.substitutions
        deletions += c.deletions
        insertions += c.insertions
    denominator = hits + substitutions + deletions
    if denominator == 0:
        raise ValueError("pooled_cer: no reference characters to pool")
    return (substitutions + deletions + insertions) / denominator


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate of hypothesis against reference: 0 is perfect; it can exceed 1."""
    return pooled_cer([char_counts(reference, hypothesis)])


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate of hypothesis against reference: 0 is perfect; it can exceed 1."""
    ref, hyp = _normalised(reference, hypothesis)
    return jiwer.process_words(ref, hyp).wer
