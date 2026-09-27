# PURPOSE: Error metrics for transcriptions.
# WHAT GOES HERE: CER and WER, both computed with jiwer.
# FILLED IN AT STEP: W2.4.
# DO NOT: Write your own edit distance; use jiwer so results match the cited library.
"""Transcription error metrics.

Learn B3 (error metrics) before implementing.
"""

from __future__ import annotations


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate of hypothesis against reference.

    Args:
        reference: ground-truth page text.
        hypothesis: model transcription.

    Returns:
        CER as a float (0 is perfect; can exceed 1).
    """
    raise NotImplementedError("TODO[W2.4]: compute CER with jiwer")


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate of hypothesis against reference.

    Args:
        reference: ground-truth page text.
        hypothesis: model transcription.

    Returns:
        WER as a float (0 is perfect; can exceed 1).
    """
    raise NotImplementedError("TODO[W2.4]: compute WER with jiwer")
