# PURPOSE: The scoring rules of ANALYSIS_PLAN_3.md section 1: split each row's character errors into reading errors and behaviour.
# WHAT GOES HERE: The format strip (R1), the one jiwer alignment (R2), anchored runs (R3), case errors split by letter shape (R4), the reading errors (R5), and the letters a font draws in lower case as its capital at a smaller size.
# FILLED IN AT STEP: Analysis plan 3 (20 October 2026), written and tested before any stage-0 or Study 2 row is scored with it.
# DO NOT: Change the primary CER (mrr.score, D13) or rewrite results.csv. Change MIN_RUN, the anchor, PREAMBLE or the shape test after a real row has been scored (ANALYSIS_PLAN_3.md section 1.3); report a sensitivity instead.
"""Reading errors and behaviour (ANALYSIS_PLAN_3.md section 1).

Above the size where a reader's cliff ends, most of its remaining error is not misreading: it skips a line inside text
it reads correctly, drops the page's unfinished last sentence, writes a capital in lower case, or opens with a chat
preamble. ``score_row`` splits one row's errors so that each class can be pooled over pages (D13):

    R1  format       a code-fence line, or a first line such as "Sure, here is the text:", removed before scoring
    R2  alignment    one jiwer character alignment of the normalised strings, as ``mrr.score.char_counts``
    R3  runs         a delete or insert chunk of at least MIN_RUN characters whose ANCHOR_CHARS neighbouring reference
                     characters on each side that exists hold at most ANCHOR_ERRORS errors: an anchored deletion is an
                     omission (head, interior or tail), an anchored insertion a runaway insertion; an unanchored run
                     stays a reading error
    R4  case         a letter written in the other case: a reading error when the font's lower case is its capital at
                     a smaller size (``shape_alike_letters``), behaviour (case normalisation) otherwise
    R5  reading      everything else

On a row that R1 leaves alone, the primary errors equal reading errors + omitted + runaway + case-normalised exactly
(``RowScore.parts_add_up``). On a row R1 touches, the primary count is the stored transcription's and the classes are
the stripped one's, so the two can differ by about the characters removed.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import jiwer
import numpy as np

from mrr.score import char_counts
from mrr.text_source import normalise_text

MIN_RUN = 15  # R3: L; in Study 1, 80% (SmolVLM-2.2B) and 60% (500M) of deleted prose characters lay in runs this long
ANCHOR_CHARS = 30  # R3: the reference characters looked at on each side of a run
ANCHOR_ERRORS = 3  # R3: the most errors among them for the run to be anchored
SHAPE_SIZE_PX = 48  # R4: the size at which a lower-case glyph is compared with its capital
SHAPE_MIN_IOU = 0.6  # R4: the overlap from which the two count as one shape; 0.8, set before measuring, admitted no
# letter in either font (scaling the lower case up thickens its strokes: o/O reach only 0.66 and 0.62), so it was
# lowered to the widest gap of Source Sans 3's ranking (u 0.64, q 0.52) before any row was scored (20 October 2026)
PREAMBLE = re.compile(
    r"^\s*(sure|certainly|of course|here is|here(?:'|’)s|here are|below is|the text|the transcription|"
    r"transcribed text)\b[^\n]*:\s*$", re.IGNORECASE)  # R1: tested on the first line that is not blank
FENCE = re.compile(r"^\s*```[\w+.-]*\s*$")  # R1: a line that is only a code fence, with or without a language name
HEAD, INTERIOR, TAIL = "head", "interior", "tail"
FONT_DIR = Path("data") / "external" / "fonts"  # under the repository root: results.csv names a font by its file name


# --------------------------------------------------------------------------- a results row's texts and font


def read_texts(run_dir: str | Path, page_id: str, transcription_file: str) -> tuple[str, str]:
    """The reference and the stored transcription of one results row: ``references/<page_id>.txt`` and the row's
    ``transcription_file``, both under its run folder (as ``mrr.study2.run_confusions`` reads them)."""
    run_dir = Path(run_dir)
    return ((run_dir / "references" / f"{page_id}.txt").read_text(encoding="utf-8"),
            (run_dir / str(transcription_file)).read_text(encoding="utf-8"))


def font_path(font_file: str, root: str | Path) -> Path:
    """The font file a results row names: a path as given when it exists under ``root``, else the one file of that name
    under ``root / FONT_DIR``. Raises FileNotFoundError when there is none, ValueError when there are several."""
    given = Path(root) / str(font_file)
    if given.is_file():
        return given
    found = sorted((Path(root) / FONT_DIR).rglob(Path(str(font_file)).name))
    if not found:
        raise FileNotFoundError(f"no font file {font_file!r} under {Path(root) / FONT_DIR}")
    if len(found) > 1:
        raise ValueError(f"{len(found)} font files named {font_file!r} under {Path(root) / FONT_DIR}")
    return found[0]


# --------------------------------------------------------------------------- R1, format


def strip_format(hypothesis: str) -> tuple[str, list[str]]:
    """The transcription without its code-fence lines and a preamble first line, and the lines removed (R1).

    The preamble is tested on the first line that is not blank once the fences are gone, so "Sure, here it is:" above
    a fenced block goes too. A transcription with nothing to remove comes back unchanged.
    """
    lines = hypothesis.splitlines()
    removed = [line for line in lines if FENCE.match(line)]
    kept = [line for line in lines if not FENCE.match(line)]
    first = next((i for i, line in enumerate(kept) if line.strip()), None)
    if first is not None and PREAMBLE.match(kept[first]):
        removed.append(kept.pop(first))
    if not removed:
        return hypothesis, []
    return "\n".join(kept), removed


# --------------------------------------------------------------------------- R4, letter shapes


def _glyph(font_file: str | Path, size_px: int, char: str) -> np.ndarray:
    """The ink of one character, rasterised as the pages were (``mrr.render.load_glyph``: unhinted FreeType, phase
    0), cropped to its inked pixels, as grey levels 0 to 255."""
    from mrr.render import load_glyph_coverage

    coverage = load_glyph_coverage(str(font_file), float(size_px), char)
    ink = np.argwhere(coverage > 0)
    if ink.size == 0:
        raise ValueError(f"the font draws no ink for {char!r}")
    (y0, x0), (y1, x1) = ink.min(axis=0), ink.max(axis=0) + 1
    return np.round(coverage[y0:y1, x0:x1] * 255.0).astype(np.uint8)


def _scaled(glyph: np.ndarray, factor: float) -> np.ndarray:
    from PIL import Image

    h, w = glyph.shape
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    return np.asarray(Image.fromarray(glyph).resize(size, Image.Resampling.LANCZOS))


def overlap(a: np.ndarray, b: np.ndarray, shift: int = 2) -> float:
    """The best intersection over union of two ink masks (grey >= 128), centred on each other and moved by up to
    ``shift`` pixels each way."""
    a, b = a >= 128, b >= 128
    pad = shift + 1
    h, w = max(a.shape[0], b.shape[0]) + 2 * pad, max(a.shape[1], b.shape[1]) + 2 * pad

    def placed(m: np.ndarray, dy: int = 0, dx: int = 0) -> np.ndarray:
        out = np.zeros((h, w), dtype=bool)
        y, x = (h - m.shape[0]) // 2 + dy, (w - m.shape[1]) // 2 + dx
        out[y:y + m.shape[0], x:x + m.shape[1]] = m
        return out

    base = placed(a)
    best = 0.0
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            moved = placed(b, dy, dx)
            union = np.logical_or(base, moved).sum()
            if union:
                best = max(best, float(np.logical_and(base, moved).sum()) / float(union))
    return best


def letter_overlaps(font_file: str | Path, size_px: int = SHAPE_SIZE_PX) -> dict[str, float]:
    """For each letter a to z: the overlap of its lower-case glyph, scaled by cap height over x-height, with its
    capital (R4). Cap height is the ink height of H, x-height that of x."""
    factor = _glyph(font_file, size_px, "H").shape[0] / _glyph(font_file, size_px, "x").shape[0]
    return {c: overlap(_glyph(font_file, size_px, c.upper()), _scaled(_glyph(font_file, size_px, c), factor))
            for c in string.ascii_lowercase}


@lru_cache(maxsize=None)
def shape_alike_letters(font_file: str, size_px: int = SHAPE_SIZE_PX, min_iou: float = SHAPE_MIN_IOU) -> frozenset[str]:
    """The lower-case letters whose glyph in ``font_file`` is its capital at a smaller size: overlap >= ``min_iou``."""
    return frozenset(c for c, iou in letter_overlaps(font_file, size_px).items() if iou >= min_iou)


# --------------------------------------------------------------------------- R2 to R5, one row


@dataclass(frozen=True)
class Run:
    """One delete or insert chunk of at least MIN_RUN characters (R3). ``start`` and ``end`` index the normalised
    reference; an insertion has start == end, the gap before reference character ``start``."""

    kind: str  # "delete" or "insert"
    start: int
    end: int
    length: int
    where: str  # HEAD, INTERIOR or TAIL
    anchored: bool


@dataclass(frozen=True)
class RowScore:
    """One row's characters by class (ANALYSIS_PLAN_3.md section 1.2). Every count is of normalised characters."""

    reference_chars: int
    errors: int  # primary: S + D + I of mrr.score.char_counts on the transcription as stored
    format_chars: int  # R1: the normalised characters of the lines removed
    omitted_head: int
    omitted_interior: int
    omitted_tail: int
    runaway_inserted: int
    case_alike: int  # R4, counted with the reading errors
    case_normalised: int  # R4, behaviour
    reading_errors: int  # R5 and case_alike
    omissions: int  # anchored deletions
    runaways: int  # anchored insertions
    format_touched: bool

    @property
    def omitted(self) -> int:
        return self.omitted_head + self.omitted_interior + self.omitted_tail

    @property
    def behaviour_errors(self) -> int:
        """Omitted, runaway and case-normalised characters; format is counted apart (``format_chars``)."""
        return self.omitted + self.runaway_inserted + self.case_normalised

    @property
    def parts_add_up(self) -> bool:
        """The primary errors equal reading + behaviour errors; true by construction on rows R1 leaves alone."""
        return self.errors == self.reading_errors + self.behaviour_errors

    @property
    def coverage(self) -> float:
        return 1.0 - self.omitted / self.reference_chars

    @property
    def reading_cer(self) -> float:
        """Reading errors over the reference characters not omitted; nan when every character was omitted."""
        read = self.reference_chars - self.omitted
        return self.reading_errors / read if read else float("nan")


def runs(ref: str, hyp: str, chunks: Iterable, min_run: int = MIN_RUN, anchor_chars: int = ANCHOR_CHARS,
         anchor_errors: int = ANCHOR_ERRORS) -> list[Run]:
    """The delete and insert chunks of at least ``min_run`` characters in jiwer's alignment of ``ref`` and ``hyp``,
    each with its place and whether it is anchored (R3).

    A side of a run exists when the reference goes on past it; its window is the ``anchor_chars`` reference
    characters next to the run, or as many as there are. The window's errors are its substituted and deleted
    characters and the characters inserted strictly inside it. A run is anchored when at least one side exists and
    every side that exists holds at most ``anchor_errors`` errors, so a run that covers the whole reference is not.
    Errors of a neighbouring run count in the window like any other.
    """
    chunks = list(chunks)
    n = len(ref)
    wrong = np.zeros(n, dtype=int)  # 1 where a reference character is substituted or deleted
    inserted = np.zeros(n + 1, dtype=int)  # characters inserted in the gap before reference character g
    for c in chunks:
        if c.type in ("substitute", "delete"):
            wrong[c.ref_start_idx:c.ref_end_idx] = 1
        elif c.type == "insert":
            inserted[c.ref_start_idx] += c.hyp_end_idx - c.hyp_start_idx

    def errors(lo: int, hi: int) -> int:
        return int(wrong[lo:hi].sum() + inserted[lo + 1:hi].sum())

    out: list[Run] = []
    for c in chunks:
        if c.type == "delete":
            start, end, length = c.ref_start_idx, c.ref_end_idx, c.ref_end_idx - c.ref_start_idx
        elif c.type == "insert":
            start = end = c.ref_start_idx
            length = c.hyp_end_idx - c.hyp_start_idx
        else:
            continue
        if length < min_run:
            continue
        sides = []
        if start > 0:
            sides.append(errors(max(0, start - anchor_chars), start))
        if end < n:
            sides.append(errors(end, min(n, end + anchor_chars)))
        anchored = bool(sides) and all(e <= anchor_errors for e in sides)
        where = HEAD if start == 0 else TAIL if end == n else INTERIOR
        out.append(Run(c.type, start, end, length, where, anchored))
    return out


def score_row(reference: str, hypothesis: str, shape_alike: Iterable[str], min_run: int = MIN_RUN,
              anchor_chars: int = ANCHOR_CHARS, anchor_errors: int = ANCHOR_ERRORS) -> RowScore:
    """One row's errors by class (R1 to R5). ``shape_alike``: the lower-case letters of R4 for the row's font, from
    ``shape_alike_letters``.

    Raises:
        ValueError: when the normalised reference is empty (as ``mrr.score.char_counts``).
    """
    primary = char_counts(reference, hypothesis)
    kept, removed = strip_format(hypothesis)
    ref, hyp = normalise_text(reference), normalise_text(kept)
    chunks = jiwer.process_characters(ref, hyp).alignments[0]
    alike = frozenset(shape_alike)

    found = runs(ref, hyp, chunks, min_run, anchor_chars, anchor_errors)
    omitted = {HEAD: 0, INTERIOR: 0, TAIL: 0}
    runaway = omissions = runaways = 0
    behaviour_spans = set()
    for r in found:
        if not r.anchored:
            continue
        behaviour_spans.add((r.kind, r.start, r.end))
        if r.kind == "delete":
            omitted[r.where] += r.length
            omissions += 1
        else:
            runaway += r.length
            runaways += 1

    case_alike = case_normalised = reading = 0
    for c in chunks:
        if c.type == "substitute":
            for a, b in zip(ref[c.ref_start_idx:c.ref_end_idx], hyp[c.hyp_start_idx:c.hyp_end_idx]):
                if a.isalpha() and a != b and a.lower() == b.lower():
                    if a.lower() in alike:
                        case_alike += 1
                    else:
                        case_normalised += 1
                else:
                    reading += 1
        elif c.type in ("delete", "insert") and (c.type, c.ref_start_idx, c.ref_end_idx) not in behaviour_spans:
            reading += (c.ref_end_idx - c.ref_start_idx) if c.type == "delete" else (c.hyp_end_idx - c.hyp_start_idx)

    return RowScore(
        reference_chars=len(ref), errors=primary.substitutions + primary.deletions + primary.insertions,
        format_chars=len(normalise_text(" ".join(removed))), omitted_head=omitted[HEAD],
        omitted_interior=omitted[INTERIOR], omitted_tail=omitted[TAIL], runaway_inserted=runaway,
        case_alike=case_alike, case_normalised=case_normalised, reading_errors=reading + case_alike,
        omissions=omissions, runaways=runaways, format_touched=bool(removed),
    )
