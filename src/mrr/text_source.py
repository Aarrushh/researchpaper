# PURPOSE: Loading and paginating source texts.
# WHAT GOES HERE: Book loading (plain text or EPUB), boilerplate and front-matter stripping, normalisation (D13), pagination by fit and random text (build guide 1A.8).
# FILLED IN AT STEP: W1.4.
# DO NOT: Modify files in data/raw/. Return cleaned text instead.
"""Source text handling.

Learn B3 (error metrics; text normalisation matters for CER) before implementing.
"""

from __future__ import annotations

import posixpath
import re
import string
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import unquote

import numpy as np


def load_book(path: str | Path) -> str:
    """Read one book as UTF-8 text.

    Plain text (.txt) is read as is. EPUB (.epub) is converted to plain text by
    walking the spine in reading order. Paragraphs and headings become blocks
    separated by one blank line, and <br/> becomes a line break. Nothing is
    removed here: the Project Gutenberg header and footer stay in, so that
    strip_gutenberg_boilerplate handles both formats the same way.

    Args:
        path: file under data/raw/texts/.

    Returns:
        The full raw text, with "\n" line endings.
    """
    path = Path(path)
    if path.suffix.lower() == ".epub":
        text = _epub_to_text(path)
    else:
        text = path.read_text(encoding="utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n")


# Elements that start and end a block of text. Inline elements (i, span, a, ...)
# do not break the line.
_BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "td", "th",
    "header", "footer", "section", "article", "blockquote", "table", "ul", "ol",
    "hr", "pre", "dt", "dd",
}
# Elements whose content is never part of the book text.
_SKIP_TAGS = {"head", "script", "style", "svg"}


class _XhtmlText(HTMLParser):
    """Collect visible text from one XHTML file as a list of blocks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._buf: list[str] = []
        self._skip_depth = 0

    def _flush(self) -> None:
        lines = [" ".join(line.split()) for line in "".join(self._buf).split("\n")]
        block = "\n".join(line for line in lines if line)
        if block:
            self.blocks.append(block)
        self._buf = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif self._skip_depth:
            return
        elif tag == "br":
            self._buf.append("\n")
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        if tag in _SKIP_TAGS:
            return
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif not self._skip_depth and tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            # A newline in the HTML source is just whitespace; only <br/> breaks a line.
            self._buf.append(data.replace("\n", " "))

    def close(self) -> None:
        super().close()
        self._flush()


def _epub_to_text(path: Path) -> str:
    """Convert an EPUB to plain text, following the OPF spine order."""
    with zipfile.ZipFile(path) as z:
        container = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = container.find(".//{*}rootfile").get("full-path")
        opf = ET.fromstring(z.read(opf_path))
        base = posixpath.dirname(opf_path)
        manifest = {
            item.get("id"): item
            for item in opf.iterfind(".//{*}manifest/{*}item")
        }
        blocks: list[str] = []
        for ref in opf.iterfind(".//{*}spine/{*}itemref"):
            item = manifest[ref.get("idref")]
            if item.get("media-type") != "application/xhtml+xml":
                continue
            href = posixpath.normpath(posixpath.join(base, unquote(item.get("href"))))
            parser = _XhtmlText()
            parser.feed(z.read(href).decode("utf-8"))
            parser.close()
            blocks.extend(parser.blocks)
    return "\n\n".join(blocks) + "\n"


# Project Gutenberg marks the book body with lines like
# "*** START OF THE PROJECT GUTENBERG EBOOK PERSUASION ***". Older files say
# "THIS" instead of "THE".
_START_RE = re.compile(r"^\*{3} *START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.IGNORECASE | re.MULTILINE)
_END_RE = re.compile(r"^\*{3} *END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.IGNORECASE | re.MULTILINE)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the Project Gutenberg header and footer.

    Everything up to and including the START marker line, and everything from
    the END marker line onward, is removed. The footer holds the Project
    Gutenberg licence and trademark text, so it must go too.

    Args:
        text: raw book text.

    Returns:
        Text between the start and end markers, with surrounding blank lines removed.

    Raises:
        ValueError: if either marker is missing, appears more than once, or END comes before START.
    """
    starts = list(_START_RE.finditer(text))
    ends = list(_END_RE.finditer(text))
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(
            f"expected exactly one START and one END marker, found {len(starts)} and {len(ends)}"
        )
    start, end = starts[0], ends[0]
    if end.start() < start.end():
        raise ValueError("END marker comes before START marker")
    return text[start.end():end.start()].strip("\n") + "\n"


def skip_front_matter(text: str, marker: str, occurrence: int) -> str:
    """Drop everything up to and including one occurrence of a chapter heading.

    Every stripped book still opens with its title and a table of contents, so the body begins after a
    heading that appears twice (build guide 1A.8; the markers are listed in data/raw/texts/README.md).
    The marker's words match across any run of whitespace, because a heading can break over lines:
    pg1661's is "I." and "A SCANDAL IN BOHEMIA" on two lines. Matching is case-sensitive and whole-word,
    so "Letter 1" does not match "Letter 12".

    Args:
        text: book text after strip_gutenberg_boilerplate.
        marker: the heading, its words separated by single spaces.
        occurrence: which match to skip past, counting from 1.

    Returns:
        The text after that match, with leading whitespace removed.

    Raises:
        ValueError: if occurrence is below 1, marker is empty, or marker occurs fewer than occurrence times.
    """
    if occurrence < 1:
        raise ValueError(f"occurrence counts from 1, got {occurrence}")
    words = marker.split()
    if not words:
        raise ValueError("marker is empty")
    pattern = re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)")
    for count, match in enumerate(pattern.finditer(text), start=1):
        if count == occurrence:
            return text[match.end():].lstrip()
    raise ValueError(f"{marker!r} occurs fewer than {occurrence} times")


# D13: curly quotes become straight ones and the em dash a hyphen, so the reference is what the page shows.
_ASCII_PUNCTUATION = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "—": "-"})


def normalise_text(text: str) -> str:
    """Normalise text before it is rendered and before it is scored (D13).

    Applies Unicode NFC, maps ‘ and ’ to ', “ and ” to " and the em dash to -, collapses every run of
    whitespace to one space and strips the ends. Case is kept, and so are the other non-ASCII characters
    (£, é, œ, ½ and the like), which both fonts can draw. Reference pages are rendered from this text, and
    a transcription passes through the same function before CER is counted, so a reader that writes
    straight quotes or breaks lines differently is not charged for it.

    Args:
        text: any text: a book body, a random string or a reader's transcription.

    Returns:
        The normalised text.
    """
    return " ".join(unicodedata.normalize("NFC", text).translate(_ASCII_PUNCTUATION).split())


def paginate(text: str, fits: Callable[[str], bool], n_pages: int, start: int = 0) -> list[str]:
    """Cut text into consecutive pages, each holding as many words as fit.

    A page takes words while fits(candidate) is true (build guide 1A.8). A fixed character count cannot
    work, because the same characters take more lines in some layouts than in others. The largest word
    count is found by doubling and then halving, so a page of k words costs about 2 log2(k) calls to fits
    instead of k. The search relies on fits being monotone, which greedy line breaking guarantees: a word
    added at the end can only lengthen the last line or start a new one.

    Args:
        text: normalised text; its words are the runs between whitespace.
        fits: true when a candidate page fits, normally render.fits_page under every layout that will be
            compared on that page.
        n_pages: the most pages to return.
        start: index of the first word to use, so that a later run can begin after the last word used by an
            earlier one.

    Returns:
        Up to n_pages pages, fewer if the text runs out. Joined with single spaces, they are the words from
        start onward, in order.

    Raises:
        ValueError: if n_pages is below 1, start lies outside the text, or a word does not fit on a page by
            itself.
    """
    words = text.split()
    if n_pages < 1:
        raise ValueError(f"n_pages must be at least 1, got {n_pages}")
    if not 0 <= start <= len(words):
        raise ValueError(f"start must be a word index from 0 to {len(words)}, got {start}")
    pages: list[str] = []
    i = start
    while len(pages) < n_pages and i < len(words):
        if not fits(words[i]):
            raise ValueError(f"word {i} ({words[i][:40]!r}) does not fit on a page by itself")
        left = len(words) - i
        # Invariant: the first lo words fit; the first hi words do not, or hi runs past the end.
        lo, hi = 1, 2
        while hi <= left and fits(" ".join(words[i : i + hi])):
            lo, hi = hi, 2 * hi
        hi = min(hi, left + 1)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if fits(" ".join(words[i : i + mid])):
                lo = mid
            else:
                hi = mid
        pages.append(" ".join(words[i : i + lo]))
        i += lo
    return pages


_RANDOM_ALPHABET = string.digits + string.ascii_uppercase + string.ascii_lowercase  # 62 characters


def random_text(n_chars: int, seed: int, word_len: int = 5) -> str:
    """Make a uniform random alphanumeric string, broken into words.

    Every character is independent and uniform over the 62 in _RANDOM_ALPHABET, so context tells the
    language decoder nothing about any of them: an error the encoder makes cannot be repaired, and the gap
    in CER between natural and random pages measures the repair (build guide 1A.8). The draw order is
    fixed: n_chars indices from np.random.default_rng(seed).integers(0, 62, size=n_chars), joined into
    words of word_len separated by single spaces, then cut to exactly n_chars. The pinned strings in
    tests/test_text_source.py depend on that order and on the NumPy version in requirements.lock.txt.

    Args:
        n_chars: length of the result, spaces included.
        seed: seed for NumPy's default generator.
        word_len: characters per word.

    Returns:
        A string of exactly n_chars characters. It ends with a space when the cut falls just after a word.

    Raises:
        ValueError: if n_chars is negative or word_len is below 1.
    """
    if n_chars < 0:
        raise ValueError(f"n_chars must not be negative, got {n_chars}")
    if word_len < 1:
        raise ValueError(f"word_len must be at least 1, got {word_len}")
    indices = np.random.default_rng(seed).integers(0, len(_RANDOM_ALPHABET), size=n_chars)
    chars = "".join(_RANDOM_ALPHABET[k] for k in indices)
    return " ".join(chars[k : k + word_len] for k in range(0, n_chars, word_len))[:n_chars]
