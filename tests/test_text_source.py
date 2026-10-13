# PURPOSE: Correctness tests for book loading, boilerplate and front-matter stripping, normalisation, pagination and random text.
# WHAT GOES HERE: Synthetic checks that need no data, plus checks on the committed books.
# FILLED IN AT STEP: W1.4.
# DO NOT: Require network. Do not weaken the "no Gutenberg text left" checks to make a book pass.
"""Tests for mrr.text_source."""

from __future__ import annotations

import string
import zipfile
from collections import Counter

import pytest

from conftest import REPO_ROOT
from mrr.render import load_glyph
from mrr.text_source import (
    load_book,
    normalise_text,
    paginate,
    random_text,
    skip_front_matter,
    strip_gutenberg_boilerplate,
)

TEXTS = REPO_ROOT / "data" / "raw" / "texts"
BOOKS = {
    "pg84-images-3.epub": "Frankenstein",
    "pg105-images-3.epub": "Persuasion",
    "pg1661-images-3.epub": "Sherlock Holmes",
}
FONTS = REPO_ROOT / "data" / "external" / "fonts"
# The table in data/raw/texts/README.md: book, marker, occurrence, first words of the body.
FRONT_MATTER = [
    ("pg105-images-3.epub", "CHAPTER I.", 2, "Sir Walter Elliot, of Kellynch Hall, in Somersetshire, was a man who"),
    ("pg84-images-3.epub", "Letter 1", 2, "To Mrs. Saville, England. St. Petersburgh, Dec. 11th, 17—."),
    ("pg1661-images-3.epub", "I. A SCANDAL IN BOHEMIA", 1, "I. To Sherlock Holmes she is always the woman."),
]

SAMPLE = (
    "The Project Gutenberg eBook of Test\n"
    "Licence text.\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
    "\n"
    "Body line one.\n"
    "\n"
    "Body line two.\n"
    "\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
    "THE FULL PROJECT GUTENBERG LICENSE\n"
)


def test_strip_keeps_only_body():
    assert strip_gutenberg_boilerplate(SAMPLE) == "Body line one.\n\nBody line two.\n"


def test_strip_accepts_old_this_wording():
    text = SAMPLE.replace("OF THE PROJECT", "OF THIS PROJECT")
    assert strip_gutenberg_boilerplate(text) == "Body line one.\n\nBody line two.\n"


@pytest.mark.parametrize("marker", ["*** START", "*** END"])
def test_strip_missing_marker_raises(marker):
    text = "\n".join(line for line in SAMPLE.splitlines() if not line.startswith(marker))
    with pytest.raises(ValueError):
        strip_gutenberg_boilerplate(text)


def test_strip_end_before_start_raises():
    lines = SAMPLE.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("*** START"))
    end = next(i for i, line in enumerate(lines) if line.startswith("*** END"))
    lines[start], lines[end] = lines[end], lines[start]
    with pytest.raises(ValueError):
        strip_gutenberg_boilerplate("\n".join(lines))


def test_load_txt_normalises_line_endings(tmp_path):
    path = tmp_path / "book.txt"
    path.write_bytes("a\r\nb\rc\n".encode("utf-8"))
    assert load_book(path) == "a\nb\nc\n"


def _write_epub(path, chapters):
    """Write a minimal EPUB 3 whose spine lists chapters in the order given."""
    manifest = "".join(
        f'<item id="c{i}" href="c{i}.xhtml" media-type="application/xhtml+xml"/>'
        for i in range(len(chapters))
    )
    spine = "".join(f'<itemref idref="c{i}"/>' for i in reversed(range(len(chapters))))
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
            '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
            "</rootfiles></container>",
        )
        z.writestr(
            "OEBPS/content.opf",
            '<package xmlns="http://www.idpf.org/2007/opf">'
            f"<manifest>{manifest}</manifest><spine>{spine}</spine></package>",
        )
        for i, body in enumerate(chapters):
            z.writestr(
                f"OEBPS/c{i}.xhtml",
                '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>skip me</title></head>'
                f"<body>{body}</body></html>",
            )


def test_load_epub_follows_spine_and_blocks(tmp_path):
    path = tmp_path / "book.epub"
    # The spine lists c1 before c0, so c1's text must come first.
    _write_epub(
        path,
        [
            "<p>Second   file,\n wrapped <i>inline</i>.</p>",
            "<h2>First</h2><p>Line one<br/>line two</p>",
        ],
    )
    assert load_book(path) == "First\n\nLine one\nline two\n\nSecond file, wrapped inline.\n"


@pytest.mark.parametrize("name", sorted(BOOKS))
def test_committed_book_strips_clean(name):
    path = TEXTS / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    raw = load_book(path)
    assert "START OF THE PROJECT GUTENBERG EBOOK" in raw
    body = strip_gutenberg_boilerplate(raw)
    assert "gutenberg" not in body.lower()
    assert len(body) > 0.9 * len(raw)


def _stripped(name):
    path = TEXTS / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return strip_gutenberg_boilerplate(load_book(path))


@pytest.mark.parametrize(("name", "marker", "occurrence", "begins"), FRONT_MATTER)
def test_skip_front_matter_on_committed_books(name, marker, occurrence, begins):
    body = skip_front_matter(_stripped(name), marker, occurrence)
    assert " ".join(body.split()).startswith(begins)


def test_skip_front_matter_matches_across_line_breaks():
    # pg1661's heading is "I." and "A SCANDAL IN BOHEMIA" on two lines.
    text = "Contents\n\nI.\nA Scandal\n\nI.\nA SCANDAL\n\nI.\n\nBody starts here."
    assert skip_front_matter(text, "I. A SCANDAL", 1) == "I.\n\nBody starts here."


def test_skip_front_matter_matches_whole_words_only():
    assert skip_front_matter("Letter 12 is not it. Letter 1\nBody.", "Letter 1", 1) == "Body."


def test_skip_front_matter_rejects_a_missing_occurrence():
    with pytest.raises(ValueError):
        skip_front_matter("CHAPTER I. Only once.", "CHAPTER I.", 2)
    with pytest.raises(ValueError):
        skip_front_matter("CHAPTER I. Only once.", "CHAPTER I.", 0)


def test_normalise_text_maps_quotes_and_dashes_and_collapses_whitespace():
    raw = "  ‘It’s’ “fine”—\n\n she\tpaid £5 ½ for café œuvre æ  "
    assert normalise_text(raw) == "'It's' \"fine\"- she paid £5 ½ for café œuvre æ"


def test_normalise_text_applies_nfc():
    assert normalise_text("café") == "café"


def test_random_text_pinned_strings():
    # Measured with NumPy 2.4.6, the version in requirements.lock.txt. NumPy does not promise the same
    # stream across versions; if this fails after an upgrade, the pages are not the preregistered ones.
    assert random_text(23, 0) == "qdVGJ 241Ao euVby jdXYv"
    assert random_text(23, 1) == "TVkw2 8pwFJ rQGpF PdY51"


@pytest.mark.parametrize("n_chars", [0, 1, 5, 6, 7, 1000])
def test_random_text_length_alphabet_and_words(n_chars):
    text = random_text(n_chars, 7)
    assert len(text) == n_chars
    assert set(text) <= set(string.digits + string.ascii_letters + " ")
    words = text.split(" ")
    assert all(len(word) == 5 for word in words[:-1]) and len(words[-1]) <= 5


def test_random_text_is_uniform_and_deterministic():
    counts = Counter(random_text(74_400, 0).replace(" ", ""))  # 62,000 characters
    assert sorted(counts) == sorted(string.digits + string.ascii_letters)
    assert 850 < min(counts.values()) and max(counts.values()) < 1150
    assert random_text(500, 3) == random_text(500, 3) != random_text(500, 4)


WORDS = " ".join(f"w{i}" + "x" * (i % 7) for i in range(300))


def _fits_40(page):
    return len(page) <= 40


def test_paginate_pages_join_back_and_each_fits():
    pages = paginate(WORDS, _fits_40, n_pages=1000)
    assert " ".join(pages) == WORDS
    assert all(_fits_40(page) for page in pages)


def test_paginate_fills_each_page_greedily():
    pages = paginate(WORDS, _fits_40, n_pages=1000)
    for page, following in zip(pages, pages[1:]):
        assert not _fits_40(page + " " + following.split()[0])


def test_paginate_stops_after_n_pages_and_resumes_at_a_word():
    words = WORDS.split()
    first = paginate(WORDS, _fits_40, n_pages=3)
    used = sum(len(page.split()) for page in first)
    later = paginate(WORDS, _fits_40, n_pages=2, start=used)
    assert len(first) == 3 and len(later) == 2
    assert later[0].split()[0] == words[used]
    assert first + later == paginate(WORDS, _fits_40, n_pages=5)


def test_paginate_searches_instead_of_adding_one_word_at_a_time():
    calls = []

    def fits(page):
        calls.append(page)
        return len(page.split()) <= 100

    text = " ".join(["word"] * 10_000)
    pages = paginate(text, fits, n_pages=1000)
    assert len(pages) == 100 and all(len(page.split()) == 100 for page in pages)
    assert len(calls) < 2_000  # adding one word at a time would call fits more than 10,000 times


def test_paginate_rejects_a_word_that_fits_no_page():
    with pytest.raises(ValueError):
        paginate("short " + "x" * 50 + " tail", _fits_40, n_pages=5)


def test_books_are_drawable():      # Review Focus 2
    chars = set()
    for name in BOOKS:
        chars |= set(normalise_text(_stripped(name)))
    for font in (FONTS / "source-sans-3" / "SourceSans3-Regular.ttf",
                 FONTS / "source-serif-4d" / "SourceSerif4Display-Regular.ttf"):
        if not font.exists():
            pytest.skip(f"{font.name} not present")
        missing = []
        for char in sorted(chars):
            try:
                load_glyph(str(font), 16, char)
            except ValueError:
                missing.append(f"U+{ord(char):04X}")
        assert not missing, (font.name, missing)
