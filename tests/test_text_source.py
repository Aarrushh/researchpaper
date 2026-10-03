# PURPOSE: Correctness tests for book loading and Project Gutenberg boilerplate stripping.
# WHAT GOES HERE: Synthetic checks that need no data, plus checks on the committed books.
# FILLED IN AT STEP: W1.4.
# DO NOT: Require network. Do not weaken the "no Gutenberg text left" checks to make a book pass.
"""Tests for mrr.text_source."""

from __future__ import annotations

import zipfile

import pytest

from conftest import REPO_ROOT
from mrr.text_source import load_book, strip_gutenberg_boilerplate

TEXTS = REPO_ROOT / "data" / "raw" / "texts"
BOOKS = {
    "pg84-images-3.epub": "Frankenstein",
    "pg105-images-3.epub": "Persuasion",
    "pg1661-images-3.epub": "Sherlock Holmes",
}

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
