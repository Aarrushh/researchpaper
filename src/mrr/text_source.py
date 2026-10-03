# PURPOSE: Loading and paginating source texts.
# WHAT GOES HERE: Book loading (plain text or EPUB), Project Gutenberg boilerplate stripping, pagination.
# FILLED IN AT STEP: W1.4.
# DO NOT: Modify files in data/raw/. Return cleaned text instead.
"""Source text handling.

Learn B3 (error metrics; text normalisation matters for CER) before implementing.
"""

from __future__ import annotations

import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote


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


def paginate(text: str, chars_per_page: int) -> list[str]:
    """Split text into pages of roughly chars_per_page characters on word boundaries.

    Args:
        text: cleaned book text.
        chars_per_page: target characters per page.

    Returns:
        List of page strings.
    """
    raise NotImplementedError("TODO[W1.4]: split on word boundaries near chars_per_page")
