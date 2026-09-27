# PURPOSE: Loading and paginating source texts.
# WHAT GOES HERE: Book loading, Project Gutenberg boilerplate stripping, pagination.
# FILLED IN AT STEP: W1.4.
# DO NOT: Modify files in data/raw/. Return cleaned text instead.
"""Source text handling.

Learn B3 (error metrics; text normalisation matters for CER) before implementing.
"""

from __future__ import annotations

from pathlib import Path


def load_book(path: str | Path) -> str:
    """Read one book as UTF-8 text.

    Args:
        path: file under data/raw/texts/.

    Returns:
        The full raw text.
    """
    raise NotImplementedError("TODO[W1.4]: read the file as UTF-8 and normalise line endings")


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the Project Gutenberg header and footer.

    Args:
        text: raw book text.

    Returns:
        Text between the start and end markers.
    """
    raise NotImplementedError("TODO[W1.4]: find the START and END markers and cut outside them")


def paginate(text: str, chars_per_page: int) -> list[str]:
    """Split text into pages of roughly chars_per_page characters on word boundaries.

    Args:
        text: cleaned book text.
        chars_per_page: target characters per page.

    Returns:
        List of page strings.
    """
    raise NotImplementedError("TODO[W1.4]: split on word boundaries near chars_per_page")
