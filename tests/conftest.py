# PURPOSE: Shared pytest setup.
# WHAT GOES HERE: Path setup so `mrr` imports without installation; shared fixtures, including each reader's native page and its processor without the model for the model-marked tests (build guide 1B.6).
# FILLED IN AT STEP: Scaffolded at setup; the reader fixtures at W1.8.
# DO NOT: Load models or GPUs in fixtures used by non-todo tests.
"""Pytest configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

READERS = {
    reader["name"]: reader
    for reader in yaml.safe_load((REPO_ROOT / "configs" / "readers.yaml").read_text(encoding="utf-8"))["readers"]
}
PERSUASION = REPO_ROOT / "data" / "raw" / "texts" / "pg105-images-3.epub"


@pytest.fixture(scope="session")
def native_page():
    """Return a function: reader name -> (reference text, page), Persuasion's first page at the native size.

    The page is rendered with configs/render_defaults.yaml, as the sweep renders a page at phase 0 (D8).
    """
    from mrr.grid import resolve_render_cfg
    from mrr.render import fits_page, render_page
    from mrr.text_source import load_book, normalise_text, paginate, skip_front_matter, strip_gutenberg_boilerplate

    defaults = yaml.safe_load((REPO_ROOT / "configs" / "render_defaults.yaml").read_text(encoding="utf-8"))
    font = REPO_ROOT / defaults["font_file"]
    if not (PERSUASION.exists() and font.exists()):
        pytest.skip("book or font not present")
    text = normalise_text(skip_front_matter(strip_gutenberg_boilerplate(load_book(PERSUASION)), "CHAPTER I.", 2))
    pages = {}

    def page(name):
        if name not in pages:
            cfg = resolve_render_cfg(defaults, READERS[name], font_file=str(font))  # a str for freetype-py
            reference = paginate(text, lambda candidate: fits_page(candidate, cfg), n_pages=1)[0]
            pages[name] = (reference, render_page(reference, cfg))
        return pages[name]

    return page


@pytest.fixture(scope="session")
def cpu_reader():
    """Return a function: reader name -> mrr.readers.Reader with its pinned processor and no model.

    Enough for everything before the model: the processor inputs, the image tokens and the chat template.
    """
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from mrr.readers import Reader

    loaded = {}

    def reader(name):
        if name not in loaded:
            cfg = READERS[name]
            processor = transformers.AutoProcessor.from_pretrained(cfg["hf_id"], revision=cfg["revision"])
            loaded[name] = Reader(
                name=name, model=None, processor=processor, cfg=cfg, image_token_id=processor.image_token_id
            )
        return loaded[name]

    return reader
