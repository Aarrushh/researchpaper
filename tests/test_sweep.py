# PURPOSE: Tests for the pagination plan of a sweep (build guide section 2.4; the block Z pages of section 3.5), the condition hash (2.5) and the runner (2.8).
# WHAT GOES HERE: CPU checks on the committed books and font: phase allowances, shared pages across layouts, text order and ids, block Z passages, placeholders, no reuse; the hash test vector; the runner on a fake backend: rows, resume, cut-off line, shuffle, deadline, refusals; Study 2's text_types and results_schema.
# FILLED IN AT STEP: W2.2 (pages), W2.3 (hash, runner).
# DO NOT: Load a model. Paginate by character count. Weaken the capacity numbers of section 2.4 to make a test pass.
"""Tests for mrr.sweep: plan_pages and its helpers, condition_hash and run_sweep."""

from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import logging
import math
import re
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from conftest import REPO_ROOT
from mrr import grid
from mrr import sweep as sweep_module
from mrr.grid import resolve_render_cfg
from mrr.manifest import RESULTS_COLUMNS, STUDY2_SCORE_COLUMNS
from mrr.readers import Transcription
from mrr.render import fits_page
from mrr.score import cer, char_counts, char_counts_letters_only, spaces_inserted, wer
from mrr.sweep import (
    RANDOM_STREAM_CHARS,
    Condition,
    Page,
    condition_hash,
    expand_conditions,
    load_results,
    pagination_key,
    phase_allowances,
    plan_pages,
    run_sweep,
)
from mrr.text_source import BOOKS, TEXTS_DIR, book_text, random_text

READERS = {
    reader["name"]: reader
    for reader in yaml.safe_load((REPO_ROOT / "configs" / "readers.yaml").read_text(encoding="utf-8"))["readers"]
}
DEFAULTS = yaml.safe_load((REPO_ROOT / "configs" / "render_defaults.yaml").read_text(encoding="utf-8"))
PILOT = yaml.safe_load((REPO_ROOT / "configs" / "sweeps" / "pilot.yaml").read_text(encoding="utf-8"))
LADDER = yaml.safe_load((REPO_ROOT / "configs" / "sweeps" / "ladder.yaml").read_text(encoding="utf-8"))
FONT = REPO_ROOT / "data" / "external" / "fonts" / "source-sans-3" / "SourceSans3-Regular.ttf"
HAVE_DATA = FONT.exists() and all((REPO_ROOT / TEXTS_DIR / book.file).exists() for book in BOOKS.values())
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="font or books not present (W1.4)")

# Section 2.4, step 1: ceil(largest phase_x) and ceil(largest phase_y) over the shared conditions of the pilot.
ALLOWANCES = {"qwen": (105, 26), "smolvlm": (40, 40), "smolvlm500m": (60, 60)}
# Section 2.4, the "Shared page" column at 16 px: characters of Persuasion's first page, spaces included.
SHARED_CHARS_16 = {"qwen": 379, "smolvlm": 298, "smolvlm500m": 419}
ROTATION = ["pg105", "pg84", "pg1661"]


def _pilot(size=16):
    """The pilot with its "from the ladder" sizes replaced by one size for every reader."""
    sweep = copy.deepcopy(PILOT)
    sweep["font_size_px"] = {name: size for name in sweep["readers"]}
    return sweep


def _plan(sweep, reader, **kwargs):
    return plan_pages(sweep, READERS[reader], DEFAULTS, root=REPO_ROOT, **kwargs)


def _words(page):
    return len(page.text.split())


@pytest.fixture(scope="module")
def pilot_plans():
    if not HAVE_DATA:
        pytest.skip("font or books not present (W1.4)")
    return {reader: _plan(_pilot(), reader) for reader in sorted(READERS)}


@pytest.fixture(scope="module")
def ladder_plan():
    if not HAVE_DATA:
        pytest.skip("font or books not present (W1.4)")
    return _plan(LADDER, "qwen")


# Allowances (section 2.4, step 1).


@pytest.mark.parametrize("reader", sorted(ALLOWANCES))
def test_phase_allowances_are_the_ceiling_of_the_largest_phase_on_each_axis(reader):
    assert phase_allowances(expand_conditions(_pilot(), READERS[reader])) == ALLOWANCES[reader]


def test_phase_allowances_count_shared_conditions_only():
    ladder = expand_conditions(LADDER, READERS["qwen"])
    assert phase_allowances(ladder) == (0, 0)
    assert phase_allowances(ladder + expand_conditions(_pilot(), READERS["qwen"])) == ALLOWANCES["qwen"]


# Shared pages (section 2.4, steps 3 to 5).


@needs_data
@pytest.mark.parametrize("reader", sorted(SHARED_CHARS_16))
def test_the_first_shared_page_at_16_px_holds_the_characters_of_the_table(pilot_plans, reader):
    plan = pilot_plans[reader]
    assert set(plan) == {"shared"}
    page = plan["shared"][0]
    assert (page.page_id, page.book_id, page.text_type, page.start_word) == ("pg105_p0000", "pg105", "natural", 0)
    assert page.text.startswith("Sir Walter Elliot, of Kellynch Hall, in Somersetshire")
    # Within a word or two of the table; a difference above 5% would mean a layout rule differs (section 2.4).
    assert len(page.text) == SHARED_CHARS_16[reader]


@needs_data
def test_shared_pages_fit_every_shared_layout_and_take_every_word_that_fits(pilot_plans):
    x, y = ALLOWANCES["qwen"]
    common = dict(font_file=str(FONT), font_size_px=16, max_phase_x_px=x, max_phase_y_px=y)
    layouts = [  # conventional unaligned (C, A), conventional aligned (V), patch_aware unaligned (H, S)
        resolve_render_cfg(DEFAULTS, READERS["qwen"], line_pitch_px=20, **common),
        resolve_render_cfg(DEFAULTS, READERS["qwen"], line_pitch_px=28, **common),
        resolve_render_cfg(DEFAULTS, READERS["qwen"], render_mode="patch_aware", line_pitch_px=20, **common),
    ]
    sources = {
        "pg105": book_text("pg105", REPO_ROOT).split(),
        "rnd20261013": random_text(RANDOM_STREAM_CHARS, 20261013).split(),
    }
    checked = 0
    for page in pilot_plans["qwen"]["shared"]:
        if page.book_id not in sources:
            continue
        assert all(fits_page(page.text, cfg) for cfg in layouts)
        words = sources[page.book_id]
        longer = " ".join(words[page.start_word : page.start_word + _words(page) + 1])
        assert not all(fits_page(longer, cfg) for cfg in layouts)
        checked += 1
    assert checked == 14  # four Persuasion pages (natural pages 0, 3, 6, 9) and ten random pages


@needs_data
def test_shared_pages_interleave_natural_and_random_with_the_books_in_rotation(pilot_plans):
    pages = pilot_plans["qwen"]["shared"]
    assert len(pages) == 20 and all(isinstance(page, Page) for page in pages)
    assert [page.text_type for page in pages] == ["natural", "random62"] * 10
    natural, random = pages[0::2], pages[1::2]
    assert [page.book_id for page in natural] == (ROTATION * 4)[:10]
    assert [page.page_id for page in natural[:4]] == ["pg105_p0000", "pg84_p0001", "pg1661_p0002", "pg105_p0003"]
    assert {page.book_id for page in random} == {"rnd20261013"}
    assert [page.page_id for page in random[:2]] == ["rnd20261013_p0000", "rnd20261013_p0001"]
    alphabet = set(string.digits + string.ascii_letters + " ")
    assert all(set(page.text) <= alphabet for page in random)
    assert len({page.page_id for page in pages}) == 20


@needs_data
def test_consecutive_pages_of_one_source_are_contiguous_from_its_start(pilot_plans):
    pages = pilot_plans["smolvlm"]["shared"]
    texts = {book: book_text(book, REPO_ROOT) for book in ROTATION}
    texts["rnd20261013"] = random_text(RANDOM_STREAM_CHARS, 20261013)
    for source, text in texts.items():
        words = text.split()
        mine = [page for page in pages if page.book_id == source]
        assert mine
        cursor = 0
        for page in mine:
            assert page.start_word == cursor
            assert page.text == " ".join(words[cursor : cursor + _words(page)])
            cursor += _words(page)


@needs_data
def test_per_reader_page_counts_are_read_for_this_reader_and_may_differ_by_text_type():
    sweep = _pilot()
    sweep["pages"]["natural"]["per_text_type"] = {"qwen": 1, "smolvlm": 2, "smolvlm500m": 1}
    sweep["pages"]["random62"]["per_text_type"] = {"qwen": 1, "smolvlm": 1, "smolvlm500m": 1}
    plan = _plan(sweep, "smolvlm")
    assert [page.text_type for page in plan["shared"]] == ["natural", "random62", "natural"]
    del sweep["pages"]["natural"]["per_text_type"]["smolvlm"]
    with pytest.raises(ValueError, match="pages.natural.per_text_type"):
        _plan(sweep, "smolvlm")


# Placeholders (section 2.4, step 7) and malformed configs.


@pytest.mark.parametrize(
    ("kind", "value"),
    [("natural", "TODO[W3.2]"), ("random62", "<N_Q>"), ("natural", 2.5), ("random62", -1), ("natural", True)],
)
def test_a_page_count_that_is_not_a_whole_number_is_refused_naming_the_key(kind, value):
    sweep = _pilot()
    sweep["pages"][kind]["per_text_type"] = value
    with pytest.raises(ValueError, match=f"pages.{kind}.per_text_type"):
        _plan(sweep, "qwen")


def test_a_seed_that_is_not_a_number_is_refused_naming_the_key():
    sweep = _pilot()
    sweep["seed"] = "<new seed>"
    with pytest.raises(ValueError, match="seed"):
        _plan(sweep, "qwen")


def test_an_unknown_book_is_refused_naming_it():
    sweep = _pilot()
    sweep["pages"]["natural"]["books"] = ["pg105", "pg9999"]
    with pytest.raises(ValueError, match="pg9999"):
        _plan(sweep, "qwen")


def test_two_per_condition_layouts_whose_pages_would_share_an_id_are_refused():
    sweep = copy.deepcopy(LADDER)
    sweep["blocks"]["Z"]["line_pitch_mode"] = ["unaligned", "aligned"]
    with pytest.raises(ValueError, match="page id"):
        _plan(sweep, "qwen")


# Block Z (section 3.5, "Block Z pages").


@needs_data
def test_block_z_pages_are_planned_per_layout_from_the_same_passages(ladder_plan):
    conditions = expand_conditions(LADDER, READERS["qwen"])
    assert set(ladder_plan) == {pagination_key(c) for c in conditions} and len(ladder_plan) == 12
    assert "shared" not in ladder_plan
    for pages in ladder_plan.values():
        assert [page.text_type for page in pages] == ["natural", "random62"] * 3
    conv10, pa10, conv16 = (ladder_plan[k] for k in ("z:conventional:10", "z:patch_aware:10", "z:conventional:16"))
    assert [page.page_id for page in conv10] == [
        "pg105_z10_conv_p0000",
        "rnd20261012_z10_conv_p0000",
        "pg84_z10_conv_p0001",
        "rnd20261012_z10_conv_p0001",
        "pg1661_z10_conv_p0002",
        "rnd20261012_z10_conv_p0002",
    ]
    assert pa10[0].page_id == "pg105_z10_pa_p0000" and conv16[1].page_id == "rnd20261012_z16_conv_p0000"
    # In the ladder the passages start at the skip points (step 4).
    assert conv10[0].start_word == 0 and conv10[0].text.startswith("Sir Walter Elliot")
    assert conv10[1].start_word == 0
    # Page i of every Z condition begins at the same word (step 2).
    for i in range(6):
        starts = {(pages[i].book_id, pages[i].start_word, pages[i].text.split()[0]) for pages in ladder_plan.values()}
        assert len(starts) == 1, starts
    # The 10 px conventional page is the fullest of the twelve, and it spaces the passages of one source (step 3).
    longest = {key: _words(pages[1]) for key, pages in ladder_plan.items()}
    assert max(longest, key=longest.get) == "z:conventional:10"
    assert conv10[3].start_word == longest["z:conventional:10"]
    assert conv10[5].start_word == conv10[3].start_word + _words(conv10[3])
    # Zero allowances: Qwen's 16 px conventional Z page holds more than the 840 characters the same layout holds
    # under the 105/26 allowances (the table of section 2.4).
    assert len(conv16[0].text) > 840


# No reuse (section 2.4, step 6; section 3.5).


@needs_data
def test_a_later_sweep_starts_each_book_after_the_last_word_any_earlier_reader_used(tmp_path):
    earlier = _pilot()
    earlier["readers"] = ["qwen", "smolvlm"]
    for kind in ("natural", "random62"):
        earlier["pages"][kind]["per_text_type"] = 1
    path = tmp_path / "earlier.yaml"
    path.write_text(yaml.safe_dump(earlier), encoding="utf-8")
    ends = {}
    for reader in earlier["readers"]:
        for page in _plan(earlier, reader)["shared"]:
            if page.text_type == "natural":
                ends[reader] = page.start_word + _words(page)
    assert ends["qwen"] > ends["smolvlm"]  # 379 against 298 characters at 16 px
    later = _pilot()
    later["seed"] = 20261016
    later["pages"]["natural"]["start_after"] = [str(path)]
    for kind in ("natural", "random62"):
        later["pages"][kind]["per_text_type"] = 1
    plan = _plan(later, "smolvlm", readers=READERS)
    first, random = plan["shared"]
    assert (first.book_id, first.page_id, first.start_word) == ("pg105", "pg105_p0000", ends["qwen"])
    assert (random.book_id, random.start_word) == ("rnd20261016", 0)
    with pytest.raises(ValueError, match="readers"):
        _plan(later, "smolvlm")
    later["seed"] = earlier["seed"]
    with pytest.raises(ValueError, match="seed"):
        _plan(later, "smolvlm", readers=READERS)


# Section 2.5: the test vector of the condition hash and its hash [Measured 10 Oct; recomputed 14 Oct].
VECTOR = {
    "aa_mode": "coverage_srgb",
    "block": "C",
    "font_file": "SourceSans3-Regular.ttf",
    "font_size_px": 16.0,
    "line_pitch_mode": "unaligned",
    "line_pitch_px": 20.0,
    "patch_aware_slot_px": None,
    "phase_x_px": 7.0,
    "phase_y_px": 0.0,
    "render_mode": "conventional",
}
VECTOR_HASH = "9c811d1777d4"


def test_condition_hash_matches_the_test_vector_of_section_2_5():
    assert condition_hash(VECTOR) == VECTOR_HASH
    # Step 2 of the rule exists because json writes 7 and 7.0 differently: the uncast string hashes to something else.
    uncast = json.dumps({**VECTOR, "phase_x_px": 7}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha1(uncast).hexdigest()[:12] == "bc2ca5dcb62f"
    assert condition_hash({**VECTOR, "phase_x_px": 7}) == VECTOR_HASH  # the cast gives the same hash
    assert condition_hash({**VECTOR, "line_pitch_px": 20}) == VECTOR_HASH


def test_condition_hash_ignores_key_order_and_changes_with_every_value():
    assert condition_hash(dict(reversed(list(VECTOR.items())))) == VECTOR_HASH
    changed = {
        "aa_mode": "binary", "block": "A", "font_file": "SourceSerif4-Regular.ttf", "font_size_px": 14.0,
        "line_pitch_mode": "aligned", "line_pitch_px": 28.0, "patch_aware_slot_px": 14, "phase_x_px": 8.0,
        "phase_y_px": 1.0, "render_mode": "patch_aware",
    }
    hashes = {key: condition_hash({**VECTOR, key: value}) for key, value in changed.items()}
    assert VECTOR_HASH not in hashes.values()
    assert len(set(hashes.values())) == len(hashes)
    assert all(len(h) == 12 and set(h) <= set("0123456789abcdef") for h in hashes.values())


def test_condition_hash_takes_a_condition_and_keeps_only_the_font_file_name():
    cond = Condition(
        block="C", render_mode="conventional", line_pitch_mode="unaligned", line_pitch_px=20,
        font_file="SourceSans3-Regular.ttf", font_size_px=16, aa_mode="coverage_srgb",
        phase_x_px=7, phase_y_px=0, patch_aware_slot_px=None, paginate="shared",
    )
    assert condition_hash(cond) == VECTOR_HASH  # paginate is not one of the ten keys
    for path in ("data/external/fonts/source-sans-3/SourceSans3-Regular.ttf", r"data\external\fonts\SourceSans3-Regular.ttf"):
        assert condition_hash({**VECTOR, "font_file": path}) == VECTOR_HASH


def test_condition_hash_refuses_a_missing_key():
    incomplete = {key: value for key, value in VECTOR.items() if key != "patch_aware_slot_px"}
    with pytest.raises(ValueError, match="patch_aware_slot_px"):
        condition_hash(incomplete)


def test_every_pilot_condition_has_its_own_hash_and_h0_differs_from_s0():
    for name in PILOT["readers"]:
        conds = expand_conditions(_pilot(16), READERS[name])
        hashes = [condition_hash(c) for c in conds]
        assert len(set(hashes)) == len(hashes)
        h0 = next(c for c in conds if c.block == "H" and c.phase_x_px == 0 and c.phase_y_px == 0)
        s0 = next(c for c in conds if c.block == "S" and c.phase_x_px == 0 and c.phase_y_px == 0)
        assert condition_hash(h0) != condition_hash(s0)


# The runner (section 2.8). A fake backend stands in for readers.py, so the whole loop runs on the CPU. The sweep is
# two blocks on one natural and one random page: H at the whole-pixel phases 0, 1 and 2 and S at 0 and 0.5, both
# patch_aware and unaligned, so that (H, 0) and (S, 0) are the determinism pair of section 2.8.

RUNNER_SWEEP = {
    "readers": ["qwen"],
    "seed": 20261014,
    "pages": {"natural": {"books": ["pg105"], "per_text_type": 1}, "random62": {"per_text_type": 1}},
    "repeats": 1,
    "font_file": "data/external/fonts/source-sans-3/SourceSans3-Regular.ttf",
    "font_size_px": 16,
    "aa_mode": "coverage_srgb",
    "blocks": {
        "H": {"render_mode": "patch_aware", "line_pitch_mode": "unaligned", "axis": "x", "phases_px": [0.0, 1.0, 2.0]},
        "S": {"render_mode": "patch_aware", "line_pitch_mode": "unaligned", "axis": "x", "phases_px": [0.0, 0.5]},
    },
}
RUNNER_ROWS = 10  # (3 + 2) conditions x 2 pages x 1 repeat
QWEN_PERIODS = (14, 28, 112)  # patch, token cell, attention window
HARMONICS = (1, 2, 3)
FIXED_TEXT = "the same fixed text on every call"
RUNNER_PAGES = ("pg105_p0000", "rnd20261014_p0000")


class FakeBackend:
    """The five functions of readers.py without a model: a fixed text, fixed counts and the grid's token count."""

    def __init__(self, fail_on_call=None, texts=(FIXED_TEXT,), extra_tokens=0):
        self.fail_on_call, self.texts, self.extra_tokens = fail_on_call, tuple(texts), extra_tokens
        self.calls, self.limits, self.loaded, self.unloaded = 0, [], [], []

    def load_reader(self, cfg):
        reader = SimpleNamespace(name=cfg["name"], cfg=cfg, model="a model")
        self.loaded.append(reader)
        return reader

    def unload_reader(self, reader):
        reader.model = None
        self.unloaded.append(reader)

    def count_vision_tokens(self, reader, image):
        return grid.token_count(reader.cfg, image.shape[0], image.shape[1]) + self.extra_tokens

    def count_text_tokens(self, reader, text):
        return len(text.split())  # one token per word stands in for the tokenizer

    def transcribe_full(self, reader, image, prompt, max_new_tokens=None):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("fake driver reset")
        assert reader.model is not None and image.shape in {(448, 448, 3), (384, 384, 3)}
        self.limits.append(max_new_tokens)
        text = self.texts[(self.calls - 1) % len(self.texts)]
        return Transcription(text=text, new_tokens=7, hit_limit=False, latency_s=0.01)


def _write_sweep(folder, **changes):
    cfg = copy.deepcopy(RUNNER_SWEEP)
    cfg.update(changes)
    path = folder / "two_blocks.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _run(path, backend, output_root, **kwargs):
    kwargs.setdefault("require_clean", False)
    return run_sweep(path, READERS, DEFAULTS, backend, output_root=output_root, root=REPO_ROOT, **kwargs)


def _csv_rows(path):
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


def _keys(rows):
    return [(r["reader"], r["page_id"], r["condition_hash"], r["repeat"]) for r in rows]


def _progress_lines(run_dir):
    return (run_dir / "progress.log").read_text(encoding="utf-8").splitlines()


def _clock(start, later):
    """A fake _now: ``start`` on its first call, ``later`` on every call after it."""
    calls = []

    def now():
        calls.append(None)
        return start if len(calls) == 1 else later

    return now


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """One complete run of RUNNER_SWEEP: (run folder, backend)."""
    if not HAVE_DATA:
        pytest.skip("font or books not present (W1.4)")
    folder = tmp_path_factory.mktemp("runner")
    backend = FakeBackend()
    dirs = _run(_write_sweep(folder), backend, folder / "runs with space")
    assert len(dirs) == 1
    return dirs[0], backend


@needs_data
def test_a_full_run_writes_every_planned_row_once_under_the_schema_header(full_run):
    run_dir, backend = full_run
    header, rows = _csv_rows(run_dir / "results.csv")
    assert header == list(RESULTS_COLUMNS)
    assert len(rows) == RUNNER_ROWS == len(set(_keys(rows)))
    assert {r["block"] for r in rows} == {"H", "S"} and {r["reader"] for r in rows} == {"qwen"}
    assert {r["page_id"] for r in rows} == set(RUNNER_PAGES)
    assert {r["text_type"] for r in rows} == {"natural", "random62"}
    assert {r["book_id"] for r in rows} == {"pg105", "rnd20261014"}
    for row in rows:
        assert (row["image_width_px"], row["image_height_px"], row["vision_tokens"]) == ("448", "448", "256")
        assert row["reader_revision"] == READERS["qwen"]["revision"] and row["run_id"] == run_dir.name
        assert (row["font_file"], row["font_size_px"], row["line_pitch_px"]) == ("SourceSans3-Regular.ttf", "16", "20")
        assert (row["line_pitch_mode"], row["aa_mode"], row["render_mode"]) == ("unaligned", "coverage_srgb", "patch_aware")
        assert (row["output_tokens"], row["hit_limit"], row["latency_s"], row["repeat"]) == ("7", "False", "0.01", "0")
        assert row["timestamp_utc"].endswith("+00:00")  # Review Focus 4
        assert row["git_commit"]
        reference = (run_dir / "references" / f"{row['page_id']}.txt").read_text(encoding="utf-8")
        assert int(row["reference_chars"]) == len(reference)
        assert row["transcription_file"] == f"transcriptions/{row['page_id']}__{row['condition_hash']}.txt"
        assert (run_dir / row["transcription_file"]).read_text(encoding="utf-8") == FIXED_TEXT
        counts = char_counts(reference, FIXED_TEXT)
        assert (int(row["hits"]), int(row["substitutions"]), int(row["deletions"]), int(row["insertions"])) == (
            counts.hits, counts.substitutions, counts.deletions, counts.insertions,
        )
        assert float(row["cer"]) == pytest.approx(cer(reference, FIXED_TEXT))
        assert float(row["wer"]) == pytest.approx(wer(reference, FIXED_TEXT))
    assert run_dir.parent.name == "runs with space"
    assert re.fullmatch(r"\d{8}-\d{4}_two_blocks_qwen_([0-9a-f]{7}|nogit)", run_dir.name)
    assert backend.calls == RUNNER_ROWS and len(backend.loaded) == 1 == len(backend.unloaded)


@needs_data
def test_each_page_gets_max_new_tokens_of_one_and_a_half_times_its_reference_tokens(full_run):
    run_dir, backend = full_run
    _, rows = _csv_rows(run_dir / "results.csv")
    expected = []
    for row in rows:  # rows are appended in call order
        reference = (run_dir / "references" / f"{row['page_id']}.txt").read_text(encoding="utf-8")
        expected.append(math.ceil(1.5 * len(reference.split())))
    assert backend.limits == expected
    assert len(set(expected)) == 2  # a natural and a random page differ in length


@needs_data
def test_the_first_pages_images_are_saved_by_default_and_every_reference_is_written(full_run):
    run_dir, _ = full_run
    _, rows = _csv_rows(run_dir / "results.csv")
    first = rows[0]["page_id"]
    images = sorted((run_dir / "images").glob("*.png"))
    assert len(images) == 5 and all(p.name.startswith(first + "__") for p in images)
    with Image.open(images[0]) as im:
        assert im.size == (448, 448) and im.mode == "RGB"
    assert sorted(p.name for p in (run_dir / "references").iterdir()) == [f"{p}.txt" for p in RUNNER_PAGES]
    natural = (run_dir / "references" / "pg105_p0000.txt").read_text(encoding="utf-8")
    random = (run_dir / "references" / "rnd20261014_p0000.txt").read_text(encoding="utf-8")
    assert natural.startswith("Sir Walter Elliot") and random_text(RANDOM_STREAM_CHARS, 20261014).startswith(random)


@needs_data
def test_coherence_rows_cover_every_block_period_and_harmonic_at_phase_0(full_run):
    run_dir, _ = full_run
    header, rows = _csv_rows(run_dir / "coherence.csv")
    assert header == ["page_id", "block", "render_mode", "line_pitch_mode", "axis", "period_px", "harmonic", "R"]
    assert len(rows) == 2 * 2 * len(QWEN_PERIODS) * len(HARMONICS)
    assert {(r["page_id"], r["block"], int(r["period_px"]), int(r["harmonic"])) for r in rows} == set(
        itertools.product(RUNNER_PAGES, ("H", "S"), QWEN_PERIODS, HARMONICS)
    )
    assert {(r["axis"], r["render_mode"], r["line_pitch_mode"]) for r in rows} == {("x", "patch_aware", "unaligned")}
    assert all(0.0 <= float(r["R"]) <= 1.0 for r in rows)
    by_block = {}
    for r in rows:
        by_block.setdefault(r["block"], []).append((r["page_id"], r["period_px"], r["harmonic"], r["R"]))
    assert by_block["H"] == by_block["S"]  # the same layout, so the same coherence
    # patch_aware: every pen position is a whole pixel and every glyph is centred in its slot (section 2.2).
    assert all(float(r["R"]) > 0.9 for r in rows if (r["period_px"], r["harmonic"]) == ("14", "1"))


@needs_data
def test_determinism_rows_compare_h0_with_s0_once_both_are_done(full_run):
    run_dir, _ = full_run
    header, rows = _csv_rows(run_dir / "determinism.csv")
    assert header == ["page_id", "condition_hash_a", "condition_hash_b", "identical"]
    conditions = expand_conditions(RUNNER_SWEEP, READERS["qwen"])
    h0 = condition_hash(next(c for c in conditions if c.block == "H" and c.phase_x_px == 0))
    s0 = condition_hash(next(c for c in conditions if c.block == "S" and c.phase_x_px == 0))
    assert sorted((r["page_id"], r["condition_hash_a"], r["condition_hash_b"], r["identical"]) for r in rows) == [
        (RUNNER_PAGES[0], h0, s0, "True"),
        (RUNNER_PAGES[1], h0, s0, "True"),
    ]


@needs_data
def test_the_manifest_records_the_sweep_file_reader_generation_and_placement(full_run):
    run_dir, _ = full_run
    data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = data["config"]
    sweep_file = Path(cfg["sweep_file"])
    assert sweep_file.name == "two_blocks.yaml" and cfg["sweep_name"] == "two_blocks"
    assert cfg["sweep_sha1"] == hashlib.sha1(sweep_file.read_bytes()).hexdigest()
    assert cfg["sweep"]["seed"] == 20261014 and cfg["reader"] == READERS["qwen"]
    render = cfg["render"]
    assert (render["page_width_px"], render["page_height_px"], render["margin_px"]) == (448, 448, 14)
    assert render["font_size_px"] == 16 and (render["max_phase_x_px"], render["max_phase_y_px"]) == (2, 0)
    generation = cfg["generation"]
    assert {k: generation[k] for k in ("do_sample", "num_beams", "repetition_penalty")} == {
        "do_sample": False, "num_beams": 1, "repetition_penalty": 1.0,
    }
    assert "1.5" in generation["max_new_tokens"]
    assert cfg["placement"]["device_map"] == "cuda" and cfg["placement"]["decision"] == "D18"
    assert (cfg["rows_planned"], cfg["pages"], cfg["save_images"]) == (RUNNER_ROWS, 2, "first")
    assert data["run_id"] == run_dir.name


@needs_data
def test_progress_log_has_one_line_per_page_with_rows_done_and_a_projection(full_run):
    run_dir, _ = full_run
    lines = _progress_lines(run_dir)
    assert len(lines) == 2
    assert "rows 5/10" in lines[0] and "rows 10/10" in lines[1]
    assert all("projected_finish" in line and "+00:00" in line for line in lines)


@needs_data
def test_an_interrupted_run_resumes_to_the_same_rows_none_repeated(tmp_path, full_run):
    out = tmp_path / "runs with space"
    path = _write_sweep(tmp_path)
    failing = FakeBackend(fail_on_call=4)
    with pytest.raises(RuntimeError, match="fake driver reset"):
        _run(path, failing, out)
    assert len(failing.unloaded) == 1  # the card is freed even when a page fails
    (run_dir,) = out.iterdir()
    _, partial = _csv_rows(run_dir / "results.csv")
    assert len(partial) == 3
    assert not (run_dir / "progress.log").exists()  # the first page never finished
    manifest_before = (run_dir / "manifest.json").read_bytes()
    resumed = FakeBackend()
    assert _run(path, resumed, out, resume=run_dir) == [run_dir]
    _, rows = _csv_rows(run_dir / "results.csv")
    assert len(rows) == RUNNER_ROWS == len(set(_keys(rows)))
    assert resumed.calls == RUNNER_ROWS - 3
    _, reference_rows = _csv_rows(full_run[0] / "results.csv")
    assert sorted(_keys(rows)) == sorted(_keys(reference_rows))
    assert (run_dir / "manifest.json").read_bytes() == manifest_before
    # The per-page files are complete and nothing in them is repeated.
    assert len(_csv_rows(run_dir / "coherence.csv")[1]) == 2 * 2 * len(QWEN_PERIODS) * len(HARMONICS)
    assert len(_csv_rows(run_dir / "determinism.csv")[1]) == 2
    assert len(_progress_lines(run_dir)) == 2


@needs_data
def test_a_cut_off_last_line_is_removed_logged_and_redone(tmp_path, caplog):
    out = tmp_path / "runs with space"
    path = _write_sweep(tmp_path)
    (run_dir,) = _run(path, FakeBackend(), out)
    results = run_dir / "results.csv"
    whole = results.read_bytes()
    assert whole.endswith(b"\n")
    body = whole.rstrip(b"\n")
    head, last = body[: body.rfind(b"\n") + 1], body[body.rfind(b"\n") + 1 :]
    results.write_bytes(head + last[:40])  # the last row loses its tail and its line break
    redo = FakeBackend()
    with caplog.at_level(logging.WARNING, logger="mrr.sweep"):
        _run(path, redo, out, resume=run_dir)
    assert "incomplete last line" in caplog.text
    _, rows = _csv_rows(results)
    assert len(rows) == RUNNER_ROWS == len(set(_keys(rows)))
    assert redo.calls == 1


@needs_data
def test_the_same_seed_gives_the_same_shuffled_order_page_by_page(tmp_path):
    path = _write_sweep(tmp_path)
    orders = []
    for name in ("first root", "second root"):
        (run_dir,) = _run(path, FakeBackend(), tmp_path / name)
        _, rows = _csv_rows(run_dir / "results.csv")
        orders.append([(r["page_id"], r["block"], r["phase_x_px"]) for r in rows])
    assert orders[0] == orders[1]
    assert [page for page, _, _ in orders[0]] == [RUNNER_PAGES[0]] * 5 + [RUNNER_PAGES[1]] * 5
    file_order = [(c.block, str(c.phase_x_px)) for c in expand_conditions(RUNNER_SWEEP, READERS["qwen"])]
    per_page = [[(b, p) for _, b, p in orders[0][i : i + 5]] for i in (0, 5)]
    assert all(sorted(page) == sorted(file_order) for page in per_page)
    assert any(page != file_order for page in per_page)  # shuffled, not file order


@needs_data
@pytest.mark.parametrize(
    "change",
    [
        {"font_size_px": "from the ladder"},
        {"pages": {"natural": {"books": ["pg105"], "per_text_type": "TODO[W3.2]"}, "random62": {"per_text_type": 1}}},
        {"repeats": "two"},
        {"seed": "<new seed>"},
    ],
    ids=["font size", "page count", "repeats", "seed"],
)
def test_a_placeholder_refuses_to_start_before_any_folder_exists(tmp_path, change):
    out = tmp_path / "runs with space"
    backend = FakeBackend()
    with pytest.raises(ValueError):
        _run(_write_sweep(tmp_path, **change), backend, out)
    assert not out.exists() and backend.loaded == []


@needs_data
def test_unknown_readers_and_an_unknown_save_images_choice_are_refused(tmp_path):
    out = tmp_path / "runs with space"
    path = _write_sweep(tmp_path)
    with pytest.raises(ValueError, match="smolvlm"):
        _run(path, FakeBackend(), out, only_readers=["smolvlm"])
    with pytest.raises(ValueError, match="save_images"):
        _run(path, FakeBackend(), out, save_images="sometimes")
    assert not out.exists()


@needs_data
def test_only_readers_picks_a_subset_and_readers_cfg_may_be_the_whole_file(tmp_path):
    path = _write_sweep(tmp_path, readers=["qwen", "smolvlm"])
    backend = FakeBackend()
    whole_file = {"readers": list(READERS.values())}  # as scripts load configs/readers.yaml
    dirs = run_sweep(
        path, whole_file, DEFAULTS, backend, only_readers=["smolvlm"], output_root=tmp_path / "runs with space",
        require_clean=False, root=REPO_ROOT,
    )
    assert len(dirs) == 1 and "_two_blocks_smolvlm_" in dirs[0].name
    _, rows = _csv_rows(dirs[0] / "results.csv")
    assert len(rows) == RUNNER_ROWS and {r["reader"] for r in rows} == {"smolvlm"}
    assert {(r["image_width_px"], r["vision_tokens"]) for r in rows} == {("384", "81")}
    assert [r.name for r in backend.loaded] == ["smolvlm"] and len(backend.unloaded) == 1


@needs_data
def test_a_deadline_already_past_leaves_a_folder_and_manifest_but_no_row(tmp_path):
    out = tmp_path / "runs with space"
    backend = FakeBackend()
    dirs = _run(_write_sweep(tmp_path), backend, out, deadline=datetime.now(timezone.utc) - timedelta(hours=1))
    assert len(dirs) == 1 and dirs[0].is_dir() and (dirs[0] / "manifest.json").exists()
    assert not (dirs[0] / "results.csv").exists() and not (dirs[0] / "progress.log").exists()
    assert backend.loaded == [] and backend.calls == 0


@needs_data
def test_a_deadline_passing_during_the_first_page_stops_at_the_page_boundary(tmp_path, monkeypatch):
    start = datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(sweep_module, "_now", _clock(start, start + timedelta(hours=1)))
    backend = FakeBackend()
    (run_dir,) = _run(_write_sweep(tmp_path), backend, tmp_path / "runs with space", deadline=start + timedelta(minutes=30))
    _, rows = _csv_rows(run_dir / "results.csv")
    assert len(rows) == 5 and {r["page_id"] for r in rows} == {RUNNER_PAGES[0]}
    assert sorted((r["block"], r["phase_x_px"]) for r in rows) == [
        ("H", "0.0"), ("H", "1.0"), ("H", "2.0"), ("S", "0.0"), ("S", "0.5"),
    ]
    assert len(_progress_lines(run_dir)) == 1
    assert len(backend.unloaded) == 1


@needs_data
def test_deadline_naive_and_aware_agree(tmp_path, monkeypatch):
    # Review Focus 4: a naive deadline is local time; both forms are compared as aware UTC.
    start = datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)
    aware = datetime.fromisoformat("2026-10-20T10:30:00Z")
    naive = aware.astimezone().replace(tzinfo=None)  # the same instant as a local wall-clock time
    path = _write_sweep(tmp_path)
    counts = {}
    for name, deadline in (("aware", aware), ("naive", naive)):
        monkeypatch.setattr(sweep_module, "_now", _clock(start, start + timedelta(hours=1)))
        (run_dir,) = _run(path, FakeBackend(), tmp_path / f"{name} root", deadline=deadline)
        _, rows = _csv_rows(run_dir / "results.csv")
        counts[name] = len(rows)
        assert all(r["timestamp_utc"].endswith("+00:00") for r in rows)
    assert counts == {"aware": 5, "naive": 5}


@needs_data
def test_dirty_code_refuses_a_start_and_changed_code_refuses_a_resume(tmp_path, monkeypatch):
    out = tmp_path / "runs with space"
    path = _write_sweep(tmp_path)
    monkeypatch.setattr(sweep_module, "code_dirty", lambda *args, **kwargs: True)
    with pytest.raises(RuntimeError, match="uncommitted"):
        _run(path, FakeBackend(), out, require_clean=True)
    monkeypatch.setattr(sweep_module, "code_dirty", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="git"):
        _run(path, FakeBackend(), out, require_clean=True)
    assert not out.exists()
    with pytest.raises(RuntimeError, match="fake driver reset"):
        _run(path, FakeBackend(fail_on_call=2), out)
    (run_dir,) = out.iterdir()
    monkeypatch.setattr(sweep_module, "code_dirty", lambda *args, **kwargs: False)
    monkeypatch.setattr(sweep_module, "code_changed_since", lambda *args, **kwargs: True)
    with pytest.raises(RuntimeError, match="changed"):
        _run(path, FakeBackend(), out, resume=run_dir, require_clean=True)
    monkeypatch.setattr(sweep_module, "code_changed_since", lambda *args, **kwargs: False)
    _run(path, FakeBackend(), out, resume=run_dir, require_clean=True)
    assert len(_csv_rows(run_dir / "results.csv")[1]) == RUNNER_ROWS


@needs_data
def test_a_token_count_the_grid_does_not_predict_stops_before_the_first_row(tmp_path):
    out = tmp_path / "runs with space"
    backend = FakeBackend(extra_tokens=1)
    with pytest.raises(RuntimeError, match="257"):
        _run(_write_sweep(tmp_path), backend, out)
    (run_dir,) = out.iterdir()
    assert not (run_dir / "results.csv").exists() and backend.calls == 0 and len(backend.unloaded) == 1


@needs_data
def test_a_resume_refuses_another_reader_or_a_changed_sweep_file(tmp_path):
    out = tmp_path / "runs with space"
    path = _write_sweep(tmp_path)
    with pytest.raises(RuntimeError, match="fake driver reset"):
        _run(path, FakeBackend(fail_on_call=2), out)
    (run_dir,) = out.iterdir()
    manifest = run_dir / "manifest.json"
    original = manifest.read_text(encoding="utf-8")
    data = json.loads(original)
    data["config"]["reader"]["name"] = "smolvlm"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="smolvlm"):
        _run(path, FakeBackend(), out, resume=run_dir)
    manifest.write_text(original, encoding="utf-8")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("# edited after the run started\n")
    with pytest.raises(RuntimeError, match="sweep file"):
        _run(path, FakeBackend(), out, resume=run_dir)
    with pytest.raises(RuntimeError, match="manifest"):
        _run(path, FakeBackend(), out, resume=tmp_path / "not a run")


@needs_data
@pytest.mark.parametrize(("choice", "count"), [("all", RUNNER_ROWS), ("none", 0)])
def test_save_images_all_or_none(tmp_path, choice, count):
    (run_dir,) = _run(_write_sweep(tmp_path), FakeBackend(), tmp_path / "runs with space", save_images=choice)
    images = list((run_dir / "images").glob("*.png")) if (run_dir / "images").exists() else []
    assert len(images) == count


@needs_data
def test_a_vertical_block_writes_its_coherence_along_y(tmp_path):
    blocks = {"V": {"render_mode": "conventional", "line_pitch_mode": "aligned", "axis": "y", "phases_px": [0.0, 2.0]}}
    (run_dir,) = _run(_write_sweep(tmp_path, blocks=blocks), FakeBackend(), tmp_path / "runs with space")
    _, rows = _csv_rows(run_dir / "coherence.csv")
    assert len(rows) == 2 * len(QWEN_PERIODS) * len(HARMONICS)
    assert {(r["block"], r["axis"], r["render_mode"], r["line_pitch_mode"]) for r in rows} == {
        ("V", "y", "conventional", "aligned"),
    }
    r1 = {(r["page_id"], r["period_px"]): float(r["R"]) for r in rows if r["harmonic"] == "1"}
    for page in RUNNER_PAGES:  # at the aligned pitch of 28 px the baselines repeat every token cell
        assert r1[(page, "28")] > r1[(page, "14")]
    _, results = _csv_rows(run_dir / "results.csv")
    assert len(results) == 4 and {(r["phase_x_px"], r["phase_y_px"]) for r in results} == {("0.0", "0.0"), ("0.0", "2.0")}
    assert {r["line_pitch_px"] for r in results} == {"28"}
    assert not (run_dir / "determinism.csv").exists()  # no two conditions differ only in their block label


@needs_data
def test_a_mismatching_determinism_pair_is_recorded_and_does_not_stop_the_run(tmp_path):
    backend = FakeBackend(texts=("alpha", "beta", "gamma"))
    (run_dir,) = _run(_write_sweep(tmp_path), backend, tmp_path / "runs with space")
    assert len(_csv_rows(run_dir / "results.csv")[1]) == RUNNER_ROWS
    _, pairs = _csv_rows(run_dir / "determinism.csv")
    assert len(pairs) == 2
    for pair in pairs:
        texts = [
            (run_dir / "transcriptions" / f"{pair['page_id']}__{pair[key]}.txt").read_text(encoding="utf-8")
            for key in ("condition_hash_a", "condition_hash_b")
        ]
        assert pair["identical"] == str(texts[0] == texts[1])


@needs_data
def test_load_results_reads_the_run_folder_with_every_id_a_string(full_run):
    run_dir, _ = full_run
    sweep_path, runs_root = run_dir.parent.parent / "two_blocks.yaml", run_dir.parent
    assert sweep_module.sweep_run_dirs(sweep_path, runs_root) == [run_dir]
    df = load_results(sweep_path, runs_root)
    assert list(df.columns) == [*RESULTS_COLUMNS, "run_dir"]
    assert len(df) == RUNNER_ROWS
    assert set(df["run_dir"]) == {str(run_dir)}
    for column in ("run_id", "reader", "page_id", "condition_hash", "transcription_file", "git_commit"):
        assert all(isinstance(value, str) for value in df[column].tolist()), column
    assert df["repeat"].tolist() == [0] * RUNNER_ROWS
    assert df["hit_limit"].dtype == bool and not df["hit_limit"].any()
    assert df["cer"].dtype == float and df["vision_tokens"].tolist() == [256] * RUNNER_ROWS
    keys = list(zip(df["reader"], df["page_id"], df["block"], df["condition_hash"]))
    assert keys == sorted(keys)


def test_load_results_without_a_run_folder_returns_an_empty_frame(tmp_path):
    df = load_results(_write_sweep(tmp_path), tmp_path / "no runs yet")
    assert len(df) == 0 and list(df.columns) == [*RESULTS_COLUMNS, "run_dir"]


# Study 2 (PREREGISTRATION_2.md sections 2 to 4): a shared count of 0, blocks that read one text type, the 38 columns.


@needs_data
def test_one_shared_page_count_may_be_zero_but_not_both():
    sweep = _pilot()
    sweep["pages"]["random62"]["per_text_type"] = 0
    pages = _plan(sweep, "qwen")["shared"]
    assert [page.text_type for page in pages] == ["natural"] * 10
    sweep["pages"]["natural"]["per_text_type"] = 0
    with pytest.raises(ValueError, match="both 0"):
        _plan(sweep, "qwen")


@pytest.mark.parametrize("value", [[], ["random62", "random62"], ["prose"], "random62"])
def test_a_bad_text_types_list_is_refused_naming_the_block(value):
    sweep = copy.deepcopy(RUNNER_SWEEP)
    sweep["blocks"]["S"]["text_types"] = value
    with pytest.raises(ValueError, match="block S.*text_types|text_types.*block S|S.*text_types"):
        expand_conditions(sweep, READERS["qwen"])


def test_text_types_are_kept_in_their_fixed_order_and_left_out_of_the_hash():
    sweep = copy.deepcopy(RUNNER_SWEEP)
    before = {condition_hash(c) for c in expand_conditions(sweep, READERS["qwen"])}
    sweep["blocks"]["S"]["text_types"] = ["random62", "natural"]
    conditions = expand_conditions(sweep, READERS["qwen"])
    assert {c.text_types for c in conditions} == {("natural", "random62")}
    assert {condition_hash(c) for c in conditions} == before


def test_text_types_on_a_per_condition_block_are_refused():
    sweep = copy.deepcopy(LADDER)
    sweep["blocks"]["Z"]["text_types"] = ["natural"]
    with pytest.raises(ValueError, match="paginate: shared"):
        expand_conditions(sweep, READERS["qwen"])


@needs_data
def test_pages_of_a_type_no_block_reads_are_refused():
    sweep = copy.deepcopy(RUNNER_SWEEP)
    for block in sweep["blocks"].values():
        block["text_types"] = ["natural"]
    with pytest.raises(ValueError, match="no shared block reads random62"):
        _plan(sweep, "qwen")


@needs_data
def test_a_block_that_reads_one_text_type_gets_rows_on_that_types_pages_only(tmp_path):
    sweep_path = _write_sweep(tmp_path, blocks={**RUNNER_SWEEP["blocks"],
                                                "S": {**RUNNER_SWEEP["blocks"]["S"], "text_types": ["random62"]}})
    backend = FakeBackend()
    (run_dir,) = _run(sweep_path, backend, tmp_path / "runs")
    _, rows = _csv_rows(run_dir / "results.csv")
    assert len(rows) == 3 * 2 + 2 * 1 == backend.calls == len(set(_keys(rows)))
    assert {r["page_id"] for r in rows if r["block"] == "S"} == {"rnd20261014_p0000"}
    assert {r["page_id"] for r in rows if r["block"] == "H"} == set(RUNNER_PAGES)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["rows_planned"] == 8
    folder = sweep_module.RunFolder(run_dir)
    assert len(folder.planned_rows("pg105_p0000", "natural")) == 3
    assert len(folder.planned_rows("rnd20261014_p0000", "random62")) == 5
    df = load_results(sweep_path, tmp_path / "runs")
    assert sweep_module.excluded_pages(df, [folder]) == {str(run_dir): set()}
    assert _run(sweep_path, FakeBackend(), tmp_path / "runs", resume=run_dir) == [run_dir]  # nothing left to write
    assert len(_csv_rows(run_dir / "results.csv")[1]) == 8


@needs_data
def test_a_study2_sweep_writes_the_letters_only_counts_and_the_inserted_spaces(tmp_path):
    texts = ("the  same text", "Sir Walter Elliot, of Kellynch Hall")
    sweep_path = _write_sweep(tmp_path, results_schema="study2")
    (run_dir,) = _run(sweep_path, FakeBackend(texts=texts), tmp_path / "runs")
    header, rows = _csv_rows(run_dir / "results.csv")
    assert header == [*RESULTS_COLUMNS, *STUDY2_SCORE_COLUMNS] and len(header) == 38
    assert len(rows) == RUNNER_ROWS
    for row in rows:
        reference = (run_dir / "references" / f"{row['page_id']}.txt").read_text(encoding="utf-8")
        hypothesis = (run_dir / row["transcription_file"]).read_text(encoding="utf-8")
        letters = char_counts_letters_only(reference, hypothesis)
        assert [int(row[c]) for c in STUDY2_SCORE_COLUMNS] == [
            letters.hits, letters.substitutions, letters.deletions, letters.insertions,
            spaces_inserted(reference, hypothesis),
        ]
    df = load_results(sweep_path, tmp_path / "runs")
    assert list(df.columns) == [*RESULTS_COLUMNS, *STUDY2_SCORE_COLUMNS, "run_dir"]
    assert all(df[c].dtype == "int64" for c in STUDY2_SCORE_COLUMNS)
    empty = load_results(sweep_path, tmp_path / "no runs yet")
    assert list(empty.columns) == [*RESULTS_COLUMNS, *STUDY2_SCORE_COLUMNS, "run_dir"]


@needs_data
def test_a_resume_under_another_schema_is_refused(full_run, tmp_path):
    run_dir, _ = full_run
    copy_dir = tmp_path / run_dir.name
    copy_dir.mkdir()
    (copy_dir / "results.csv").write_bytes((run_dir / "results.csv").read_bytes())
    with pytest.raises(RuntimeError, match="schema changed"):
        sweep_module._done_keys(copy_dir / "results.csv", [*RESULTS_COLUMNS, *STUDY2_SCORE_COLUMNS])


def test_an_unknown_results_schema_is_refused_before_any_folder(tmp_path):
    sweep_path = _write_sweep(tmp_path, results_schema="study3")
    with pytest.raises(ValueError, match="results_schema"):
        _run(sweep_path, FakeBackend(), tmp_path / "runs")
    assert not (tmp_path / "runs").exists()
