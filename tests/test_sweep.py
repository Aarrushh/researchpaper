# PURPOSE: Tests for the pagination plan of a sweep (build guide section 2.4; the block Z pages of section 3.5).
# WHAT GOES HERE: CPU checks on the committed books and font: phase allowances, shared pages across layouts, text order and ids, block Z passages, placeholders, no reuse.
# FILLED IN AT STEP: W2.2.
# DO NOT: Load a model. Paginate by character count. Weaken the capacity numbers of section 2.4 to make a test pass.
"""Tests for mrr.sweep.plan_pages and its helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import string

import pytest
import yaml

from conftest import REPO_ROOT
from mrr.grid import resolve_render_cfg
from mrr.render import fits_page
from mrr.sweep import (
    RANDOM_STREAM_CHARS,
    Condition,
    Page,
    condition_hash,
    expand_conditions,
    pagination_key,
    phase_allowances,
    plan_pages,
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
    [("natural", "TODO[W3.2]"), ("random62", "<N_Q>"), ("natural", 2.5), ("random62", 0), ("natural", True)],
)
def test_a_page_count_that_is_not_a_whole_positive_number_is_refused_naming_the_key(kind, value):
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
