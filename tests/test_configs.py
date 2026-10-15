# PURPOSE: Tests for every sweep config in configs/sweeps/ (build guide section 2.3).
# WHAT GOES HERE: CPU checks of the blocks: phase levels, condition counts, the header's run count, the determinism pair, placeholders; the full sweep against the ladder and the pilot: their blocks, no shared text, block Z pages (section 3.5).
# FILLED IN AT STEP: W2.2; W3.2 (the full sweep).
# DO NOT: Load a model. Edit a config to make a test pass after its run has started; copy it to a new file instead.
"""Tests for the sweep configs."""

from __future__ import annotations

import copy
import dataclasses
import math
import pathlib
import re

import pytest
import yaml

from mrr import grid
from mrr.sweep import HASH_KEYS, Z_MODE_TAGS, expand_conditions, pagination_key, plan_pages
from mrr.text_source import BOOKS, TEXTS_DIR

ROOT = pathlib.Path(__file__).resolve().parents[1]
SWEEPS = sorted((ROOT / "configs" / "sweeps").glob("*.yaml"))
PILOT = ROOT / "configs" / "sweeps" / "pilot.yaml"
LADDER = ROOT / "configs" / "sweeps" / "ladder.yaml"
FULL = ROOT / "configs" / "sweeps" / "full.yaml"
FONT = ROOT / "data" / "external" / "fonts" / "source-sans-3" / "SourceSans3-Regular.ttf"
HAVE_DATA = FONT.exists() and all((ROOT / TEXTS_DIR / book.file).exists() for book in BOOKS.values())
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="font or books not present (W1.4)")
Z_SIZES = (10, 12, 14, 16, 20, 24)  # block Z, the ladder's design (section 2.9)
READERS = {
    reader["name"]: reader
    for reader in yaml.safe_load((ROOT / "configs" / "readers.yaml").read_text(encoding="utf-8"))["readers"]
}
DEFAULTS = yaml.safe_load((ROOT / "configs" / "render_defaults.yaml").read_text(encoding="utf-8"))
PILOT_CONDITIONS = {"qwen": 50, "smolvlm": 62, "smolvlm500m": 54}  # per page per text type, section 2.3
FULL_CONDITIONS = {name: n + 12 for name, n in PILOT_CONDITIONS.items()}  # plus the twelve block Z layouts (section 3.5)


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _concrete(sweep, size=16):
    """Return a copy whose placeholder font sizes (strings such as "from the ladder") are replaced by ``size``."""
    sweep = copy.deepcopy(sweep)
    sizes = sweep.get("font_size_px")
    if isinstance(sizes, str):
        sweep["font_size_px"] = size
    elif isinstance(sizes, dict):
        sweep["font_size_px"] = {name: size if isinstance(value, str) else value for name, value in sizes.items()}
    return sweep


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_every_block_passes_phase_levels_for_every_reader(path):
    sweep = _load(path)
    for reader in sweep["readers"]:
        for name, block in sweep["blocks"].items():
            levels = grid.phase_levels(block, READERS[reader], block=name)
            assert levels and levels[0] == 0.0 and len(set(levels)) == len(levels)


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_every_block_expands_once_the_sizes_are_numbers(path):
    sweep = _concrete(_load(path))
    for reader in sweep["readers"]:
        conditions = expand_conditions(sweep, READERS[reader])
        assert len({dataclasses.astuple(c) for c in conditions}) == len(conditions)
        assert {c.block for c in conditions} == set(sweep["blocks"])


@pytest.mark.parametrize(("reader", "expected"), sorted(PILOT_CONDITIONS.items()))
def test_pilot_conditions_per_page_per_text_type(reader, expected):
    assert len(expand_conditions(_concrete(_load(PILOT)), READERS[reader])) == expected


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_header_run_count_equals_the_computed_count(path):
    # "total runs = 3320" on its own line, or "...; total runs = 216." at the end of the ladder's one-line formula.
    header = re.search(r"total runs = (TODO\S*|\d+)", path.read_text(encoding="utf-8"))
    assert header, "the header comment must state the total run count"
    sweep = _load(path)
    if header.group(1).startswith("TODO"):
        # A config whose counts or seed are still placeholders must refuse to plan (sections 2.3 and 2.4).
        with pytest.raises(ValueError):
            plan_pages(_concrete(sweep), READERS[sweep["readers"][0]], DEFAULTS, root=ROOT, readers=READERS)
        pytest.skip(f"{path.name}: the run count is not set yet")
    sweep = _concrete(sweep)
    total = 0  # rows = conditions x the pages each condition reads x repeats, as the runner counts them
    for reader in sweep["readers"]:
        plan = plan_pages(sweep, READERS[reader], DEFAULTS, root=ROOT, readers=READERS)  # readers: start_after
        rows = sum(len(plan[pagination_key(c)]) for c in expand_conditions(sweep, READERS[reader]))
        total += rows * sweep["repeats"]
    assert int(header.group(1)) == total


def test_h0_and_s0_differ_only_in_their_block_label():
    sweep = _concrete(_load(PILOT))
    for reader in sweep["readers"]:
        conditions = expand_conditions(sweep, READERS[reader])
        at_zero = {c.block: c for c in conditions if c.block in "HS" and (c.phase_x_px, c.phase_y_px) == (0.0, 0.0)}
        assert set(at_zero) == {"H", "S"}
        assert dataclasses.replace(at_zero["H"], block="S") == at_zero["S"]


def test_block_v_uses_the_aligned_pitch_and_block_h_the_patch_slot():
    sweep = _concrete(_load(PILOT))
    for reader, cell in (("qwen", 28), ("smolvlm", 42), ("smolvlm500m", 64)):
        conditions = expand_conditions(sweep, READERS[reader])
        v = [c for c in conditions if c.block == "V"]
        h = [c for c in conditions if c.block == "H"]
        assert {c.line_pitch_px for c in v} == {cell}
        assert all(c.patch_aware_slot_px is None and c.phase_x_px == 0.0 for c in v)
        assert {c.patch_aware_slot_px for c in h} == {READERS[reader]["patch_size"]}
        assert all(c.phase_y_px == 0.0 and c.line_pitch_px == 20 and c.paginate == "shared" for c in h)


def test_conditions_carry_the_hash_keys_as_floats_and_the_file_name_only():
    conditions = expand_conditions(_concrete(_load(PILOT)), READERS["qwen"])
    cond = next(c for c in conditions if c.block == "C" and c.phase_x_px == 7.0)
    assert tuple(cond.hash_keys()) == HASH_KEYS
    assert cond.hash_keys() == {  # the test vector of section 2.5
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


def test_ladder_block_z_gives_twelve_conditions_per_reader_each_on_its_own_pages():
    sweep = _load(LADDER)
    for reader in sweep["readers"]:
        conditions = expand_conditions(sweep, READERS[reader])
        assert len(conditions) == 12
        pairs = {(mode, size) for mode in ("conventional", "patch_aware") for size in (10, 12, 14, 16, 20, 24)}
        assert {(c.render_mode, c.font_size_px) for c in conditions} == pairs
        assert all(c.paginate == "per_condition" and (c.phase_x_px, c.phase_y_px) == (0.0, 0.0) for c in conditions)
        assert all(c.line_pitch_px == math.floor(1.25 * c.font_size_px + 0.5) for c in conditions)


def test_pilot_placeholders_stop_the_expansion_naming_the_key():
    # The pilot's sizes read "from the ladder" until the ladder has run; a string where a number belongs must raise.
    sweep = _concrete(_load(PILOT))
    sweep["font_size_px"] = {reader: 16 for reader in sweep["readers"]}
    sweep["font_size_px"]["qwen"] = "from the ladder"
    with pytest.raises(ValueError, match="font_size_px"):
        expand_conditions(sweep, READERS["qwen"])
    assert len(expand_conditions(sweep, READERS["smolvlm"])) == PILOT_CONDITIONS["smolvlm"]


@pytest.mark.parametrize(
    ("block", "key", "value"),
    [
        ("H", "step_px", "one"),
        ("S", "phases_px", ["TODO"]),
        ("A", "aa_mode", ["binary", "coverage"]),
        ("V", "line_pitch_mode", "snapped"),
        ("H", "render_mode", "mono"),
        ("C", "paginate", "per_page"),
        ("H", "step", 1),
    ],
)
def test_a_bad_block_value_stops_the_expansion_naming_the_key(block, key, value):
    sweep = _concrete(_load(PILOT))
    sweep["blocks"][block][key] = value
    with pytest.raises(ValueError, match=key):
        expand_conditions(sweep, READERS["qwen"])


def test_a_sweep_without_blocks_is_refused():
    with pytest.raises(ValueError, match="blocks"):
        expand_conditions({"factors": {"phase_x_px": [0, 7]}}, READERS["qwen"])


# The full sweep (section 3.5): the pilot's blocks and the ladder's block Z, none of their text, block Z pages.


def _placeholders(sweep):
    """True while a page count or a font size of the sweep is still a string (the TODO[W3.2] placeholders)."""
    pages = sweep["pages"]
    values = [pages[kind]["per_text_type"] for kind in ("natural", "random62")]
    values += [pages.get("z_per_text_type", 1), sweep.get("font_size_px")]
    flat = []
    for value in values:
        flat.extend(value.values() if isinstance(value, dict) else [value])
    return any(isinstance(value, str) for value in flat)


def _end(page):
    """The index of the first word after the page in its source."""
    return page.start_word + len(page.text.split())


@pytest.fixture(scope="module")
def full_sweep(tmp_path_factory):
    """The full sweep as it will run or, while W3.2's placeholders remain, a stand-in with small counts.

    The stand-in keeps every structural choice of full.yaml (its blocks, its seed, start_after naming the ladder,
    D26) and replaces only what the ladder sets: 16 px for every reader, 2 shared pages and 1 block Z page per text
    type. The earlier sweeps are replanned from concrete copies with one page per text type. Returns (sweep, stand_in).
    """
    if not HAVE_DATA:
        pytest.skip("font or books not present (W1.4)")
    sweep = _load(FULL)
    if not _placeholders(sweep):
        return sweep, False
    folder = tmp_path_factory.mktemp("full_stand_in")
    earlier = []
    for entry in sweep["pages"]["natural"]["start_after"]:
        cfg = _concrete(_load(ROOT / entry))
        for kind in ("natural", "random62"):
            cfg["pages"][kind]["per_text_type"] = 1
        path = folder / pathlib.Path(entry).name
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        earlier.append(str(path))
    sweep = _concrete(sweep)
    for kind in ("natural", "random62"):
        sweep["pages"][kind]["per_text_type"] = 2
    sweep["pages"]["z_per_text_type"] = 1
    sweep["pages"]["natural"]["start_after"] = earlier
    return sweep, True


@pytest.fixture(scope="module")
def full_plans(full_sweep):
    sweep, _ = full_sweep
    return {
        reader: plan_pages(sweep, READERS[reader], DEFAULTS, root=ROOT, readers=READERS) for reader in sweep["readers"]
    }


def test_the_full_sweep_starts_after_the_ladder_with_a_new_seed_and_five_z_pages_per_text_type():
    # D26: the pilot is not run, so the full sweep starts after the ladder only; its seed still differs from both.
    sweep, ladder, pilot = _load(FULL), _load(LADDER), _load(PILOT)
    assert sweep["pages"]["natural"]["start_after"] == ["configs/sweeps/ladder.yaml"]
    assert all((ROOT / entry).is_file() for entry in sweep["pages"]["natural"]["start_after"])
    assert sweep["seed"] not in {ladder["seed"], pilot["seed"]}
    assert sweep["pages"]["z_per_text_type"] == 5 and sweep["repeats"] == 1
    assert list(sweep["blocks"]) == ["V", "H", "C", "S", "A", "Z"]
    assert set(sweep["font_size_px"]) == set(sweep["readers"])
    for kind in ("natural", "random62"):
        assert set(sweep["pages"][kind]["per_text_type"]) == set(sweep["readers"])


@pytest.mark.parametrize("reader", _load(FULL)["readers"])
def test_the_full_sweep_reads_the_pilot_blocks_and_the_ladder_block_z_unchanged(reader):
    full = expand_conditions(_concrete(_load(FULL)), READERS[reader])
    assert [c for c in full if c.block != "Z"] == expand_conditions(_concrete(_load(PILOT)), READERS[reader])
    assert [c for c in full if c.block == "Z"] == expand_conditions(_load(LADDER), READERS[reader])
    assert len(full) == FULL_CONDITIONS[reader]


@needs_data
def test_the_full_sweep_shares_no_text_with_the_ladder_or_the_pilot(full_sweep, full_plans):
    sweep, _ = full_sweep
    used = {}  # book -> the first word no reader of the earlier sweeps read
    texts = set()
    for entry in sweep["pages"]["natural"]["start_after"]:
        earlier = _load(ROOT / entry)
        for reader in earlier["readers"]:
            for pages in plan_pages(earlier, READERS[reader], DEFAULTS, root=ROOT, readers=READERS).values():
                for page in pages:
                    texts.add((page.book_id, page.text))
                    if page.text_type == "natural":
                        used[page.book_id] = max(used.get(page.book_id, 0), _end(page))
    assert used and texts
    for reader, plan in full_plans.items():
        for key, pages in plan.items():
            for page in pages:
                assert (page.book_id, page.text) not in texts, (reader, key, page.page_id)
                if page.text_type == "natural":
                    assert page.start_word >= used.get(page.book_id, 0), (reader, key, page.page_id)
                else:
                    assert page.book_id == f"rnd{sweep['seed']}"  # a new seed, so a new random stream


@needs_data
def test_block_z_of_the_full_sweep_has_its_pages_per_layout_after_the_shared_pages(full_sweep, full_plans):
    sweep, _ = full_sweep
    n_z = sweep["pages"]["z_per_text_type"]
    layouts = {f"z:{mode}:{size}" for mode in ("conventional", "patch_aware") for size in Z_SIZES}
    for reader, plan in full_plans.items():
        assert set(plan) == layouts | {"shared"}, reader
        shared_end = {}
        for page in plan["shared"]:
            shared_end[page.book_id] = max(shared_end.get(page.book_id, 0), _end(page))
        for key in layouts:
            pages = plan[key]
            _, mode, size = key.split(":")
            assert len(pages) == 2 * n_z and [p.text_type for p in pages] == ["natural", "random62"] * n_z
            for page in pages:
                assert re.fullmatch(rf"(pg\d+|rnd{sweep['seed']})_z{size}_{Z_MODE_TAGS[mode]}_p\d{{4}}", page.page_id)
                # Block Z never repeats a shared page (section 3.5, step 4).
                assert page.start_word >= shared_end.get(page.book_id, 0), (reader, key, page.page_id)


@needs_data
def test_page_i_of_every_block_z_layout_of_the_full_sweep_starts_with_the_same_word(full_sweep, full_plans):
    sweep, _ = full_sweep
    for reader, plan in full_plans.items():
        z_plans = [pages for key, pages in plan.items() if key != "shared"]
        assert len(z_plans) == 12
        for i in range(2 * sweep["pages"]["z_per_text_type"]):
            starts = {(pages[i].book_id, pages[i].start_word, pages[i].text.split()[0]) for pages in z_plans}
            assert len(starts) == 1, (reader, i, starts)
