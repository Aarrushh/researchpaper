# PURPOSE: Tests for every sweep config in configs/sweeps/ (build guide section 2.3).
# WHAT GOES HERE: CPU checks of the blocks: phase levels, condition counts, the header's run count, the determinism pair, placeholders; the full sweep against the ladder and the pilot: their blocks, no shared text, block Z pages (section 3.5). Each sweep's readers come from its readers_config.
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
from mrr.sweep import HASH_KEYS, Z_MODE_TAGS, expand_conditions, pagination_key, plan_pages, readers_config
from mrr.text_source import BOOKS, TEXTS_DIR, book_text

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


def _readers(sweep):
    """The reader configs by name of the file the sweep names (``readers_config``), configs/readers.yaml by default."""
    return {reader["name"]: reader for reader in _load(ROOT / readers_config(sweep))["readers"]}


def _concrete(sweep, size=16):
    """Return a copy whose placeholder font sizes (strings such as "from the ladder"), at the top or in a block, are
    replaced by ``size``."""
    sweep = copy.deepcopy(sweep)
    holders = [sweep] + [block for block in sweep.get("blocks", {}).values() if isinstance(block, dict)]
    for holder in holders:
        sizes = holder.get("font_size_px")
        if isinstance(sizes, str):
            holder["font_size_px"] = size
        elif isinstance(sizes, dict):
            holder["font_size_px"] = {name: size if isinstance(value, str) else value for name, value in sizes.items()}
    return sweep


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_every_block_passes_phase_levels_for_every_reader(path):
    sweep = _load(path)
    readers = _readers(sweep)
    for reader in sweep["readers"]:
        for name, block in sweep["blocks"].items():
            levels = grid.phase_levels(block, readers[reader], block=name)
            assert levels and levels[0] == 0.0 and len(set(levels)) == len(levels)


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_every_block_expands_once_the_sizes_are_numbers(path):
    sweep = _concrete(_load(path))
    readers = _readers(sweep)
    for reader in sweep["readers"]:
        conditions = expand_conditions(sweep, readers[reader])
        assert len({dataclasses.astuple(c) for c in conditions}) == len(conditions)
        assert {c.block for c in conditions} == set(sweep["blocks"])


@pytest.mark.parametrize(("reader", "expected"), sorted(PILOT_CONDITIONS.items()))
def test_pilot_conditions_per_page_per_text_type(reader, expected):
    assert len(expand_conditions(_concrete(_load(PILOT)), READERS[reader])) == expected


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_header_run_count_equals_the_computed_count(path):
    # "total runs = 3320" on its own line, or "...; total runs = 216." at the end of the ladder's one-line formula.
    # The stage-0 sweeps of Study 3 end their "Rows per reader" line with "...; total 392." instead; they were frozen
    # by their run (the manifest holds their SHA-1) before this test saw them, so the wording stays and the count is
    # checked all the same.
    text = path.read_text(encoding="utf-8")
    header = re.search(r"total runs = (TODO\S*|\d+)", text) or re.search(r"^# Rows per reader .*; total (\d+)\.$",
                                                                          text, re.MULTILINE)
    assert header, "the header comment must state the total run count"
    sweep = _load(path)
    readers = _readers(sweep)
    if header.group(1).startswith("TODO"):
        # A config whose counts or seed are still placeholders must refuse to plan (sections 2.3 and 2.4).
        with pytest.raises(ValueError):
            plan_pages(_concrete(sweep), readers[sweep["readers"][0]], DEFAULTS, root=ROOT, readers=readers)
        pytest.skip(f"{path.name}: the run count is not set yet")
    sweep = _concrete(sweep)
    total = 0  # rows = conditions x the pages of its text types each condition reads x repeats, as the runner counts
    for reader in sweep["readers"]:
        plan = plan_pages(sweep, readers[reader], DEFAULTS, root=ROOT, readers=readers)  # readers: start_after
        rows = sum(sum(page.text_type in c.text_types for page in plan[pagination_key(c)])
                   for c in expand_conditions(sweep, readers[reader]))
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
    full_cfg = _concrete(_load(FULL))
    full = expand_conditions(full_cfg, READERS[reader])
    pilot = _load(PILOT)  # never run (D26); its blocks at the full sweep's sizes, which the ladder chose
    pilot["font_size_px"] = full_cfg["font_size_px"]
    assert [c for c in full if c.block != "Z"] == expand_conditions(pilot, READERS[reader])
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


# Study 2 (PREREGISTRATION_2.md sections 2, 3 and 5).

STUDY2 = ROOT / "configs" / "sweeps" / "study2.yaml"
STUDY2_RANDOM = {"S": [0.0, 5.0, 10.0], "Q": [0.0], "M": [0.0, 3.0, 7.0], "D": [0.0]}  # P = 14; M7 is 8 px at P = 16
STUDY1_ALLOWANCES = {"qwen": [105, 26], "smolvlm": [40, 40], "smolvlm500m": [60, 60]}
PILOT_PERSUASION_OFFSET = 200_000  # the pilots' prose came from Persuasion's skipped text at this character onward


def test_study2_names_its_readers_schema_and_a_new_seed():
    sweep = _load(STUDY2)
    assert sweep["readers"] == ["qwen", "smolvlm", "smolvlm500m", "qwen7b", "internvl3_2b"]
    assert readers_config(sweep) == "configs/readers_study2.yaml" and sweep["results_schema"] == "study2"
    used = {_load(path)["seed"] for path in SWEEPS if path != STUDY2}
    assert sweep["seed"] == 20261020 and sweep["seed"] not in used | {20261018, 20261019, 20261021, 20261022}
    assert sweep["pages"]["natural"]["start_after"] == ["configs/sweeps/ladder.yaml", "configs/sweeps/full.yaml"]
    assert sweep["pages"]["random62"]["per_text_type"]["smolvlm500m"] == 0


def test_study2_keeps_study1_allowances_and_gives_the_validation_readers_qwens():
    allowances = _load(STUDY2)["phase_allowances_px"]
    assert {name: allowances[name] for name in STUDY1_ALLOWANCES} == STUDY1_ALLOWANCES
    assert allowances["qwen7b"] == allowances["internvl3_2b"] == STUDY1_ALLOWANCES["qwen"]


@pytest.mark.parametrize("reader", _load(STUDY2)["readers"])
def test_study2_conditions_are_the_eight_renders_of_section_3(reader):
    sweep = _load(STUDY2)
    readers = _readers(sweep)
    cfg = readers[reader]
    conditions = expand_conditions(sweep, cfg)
    by_block = {}
    for c in conditions:
        by_block.setdefault(c.block, []).append(c)
    p = cfg["patch_size"]
    expected = dict(STUDY2_RANDOM, M=[0.0, 3.0, p / 2])
    assert {b: [c.phase_x_px for c in cs] for b, cs in by_block.items()} == expected
    assert all(c.phase_y_px == 0.0 and c.aa_mode == "coverage_srgb" and c.line_pitch_mode == "unaligned"
               for c in conditions)
    mono, sans = by_block["M"][0], by_block["S"][0]
    assert mono.font_file == "FiraMono-Regular.ttf" and mono.render_mode == "patch_aware"
    assert mono.patch_aware_slot_px == p and mono.font_size_px == {14: 23.333, 16: 26.651}[p]
    assert sans.font_file == "SourceSans3-Regular.ttf" and sans.font_size_px == {14: 25, 16: 29}[p]
    assert by_block["Q"][0].font_size_px == sans.font_size_px and by_block["Q"][0].render_mode == "patch_aware"
    density = by_block["D"][0]  # S28, the owner's choice of 19 October (section 11)
    assert density.font_file == sans.font_file and density.font_size_px == 28 and density.render_mode == "conventional"
    assert {b: cs[0].text_types for b, cs in by_block.items()} == {
        "S": ("natural", "random62"), "Q": ("random62",), "M": ("natural", "random62"), "D": ("random62",)}


@needs_data
def test_the_mono_size_puts_the_advance_at_most_one_patch_and_the_sans_matches_its_x_height():
    from mrr.render import load_glyph

    mono = str(ROOT / "data" / "external" / "fonts" / "fira-mono" / "FiraMono-Regular.ttf")
    for p, size, sans_size in ((14, 23.333, 25), (16, 26.651, 29)):
        advance = load_glyph(mono, size, "x").advance_px
        assert p - 0.02 < advance <= p
        assert load_glyph(mono, size + 1 / 64, "x").advance_px > p  # the largest such size, to 1/64 px
        assert load_glyph(str(FONT), sans_size, "x").coverage.shape[0] == load_glyph(mono, size, "x").coverage.shape[0]


@pytest.fixture(scope="module")
def study2_plans():
    if not HAVE_DATA:
        pytest.skip("font or books not present (W1.4)")
    sweep = _load(STUDY2)
    readers = _readers(sweep)
    return sweep, {name: plan_pages(sweep, readers[name], DEFAULTS, root=ROOT, readers=readers)
                   for name in sweep["readers"]}


@needs_data
def test_study2_pages_are_planned_per_reader_text_types_alternating(study2_plans):
    sweep, plans = study2_plans
    for reader, plan in plans.items():
        assert set(plan) == {"shared"}
        pages = plan["shared"]
        counts = {kind: sweep["pages"][kind]["per_text_type"][reader] for kind in ("natural", "random62")}
        assert [sum(p.text_type == kind for p in pages) for kind in counts] == list(counts.values())
        if counts["random62"]:
            assert [p.text_type for p in pages[:4]] == ["natural", "random62"] * 2
        assert all(p.book_id == f"rnd{sweep['seed']}" for p in pages if p.text_type == "random62")


@needs_data
def test_study2_prose_starts_after_study1_and_stays_below_the_pilots_offset(study2_plans):
    sweep, plans = study2_plans
    used = {}
    readers = _readers(sweep)
    for entry in sweep["pages"]["natural"]["start_after"]:
        earlier = _load(ROOT / entry)
        for reader in earlier["readers"]:
            for pages in plan_pages(earlier, readers[reader], DEFAULTS, root=ROOT, readers=readers).values():
                for page in pages:
                    if page.text_type == "natural":
                        used[page.book_id] = max(used.get(page.book_id, 0), _end(page))
    words = book_text("pg105", ROOT).split()
    for reader, plan in plans.items():
        for page in plan["shared"]:
            if page.text_type != "natural":
                continue
            assert page.start_word >= used[page.book_id], (reader, page.page_id)
            if page.book_id == "pg105":
                assert len(" ".join(words[: _end(page)])) < PILOT_PERSUASION_OFFSET, (reader, page.page_id)

