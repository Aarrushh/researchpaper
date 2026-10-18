# PURPOSE: Turn a sweep config into conditions and pages, hash a condition, and run the sweep.
# WHAT GOES HERE: Condition and expand_conditions (section 2.3); Page and plan_pages (2.4); condition_hash (2.5); run_sweep and load_results (2.8).
# FILLED IN AT STEP: W2.2 (conditions, pages), W2.3 (hash, runner).
# DO NOT: Load a model here. Accept a string where a number belongs: a placeholder must stop a run. Change the hash keys after the pilot without an amendment.
"""Sweep definitions: blocks to conditions (build guide section 2.3, D10) and conditions to pages (section 2.4).

A sweep config is a mapping of blocks. Each block names one layout (``render_mode`` and ``line_pitch_mode``) swept
along one axis over a range of phases at a step, or at an explicit list of phases. ``expand_conditions`` turns the
blocks of one reader into ``Condition`` records in a fixed order. The ten fields the condition hash of section 2.5
reads are the ten keys of that hash; ``paginate`` says whether the block reads the reader's shared pages or its own.

``plan_pages`` gives the pages: one shared pagination per reader that fits every shared layout under the phase
allowances of section 2.4, and one pagination per block Z layout, on its own pages (section 3.5).
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import itertools
import json
import logging
import math
import numbers
import os
import re
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image

from mrr import grid, render, score, text_source
from mrr.manifest import (
    CODE_PATHS,
    RESULTS_COLUMNS,
    RESULTS_SCHEMAS,
    STUDY2_SCORE_COLUMNS,
    code_changed_since,
    code_dirty,
    git_commit,
    new_run_id,
    write_manifest,
)
from mrr.readers import GREEDY

LINE_PITCH_MODES = ("aligned", "unaligned")
PAGINATION_MODES = ("shared", "per_condition")
AXES = ("x", "y")
# Every key a block may hold. Anything else is a typo, and a typo must not silently become a default.
BLOCK_KEYS = frozenset(
    {"render_mode", "line_pitch_mode", "axis", "range", "step_px", "phases_px", "aa_mode", "font_size_px", "font_file", "paginate",
     "text_types"}
)
# The two text types, the keys under pages: in a sweep config. A block reads both unless its text_types names one
# (Study 2: Q0 and S18 read random characters only, PREREGISTRATION_2.md section 3). Not a hash key.
TEXT_TYPES = ("natural", "random62")
# The keys a block inherits from the top of the sweep when it does not set them itself (section 2.3, step 1).
INHERITED_KEYS = ("font_file", "font_size_px", "aa_mode")
# The keys that may hold a list. The lists are crossed in this order, with the phase levels innermost (step 3).
CROSSED_KEYS = ("render_mode", "line_pitch_mode", "font_size_px", "aa_mode")
# The ten keys of the condition hash, in the order json.dumps(sort_keys=True) writes them (section 2.5).
HASH_KEYS = (
    "aa_mode",
    "block",
    "font_file",
    "font_size_px",
    "line_pitch_mode",
    "line_pitch_px",
    "patch_aware_slot_px",
    "phase_x_px",
    "phase_y_px",
    "render_mode",
)


@dataclasses.dataclass(frozen=True)
class Condition:
    """One rendering condition: the ten keys of the condition hash (section 2.5) and the block's pagination mode.

    ``font_file`` is the file name only, never a path, so that the hash does not depend on the machine.
    ``patch_aware_slot_px`` is the reader's patch for ``patch_aware`` and None for ``conventional``.
    ``paginate`` is ``shared`` (the reader's one pagination, section 2.4) or ``per_condition`` (block Z).
    ``text_types`` are the text types whose pages the condition reads, both unless its block names one; like
    ``paginate`` it is not hashed.
    """

    block: str
    render_mode: str
    line_pitch_mode: str
    line_pitch_px: int
    font_file: str
    font_size_px: float
    aa_mode: str
    phase_x_px: float
    phase_y_px: float
    patch_aware_slot_px: int | None
    paginate: str = "shared"
    text_types: tuple[str, ...] = TEXT_TYPES

    def hash_keys(self) -> dict[str, Any]:
        """Return the ten keys of the condition hash, every number cast to float (section 2.5, step 2)."""
        out: dict[str, Any] = {}
        for key in HASH_KEYS:
            value = getattr(self, key)
            out[key] = float(value) if isinstance(value, numbers.Real) and not isinstance(value, bool) else value
        return out


def _number(value: Any, key: str, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label}: {key} must be a number, not {value!r}")
    return value


def _choice(value: Any, allowed: tuple[str, ...], key: str, label: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{label}: {key} must be one of {', '.join(allowed)}, not {value!r}")
    return value


def _text_types(value: Any, label: str) -> tuple[str, ...]:
    """A block's text types: both when absent, else a non-empty list of them without repeats, in TEXT_TYPES order."""
    if value is None:
        return TEXT_TYPES
    if not isinstance(value, (list, tuple)) or not value or len(set(map(str, value))) != len(value):
        raise ValueError(f"{label}: text_types must be a non-empty list of {', '.join(TEXT_TYPES)} without repeats, "
                         f"not {value!r}")
    for v in value:
        _choice(v, TEXT_TYPES, "text_types", label)
    return tuple(t for t in TEXT_TYPES if t in value)


def _per_reader(value: Any, reader: str, key: str) -> Any:
    """Resolve a top-level value that may be a map from reader name to value."""
    if isinstance(value, Mapping):
        if reader not in value:
            raise ValueError(f"{key} has no entry for reader {reader}")
        return value[reader]
    return value


def _file_name(path: str) -> str:
    """Return the last component of a path written with either separator."""
    return PureWindowsPath(path).name


def condition_hash(cond: Mapping[str, Any] | Condition) -> str:
    """Return the 12-character condition hash of section 2.5: the row's identity across runs and machines.

    Exactly the ten ``HASH_KEYS`` are hashed, every number cast to float and ``font_file`` reduced to its file name,
    through ``json.dumps(..., sort_keys=True, separators=(",", ":"))`` and SHA-1. Nothing else goes in: a path or a
    page size would tie the hash to the machine, and Python's ``hash()`` changes between processes. A missing key
    raises ``ValueError``; extra keys are ignored. The key set is frozen once the pilot has run.
    """
    source = cond.hash_keys() if isinstance(cond, Condition) else cond
    missing = [key for key in HASH_KEYS if key not in source]
    if missing:
        raise ValueError(f"condition_hash: missing {', '.join(missing)}")
    keyed: dict[str, Any] = {}
    for key in HASH_KEYS:
        value = source[key]
        if key == "font_file":
            value = _file_name(str(value))
        elif isinstance(value, numbers.Real) and not isinstance(value, bool):
            value = float(value)
        keyed[key] = value
    canonical = json.dumps(keyed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def expand_conditions(sweep: Mapping[str, Any], reader_cfg: Mapping[str, Any]) -> list[Condition]:
    """Return the conditions of every block of ``sweep`` for one reader, blocks in file order (section 2.3).

    Each block starts from the sweep's top-level ``font_file``, this reader's ``font_size_px`` and ``aa_mode``, and
    its own keys override them; a block's value may itself be a map from reader name to value (Study 2's sizes differ
    between P = 14 and P = 16). Its phase levels come from ``grid.phase_levels`` on its axis (the other axis is 0),
    crossed with every list-valued key in the order of ``CROSSED_KEYS``, phases innermost. ``line_pitch_px`` follows
    section 1A.4: floor(1.25 * size + 0.5) unaligned, the reader's ``aligned_pitch_px`` aligned.

    Raises ValueError naming the key when a string stands where a number belongs (the "from the ladder"
    placeholders of section 2.7), when a block holds an unknown key, or when an enumerated value is not allowed.
    """
    reader = str(reader_cfg["name"])
    patch = int(reader_cfg["patch_size"])
    blocks = sweep.get("blocks")
    if not isinstance(blocks, Mapping) or not blocks:
        raise ValueError("a sweep is a mapping of blocks under the key blocks (section 2.7); none found")
    conditions: list[Condition] = []
    for name, block in blocks.items():
        label = f"block {name}"
        if not isinstance(block, Mapping):
            raise ValueError(f"{label}: must be a mapping of keys, not {block!r}")
        unknown = sorted(set(block) - BLOCK_KEYS)
        if unknown:
            raise ValueError(f"{label}: unknown key(s) {unknown}; a block may hold {sorted(BLOCK_KEYS)}")
        settings = {key: _per_reader(sweep.get(key), reader, key) for key in INHERITED_KEYS}
        for key in INHERITED_KEYS + ("render_mode", "line_pitch_mode"):
            if key in block:
                settings[key] = _per_reader(block[key], reader, f"{label}: {key}")
        for key in ("render_mode", "line_pitch_mode", "font_file", "font_size_px", "aa_mode"):
            if settings.get(key) is None:
                raise ValueError(f"{label}: {key} is missing; set it in the block or at the top of the sweep")
        axis = _choice(block.get("axis", "x"), AXES, "axis", label)
        paginate = _choice(block.get("paginate", "shared"), PAGINATION_MODES, "paginate", label)
        text_types = _text_types(block.get("text_types"), label)
        if paginate != "shared" and text_types != TEXT_TYPES:
            raise ValueError(f"{label}: text_types needs paginate: shared; a per-condition block reads both text types")
        font_file = settings["font_file"]
        if not isinstance(font_file, str):
            raise ValueError(f"{label}: font_file must be one path, not {font_file!r} (one typeface per sweep)")
        levels = grid.phase_levels(block, reader_cfg, block=str(name))
        choices = []
        for key in CROSSED_KEYS:
            value = settings[key]
            values = list(value) if isinstance(value, (list, tuple)) else [value]
            if not values:
                raise ValueError(f"{label}: {key} is an empty list")
            choices.append([(key, v) for v in values])
        for combination in itertools.product(*choices):
            chosen = dict(combination)
            render_mode = _choice(chosen["render_mode"], render.RENDER_MODES, "render_mode", label)
            pitch_mode = _choice(chosen["line_pitch_mode"], LINE_PITCH_MODES, "line_pitch_mode", label)
            aa_mode = _choice(chosen["aa_mode"], render.AA_MODES, "aa_mode", label)
            size = _number(chosen["font_size_px"], "font_size_px", label)
            if size <= 0:
                raise ValueError(f"{label}: font_size_px must be positive, not {size!r}")
            if pitch_mode == "unaligned":
                pitch = render.unaligned_pitch_px(size)
            else:
                pitch = int(_number(reader_cfg.get("aligned_pitch_px"), "aligned_pitch_px", f"reader {reader}"))
            slot = patch if render_mode == "patch_aware" else None
            for phase in levels:
                phase_x, phase_y = (phase, 0.0) if axis == "x" else (0.0, phase)
                conditions.append(
                    Condition(
                        block=str(name),
                        render_mode=render_mode,
                        line_pitch_mode=pitch_mode,
                        line_pitch_px=pitch,
                        font_file=_file_name(font_file),
                        font_size_px=size,
                        aa_mode=aa_mode,
                        phase_x_px=phase_x,
                        phase_y_px=phase_y,
                        patch_aware_slot_px=slot,
                        paginate=paginate,
                        text_types=text_types,
                    )
                )
    return conditions


# Pages (section 2.4; the block Z pages of section 3.5).

SHARED = "shared"  # the pagination key of the pages every shared block reads
NATURAL, RANDOM62 = TEXT_TYPES  # the two text types: the keys under pages: in a sweep config
RANDOM_STREAM_CHARS = 200_000  # one random_text stream per sweep (section 2.4, step 4)
Z_MODE_TAGS = {"conventional": "conv", "patch_aware": "pa"}  # <mode> in a block Z page id


@dataclasses.dataclass(frozen=True)
class Page:
    """One page of text and where it came from (section 2.4).

    ``page_id`` is ``<book_id>_p<NNNN>`` for a shared page and ``<book_id>_z<size>_<mode>_p<NNNN>`` for a block Z
    page, ``<mode>`` being ``conv`` or ``pa``. NNNN is the page's index among the pages of its text type in the plan,
    from 0000, so natural pages carry the books in rotation (pg105_p0000, pg84_p0001, pg1661_p0002, pg105_p0003).
    ``book_id`` is a key of ``text_source.BOOKS`` or ``rnd<seed>`` for the sweep's random stream. ``start_word`` is
    the index of the page's first word in its source, the book after its front-matter skip or the random stream,
    so that a later sweep can start after it (step 6).
    """

    page_id: str
    book_id: str
    text_type: str
    start_word: int
    text: str


@dataclasses.dataclass
class _Stream:
    """A source of words with a cursor: one book after its skip point, or the sweep's random stream."""

    book_id: str
    text_type: str
    text: str
    cursor: int = 0

    def page_from(self, start: int, fits: Callable[[str], bool]) -> str:
        pages = text_source.paginate(self.text, fits, n_pages=1, start=start)
        if not pages:
            raise ValueError(f"{self.book_id} ran out of text at word {start}; plan fewer pages or add a book")
        return pages[0]


def phase_allowances(conditions: Iterable[Condition]) -> tuple[int, int]:
    """Return ``(max_phase_x_px, max_phase_y_px)`` for the shared pages (section 2.4, step 1).

    The ceiling of the largest ``phase_x_px`` and of the largest ``phase_y_px`` over the conditions that read the
    shared pages: 105 and 26 px for Qwen, 40 and 40 for SmolVLM-2.2B, 60 and 60 for SmolVLM-500M with the blocks of
    the pilot. (0, 0) when no condition is shared.
    """
    shared = [c for c in conditions if c.paginate == SHARED]
    if not shared:
        return (0, 0)
    return (math.ceil(max(c.phase_x_px for c in shared)), math.ceil(max(c.phase_y_px for c in shared)))


def shared_allowances(sweep: Mapping[str, Any], reader: str, conditions: Iterable[Condition]) -> tuple[int, int]:
    """Return the phase allowances of one reader's shared pages: the sweep's ``phase_allowances_px``, else computed.

    Without the key, ``phase_allowances`` of the conditions. With it, a map from reader name to [x, y] whole pixels,
    which must be at least the computed allowance on each axis, or a phase would move text out of the block. Study 2
    fixes Study 1's allowances (PREREGISTRATION_2.md section 3), so its pages have Study 1's text block although its
    own phases reach only 10 px.
    """
    computed = phase_allowances(conditions)
    if "phase_allowances_px" not in sweep:
        return computed
    value = _per_reader(sweep["phase_allowances_px"], reader, "phase_allowances_px")
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, numbers.Integral) for v in value)):
        raise ValueError(f"phase_allowances_px for {reader} must be two whole numbers [x, y], not {value!r}")
    x, y = int(value[0]), int(value[1])
    if x < computed[0] or y < computed[1]:
        raise ValueError(f"phase_allowances_px for {reader} is {[x, y]}, below the largest phases {list(computed)} of "
                         "its shared conditions")
    return (x, y)


def pagination_key(condition: Condition) -> str:
    """Return the key under which ``plan_pages`` lists the pages this condition reads.

    ``shared`` for the reader's one pagination; ``z:<render_mode>:<size>`` for a block Z layout, paginated on its
    own (section 3.5). The phase, the anti-aliasing and the block label do not change a pagination.
    """
    if condition.paginate == SHARED:
        return SHARED
    return f"z:{condition.render_mode}:{condition.font_size_px:g}"


def _layout_id(condition: Condition) -> tuple[str, int, float, str]:
    """What decides a pagination: the render mode, the pitch, the size and the font."""
    return (condition.render_mode, condition.line_pitch_px, condition.font_size_px, condition.font_file)


def _font_paths(sweep: Mapping[str, Any], root: Path) -> dict[str, str]:
    """Map each font file name of the sweep to its path, a str for freetype-py; relative paths start at root."""
    blocks = sweep.get("blocks") if isinstance(sweep.get("blocks"), Mapping) else {}
    candidates = []
    for value in [sweep.get("font_file")] + [b.get("font_file") for b in blocks.values() if isinstance(b, Mapping)]:
        candidates.extend(value.values() if isinstance(value, Mapping) else [value])  # a map by reader, or one path
    out: dict[str, str] = {}
    for path in candidates:
        if isinstance(path, str):
            name, resolved = _file_name(path), str(root / path)
            if out.get(name, resolved) != resolved:
                raise ValueError(f"two font files share the name {name}: {out[name]} and {resolved}")
            out[name] = resolved
    return out


def _layout_cfg(
    condition: Condition,
    reader_cfg: Mapping[str, Any],
    render_cfg: Mapping[str, Any],
    font_paths: Mapping[str, str],
    allowances: tuple[int, int],
    phase: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    """Return the render config of this condition's layout under ``allowances``, at ``phase`` (x, y; 0 by default).

    plan_pages paginates at phase 0; the runner renders each row at the condition's own phases (section 2.8).
    """
    max_x, max_y = allowances
    return grid.resolve_render_cfg(
        render_cfg,
        reader_cfg,
        font_file=font_paths[condition.font_file],
        font_size_px=condition.font_size_px,
        line_pitch_px=condition.line_pitch_px,
        render_mode=condition.render_mode,
        aa_mode=condition.aa_mode,
        phase_x_px=phase[0],
        phase_y_px=phase[1],
        max_phase_x_px=max_x,
        max_phase_y_px=max_y,
    )


def _count(value: Any, label: str, reader: str, minimum: int = 1) -> int:
    value = _per_reader(value, reader, label)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{label} must be a whole number of pages, {minimum} or more, not {value!r}")
    return int(value)


def _page_counts(pages_cfg: Any, reader: str) -> tuple[dict[str, int], dict[str, int]]:
    """Return the shared and the block Z page counts by text type (section 2.4, step 7 on placeholders).

    The shared counts are ``pages.<text type>.per_text_type``; the Z counts are ``pages.z_per_text_type`` for both
    text types, or the shared counts when that key is absent (the ladder). Any may be a map by reader. A string or a
    fraction raises ValueError naming the key, and so does zero, except that one shared count of a reader may be 0
    while the other is not (Study 2: smolvlm500m reads prose only, PREREGISTRATION_2.md section 2).
    """
    if not isinstance(pages_cfg, Mapping):
        raise ValueError("pages must be a mapping with natural and random62 (section 2.7)")
    shared = {}
    for kind in (NATURAL, RANDOM62):
        label = f"pages.{kind}.per_text_type"
        section = pages_cfg.get(kind)
        if not isinstance(section, Mapping) or "per_text_type" not in section:
            raise ValueError(f"{label} is missing")
        shared[kind] = _count(section["per_text_type"], label, reader, minimum=0)
    if not any(shared.values()):
        raise ValueError(f"pages.natural.per_text_type and pages.random62.per_text_type are both 0 for {reader}")
    if "z_per_text_type" in pages_cfg:
        n = _count(pages_cfg["z_per_text_type"], "pages.z_per_text_type", reader)
        return shared, {NATURAL: n, RANDOM62: n}
    return shared, dict(shared)


def _interleave(first: list, second: list) -> list:
    """Return first[0], second[0], first[1], second[1], ... then whatever is left (section 2.4, step 5)."""
    out = []
    for a, b in itertools.zip_longest(first, second):
        out.extend(x for x in (a, b) if x is not None)
    return out


def _start_after(
    sweep: Mapping[str, Any],
    pages_cfg: Mapping[str, Any],
    render_cfg: Mapping[str, Any],
    root: Path,
    readers: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, int]:
    """Return the first unused word of each book after the sweeps in ``pages.natural.start_after`` (step 6).

    Each earlier sweep is replanned for every reader it lists, so the later sweep begins each book after the last
    word any of them read, and its seed must differ from each earlier seed so that the random stream is new too.
    """
    entries = pages_cfg[NATURAL].get("start_after") or []
    if not entries:
        return {}
    if not isinstance(entries, (list, tuple)):
        raise ValueError("pages.natural.start_after must be a list of sweep files")
    if readers is None:
        raise ValueError(
            "pages.natural.start_after needs readers, the reader configs by name, to replan the earlier sweeps"
        )
    starts: dict[str, int] = {}
    for entry in entries:
        earlier = yaml.safe_load((root / str(entry)).read_text(encoding="utf-8"))
        if earlier.get("seed") == sweep.get("seed"):
            raise ValueError(
                f"seed {sweep.get('seed')!r} is also the seed of {entry}; a later sweep needs a new seed "
                "(section 2.4, step 6)"
            )
        for name in earlier.get("readers") or []:
            if name not in readers:
                raise ValueError(f"{entry} lists reader {name!r}, which readers does not have")
            for pages in plan_pages(earlier, readers[name], render_cfg, root=root, readers=readers).values():
                for page in pages:
                    if page.text_type == NATURAL:
                        end = page.start_word + len(page.text.split())
                        starts[page.book_id] = max(starts.get(page.book_id, 0), end)
    return starts


def plan_pages(
    sweep: Mapping[str, Any],
    reader_cfg: Mapping[str, Any],
    render_cfg: Mapping[str, Any],
    *,
    root: str | Path = ".",
    readers: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, list[Page]]:
    """Return the pages of one reader for every pagination of ``sweep`` (section 2.4; block Z, section 3.5).

    Under ``shared`` are the pages every block with ``paginate: shared`` reads. A page fits when ``render.fits_page``
    holds for each distinct shared layout that reads its text type (in the pilot: conventional unaligned for C and A,
    conventional aligned for V, patch_aware unaligned for H and S; a block's ``text_types`` can leave a layout out of
    one type) with the allowances of ``phase_allowances``. Natural pages come from
    ``pages.natural.books`` in rotation, each book from its front-matter skip (``text_source.book_text``); random
    pages from one ``random_text(RANDOM_STREAM_CHARS, seed)`` stream; the list interleaves them, natural then random.

    Under ``pagination_key(condition)`` for each block Z layout are that layout's own pages, paginated alone at phase
    0 with zero allowances. Page i of every Z layout starts at the same word, the start of passage i; passages of one
    source are spaced by the fullest page among the layouts, so no two pages of one layout overlap. Passages start
    where this sweep's shared pages ended, so block Z never repeats a shared page; in a sweep without shared blocks
    (the ladder) they start at the skip points. ``pages.z_per_text_type`` sets their count; without it the shared
    counts apply.

    ``pages.natural.start_after`` lists earlier sweep files: each book then starts after the last word any reader of
    those sweeps used, and ``seed`` must be new (step 6); ``readers`` is required for it.

    Args:
        sweep: a sweep config (configs/sweeps/*.yaml).
        reader_cfg: one reader block of configs/readers.yaml.
        render_cfg: the mapping of configs/render_defaults.yaml; every layout is resolved from it with
            grid.resolve_render_cfg.
        root: the directory relative paths resolve against (font_file, the books under text_source.TEXTS_DIR,
            start_after); scripts run from the repository root.
        readers: every reader config by name, needed only with start_after.

    Returns:
        The lists of Page by pagination key, each in the order the runner visits its pages.

    Raises:
        ValueError: as expand_conditions; or if a page count or the seed is a string (a placeholder), a book is not
            registered, two block Z layouts would share page ids, or a source runs out of text.
    """
    reader = str(reader_cfg["name"])
    root = Path(root)
    pages_cfg = sweep.get("pages")
    shared_counts, z_counts = _page_counts(pages_cfg, reader)
    seed = sweep.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError(f"seed must be a whole number, not {seed!r}")
    books = pages_cfg[NATURAL].get("books")
    if not isinstance(books, (list, tuple)) or not books:
        raise ValueError("pages.natural.books must list at least one book id")
    unknown = [book for book in books if book not in text_source.BOOKS]
    if unknown:
        raise ValueError(
            f"pages.natural.books: unknown book(s) {unknown}; the registered books are {sorted(text_source.BOOKS)}"
        )
    conditions = expand_conditions(sweep, reader_cfg)
    shared = [c for c in conditions if c.paginate == SHARED]
    z_layouts: dict[str, Condition] = {}
    for c in conditions:
        if c.paginate == SHARED:
            continue
        first = z_layouts.setdefault(pagination_key(c), c)
        if _layout_id(first) != _layout_id(c):
            raise ValueError(
                f"blocks {first.block} and {c.block}: two per-condition layouts at {c.render_mode} "
                f"{c.font_size_px:g} px differ in pitch or font, so their pages would share a page id"
            )
    font_paths = _font_paths(sweep, root)
    starts = _start_after(sweep, pages_cfg, render_cfg, root, readers)
    natural = {b: _Stream(b, NATURAL, text_source.book_text(b, root), starts.get(b, 0)) for b in dict.fromkeys(books)}
    random = _Stream(f"rnd{seed}", RANDOM62, text_source.random_text(RANDOM_STREAM_CHARS, int(seed)))

    def stream(kind: str, index: int) -> _Stream:
        return natural[books[index % len(books)]] if kind == NATURAL else random

    plan: dict[str, list[Page]] = {}
    if shared:
        allowances = shared_allowances(sweep, reader, shared)
        layouts: dict[tuple, dict[str, Any]] = {}
        read_by: dict[str, dict[tuple, dict[str, Any]]] = {kind: {} for kind in TEXT_TYPES}  # the layouts of each type
        for c in shared:
            if _layout_id(c) not in layouts:
                layouts[_layout_id(c)] = _layout_cfg(c, reader_cfg, render_cfg, font_paths, allowances)
            for kind in c.text_types:
                read_by[kind][_layout_id(c)] = layouts[_layout_id(c)]

        by_type: dict[str, list[Page]] = {}
        for kind in (NATURAL, RANDOM62):
            by_type[kind] = []
            cfgs = list(read_by[kind].values())
            if shared_counts[kind] and not cfgs:
                raise ValueError(f"pages.{kind}.per_text_type plans {shared_counts[kind]} pages for {reader}, "
                                 f"but no shared block reads {kind}")

            def fits(page: str, cfgs: list = cfgs) -> bool:
                return all(render.fits_page(page, cfg) for cfg in cfgs)

            for i in range(shared_counts[kind]):
                source = stream(kind, i)
                text = source.page_from(source.cursor, fits)
                by_type[kind].append(Page(f"{source.book_id}_p{i:04d}", source.book_id, kind, source.cursor, text))
                source.cursor += len(text.split())
        plan[SHARED] = _interleave(by_type[NATURAL], by_type[RANDOM62])
    if z_layouts:
        z_cfgs = {key: _layout_cfg(c, reader_cfg, render_cfg, font_paths, (0, 0)) for key, c in z_layouts.items()}
        z_pages: dict[str, dict[str, list[Page]]] = {key: {NATURAL: [], RANDOM62: []} for key in z_layouts}
        for kind in (NATURAL, RANDOM62):
            for i in range(z_counts[kind]):
                source = stream(kind, i)
                start, longest = source.cursor, 0
                for key, c in z_layouts.items():
                    cfg = z_cfgs[key]
                    text = source.page_from(start, lambda page, cfg=cfg: render.fits_page(page, cfg))
                    longest = max(longest, len(text.split()))
                    page_id = f"{source.book_id}_z{c.font_size_px:g}_{Z_MODE_TAGS[c.render_mode]}_p{i:04d}"
                    z_pages[key][kind].append(Page(page_id, source.book_id, kind, start, text))
                source.cursor = start + longest
        for key in z_layouts:
            plan[key] = _interleave(z_pages[key][NATURAL], z_pages[key][RANDOM62])
    return plan


# The runner (section 2.8).

SAVE_IMAGES = ("all", "first", "none")  # which rendered pages a run keeps under images/
HARMONICS = (1, 2, 3)  # of the coherence rows (section 2.2)
MAX_NEW_TOKENS_FACTOR = 1.5  # section 1B.9, applied page by page
MAX_NEW_TOKENS_RULE = "ceil(1.5 x the reference's length in the reader's tokens), per page (sections 1B.9 and 2.8)"
COHERENCE_COLUMNS = ("page_id", "block", "render_mode", "line_pitch_mode", "axis", "period_px", "harmonic", "R")
DETERMINISM_COLUMNS = ("page_id", "condition_hash_a", "condition_hash_b", "identical")
RESULTS_FILE, COHERENCE_FILE, DETERMINISM_FILE, PROGRESS_FILE = "results.csv", "coherence.csv", "determinism.csv", "progress.log"
PROJECTION_PAGES = 10  # the projected finish uses the mean row latency of the last ten pages

_log = logging.getLogger(__name__)


def _now() -> datetime:
    """Return the current time as aware UTC: the runner's only clock, so that tests can replace it (Review Focus 4)."""
    return datetime.now(timezone.utc)


def _readers_by_name(readers_cfg: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Accept the mapping of configs/readers.yaml (its ``readers`` list) or a mapping from reader name to config."""
    listed = readers_cfg.get("readers")
    entries = listed if isinstance(listed, list) else list(readers_cfg.values())
    out: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or "name" not in entry:
            raise ValueError("readers_cfg must be the mapping of configs/readers.yaml or a mapping from name to reader config")
        out[str(entry["name"])] = entry
    return out


def _repeats(sweep: Mapping[str, Any]) -> int:
    value = sweep.get("repeats", 1)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"repeats must be a whole number, 1 or more, not {value!r}")
    return int(value)


def reader_periods(reader_cfg: Mapping[str, Any]) -> list[int]:
    """The periods of the reader's grid: the patch, the token cell and, when it has one, the attention window."""
    periods = {int(reader_cfg["patch_size"]), grid.token_cell_px(reader_cfg)}
    if "window_px" in reader_cfg:
        periods.add(int(reader_cfg["window_px"]))
    return sorted(periods)


def _page_sequence(plan: Mapping[str, list[Page]]) -> list[tuple[str, Page]]:
    """The run's pages in order, as (pagination key, page): the paginations interleaved page index by page index.

    The shared pages keep their plan order. Block Z layouts take turns at each index, so an interrupted ladder holds
    every layout on the same passages rather than every passage of a few layouts.
    """
    out: list[tuple[str, Page]] = []
    longest = max((len(pages) for pages in plan.values()), default=0)
    for i in range(longest):
        for key, pages in plan.items():
            if i < len(pages):
                out.append((key, pages[i]))
    return out


def _twin_key(condition: Condition) -> tuple:
    """Everything the hash reads except the block label: two conditions with one key render the same pixels."""
    keys = condition.hash_keys()
    return tuple(keys[k] for k in HASH_KEYS if k != "block")


def _repair_results(path: Path) -> None:
    """Cut off an incomplete last line of results.csv, one without a line break, and log it (section 2.8, step 7)."""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n") + 1
    _log.warning("%s: cut off an incomplete last line of %d bytes: %r", path, len(data) - cut, data[cut : cut + 200])
    with open(path, "r+b") as fh:
        fh.truncate(cut)


DEFAULT_READERS_CONFIG = "configs/readers.yaml"


def readers_config(sweep: Mapping[str, Any]) -> str:
    """The reader file a sweep's readers are configured in: its ``readers_config``, else configs/readers.yaml.

    Study 2 names configs/readers_study2.yaml, which adds the validation readers to Study 1's three (whose blocks it
    copies unchanged), so that configs/readers.yaml and ``make contract`` keep Study 1's readers only.
    """
    path = sweep.get("readers_config", DEFAULT_READERS_CONFIG)
    if not isinstance(path, str) or not path:
        raise ValueError(f"readers_config must be one path, not {path!r}")
    return path


def results_columns(sweep: Mapping[str, Any]) -> tuple[str, ...]:
    """The columns of results.csv a sweep writes: RESULTS_COLUMNS, or Study 2's 38 under ``results_schema: study2``."""
    name = sweep.get("results_schema", "study1")
    if name not in RESULTS_SCHEMAS:
        raise ValueError(f"results_schema must be one of {', '.join(RESULTS_SCHEMAS)}, not {name!r}")
    return RESULTS_SCHEMAS[name]


def _done_keys(path: Path, columns: Sequence[str] = RESULTS_COLUMNS) -> set[tuple[str, str, str, str]]:
    """The resume keys (reader, page_id, condition_hash, repeat) of the rows already in results.csv."""
    if not path.exists():
        return set()
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return set()
        if list(reader.fieldnames) != list(columns):
            raise RuntimeError(f"{path}: its header is not the sweep's results columns; the schema changed since this "
                               "run started")
        return {(row["reader"], row["page_id"], row["condition_hash"], row["repeat"]) for row in reader}


def _csv_tuples(path: Path, columns: Sequence[str]) -> set[tuple[str, ...]]:
    """The values of ``columns`` in every row of an existing CSV, as tuples; an empty set without the file."""
    if not path.exists():
        return set()
    with open(path, encoding="utf-8", newline="") as fh:
        return {tuple(row[c] for c in columns) for row in csv.DictReader(fh)}


def _append_rows(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Append rows to a CSV, the header only when the file is new, then flush and fsync (section 2.8, step 6)."""
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def _check_resume(run_dir: Path, reader: str, sweep_sha1: str, require_clean: bool, root: Path) -> None:
    """Refuse a resume folder whose manifest names another reader or sweep file, or whose code moved."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"{run_dir}: no manifest.json; resume needs a run folder")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = manifest.get("config") or {}
    named = (cfg.get("reader") or {}).get("name")
    if named != reader:
        raise RuntimeError(f"{run_dir}: its manifest names reader {named!r}, not {reader!r}")
    if cfg.get("sweep_sha1") != sweep_sha1:
        raise RuntimeError(
            f"{run_dir}: the sweep file changed since this run started (its manifest holds SHA-1 "
            f"{cfg.get('sweep_sha1')}); a changed config is a new run (the DO NOT of configs/sweeps)"
        )
    if require_clean:
        commit = manifest.get("git_commit")
        changed = code_changed_since(commit, cwd=root) if commit else None
        if changed is None:
            raise RuntimeError(f"{run_dir}: cannot compare the code with the manifest's commit {commit!r} through git")
        if changed:
            raise RuntimeError(
                f"{run_dir}: {', '.join(CODE_PATHS)} changed since commit {commit[:7]}; a resumed run must use the code "
                "that started it (section 2.8)"
            )


def _manifest_config(
    sweep_path: Path,
    sweep: Mapping[str, Any],
    sweep_sha1: str,
    reader_cfg: Mapping[str, Any],
    conditions: Sequence[Condition],
    render_defaults: Mapping[str, Any],
    allowances: tuple[int, int],
    repeats: int,
    n_pages: int,
    rows_planned: int,
    save_images: str,
    deadline: datetime | None,
) -> dict[str, Any]:
    """The resolved config of one reader's run, for the manifest (section 2.8, step 2)."""
    sizes = sorted({c.font_size_px for c in conditions})
    fonts = sorted({c.font_file for c in conditions})
    render_cfg = grid.resolve_render_cfg(
        render_defaults, reader_cfg, max_phase_x_px=allowances[0], max_phase_y_px=allowances[1]
    )
    render_cfg["font_size_px"] = sizes[0] if len(sizes) == 1 else sizes
    render_cfg["font_file"] = fonts[0] if len(fonts) == 1 else fonts  # the names, as the rows carry them
    return {
        "sweep_file": str(sweep_path),
        "sweep_sha1": sweep_sha1,
        "sweep_name": sweep_path.stem,
        "sweep": dict(sweep),
        "reader": dict(reader_cfg),
        "render": render_cfg,
        "generation": {**GREEDY, "max_new_tokens": MAX_NEW_TOKENS_RULE},
        "placement": {"device_map": reader_cfg.get("device_map"), "dtype": reader_cfg.get("dtype"), "decision": "D18"},
        "conditions": len(conditions),
        "pages": n_pages,
        "repeats": repeats,
        "rows_planned": rows_planned,
        "save_images": save_images,
        "deadline": deadline.isoformat() if deadline is not None else None,
    }


@dataclasses.dataclass
class _ReaderRun:
    """One reader's share of a run: its folder, its conditions and pages, and the loop over them."""

    reader_cfg: Mapping[str, Any]
    sweep: Mapping[str, Any]
    conditions: list[Condition]
    plan: Mapping[str, list[Page]]
    repeats: int
    run_dir: Path
    render_defaults: Mapping[str, Any]
    font_paths: Mapping[str, str]
    allowances: tuple[int, int]
    save_images: str
    commit: str

    def __post_init__(self) -> None:
        self.name = str(self.reader_cfg["name"])
        self.sequence = _page_sequence(self.plan)
        self.columns = results_columns(self.sweep)
        self.rows_planned = self.repeats * sum(
            sum(page.text_type in c.text_types for page in self.plan[pagination_key(c)]) for c in self.conditions
        )
        self.hashes = {c: condition_hash(c) for c in self.conditions}
        self.blocks = {str(k): v for k, v in self.sweep["blocks"].items()}
        self.periods = reader_periods(self.reader_cfg)
        self.seed = int(self.sweep["seed"])
        twins: dict[tuple, list[Condition]] = {}
        for c in self.conditions:
            twins.setdefault(_twin_key(c), []).append(c)
        self.twins = {key: group for key, group in twins.items() if len(group) > 1}

    def _key(self, page: Page, condition: Condition, repeat: int) -> tuple[str, str, str, str]:
        return (self.name, page.page_id, self.hashes[condition], str(repeat))

    def _cfg(self, condition: Condition, key: str, at_phase: bool) -> dict[str, Any]:
        allow = self.allowances if key == SHARED else (0, 0)
        phase = (condition.phase_x_px, condition.phase_y_px) if at_phase else (0.0, 0.0)
        return _layout_cfg(condition, self.reader_cfg, self.render_defaults, self.font_paths, allow, phase=phase)

    def _conditions_of(self, key: str, text_type: str) -> list[Condition]:
        """The conditions that read the pages of pagination ``key`` and of ``text_type``, in condition order."""
        return [c for c in self.conditions if pagination_key(c) == key and text_type in c.text_types]

    def pending(self, done: set[tuple[str, str, str, str]]) -> list[tuple[int, str, Page]]:
        """The pages with at least one row still to write: (index in the sequence, pagination key, page)."""
        out = []
        for i, (key, page) in enumerate(self.sequence):
            conds = self._conditions_of(key, page.text_type)
            if any(self._key(page, c, r) not in done for c in conds for r in range(self.repeats)):
                out.append((i, key, page))
        return out

    def coherence_rows(self, page: Page, key: str) -> list[dict[str, Any]]:
        """The page's coherence at phase 0: one row per block layout, period and harmonic (section 2.2)."""
        rows = []
        seen = set()
        for c in self._conditions_of(key, page.text_type):
            ident = (c.block, _layout_id(c))
            if ident in seen:
                continue
            seen.add(ident)
            axis = str(self.blocks[c.block].get("axis", "x"))
            boxes = render.glyph_ink_boxes(page.text, self._cfg(c, key, at_phase=False), axis)
            centres = [b.start_px + b.size_px / 2 for b in boxes]
            for period in self.periods:
                for k in HARMONICS:
                    rows.append({
                        "page_id": page.page_id, "block": c.block, "render_mode": c.render_mode,
                        "line_pitch_mode": c.line_pitch_mode, "axis": axis, "period_px": period, "harmonic": k,
                        "R": grid.coherence(centres, period, k),
                    })
        return rows

    def check_tokens(self, backend: Any, reader: Any, page: Page, key: str) -> None:
        """Before the first row: the processor's token count must be the grid's (section 2.8, step 3; D8)."""
        image = render.render_page(page.text, self._cfg(self._conditions_of(key, page.text_type)[0], key, at_phase=True))
        height, width = image.shape[:2]
        counted, expected = backend.count_vision_tokens(reader, image), grid.token_count(self.reader_cfg, height, width)
        if counted != expected:
            raise RuntimeError(
                f"{self.name}: the processor made {counted} vision tokens from a {height} x {width} page, the grid "
                f"predicts {expected}: the page is being resized (D8, section 1B.6)"
            )

    def run(self, backend: Any, deadline: datetime | None) -> bool:
        """Write every missing row of this reader; return True if the deadline stopped the run (section 2.8)."""
        results_path = self.run_dir / RESULTS_FILE
        _repair_results(results_path)
        done = _done_keys(results_path, self.columns)
        coherence_done = {t[0] for t in _csv_tuples(self.run_dir / COHERENCE_FILE, ("page_id",))}
        determinism_done = _csv_tuples(self.run_dir / DETERMINISM_FILE, DETERMINISM_COLUMNS[:3])
        pending = self.pending(done)
        _log.info("%s: %s: %d of %d rows done, %d of %d pages to go", self.run_dir.name, self.name, len(done),
                  self.rows_planned, len(pending), len(self.sequence))
        if not pending:
            return False
        prompt = str(self.reader_cfg["prompt"])
        recent: deque[tuple[float, int]] = deque(maxlen=PROJECTION_PAGES)
        reader = None
        try:
            for i, key, page in pending:
                if deadline is not None and _now() >= deadline:
                    _log.info("%s: deadline %s reached before page %s; stopping at the page boundary",
                              self.name, deadline.isoformat(), page.page_id)
                    return True
                if reader is None:
                    reader = backend.load_reader(self.reader_cfg)
                    self.check_tokens(backend, reader, page, key)
                if page.page_id not in coherence_done:
                    _append_rows(self.run_dir / COHERENCE_FILE, COHERENCE_COLUMNS, self.coherence_rows(page, key))
                    coherence_done.add(page.page_id)
                conds = self._conditions_of(key, page.text_type)
                limit = max(1, math.ceil(MAX_NEW_TOKENS_FACTOR * backend.count_text_tokens(reader, page.text)))
                latencies: list[float] = []
                for j in np.random.default_rng([self.seed, i]).permutation(len(conds) * self.repeats):
                    c, r = conds[j % len(conds)], int(j // len(conds))
                    if self._key(page, c, r) in done:
                        continue
                    latencies.append(self._row(backend, reader, page, key, c, r, i == 0, limit, prompt, done))
                for c in conds:
                    self._determinism(page, c, done, determinism_done)
                recent.append((sum(latencies), len(latencies)))
                self._progress(i, page, done, latencies, recent)
        finally:
            if reader is not None:
                backend.unload_reader(reader)
        return False

    def _row(self, backend, reader, page: Page, key: str, c: Condition, r: int, first_page: bool, limit: int,
             prompt: str, done: set) -> float:
        """Render, transcribe and score one row; write its transcription, then the row. Return its latency."""
        image = render.render_page(page.text, self._cfg(c, key, at_phase=True))
        result = backend.transcribe_full(reader, image, prompt, max_new_tokens=limit)
        counts = score.char_counts(page.text, result.text)
        suffix = "" if r == 0 else f"__r{r}"
        name = f"{page.page_id}__{self.hashes[c]}{suffix}"
        transcription_file = f"transcriptions/{name}.txt"
        (self.run_dir / transcription_file).write_text(result.text, encoding="utf-8")
        if self.save_images == "all" or (self.save_images == "first" and first_page):
            (self.run_dir / "images").mkdir(exist_ok=True)
            Image.fromarray(image).save(self.run_dir / "images" / f"{name}.png")
        height, width = image.shape[:2]
        row = {
            "run_id": self.run_dir.name, "reader": self.name, "reader_revision": self.reader_cfg.get("revision"),
            "page_id": page.page_id, "book_id": page.book_id, "text_type": page.text_type,
            "block": c.block, "condition_hash": self.hashes[c], "repeat": r,
            "font_file": c.font_file, "font_size_px": c.font_size_px, "line_pitch_px": c.line_pitch_px,
            "line_pitch_mode": c.line_pitch_mode, "aa_mode": c.aa_mode, "render_mode": c.render_mode,
            "phase_x_px": c.phase_x_px, "phase_y_px": c.phase_y_px,
            "image_width_px": width, "image_height_px": height,
            "vision_tokens": grid.token_count(self.reader_cfg, height, width),
            "reference_chars": len(page.text), "transcription_file": transcription_file,
            "output_tokens": result.new_tokens, "hit_limit": result.hit_limit,
            "cer": score.pooled_cer([counts]), "wer": score.wer(page.text, result.text),
            "hits": counts.hits, "substitutions": counts.substitutions,
            "deletions": counts.deletions, "insertions": counts.insertions,
            "latency_s": result.latency_s, "timestamp_utc": _now().isoformat(timespec="seconds"),
            "git_commit": self.commit,
        }
        if "spaces_inserted" in self.columns:  # Study 2's scores (PREREGISTRATION_2.md section 4)
            letters = score.char_counts_letters_only(page.text, result.text)
            row.update({"hits_ns": letters.hits, "substitutions_ns": letters.substitutions,
                        "deletions_ns": letters.deletions, "insertions_ns": letters.insertions,
                        "spaces_inserted": score.spaces_inserted(page.text, result.text)})
        _append_rows(self.run_dir / RESULTS_FILE, self.columns, [row])
        done.add(self._key(page, c, r))
        return float(result.latency_s)

    def _determinism(self, page: Page, c: Condition, done: set, recorded: set) -> None:
        """Once both conditions of a pair that differ only in their block are done, compare their transcriptions."""
        for other in self.twins.get(_twin_key(c), ()):
            if other is c or self._key(page, c, 0) not in done or self._key(page, other, 0) not in done:
                continue
            a, b = sorted((c, other), key=self.conditions.index)
            pair = (page.page_id, self.hashes[a], self.hashes[b])
            if pair in recorded:
                continue
            texts = [
                (self.run_dir / "transcriptions" / f"{page.page_id}__{self.hashes[x]}.txt").read_text(encoding="utf-8")
                for x in (a, b)
            ]
            _append_rows(self.run_dir / DETERMINISM_FILE, DETERMINISM_COLUMNS, [{
                "page_id": page.page_id, "condition_hash_a": self.hashes[a], "condition_hash_b": self.hashes[b],
                "identical": texts[0] == texts[1],
            }])
            recorded.add(pair)

    def _progress(self, i: int, page: Page, done: set, latencies: list[float], recent: deque) -> None:
        """One line per page: time, rows done of planned, the page's mean latency, the projected finish (step 8)."""
        now = _now()
        rows_recent = sum(n for _, n in recent)
        mean_recent = sum(s for s, _ in recent) / rows_recent if rows_recent else 0.0
        projected = now + timedelta(seconds=(self.rows_planned - len(done)) * mean_recent)
        page_mean = sum(latencies) / len(latencies) if latencies else 0.0
        line = (
            f"{now.isoformat(timespec='seconds')} page {i + 1}/{len(self.sequence)} {page.page_id} "
            f"rows {len(done)}/{self.rows_planned} page_latency_s {page_mean:.3f} "
            f"projected_finish {projected.isoformat(timespec='seconds')}"
        )
        with open(self.run_dir / PROGRESS_FILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        _log.info(line)


def run_sweep(
    sweep_path: Path,
    readers_cfg: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    backend: Any,
    only_readers: Sequence[str] | None = None,
    resume: Path | None = None,
    save_images: str = "first",
    output_root: Path | None = None,
    require_clean: bool = True,
    deadline: datetime | None = None,
    *,
    root: str | Path | None = None,
) -> list[Path]:
    """Run a sweep: render, transcribe and score every page under every condition, one row at a time (section 2.8).

    One reader at a time, in the sweep's order. Each reader gets a run folder ``output_root / new_run_id(...)`` with
    ``manifest.json``, ``results.csv`` (``results_columns(sweep)``: the 33 of ``RESULTS_COLUMNS``, 38 under
    ``results_schema: study2``), ``coherence.csv``, ``determinism.csv``,
    ``progress.log``, ``references/`` and ``transcriptions/``, and ``images/`` under ``save_images``. The pages come
    from ``plan_pages``; within a page the (condition, repeat) pairs are shuffled with
    ``np.random.default_rng([seed, page_index]).permutation(n)``. Each page's ``max_new_tokens`` is
    ceil(1.5 x its reference in the reader's tokens). A transcription file is written before its row, and every row is
    flushed and fsynced, so an interrupted run resumes from ``results.csv``: the key is (reader, page_id,
    condition_hash, repeat), and an incomplete last line is cut off and logged.

    Args:
        sweep_path: a sweep config (configs/sweeps/*.yaml).
        readers_cfg: the mapping of configs/readers.yaml, or a mapping from reader name to reader config.
        render_defaults: the mapping of configs/render_defaults.yaml.
        backend: an object with load_reader, unload_reader, transcribe_full, count_vision_tokens and
            count_text_tokens, the five functions of mrr.readers; the command line passes that module.
        only_readers: a subset of the sweep's readers to run; None runs them all.
        resume: an existing run folder to continue; its manifest must name one of the readers and the same sweep file.
        save_images: "all", "first" (every condition of the first page) or "none".
        output_root: where run folders go; None takes the sweep's ``output_root`` (results/runs) under ``root``.
        require_clean: refuse to start while src, scripts, configs, pyproject.toml or requirements.lock.txt have
            uncommitted changes, and refuse to resume if they changed since the manifest's commit. Tests pass False.
        deadline: stop at the first page boundary after this time; a naive datetime is local time, an aware one is
            converted to UTC (Review Focus 4). The clock is ``_now()``.
        root: the directory relative paths resolve against (fonts, books, output_root); the current directory by
            default, which is the repository root when the scripts run.

    Returns:
        The run folders, one per reader started, in order.

    Raises:
        ValueError: a bad argument, or as expand_conditions and plan_pages (a placeholder stops the run before any
            folder is made).
        RuntimeError: dirty or changed code, a resume folder that does not match, a run folder that already exists,
            or a vision token count the grid does not predict.
        Whatever the backend raises: a failed page stops the run, it is never skipped; fix and resume.
    """
    if save_images not in SAVE_IMAGES:
        raise ValueError(f"save_images must be one of {', '.join(SAVE_IMAGES)}, not {save_images!r}")
    sweep_path = Path(sweep_path)
    root = Path.cwd() if root is None else Path(root)
    raw = sweep_path.read_bytes()
    sweep = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(sweep, Mapping):
        raise ValueError(f"{sweep_path}: a sweep config is a mapping")
    sweep_sha1 = hashlib.sha1(raw).hexdigest()
    readers = _readers_by_name(readers_cfg)
    listed = sweep.get("readers")
    if not isinstance(listed, (list, tuple)) or not listed:
        raise ValueError(f"{sweep_path}: readers must list at least one reader")
    listed = [str(r) for r in listed]
    selected = listed if only_readers is None else [str(r) for r in only_readers]
    unknown = [r for r in selected if r not in listed]
    if unknown:
        raise ValueError(f"reader(s) {unknown} are not among the sweep's readers {listed}")
    missing = [r for r in selected if r not in readers]
    if missing:
        raise ValueError(f"reader(s) {missing} are not configured; the readers are {sorted(readers)}")
    if deadline is not None:
        deadline = deadline.astimezone(timezone.utc)  # a naive deadline is local time
    repeats = _repeats(sweep)
    results_columns(sweep)  # an unknown results_schema stops the run before any folder is made
    if require_clean:
        dirty = code_dirty(cwd=root)
        if dirty is None:
            raise RuntimeError(f"cannot check {', '.join(CODE_PATHS)} through git in {root}; a run must record a commit")
        if dirty:
            raise RuntimeError(f"{', '.join(CODE_PATHS)} have uncommitted changes; commit them before a run (section 2.8)")
    resume_reader = None
    if resume is not None:
        resume = Path(resume)
        manifest_path = resume / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"{resume}: no manifest.json; resume needs a run folder")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resume_reader = ((manifest.get("config") or {}).get("reader") or {}).get("name")
        if resume_reader not in selected:
            raise RuntimeError(f"{resume}: its manifest names reader {resume_reader!r}, not one of {selected}")
    # Every reader is planned before any folder is made, so a placeholder stops the run at once (section 2.7).
    planned = {
        name: (
            expand_conditions(sweep, readers[name]),
            plan_pages(sweep, readers[name], render_defaults, root=root, readers=readers),
        )
        for name in selected
    }
    font_paths = _font_paths(sweep, root)
    out_root = Path(output_root) if output_root is not None else root / str(sweep.get("output_root", "results/runs"))
    commit = git_commit(cwd=root) or "nogit"
    run_dirs: list[Path] = []
    for name in selected:
        conditions, plan = planned[name]
        allowances = shared_allowances(sweep, name, conditions)
        if name == resume_reader:
            run_dir = resume
            _check_resume(run_dir, name, sweep_sha1, require_clean, root)
        else:
            run_dir = out_root / new_run_id(sweep_path.stem, name)
            if run_dir.exists():
                raise RuntimeError(f"{run_dir} exists; resume it, or start a minute later for a new run id")
            run_dir.mkdir(parents=True)
        job = _ReaderRun(readers[name], sweep, conditions, plan, repeats, run_dir, render_defaults, font_paths,
                         allowances, save_images, commit)
        if name != resume_reader:
            write_manifest(run_dir, _manifest_config(sweep_path, sweep, sweep_sha1, readers[name], conditions,
                                                     render_defaults, allowances, repeats, len(job.sequence),
                                                     job.rows_planned, save_images, deadline))
        for folder in ("references", "transcriptions"):
            (run_dir / folder).mkdir(exist_ok=True)
        for _, page in job.sequence:
            (run_dir / "references" / f"{page.page_id}.txt").write_text(page.text, encoding="utf-8")
        run_dirs.append(run_dir)
        if job.run(backend, deadline):
            break
    return run_dirs


# The dtypes results.csv is read with (section 3.2, step 4): a string for every id, hash and name, so that a hash such
# as 1e5123456789 is never read as a number and an all-digit one never as an integer. The keys are RESULTS_COLUMNS and
# STUDY2_SCORE_COLUMNS.
RESULTS_DTYPES: dict[str, Any] = {
    "run_id": str, "reader": str, "reader_revision": str, "page_id": str, "book_id": str, "text_type": str,
    "block": str, "condition_hash": str, "repeat": "int64", "font_file": str, "font_size_px": "float64",
    "line_pitch_px": "float64", "line_pitch_mode": str, "aa_mode": str, "render_mode": str, "phase_x_px": "float64",
    "phase_y_px": "float64", "image_width_px": "int64", "image_height_px": "int64", "vision_tokens": "int64",
    "reference_chars": "int64", "transcription_file": str, "output_tokens": "int64", "hit_limit": bool,
    "cer": "float64", "wer": "float64", "hits": "int64", "substitutions": "int64", "deletions": "int64",
    "insertions": "int64", "latency_s": "float64", "timestamp_utc": str, "git_commit": str,
    **{column: "int64" for column in STUDY2_SCORE_COLUMNS},
}
RESULTS_KEY = ("reader", "page_id", "condition_hash", "repeat")  # the resume key of a row (section 2.8)


def sweep_run_dirs(sweep_path: Path, runs_root: Path) -> list[Path]:
    """Return the run folders of ``sweep_path`` under ``runs_root``, in sorted order (section 3.2, step 3).

    A folder belongs to the sweep when its ``manifest.json`` names the sweep file's SHA-1 and stem, which the runner
    writes into every manifest. Folders of other sweeps, and folders without a manifest, are ignored. A folder with
    the sweep's name and another SHA-1 raises ValueError: someone edited a frozen config.
    """
    sweep_path, runs_root = Path(sweep_path), Path(runs_root)
    sweep_sha1 = hashlib.sha1(sweep_path.read_bytes()).hexdigest()
    if not runs_root.is_dir():
        return []
    out: list[Path] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        cfg = (json.loads(manifest_path.read_text(encoding="utf-8")) or {}).get("config") or {}
        if cfg.get("sweep_name") != sweep_path.stem:
            continue
        if cfg.get("sweep_sha1") != sweep_sha1:
            raise ValueError(
                f"{run_dir}: its manifest names sweep {sweep_path.stem} with SHA-1 {cfg.get('sweep_sha1')}, but "
                f"{sweep_path} has {sweep_sha1}: a frozen config was edited (the DO NOT of configs/sweeps)"
            )
        out.append(run_dir)
    return out


def load_results(sweep_path: Path, runs_root: Path) -> "pandas.DataFrame":
    """Return every row of every run of ``sweep_path`` under ``runs_root`` (section 3.2).

    Only the run folders of ``sweep_run_dirs`` are read, each ``results.csv`` with ``RESULTS_DTYPES``, so every id
    and hash stays a string. A ``run_dir`` column names each row's folder. Raises ValueError on a header that is
    not the sweep's ``results_columns`` and on a repeated key (reader, page_id, condition_hash, repeat), naming it. The frame
    is sorted by reader, page_id, block and condition_hash. pandas is imported here, as plotting.py imports
    matplotlib, so the runner needs neither.
    """
    import pandas as pd  # noqa: PLC0415  (the analysis extra, not a runner dependency)

    expected = results_columns(yaml.safe_load(Path(sweep_path).read_text(encoding="utf-8")) or {})
    frames = []
    for run_dir in sweep_run_dirs(sweep_path, runs_root):
        results_path = run_dir / RESULTS_FILE
        if not results_path.exists() or results_path.stat().st_size == 0:
            continue  # a run that has not written a row yet
        frame = pd.read_csv(results_path, dtype=RESULTS_DTYPES, keep_default_na=False)
        if list(frame.columns) != list(expected):
            raise ValueError(f"{results_path}: its header is not the sweep's results columns; the schema changed since "
                             "this run")
        frame["run_dir"] = str(run_dir)
        frames.append(frame)
    columns = [*expected, "run_dir"]
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype=RESULTS_DTYPES.get(c, str)) for c in columns})
    df = pd.concat(frames, ignore_index=True)
    repeated = df.duplicated(list(RESULTS_KEY), keep=False)
    if repeated.any():
        first = df.loc[repeated, list(RESULTS_KEY)].iloc[0]
        folders = sorted({Path(d).name for d in df.loc[repeated, "run_dir"]})
        raise ValueError(
            f"repeated key (reader, page_id, condition_hash, repeat) = {tuple(first.tolist())} in {folders}: the same row "
            "was written twice (section 8 of PREREGISTRATION.md); resume a run, never start it again"
        )
    return df.sort_values(["reader", "page_id", "block", "condition_hash"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- the run folders of a sweep (sections 3.2, 3.3)

Z_PAGE = re.compile(r"_z(?P<size>[0-9.]+)_(?P<mode>conv|pa)_p\d{4}$")  # <book>_z<size>_<conv|pa>_p<NNNN>
_Z_MODES = {tag: mode for mode, tag in Z_MODE_TAGS.items()}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Every row of a small CSV as strings; an empty list without the file."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class RunFolder:
    """One run folder: its manifest, the conditions its reader was to run, and its coherence and determinism rows.

    The plan is rebuilt from the manifest's ``config.sweep`` and ``config.reader`` as the runner froze them, so a
    page's planned rows are what that run set out to write. ``scripts/20_analyze.py`` and
    ``scripts/30_make_figures.py`` both read run folders through this class, so the tables and the figures drop the
    same pages.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.manifest = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        cfg = self.manifest["config"]
        self.reader_cfg: Mapping[str, Any] = cfg["reader"]
        self.reader = str(self.reader_cfg["name"])
        self.sweep: Mapping[str, Any] = cfg["sweep"]
        self.blocks: dict[str, Mapping[str, Any]] = {str(k): v for k, v in self.sweep["blocks"].items()}
        self.repeats = int(cfg.get("repeats", 1))
        self.rows_planned = int(cfg.get("rows_planned", 0))
        self.conditions = expand_conditions(self.sweep, self.reader_cfg)
        self.hashes_by_key: dict[str, list[tuple[str, tuple[str, ...]]]] = {}  # (hash, text types) by pagination key
        for c in self.conditions:
            self.hashes_by_key.setdefault(pagination_key(c), []).append((condition_hash(c), c.text_types))
        self.coherence = _read_csv_rows(self.run_dir / COHERENCE_FILE)
        self.determinism = _read_csv_rows(self.run_dir / DETERMINISM_FILE)

    @staticmethod
    def page_key(page_id: str) -> str:
        """The pagination key a page id belongs to: ``shared`` or ``z:<render_mode>:<size>`` (``pagination_key``)."""
        m = Z_PAGE.search(page_id)
        if m is None:
            return SHARED
        return f"z:{_Z_MODES[m['mode']]}:{float(m['size']):g}"

    def planned_rows(self, page_id: str, text_type: str) -> set[tuple[str, int]]:
        """The (condition hash, repeat) pairs the runner was to write for this page, whose text is ``text_type``.

        A condition whose block names ``text_types`` without it reads none of that type's pages.
        """
        return {(h, r) for h, kinds in self.hashes_by_key.get(self.page_key(page_id), []) if text_type in kinds
                for r in range(self.repeats)}

    def reference_symbols(self, page_id: str) -> float:
        """The page's reference length without whitespace, from references/<page_id>.txt; NaN without the file."""
        path = self.run_dir / "references" / f"{page_id}.txt"
        if not path.exists():
            return float("nan")
        return float(len("".join(path.read_text(encoding="utf-8").split())))


def excluded_pages(df, folders: Sequence[RunFolder]) -> dict[str, set[str]]:
    """Section 8 of PREREGISTRATION.md: the pages of each run folder that miss a planned (condition, repeat) row.

    Keyed by run folder path, as the ``run_dir`` column of ``load_results`` writes it. Nothing else is ever excluded.
    """
    out: dict[str, set[str]] = {}
    for folder in folders:
        rows = df[df["run_dir"] == str(folder.run_dir)]
        missing: set[str] = set()
        for page_id, group in rows.groupby("page_id", sort=True):
            done = set(zip(group["condition_hash"].tolist(), (int(r) for r in group["repeat"].tolist())))
            if folder.planned_rows(str(page_id), str(group["text_type"].iloc[0])) - done:
                missing.add(str(page_id))
        out[str(folder.run_dir)] = missing
    return out


def complete_rows(df, folders: Sequence[RunFolder]) -> tuple[Any, dict[str, set[str]]]:
    """``df`` without the rows of the pages ``excluded_pages`` names, and those pages by run folder path."""
    excluded = excluded_pages(df, folders)
    keep = np.ones(len(df), dtype=bool)
    for run_dir, pages in excluded.items():
        if pages:
            keep &= ~((df["run_dir"] == run_dir) & df["page_id"].isin(pages)).to_numpy()
    return df[keep].reset_index(drop=True), excluded
