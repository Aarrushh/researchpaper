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

import dataclasses
import itertools
import math
import numbers
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping

import yaml

from mrr import grid, render, text_source

LINE_PITCH_MODES = ("aligned", "unaligned")
PAGINATION_MODES = ("shared", "per_condition")
AXES = ("x", "y")
# Every key a block may hold. Anything else is a typo, and a typo must not silently become a default.
BLOCK_KEYS = frozenset(
    {"render_mode", "line_pitch_mode", "axis", "range", "step_px", "phases_px", "aa_mode", "font_size_px", "font_file", "paginate"}
)
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


def expand_conditions(sweep: Mapping[str, Any], reader_cfg: Mapping[str, Any]) -> list[Condition]:
    """Return the conditions of every block of ``sweep`` for one reader, blocks in file order (section 2.3).

    Each block starts from the sweep's top-level ``font_file``, this reader's ``font_size_px`` and ``aa_mode``, and
    its own keys override them. Its phase levels come from ``grid.phase_levels`` on its axis (the other axis is 0),
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
                settings[key] = block[key]
        for key in ("render_mode", "line_pitch_mode", "font_file", "font_size_px", "aa_mode"):
            if settings.get(key) is None:
                raise ValueError(f"{label}: {key} is missing; set it in the block or at the top of the sweep")
        axis = _choice(block.get("axis", "x"), AXES, "axis", label)
        paginate = _choice(block.get("paginate", "shared"), PAGINATION_MODES, "paginate", label)
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
                    )
                )
    return conditions


# Pages (section 2.4; the block Z pages of section 3.5).

SHARED = "shared"  # the pagination key of the pages every shared block reads
NATURAL, RANDOM62 = "natural", "random62"  # the two text types: the keys under pages: in a sweep config
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
    candidates = [sweep.get("font_file")] + [b.get("font_file") for b in blocks.values() if isinstance(b, Mapping)]
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
) -> dict[str, Any]:
    """Return the render config that lays out this condition's pages at phase 0 under ``allowances``."""
    max_x, max_y = allowances
    return grid.resolve_render_cfg(
        render_cfg,
        reader_cfg,
        font_file=font_paths[condition.font_file],
        font_size_px=condition.font_size_px,
        line_pitch_px=condition.line_pitch_px,
        render_mode=condition.render_mode,
        aa_mode=condition.aa_mode,
        phase_x_px=0,
        phase_y_px=0,
        max_phase_x_px=max_x,
        max_phase_y_px=max_y,
    )


def _count(value: Any, label: str, reader: str) -> int:
    value = _per_reader(value, reader, label)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{label} must be a whole number of pages, 1 or more, not {value!r}")
    return int(value)


def _page_counts(pages_cfg: Any, reader: str) -> tuple[dict[str, int], dict[str, int]]:
    """Return the shared and the block Z page counts by text type (section 2.4, step 7 on placeholders).

    The shared counts are ``pages.<text type>.per_text_type``; the Z counts are ``pages.z_per_text_type`` for both
    text types, or the shared counts when that key is absent (the ladder). Any may be a map by reader. A string, a
    fraction or zero raises ValueError naming the key.
    """
    if not isinstance(pages_cfg, Mapping):
        raise ValueError("pages must be a mapping with natural and random62 (section 2.7)")
    shared = {}
    for kind in (NATURAL, RANDOM62):
        label = f"pages.{kind}.per_text_type"
        section = pages_cfg.get(kind)
        if not isinstance(section, Mapping) or "per_text_type" not in section:
            raise ValueError(f"{label} is missing")
        shared[kind] = _count(section["per_text_type"], label, reader)
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
    holds for each distinct shared layout (in the pilot: conventional unaligned for C and A, conventional aligned for
    V, patch_aware unaligned for H and S) with the allowances of ``phase_allowances``. Natural pages come from
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
        allowances = phase_allowances(shared)
        layouts: dict[tuple, dict[str, Any]] = {}
        for c in shared:
            if _layout_id(c) not in layouts:
                layouts[_layout_id(c)] = _layout_cfg(c, reader_cfg, render_cfg, font_paths, allowances)
        cfgs = list(layouts.values())

        def fits(page: str) -> bool:
            return all(render.fits_page(page, cfg) for cfg in cfgs)

        by_type: dict[str, list[Page]] = {}
        for kind in (NATURAL, RANDOM62):
            by_type[kind] = []
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
