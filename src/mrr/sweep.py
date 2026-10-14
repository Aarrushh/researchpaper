# PURPOSE: Turn a sweep config into conditions and pages, hash a condition, and run the sweep.
# WHAT GOES HERE: Condition and expand_conditions (section 2.3); Page and plan_pages (2.4); condition_hash (2.5); run_sweep and load_results (2.8).
# FILLED IN AT STEP: W2.2 (conditions, pages), W2.3 (hash, runner).
# DO NOT: Load a model here. Accept a string where a number belongs: a placeholder must stop a run. Change the hash keys after the pilot without an amendment.
"""Sweep definitions: blocks to conditions (build guide section 2.3, D10).

A sweep config is a mapping of blocks. Each block names one layout (``render_mode`` and ``line_pitch_mode``) swept
along one axis over a range of phases at a step, or at an explicit list of phases. ``expand_conditions`` turns the
blocks of one reader into ``Condition`` records in a fixed order. The ten fields the condition hash of section 2.5
reads are the ten keys of that hash; ``paginate`` says whether the block reads the reader's shared pages or its own.
"""

from __future__ import annotations

import dataclasses
import itertools
import numbers
from pathlib import PureWindowsPath
from typing import Any, Mapping

from mrr import grid, render

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
