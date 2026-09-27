# PURPOSE: Geometry of the encoder's token grid.
# WHAT GOES HERE: Cell size lookup, phase offsets, snapping, native (no-resize) page sizes.
# FILLED IN AT STEP: W1.5 (phase_offsets, snap_to_grid), W1.6 (token_cell_px, native_size).
# DO NOT: Hard-code a cell size; always read it from the reader config.
"""Token grid geometry.

Learn A1 (ViT patches) and A2 (Qwen resize and tiling) before implementing.
"""

from __future__ import annotations

from typing import Any, Mapping


def token_cell_px(reader_cfg: Mapping[str, Any]) -> int:
    """Return the side length in pixels of one vision token's cell.

    For Qwen2.5-VL this is patch_size * merge_size = 14 * 2 = 28.

    Args:
        reader_cfg: one reader block from configs/readers.yaml.

    Returns:
        Cell side length in pixels.
    """
    raise NotImplementedError("TODO[W1.6]: derive cell size from patch_size and merge_size and check it equals token_cell_px")


def phase_offsets(cell_px: int, n_steps: int) -> list[float]:
    """Return evenly spaced phase offsets covering one cell, excluding the endpoint.

    Args:
        cell_px: cell side length in pixels.
        n_steps: number of phases.

    Returns:
        List of n_steps offsets in pixels, in [0, cell_px).
    """
    raise NotImplementedError("TODO[W1.5]: return n_steps evenly spaced offsets in [0, cell_px)")


def snap_to_grid(x: float, cell_px: int) -> float:
    """Snap a coordinate down to the nearest cell boundary.

    Args:
        x: coordinate in pixels.
        cell_px: cell side length in pixels.

    Returns:
        The largest multiple of cell_px that is less than or equal to x.
    """
    raise NotImplementedError("TODO[W1.5]: snap x down to a multiple of cell_px")


def native_size(width: int, height: int, cell_px: int) -> tuple[int, int]:
    """Return the nearest page size the processor will not resize.

    Args:
        width: desired width in pixels.
        height: desired height in pixels.
        cell_px: cell side length in pixels.

    Returns:
        (width, height), each a multiple of cell_px and within the processor's pixel limits.
    """
    raise NotImplementedError("TODO[W1.6]: round to multiples of cell_px and respect the processor min/max pixel limits")
