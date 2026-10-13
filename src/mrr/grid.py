# PURPOSE: Geometry of the encoder's token grid.
# WHAT GOES HERE: Each reader's periods, token count, blind strip and native page (D7, D8, D9); phase splitting, snapping and the per-reader render config.
# FILLED IN AT STEP: W1.5 (phase_offsets, snap_to_grid, split_phase_64), W1.6 (reader geometry, page_size_px, resolve_render_cfg).
# DO NOT: Hard-code a cell size; always read it from the reader config.
"""Token grid geometry.

Learn A1 (ViT patches), A2 (each reader's patch, merge and window) and A7 (why the sweep covers the
longest period) before implementing.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping


def token_cell_px(reader_cfg: Mapping[str, Any]) -> int:
    """Return the side length in pixels of one vision token's cell.

    The cell is patch_size * merge_size: 14 * 2 = 28 for Qwen2.5-VL, 14 * 3 = 42 for SmolVLM-2.2B and
    16 * 4 = 64 for SmolVLM-500M. merge_size means the spatial merge for Qwen and the pixel-shuffle
    factor (scale_factor in the Hub config) for SmolVLM; the key keeps one name for both.

    Args:
        reader_cfg: one reader block from configs/readers.yaml.

    Returns:
        Cell side length in pixels.

    Raises:
        ValueError: if patch_size * merge_size differs from the block's token_cell_px.
    """
    cell = int(reader_cfg["patch_size"]) * int(reader_cfg["merge_size"])
    if cell != int(reader_cfg["token_cell_px"]):
        raise ValueError(
            f"{reader_cfg.get('name')}: patch_size * merge_size = {cell}, "
            f"but token_cell_px = {reader_cfg['token_cell_px']}"
        )
    return cell


def sweep_period_px(reader_cfg: Mapping[str, Any]) -> int:
    """Return the longest period of the reader's grid, which a phase sweep must cover (D9).

    The least common multiple of the patch, the token cell and, for a reader with attention windows,
    the window: 112 px for Qwen, 42 px for SmolVLM-2.2B and 64 px for SmolVLM-500M.

    Args:
        reader_cfg: one reader block from configs/readers.yaml.

    Returns:
        Period in pixels.
    """
    periods = [int(reader_cfg["patch_size"]), token_cell_px(reader_cfg)]
    if "window_px" in reader_cfg:
        periods.append(int(reader_cfg["window_px"]))
    return math.lcm(*periods)


def token_count(reader_cfg: Mapping[str, Any], height: int, width: int) -> int:
    """Return how many vision tokens the encoder makes from a page of height x width pixels.

    The patch embedding drops the patches that do not fit, and each token merges m x m patches:
    ((h // P) // m) * ((w // P) // m). The scaffold's (h / cell) * (w / cell) gives 83.6 for SmolVLM-2.2B's
    384 px frame, where the encoder makes 81.

    Args:
        reader_cfg: one reader block from configs/readers.yaml.
        height: page height in pixels.
        width: page width in pixels.

    Returns:
        Number of vision tokens.
    """
    patch = int(reader_cfg["patch_size"])
    merge = int(reader_cfg["merge_size"])
    return ((height // patch) // merge) * ((width // patch) // merge)


def blind_strip_px(reader_cfg: Mapping[str, Any]) -> int:
    """Return the width of the strip at the right and bottom of the frame that no patch covers.

    frame - P * (frame // P), with frame = page_px[0]: 0 for Qwen, 6 for SmolVLM-2.2B (384 = 27 * 14 + 6)
    and 0 for SmolVLM-500M. Ink in that strip never reaches the encoder (D8).

    Args:
        reader_cfg: one reader block from configs/readers.yaml.

    Returns:
        Strip width in pixels.

    Raises:
        ValueError: if the block's blind_from_px is not where the patches end.
    """
    frame = int(reader_cfg["page_px"][0])
    patch = int(reader_cfg["patch_size"])
    covered = patch * (frame // patch)
    if "blind_from_px" in reader_cfg and int(reader_cfg["blind_from_px"]) != covered:
        raise ValueError(
            f"{reader_cfg.get('name')}: the patches end at {covered} px, "
            f"but blind_from_px = {reader_cfg['blind_from_px']}"
        )
    return frame - covered


def page_size_px(reader_cfg: Mapping[str, Any]) -> tuple[int, int]:
    """Return the reader's native page, after the checks against a resize that its config allows (D8).

    A reader with min_pixels and max_pixels (Qwen) resizes every image to a fixed point of smart_resize,
    so its page must be one: 448 stays 448, but 384 would become 392 and 512 would become 504. A reader
    without them (SmolVLM) resizes every image to its square frame, so its page must be square; whether
    the side equals the frame is not checked here. The no-resize test (W1.8) checks both pages against the
    real processors, and readers.processor_inputs checks every page it is given.

    Args:
        reader_cfg: one reader block from configs/readers.yaml.

    Returns:
        (height, width) in pixels, from page_px.

    Raises:
        ValueError: if smart_resize would resize a Qwen page, or a SmolVLM page is not square.
    """
    height, width = (int(side) for side in reader_cfg["page_px"])
    if "min_pixels" in reader_cfg:
        resized = qwen_smart_resize(
            height, width, token_cell_px(reader_cfg), int(reader_cfg["min_pixels"]), int(reader_cfg["max_pixels"])
        )
        if resized != (height, width):
            raise ValueError(
                f"{reader_cfg.get('name')}: smart_resize turns a {height} x {width} page into "
                f"{resized[0]} x {resized[1]}"
            )
    elif height != width:
        raise ValueError(
            f"{reader_cfg.get('name')}: a {height} x {width} page is not square, so the processor resizes it"
        )
    return height, width


def qwen_smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 3136, max_pixels: int = 12845056
) -> tuple[int, int]:
    """Return the (height, width) to which Qwen2.5-VL's image processor resizes an image.

    A copy of smart_resize in transformers 5.19.0 (models/qwen2_vl/image_processing_qwen2_vl.py), Python's
    round included, which takes halves to even: 70 px becomes 56. The defaults are the min_pixels and
    max_pixels of the pinned Qwen2.5-VL-3B-Instruct snapshot. This replicates the processor and is not
    part of the renderer, so the renderer's rule of rounding halves up does not apply.

    Args:
        height: image height in pixels.
        width: image width in pixels.
        factor: patch_size * merge_size.
        min_pixels: the processor's smallest pixel count.
        max_pixels: the processor's largest pixel count.

    Returns:
        (height, width) after resizing.

    Raises:
        ValueError: if the aspect ratio exceeds 200, as in transformers.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def split_phase_64(phase_px: float) -> tuple[int, int]:
    """Split a phase into whole pixels and 64ths of a pixel.

    X = floor(64 * phase + 0.5), which rounds halves up like every rounding in the layout; the result is
    (X >> 6, X & 63). 3.25 gives (3, 16); 13.0078125 gives (13, 1), where rounding halves to even would
    give (13, 0).

    Args:
        phase_px: phase in pixels.

    Returns:
        (whole pixels, 64ths of a pixel).
    """
    x_64 = math.floor(64 * phase_px + 0.5)
    return x_64 >> 6, x_64 & 63


def phase_offsets(cell_px: int, n_steps: int) -> list[float]:
    """Return evenly spaced phase offsets covering one cell, excluding the endpoint.

    k * cell_px / n_steps for k = 0 to n_steps - 1, for the render check (scripts/02_render_check.py).

    Args:
        cell_px: cell side length in pixels.
        n_steps: number of phases.

    Returns:
        List of n_steps offsets in pixels, in [0, cell_px).

    Raises:
        ValueError: if n_steps is less than 1.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    return [k * cell_px / n_steps for k in range(n_steps)]


def snap_to_grid(x: float, cell_px: int) -> int:
    """Snap a coordinate up to the nearest cell boundary.

    Returns the smallest multiple of cell_px at or above x, as section 1A.7 of the build guide specifies:
    30 with a 28 px cell gives 56, and 28 gives 28.

    Args:
        x: coordinate in pixels.
        cell_px: cell side length in pixels.

    Returns:
        The smallest multiple of cell_px that is greater than or equal to x.
    """
    return cell_px * math.ceil(x / cell_px)


def resolve_render_cfg(
    render_defaults: Mapping[str, Any], reader_cfg: Mapping[str, Any], **overrides: Any
) -> dict[str, Any]:
    """Return the render config for one reader: the defaults, then the reader's geometry, then overrides.

    page_height_px and page_width_px come from page_size_px, and margin_px is the patch size (14, 14 and
    16 px). max_phase_x_px and max_phase_y_px become sweep_period_px - 1, and patch_aware_slot_px the
    patch size, wherever the defaults leave them null. blind_strip_px is added. The defaults are not
    changed.

    Args:
        render_defaults: the mapping from configs/render_defaults.yaml.
        reader_cfg: one reader block from configs/readers.yaml.
        **overrides: keys to set last, such as a sweep's phase allowances.

    Returns:
        A new dict.

    Raises:
        KeyError: if an override names a key the resolved config does not have.
    """
    cfg = copy.deepcopy(dict(render_defaults))
    patch = int(reader_cfg["patch_size"])
    cfg["page_height_px"], cfg["page_width_px"] = page_size_px(reader_cfg)
    cfg["margin_px"] = patch
    for key in ("max_phase_x_px", "max_phase_y_px"):
        if cfg.get(key) is None:
            cfg[key] = sweep_period_px(reader_cfg) - 1
    if cfg.get("patch_aware_slot_px") is None:
        cfg["patch_aware_slot_px"] = patch
    cfg["blind_strip_px"] = blind_strip_px(reader_cfg)
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise KeyError(f"unknown render config keys: {unknown}")
    cfg.update(overrides)
    return cfg
