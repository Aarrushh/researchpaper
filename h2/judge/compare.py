# PURPOSE: Judge an H2 kernel against the reference renderer.
# WHAT GOES HERE: Pixel-level comparison metrics between two rendered images, and the preregistered tolerance.
# FILLED IN AT STEP: W4.4.
# DO NOT: Loosen the tolerance after seeing results. Do not modify kernels to make them pass.
"""Step W4.4: compare kernel output with the reference renderer.

Learn A11 (Triton) before implementing.
"""

from __future__ import annotations

import numpy as np

# Preregistered at W2.1 and copied into judge.yaml; never changed after attempt 1.
TOLERANCE = {"binary": (0, 0.0), "coverage_srgb": (0, 0.0), "coverage_linear": (1, 1e-3)}


def compare(kernel_output: np.ndarray, reference_output: np.ndarray) -> dict:
    """Compare two rendered images of the same shape.

    Args:
        kernel_output: uint8 array (H, W, 3) from the H2 kernel.
        reference_output: uint8 array (H, W, 3) from the reference renderer.

    Returns:
        Dict with keys "max_abs_error" (int, 0 to 255) and
        "frac_pixels_differing" (float, 0 to 1).

    Raises:
        ValueError: if the shapes or dtypes differ, or the reference is not (H, W, 3).
    """
    if reference_output.ndim != 3 or reference_output.shape[-1] != 3:
        raise ValueError(f"reference shape {reference_output.shape}; expected (H, W, 3)")
    if kernel_output.shape != reference_output.shape:
        raise ValueError(f"kernel shape {kernel_output.shape}, reference shape {reference_output.shape}")
    if kernel_output.dtype != np.uint8 or reference_output.dtype != np.uint8:
        raise ValueError(f"dtypes {kernel_output.dtype} and {reference_output.dtype}; both must be uint8")
    diff = np.abs(kernel_output.astype(np.int16) - reference_output.astype(np.int16))   # never subtract uint8
    return {"max_abs_error": int(diff.max()),
            "frac_pixels_differing": float((diff > 0).any(axis=-1).mean())}


def page_passes(result: dict, aa_mode: str) -> bool:
    max_error, max_frac = TOLERANCE[aa_mode]
    return result["max_abs_error"] <= max_error and result["frac_pixels_differing"] <= max_frac
