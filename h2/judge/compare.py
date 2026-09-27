# PURPOSE: Judge an H2 kernel against the reference renderer.
# WHAT GOES HERE: Pixel-level comparison metrics between two rendered images.
# FILLED IN AT STEP: W4.4.
# DO NOT: Loosen the tolerance after seeing results. Do not modify kernels to make them pass.
"""Step W4.4: compare kernel output with the reference renderer.

Learn A8 (Triton) before implementing.
"""

from __future__ import annotations

import numpy as np


def compare(kernel_output: np.ndarray, reference_output: np.ndarray) -> dict:
    """Compare two rendered images of the same shape.

    Args:
        kernel_output: uint8 array (H, W, 3) from the H2 kernel.
        reference_output: uint8 array (H, W, 3) from mrr.render.render_page.

    Returns:
        Dict with keys "max_abs_error" (int, 0 to 255) and
        "frac_pixels_differing" (float, 0 to 1).
    """
    raise NotImplementedError("TODO[W4.4]: check shapes match, then compute max abs error and fraction of differing pixels")
