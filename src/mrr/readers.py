# PURPOSE: Wrappers around the VLM readers.
# WHAT GOES HERE: Model loading, vision token counting, transcription.
# FILLED IN AT STEP: W1.6.
# DO NOT: Import torch or transformers at module top level; CI has no GPU. Import inside functions.
"""VLM readers.

Learn A2 (Qwen resize and tiling) and B1 (readers) before implementing.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def load_reader(cfg: Mapping[str, Any]) -> Any:
    """Load a reader model and its processor.

    Args:
        cfg: one reader block from configs/readers.yaml.

    Returns:
        An object holding the model, processor and cfg.
    """
    raise NotImplementedError("TODO[W1.6]: load model and processor at the pinned revision in bf16")


def count_vision_tokens(reader: Any, image: np.ndarray) -> int:
    """Return the number of vision tokens the processor produces for this image.

    Args:
        reader: object returned by load_reader.
        image: uint8 array (H, W, 3).

    Returns:
        Vision token count as reported by the processor.
    """
    raise NotImplementedError("TODO[W1.6]: run the processor and read the image grid to count tokens")


def transcribe(reader: Any, image: np.ndarray, prompt: str) -> str:
    """Transcribe the text in one image.

    Args:
        reader: object returned by load_reader.
        image: uint8 array (H, W, 3).
        prompt: instruction from configs/readers.yaml.

    Returns:
        The model's raw text output.
    """
    raise NotImplementedError("TODO[W1.6]: build the chat input, generate greedily, decode only new tokens")
