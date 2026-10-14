# PURPOSE: Wrappers around the VLM readers.
# WHAT GOES HERE: Model loading at the pinned revision, the SmolVLM position-id fix, vision token counting, greedy transcription with input checks (D15; build guide 1B.4-1B.5).
# FILLED IN AT STEP: W1.6.
# DO NOT: Import torch or transformers at module top level; CI has no GPU. Import inside functions.
"""VLM readers.

Learn A2 (Qwen resize and tiling) and B1 (readers) before implementing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image

from mrr import grid

# The Hub names each reader's processor; the guide names the model class loaded with it (build guide 1B.3).
MODEL_CLASSES = {
    "Qwen2_5_VLProcessor": "Qwen2_5_VLForConditionalGeneration",
    "Idefics3Processor": "Idefics3ForConditionalGeneration",
}
# Passed on every call: the Hub generation settings differ by reader, and Qwen's sample.
GREEDY = {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0}


@dataclass(frozen=True)
class Transcription:
    """One page read by one reader."""

    text: str
    new_tokens: int  # including the end token, when there is one
    hit_limit: bool  # stopped at max_new_tokens, not at an end token
    latency_s: float  # generate only, between two CUDA synchronisations


@dataclass
class Reader:
    """A loaded reader: model and processor at the pinned revision, its config and its image token id."""

    name: str
    model: Any
    processor: Any
    cfg: Mapping[str, Any]
    image_token_id: int


def load_reader(cfg: Mapping[str, Any]) -> Reader:
    """Load a reader model and its processor.

    The model is loaded at the pinned revision in the config's dtype, with SDPA attention, and put in
    eval mode. For the SmolVLM readers the position-id fix goes on before anything else can attach a
    hook (D15). A SmolVLM block must name its position_id_fix, pre_hook or null, so that a block
    without the key cannot load without the fix unnoticed.

    Args:
        cfg: one reader block from configs/readers.yaml.

    Returns:
        The model, processor, cfg and image token id.
    """
    import torch
    import transformers

    if not re.fullmatch(r"[0-9a-f]{40}", str(cfg["revision"])):
        raise ValueError(f"{cfg['name']}: revision must be a 40-character commit hash, got {cfg['revision']!r}")
    dtype = getattr(torch, str(cfg["dtype"]), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{cfg['name']}: unknown dtype {cfg['dtype']!r}")
    processor = transformers.AutoProcessor.from_pretrained(cfg["hf_id"], revision=cfg["revision"])
    model_class = MODEL_CLASSES.get(type(processor).__name__)
    if model_class is None:
        raise ValueError(f"{cfg['name']}: no model class for {type(processor).__name__}")
    if type(processor).__name__ == "Idefics3Processor" and "position_id_fix" not in cfg:
        raise ValueError(f"{cfg['name']}: a SmolVLM block needs position_id_fix: pre_hook, or null to read without it")
    placement = {"max_memory": cfg["max_memory"]} if "max_memory" in cfg else {}  # the fallback of 1B.8 (D18)
    model = getattr(transformers, model_class).from_pretrained(
        cfg["hf_id"],
        revision=cfg["revision"],
        dtype=dtype,
        device_map=cfg["device_map"],
        attn_implementation="sdpa",
        **placement,
    ).eval()

    fix = cfg.get("position_id_fix")
    if fix == "pre_hook":
        install_position_id_fix(model)
    elif fix is not None:
        raise ValueError(f"{cfg['name']}: unknown position_id_fix {fix!r}")

    # An unknown token would map to <unk> and every count would be wrong.
    image_token_id = processor.tokenizer.convert_tokens_to_ids(cfg["image_token"])
    if image_token_id != processor.image_token_id:
        raise ValueError(
            f"{cfg['name']}: {cfg['image_token']!r} has id {image_token_id}, "
            f"but the processor expands {processor.image_token_id}"
        )
    return Reader(name=cfg["name"], model=model, processor=processor, cfg=cfg, image_token_id=image_token_id)


def install_position_id_fix(model: Any) -> None:
    """Give the patches of a full SmolVLM frame the position ids 0 to n - 1, in raster order (D15).

    The pinned Idefics3VisionEmbeddings casts each patch's coordinate k/27 to the pixel dtype before
    bucketing it. In bfloat16 neighbouring patches land in the same bucket, and a 384 px frame gets 441
    distinct ids instead of 729. A forward pre-hook on the position table replaces the ids of a full
    frame and leaves any other input alone, so it is right only for a full frame without padding;
    processor_inputs checks that. The hook sees only the ids, so it takes a full frame to be n ids of
    which exactly one is 0 in each image: padded patches get id 0, and a frame that is not square puts
    several patches in bucket 0. Pre-hooks run in registration order, so install this one before any
    hook that records ids.

    Args:
        model: an Idefics3ForConditionalGeneration.
    """
    import torch

    emb = model.model.vision_model.embeddings.position_embedding  # nn.Embedding
    n = emb.num_embeddings  # 729 for SmolVLM-Instruct, 1024 for the 500M

    def fix_ids(module, args):
        (ids,) = args
        if ids.shape[-1] == n and bool((ids == 0).sum(-1).eq(1).all()):  # one full frame: ids must be 0..n-1
            return (torch.arange(n, device=ids.device).expand_as(ids),)
        return None  # anything else is left alone

    emb.register_forward_pre_hook(fix_ids)


def count_vision_tokens(reader: Reader, image: np.ndarray) -> int:
    """Return the number of vision tokens the language model receives for this image.

    The inputs are built exactly as transcribe_full builds them, with the reader's prompt, and the
    image tokens in input_ids are counted: 256, 81 and 64 on the native pages.

    Args:
        reader: object returned by load_reader.
        image: uint8 array (H, W, 3).

    Returns:
        Vision token count as reported by the processor.
    """
    inputs = processor_inputs(reader, image, reader.cfg["prompt"])
    return int((inputs["input_ids"] == reader.image_token_id).sum())


def transcribe_full(
    reader: Reader, image: np.ndarray, prompt: str, max_new_tokens: int | None = None
) -> Transcription:
    """Transcribe one page greedily and say how long it took and whether the limit cut it off.

    One page per call: the tokenizers pad on the right, and a page's transcription should not depend
    on the pages batched with it.

    Args:
        reader: object returned by load_reader.
        image: uint8 array (H, W, 3).
        prompt: instruction from configs/readers.yaml.
        max_new_tokens: generation limit; None takes the reader's config value (set at W1.7).

    Returns:
        The decoded new tokens, their number, whether generation hit the limit, and the latency.
    """
    import torch

    limit = reader.cfg["max_new_tokens"] if max_new_tokens is None else max_new_tokens
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError(f"{reader.name}: max_new_tokens must be a positive integer, got {limit!r}")
    inputs = processor_inputs(reader, image, prompt).to(reader.model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        torch.cuda.synchronize()
        start = time.perf_counter()
        # Qwen gets the whole processor output: generation needs its mm_token_type_ids.
        out = reader.model.generate(**inputs, **GREEDY, max_new_tokens=limit)
        torch.cuda.synchronize()
        latency_s = time.perf_counter() - start
    new = out[:, input_len:]
    text = reader.processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    eos = reader.model.generation_config.eos_token_id
    ends = {eos} if isinstance(eos, int) else set(eos or ())
    n = int(new.shape[1])
    hit_limit = n == limit and int(new[0, -1]) not in ends
    return Transcription(text=text, new_tokens=n, hit_limit=hit_limit, latency_s=latency_s)


def transcribe(reader: Reader, image: np.ndarray, prompt: str) -> str:
    """Transcribe the text in one image, with the limit of configs/readers.yaml.

    The stub of the scaffold, kept with its signature (build guide 1B.4). It drops new_tokens, hit_limit
    and latency_s, so the sweep runner calls transcribe_full with a per-page limit instead.

    Args:
        reader: object returned by load_reader.
        image: uint8 array (H, W, 3).
        prompt: instruction from configs/readers.yaml.

    Returns:
        The model's raw text output.
    """
    return transcribe_full(reader, image, prompt).text


def processor_inputs(reader: Reader, image: np.ndarray, prompt: str) -> Any:
    """Build the processor's inputs for one page and prompt, as every call in this module builds them.

    The prompt goes into the reader's chat template after the image, and the processor runs with
    do_resize=False. The result is checked before it is returned: the encoder must get the page itself.

    Args:
        reader: object returned by load_reader; only its name, processor, cfg and image token id are used.
        image: uint8 array (H, W, 3).
        prompt: instruction from configs/readers.yaml.

    Returns:
        The processor's output on the CPU: input_ids, attention_mask and the vision inputs.

    Raises:
        ValueError: if the image is not a uint8 array of shape (H, W, 3).
        RuntimeError: if the vision input is not the page itself (see _check_vision_input), or if a Qwen
            page is not whole token cells, which Qwen's processor cannot cut into merge groups.
    """
    if not (isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3):
        got = f"{image.dtype} {image.shape}" if isinstance(image, np.ndarray) else type(image).__name__
        raise ValueError(f"{reader.name}: expected a uint8 page of shape (H, W, 3), got {got}")
    height, width = image.shape[:2]
    if "min_pixels" in reader.cfg:  # Qwen, the reader with smart_resize bounds (as in grid.page_size_px)
        cell = grid.token_cell_px(reader.cfg)
        if height % cell or width % cell:  # otherwise the processor fails with a bare reshape error
            raise RuntimeError(f"{reader.name}: a {height}x{width} page is not whole {cell} px token cells")
    pil = Image.fromarray(image)  # Pillow for input and output only
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = reader.processor.apply_chat_template(messages, add_generation_prompt=True)
    # Keywords only: the processor's first positional argument is images.
    inputs = reader.processor(text=[text], images=[pil], return_tensors="pt", do_resize=False)
    _check_vision_input(reader, inputs, height, width)
    return inputs


def _check_vision_input(reader: Reader, inputs: Mapping[str, Any], height: int, width: int) -> None:
    """Raise unless the encoder gets the page itself: not resized, not tiled, not padded (D8, D15).

    Qwen's image_grid_thw must be [[1, H/P, W/P]]. SmolVLM's pixel_values must be one frame,
    (1, 1, 3, H, W), with an all-true pixel_attention_mask, which the position-id fix relies on. For
    both, input_ids must hold grid.token_count image tokens.
    """
    if "image_grid_thw" in inputs:
        patch = int(reader.cfg["patch_size"])
        want, got = [[1, height // patch, width // patch]], inputs["image_grid_thw"].tolist()
        if got != want:
            raise RuntimeError(f"{reader.name}: image_grid_thw {got} for a {height}x{width} page; want {want}")
    else:
        want, got = (1, 1, 3, height, width), tuple(inputs["pixel_values"].shape)
        if got != want or not bool(inputs["pixel_attention_mask"].all()):
            raise RuntimeError(
                f"{reader.name}: pixel_values {got} for a {height}x{width} page; want {want}, unpadded"
            )
    tokens = int((inputs["input_ids"] == reader.image_token_id).sum())
    expected = grid.token_count(reader.cfg, height, width)
    if tokens != expected:
        raise RuntimeError(f"{reader.name}: {tokens} image tokens for a {height}x{width} page; want {expected}")
