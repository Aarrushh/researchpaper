# PURPOSE: Wrappers around the VLM readers.
# WHAT GOES HERE: Model loading at the pinned revision, the SmolVLM position-id fix, vision token counting, greedy transcription with input checks (D15; build guide 1B.4-1B.5); each reader's vision tower alone in float32 and its hidden states in raster order, for the shift-equivariance error (build guide 2.1).
# FILLED IN AT STEP: W1.6; W1.9 (vision towers).
# DO NOT: Import torch or transformers at module top level; CI has no GPU. Import inside functions.
"""VLM readers.

Learn A2 (Qwen resize and tiling) and B1 (readers) before implementing; A7 (shift variance) for the towers.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from mrr import grid

# The Hub names each reader's processor; the guide names the model class loaded with it (build guide 1B.3).
MODEL_CLASSES = {
    "Qwen2_5_VLProcessor": "Qwen2_5_VLForConditionalGeneration",
    "Idefics3Processor": "Idefics3ForConditionalGeneration",
    "InternVLProcessor": "InternVLForConditionalGeneration",  # Study 2's validation reader (PREREGISTRATION_2.md section 2)
}
# A reader block may ask for 4-bit weights with quantization: nf4 (Study 2, the 7B validation reader on an 8 GiB card).
QUANTIZATIONS = ("nf4",)
# Passed on every call: the Hub generation settings differ by reader, and Qwen's sample.
GREEDY = {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0}
# The vision towers' depth and patch-embedding weight shape [Verified in the notes, weights record]; build guide 2.1.
TOWER_FACTS = {
    "qwen": (32, (1280, 3, 2, 14, 14)),
    "smolvlm": (27, (1152, 3, 14, 14)),
    "smolvlm500m": (12, (768, 3, 16, 16)),
}


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


@dataclass(frozen=True)
class VisionTower:
    """A reader's vision encoder alone, in float32, with the reader's processor and config (build guide 2.1)."""

    name: str
    module: Any  # model.model.visual (Qwen) or model.model.vision_model (SmolVLM)
    processor: Any
    cfg: Mapping[str, Any]
    patch_px: int
    position_ids_fixed: bool = False  # True if load_vision_tower had to add the fix of section 1B.5 (D15)


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
    quantization = _quantization_config(cfg, dtype)
    if quantization is not None:
        placement["quantization_config"] = quantization
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


def _quantization_config(cfg: Mapping[str, Any], dtype: Any) -> Any:
    """The BitsAndBytesConfig a reader block asks for with ``quantization``, or None (Study 2, section 2).

    ``nf4``: 4-bit NF4 weights with double quantisation for the language model, computed in the block's dtype;
    the vision tower and the output head stay in that dtype (``llm_int8_skip_modules``), so the encoder the
    paper measures is the released one. Any other value is refused.
    """
    name = cfg.get("quantization")
    if name is None:
        return None
    if name not in QUANTIZATIONS:
        raise ValueError(f"{cfg['name']}: unknown quantization {name!r}; one of {QUANTIZATIONS} or absent")
    import transformers

    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
        llm_int8_skip_modules=["visual", "vision_model", "vision_tower", "lm_head"],
    )


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
    _fix_position_ids(model.model.vision_model.embeddings.position_embedding)


def _fix_position_ids(table: Any) -> None:
    """Register install_position_id_fix's pre-hook on one SmolVLM position table (an nn.Embedding)."""
    import torch

    n = table.num_embeddings  # 729 for SmolVLM-Instruct, 1024 for the 500M

    def fix_ids(module, args):
        (ids,) = args
        if ids.shape[-1] == n and bool((ids == 0).sum(-1).eq(1).all()):  # one full frame: ids must be 0..n-1
            return (torch.arange(n, device=ids.device).expand_as(ids),)
        return None  # anything else is left alone

    table.register_forward_pre_hook(fix_ids)


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
    elif "pixel_attention_mask" in inputs:  # SmolVLM: frames with a mask
        want, got = (1, 1, 3, height, width), tuple(inputs["pixel_values"].shape)
        if got != want or not bool(inputs["pixel_attention_mask"].all()):
            raise RuntimeError(
                f"{reader.name}: pixel_values {got} for a {height}x{width} page; want {want}, unpadded"
            )
    else:  # InternVL: tiles without a mask; one tile is the page itself (crop_to_patches off)
        want, got = (1, 3, height, width), tuple(inputs["pixel_values"].shape)
        if got != want:
            raise RuntimeError(f"{reader.name}: pixel_values {got} for a {height}x{width} page; want {want}, one tile")
    tokens = int((inputs["input_ids"] == reader.image_token_id).sum())
    expected = grid.token_count(reader.cfg, height, width)
    if tokens != expected:
        raise RuntimeError(f"{reader.name}: {tokens} image tokens for a {height}x{width} page; want {expected}")


# The vision towers alone, for the shift-equivariance error (build guide 2.1).


def load_vision_tower(cfg: Mapping[str, Any], device: str = "cuda") -> VisionTower:
    """Load a reader's vision tower alone, in float32, for the shift-equivariance error (build guide 2.1).

    The whole model is loaded on the CPU in bfloat16 at the pinned revision. Its vision tower is kept, cast
    to float32 and moved to the device, and the rest is freed, so the language model never reaches the GPU.
    The checkpoints store these weights in bfloat16, so the cast is exact. The depth and the patch-embedding
    weight are checked against TOWER_FACTS.

    For SmolVLM, D15 names a float32 tower as the alternative to the position-id fix. The ids that reach the
    position table are recorded on a blank page of the native size (they depend on the frame's size and
    dtype, not on its content); if they are not 0 to n - 1, the fix of install_position_id_fix goes on the
    tower too, and the ids are checked again.

    Args:
        cfg: one reader block from configs/readers.yaml.
        device: where the tower runs.

    Returns:
        The tower with the reader's name, processor, config and patch size.

    Raises:
        ValueError: if the revision or the processor is not one load_reader accepts.
        RuntimeError: if the tower's structure differs from TOWER_FACTS, or the ids stay wrong.
    """
    import gc

    import torch
    import transformers

    if not re.fullmatch(r"[0-9a-f]{40}", str(cfg["revision"])):
        raise ValueError(f"{cfg['name']}: revision must be a 40-character commit hash, got {cfg['revision']!r}")
    processor = transformers.AutoProcessor.from_pretrained(cfg["hf_id"], revision=cfg["revision"])
    model_class = MODEL_CLASSES.get(type(processor).__name__)
    if model_class is None:
        raise ValueError(f"{cfg['name']}: no model class for {type(processor).__name__}")
    model = getattr(transformers, model_class).from_pretrained(
        cfg["hf_id"], revision=cfg["revision"], dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa"
    ).eval()
    is_qwen = hasattr(model.model, "visual")
    module = (model.model.visual if is_qwen else model.model.vision_model).float().to(device)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    names = dict(module.named_modules())
    if any(hasattr(m, "_hf_hook") for m in names.values()):
        raise RuntimeError(f"{cfg['name']}: the tower carries accelerate hooks; it would not run where it was moved")
    depth, weight_shape = TOWER_FACTS[cfg["name"]]
    block = re.compile(r"blocks\.\d+" if is_qwen else r"encoder\.layers\.\d+")
    got_depth = sum(1 for name in names if block.fullmatch(name))
    weight = (module.patch_embed.proj if is_qwen else module.embeddings.patch_embedding).weight
    if got_depth != depth or tuple(weight.shape) != weight_shape:
        raise RuntimeError(
            f"{cfg['name']}: tower has {got_depth} blocks and patch weight {tuple(weight.shape)}; "
            f"want {depth} and {weight_shape}"
        )
    tower = VisionTower(name=cfg["name"], module=module, processor=processor, cfg=cfg, patch_px=int(cfg["patch_size"]))
    if not is_qwen:
        height, width = grid.page_size_px(cfg)
        blank = np.full((height, width, 3), 255, dtype=np.uint8)
        n = module.embeddings.position_embedding.num_embeddings
        if _position_id_count(tower, blank) != n:
            _fix_position_ids(module.embeddings.position_embedding)
            got = _position_id_count(tower, blank)
            if got != n:
                raise RuntimeError(f"{cfg['name']}: {got} distinct position ids with the fix; want {n}")
            tower = replace(tower, position_ids_fixed=True)
    return tower


def _position_id_count(tower: VisionTower, image: np.ndarray) -> int:
    """The number of distinct position ids that reach a SmolVLM tower's position table for one page."""
    seen = []
    table = tower.module.embeddings.position_embedding
    handle = table.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().cpu()))
    try:
        vision_hidden_states(tower, image, [0])
    finally:
        handle.remove()
    (ids,) = seen
    return int(ids.unique().numel())


def vision_hidden_states(tower: VisionTower, image: np.ndarray, layers: Sequence[int]) -> dict[int, np.ndarray]:
    """Return the tower's hidden states for one page at the given layers: (rows, cols, D) float64, raster order.

    Layer 0 is the patch embedding, before any position encoding; layer l >= 1 is the output of block l - 1
    (Qwen) or encoder layer l - 1 (SmolVLM). The inputs are built as transcribe_full builds them, with
    do_resize=False and the same checks. Qwen's visual module runs as get_image_features runs it. SmolVLM's
    vision model is called directly, with the full-frame patch_attention_mask that get_image_features would
    compute, because get_image_features drops an image whose normalised values are all exactly zero. Forward
    hooks copy each output to the CPU as float64 and are removed afterwards. The forward runs under
    _ieee_float32, because torch otherwise lets cuDNN run float32 convolutions, the patch embedding among
    them, in TF32.

    Args:
        tower: object returned by load_vision_tower.
        image: uint8 array (H, W, 3), the reader's native page.
        layers: layer numbers, each from 0 to the tower's depth, without repeats.

    Returns:
        layer -> hidden states, in the order of layers.
    """
    import torch

    module = tower.module
    is_qwen = hasattr(module, "blocks")
    blocks = module.blocks if is_qwen else module.encoder.layers
    layers = [int(layer) for layer in layers]
    if not layers or len(set(layers)) != len(layers) or not all(0 <= layer <= len(blocks) for layer in layers):
        raise ValueError(f"{tower.name}: layers must be distinct numbers from 0 to {len(blocks)}, got {layers}")
    reader = Reader(
        name=tower.name, model=None, processor=tower.processor, cfg=tower.cfg,
        image_token_id=tower.processor.image_token_id,
    )
    inputs = processor_inputs(reader, image, tower.cfg["prompt"])
    parameter = next(module.parameters())
    captured: dict[int, Any] = {}

    def keep(layer: int):
        def hook(_module, _args, output):
            out = output[0] if isinstance(output, tuple) else output
            captured[layer] = out.detach().to("cpu", torch.float64)

        return hook

    embedding = module.patch_embed if is_qwen else module.embeddings.patch_embedding
    handles = [(embedding if layer == 0 else blocks[layer - 1]).register_forward_hook(keep(layer)) for layer in layers]
    rows, cols = image.shape[0] // tower.patch_px, image.shape[1] // tower.patch_px
    try:
        with torch.inference_mode(), _ieee_float32():
            if is_qwen:
                pixel_values = inputs["pixel_values"].to(parameter.device).type(parameter.dtype)
                module(pixel_values, grid_thw=inputs["image_grid_thw"].to(parameter.device))
            else:
                p = tower.patch_px
                pixel_values = inputs["pixel_values"].flatten(0, 1).to(parameter.device, parameter.dtype)
                mask = inputs["pixel_attention_mask"].flatten(0, 1).to(parameter.device)
                patches = mask.unfold(1, p, p).unfold(2, p, p)  # as get_image_features computes it
                patch_attention_mask = (patches.sum(dim=(-1, -2)) > 0).bool()
                if tuple(patch_attention_mask.shape) != (1, rows, cols) or not bool(patch_attention_mask.all()):
                    raise RuntimeError(
                        f"{tower.name}: patch_attention_mask {tuple(patch_attention_mask.shape)} "
                        f"is not one full frame of {rows} x {cols} patches"
                    )
                module(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)
    finally:
        for handle in handles:
            handle.remove()

    states = {}
    for layer in layers:
        h = captured[layer]
        if is_qwen:
            states[layer] = qwen_to_raster(h, inputs["image_grid_thw"], module, window_order=layer > 0).numpy()
        elif layer == 0:
            states[layer] = h.permute(0, 2, 3, 1)[0].numpy()  # (1, D, rows, cols) -> (rows, cols, D)
        else:
            states[layer] = h.reshape(rows, cols, -1).numpy()  # (1, rows x cols, D) -> (rows, cols, D)
    return states


def qwen_to_raster(h: Any, grid_thw: Any, visual: Any, window_order: bool = True) -> Any:
    """(n_patches, d) from patch_embed (window_order=False) or from a block (True) -> (gh, gw, d) in raster order.

    Copied from build guide section 2.1. Qwen's vision forward reorders the merge units with
    new = old[window_index] before the blocks, so old = new[argsort(window_index)]; then the processor's
    grouping of patches into merge units of 2 x 2 is undone.

    Args:
        h: torch tensor (n_patches, d).
        grid_thw: the processor's image_grid_thw, [[1, gh, gw]].
        visual: the Qwen vision module, for its merge size, window size and patch size.
        window_order: whether h is in window order (every block output is).

    Returns:
        torch float64 tensor (gh, gw, d) on the CPU.
    """
    import torch
    from transformers.vision_utils import get_vision_window_index

    h = h.detach().to("cpu", torch.float64)
    _, gh, gw = (int(v) for v in grid_thw[0])
    d, m = h.shape[-1], visual.spatial_merge_size
    if window_order:
        wi, _ = get_vision_window_index(
            grid_thw, spatial_merge_size=m, window_size=visual.window_size, patch_size=visual.patch_size
        )
        h = h.reshape(-1, visual.spatial_merge_unit, d)[torch.argsort(wi.cpu())].reshape(-1, d)
    return h.reshape(gh // m, gw // m, m, m, d).permute(0, 2, 1, 3, 4).reshape(gh, gw, d)


def count_text_tokens(reader: Reader, text: str) -> int:
    """Return the length of text in the reader's own tokens, without special tokens (build guide 1B.9 and 2.8).

    Args:
        reader: object returned by load_reader; only its processor is used.
        text: a page's reference text.

    Returns:
        len(processor.tokenizer(text, add_special_tokens=False)["input_ids"]).
    """
    return len(reader.processor.tokenizer(text, add_special_tokens=False)["input_ids"])


def unload_reader(reader: Reader) -> None:
    """Free a reader's model, so that the next reader has the card to itself (build guide 2.8).

    Drops the reader's reference to the model, then runs gc.collect() and torch.cuda.empty_cache(). The
    memory comes back only if the caller holds no other reference to the model.

    Args:
        reader: object returned by load_reader; its model becomes None.
    """
    import gc

    import torch

    reader.model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def _ieee_float32():
    """Run float32 convolutions and matrix products in full float32 inside the block, then restore the flags.

    torch lets cuDNN run float32 convolutions in TF32 by default (torch.backends.cudnn.conv.fp32_precision
    reads "tf32" at torch 2.14), which keeps 10 of float32's 23 mantissa bits; matrix products default to
    float32 but a library may change that. The legacy switches are used because setting only the new
    per-operator one leaves cuDNN's flags mixed, and torch then refuses to read torch.backends.cudnn.allow_tf32.
    """
    import torch

    previous = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous
