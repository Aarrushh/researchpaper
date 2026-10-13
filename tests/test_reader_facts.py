# PURPOSE: Facts about the three readers that the experiment relies on, checked on the pinned processors and models.
# WHAT GOES HERE: Tokens per native page, image-token ids, Qwen's smart_resize, one <|im_start|>; without weights, the frames the position-id fix changes and load_reader's check for its key; on the GPU, SmolVLM's position ids without and with the fix and as load_reader installs it (D15), the 6 px strip SmolVLM-2.2B never sees, and determinism (build guide 1B.6).
# FILLED IN AT STEP: W1.6/W1.8.
# DO NOT: Hold two models on the GPU at once; the laptop has 8 GiB. Do not run in CI: every test here is marked model.
"""Reader facts (build guide 1B.6)."""

from __future__ import annotations

import gc
import inspect
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import READERS
from mrr import grid, readers

pytestmark = pytest.mark.model

# max_new_tokens is set at W1.7; until then the determinism test reads 256 tokens, several lines of a page.
DETERMINISM_TOKENS = 256
SIZES = (448, 384, 512, 1000, 70)


# Before the model: the processors alone.


@pytest.mark.parametrize(("name", "expected"), [("qwen", 256), ("smolvlm", 81), ("smolvlm500m", 64)])
def test_tokens_per_native_page(cpu_reader, native_page, name, expected):
    pytest.importorskip("torch")
    _, page = native_page(name)
    assert readers.count_vision_tokens(cpu_reader(name), page) == expected
    assert grid.token_count(READERS[name], *page.shape[:2]) == expected


@pytest.mark.parametrize(("name", "expected"), [("qwen", 151655), ("smolvlm", 49153), ("smolvlm500m", 49190)])
def test_image_token_id(cpu_reader, name, expected):
    pytest.importorskip("torch")
    processor = cpu_reader(name).processor
    assert processor.tokenizer.convert_tokens_to_ids(READERS[name]["image_token"]) == expected
    assert processor.image_token_id == expected


def _transformers_smart_resize(cpu_reader, height, width):
    """transformers' smart_resize with the pinned Qwen processor's factor and pixel bounds."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    image_processor = cpu_reader("qwen").processor.image_processor
    factor = image_processor.patch_size * image_processor.merge_size
    bounds = image_processor.size["shortest_edge"], image_processor.size["longest_edge"]
    return smart_resize(height, width, factor, *bounds)


@pytest.mark.parametrize(("size", "expected"), [(448, 448), (1000, 1008), (70, 56)])
def test_qwen_smart_resize(cpu_reader, size, expected):
    pytest.importorskip("torch")
    assert _transformers_smart_resize(cpu_reader, size, size) == (expected, expected)


@pytest.mark.parametrize(("height", "width"), list(itertools.product(SIZES, SIZES)))
def test_grid_copy_of_smart_resize_agrees_with_transformers(cpu_reader, height, width):
    pytest.importorskip("torch")
    assert grid.qwen_smart_resize(height, width) == _transformers_smart_resize(cpu_reader, height, width)


def test_qwen_pixel_bounds_are_the_processor_bounds(cpu_reader):
    """grid.qwen_smart_resize's defaults and the copy in readers.yaml are the pinned processor's values."""
    pytest.importorskip("torch")
    image_processor = cpu_reader("qwen").processor.image_processor
    bounds = (image_processor.size["shortest_edge"], image_processor.size["longest_edge"])
    defaults = inspect.signature(grid.qwen_smart_resize).parameters
    assert (defaults["min_pixels"].default, defaults["max_pixels"].default) == bounds
    assert (READERS["qwen"]["min_pixels"], READERS["qwen"]["max_pixels"]) == bounds
    assert defaults["factor"].default == image_processor.patch_size * image_processor.merge_size


@pytest.mark.parametrize("name", ["smolvlm", "smolvlm500m"])
def test_smolvlm_prompt_holds_one_im_start(cpu_reader, native_page, name):
    pytest.importorskip("torch")
    reader = cpu_reader(name)
    tokenizer = reader.processor.tokenizer
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assert im_start != tokenizer.unk_token_id
    inputs = readers.processor_inputs(reader, native_page(name)[1], reader.cfg["prompt"])
    assert int((inputs["input_ids"] == im_start).sum()) == 1


def test_load_reader_wants_a_smolvlm_block_to_name_its_position_id_fix():
    """D15: without the key a SmolVLM block would load without the fix; load_reader stops before the weights."""
    pytest.importorskip("torch")
    cfg = {key: value for key, value in READERS["smolvlm"].items() if key != "position_id_fix"}
    with pytest.raises(ValueError, match="position_id_fix"):
        readers.load_reader(cfg)


@pytest.mark.parametrize(
    ("patches_h", "patches_w", "valid_w", "fixed"),
    [(27, 27, 27, True), (27, 27, 14, False), (9, 81, 81, False)],
    ids=["full-frame", "padded-frame", "frame-not-square"],
)
def test_position_id_fix_changes_only_a_full_frame(patches_h, patches_w, valid_w, fixed):
    """D15, on the pinned embeddings in bfloat16 without weights: the hook replaces the ids of a full frame only.

    A padded frame and a frame of 9 x 81 patches also hold 729 ids; they keep the ids the embeddings give them.
    """
    torch = pytest.importorskip("torch")
    from transformers.models.idefics3.configuration_idefics3 import Idefics3VisionConfig
    from transformers.models.idefics3.modeling_idefics3 import Idefics3VisionEmbeddings

    patch = READERS["smolvlm"]["patch_size"]
    config = Idefics3VisionConfig(hidden_size=8, num_attention_heads=1, image_size=384, patch_size=patch)
    embeddings = Idefics3VisionEmbeddings(config).to(torch.bfloat16)
    pixels = torch.zeros(1, 3, patches_h * patch, patches_w * patch, dtype=torch.bfloat16)
    mask = torch.zeros(1, patches_h, patches_w, dtype=torch.bool)
    mask[:, :, :valid_w] = True

    def ids_seen():
        seen = []
        handle = embeddings.position_embedding.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        try:
            with torch.inference_mode():
                embeddings(pixels, mask)
        finally:
            handle.remove()
        (ids,) = seen
        return ids

    before = ids_seen()
    model = SimpleNamespace(model=SimpleNamespace(vision_model=SimpleNamespace(embeddings=embeddings)))
    readers.install_position_id_fix(model)  # the path it takes on an Idefics3ForConditionalGeneration
    after = ids_seen()
    if fixed:
        assert before.unique().numel() == 441  # the bfloat16 bug, as on the GPU
        assert torch.equal(after[0], torch.arange(729))
    else:
        assert torch.equal(after, before)


# On the GPU, one model at a time.

_ON_GPU: dict[str, readers.Reader] = {}  # at most one reader: the laptop's 8 GiB holds one at a time


def _free_gpu() -> None:
    import torch

    _ON_GPU.clear()
    gc.collect()
    torch.cuda.empty_cache()


def _gpu_reader(name: str) -> readers.Reader:
    """Return the reader as configs/readers.yaml loads it, after freeing any other reader."""
    if name not in _ON_GPU:
        _free_gpu()
        _ON_GPU[name] = readers.load_reader(READERS[name])
    return _ON_GPU[name]


@pytest.fixture(scope="module", autouse=True)
def _free_gpu_after_the_module():
    yield
    if _ON_GPU:
        _free_gpu()


def _smolvlm_features(reader, page):
    """SmolVLM's vision encoder output and the connector's output, the image tokens the language model gets."""
    import torch

    inputs = readers.processor_inputs(reader, page, reader.cfg["prompt"]).to(reader.model.device)
    with torch.inference_mode():
        out = reader.model.get_image_features(
            pixel_values=inputs["pixel_values"], pixel_attention_mask=inputs["pixel_attention_mask"]
        )
    return out.last_hidden_state, out.pooler_output


def _position_ids(reader, page):
    """The position ids that reach SmolVLM's position table for one page, seen by a hook registered last."""
    seen = []
    table = reader.model.model.vision_model.embeddings.position_embedding
    handle = table.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().cpu()))
    try:
        _smolvlm_features(reader, page)
    finally:
        handle.remove()
    (ids,) = seen
    return ids


@pytest.mark.parametrize(("name", "without_fix", "with_fix"), [("smolvlm", 441, 729), ("smolvlm500m", 1024, 1024)])
def test_position_ids_without_and_with_the_fix(native_page, name, without_fix, with_fix):
    """D15: in bfloat16, k/27 falls below its bucket edge for six rows and six columns; k/32 is exact."""
    torch = pytest.importorskip("torch")
    _free_gpu()
    reader = readers.load_reader({**READERS[name], "position_id_fix": None})
    try:
        _, page = native_page(name)
        assert _position_ids(reader, page).unique().numel() == without_fix
        readers.install_position_id_fix(reader.model)  # before the recording hook, as in load_reader
        ids = _position_ids(reader, page)
        assert ids.unique().numel() == with_fix
        assert torch.equal(ids[0], torch.arange(with_fix))
    finally:
        del reader
        _free_gpu()


# smolvlm last: the next test reads it too, so it stays loaded
@pytest.mark.parametrize(("name", "n"), [("smolvlm500m", 1024), ("smolvlm", 729)])
def test_load_reader_installs_the_position_id_fix(native_page, name, n):
    """D15: the reader as configs/readers.yaml loads it gives a full frame the ids 0 to n - 1."""
    torch = pytest.importorskip("torch")
    _, page = native_page(name)
    assert torch.equal(_position_ids(_gpu_reader(name), page)[0], torch.arange(n))


@pytest.mark.parametrize(
    ("region", "seen"),
    [
        ((slice(None), slice(378, 384)), False),
        ((slice(378, 384), slice(None)), False),
        ((slice(None), slice(377, 378)), True),
    ],
    ids=["columns-378-383", "rows-378-383", "column-377-control"],
)
def test_smolvlm_encoder_never_sees_the_last_6_px(native_page, region, seen):
    """27 patches of 14 px cover 378 of SmolVLM-2.2B's 384 px; the rest never reaches the encoder (D8)."""
    torch = pytest.importorskip("torch")
    assert grid.blind_strip_px(READERS["smolvlm"]) == 384 - 378
    reader = _gpu_reader("smolvlm")
    _, page = native_page("smolvlm")
    changed = page.copy()
    changed[region] = np.random.default_rng(2).integers(0, 256, size=changed[region].shape, dtype=np.uint8)
    for original, after in zip(_smolvlm_features(reader, page), _smolvlm_features(reader, changed)):
        assert torch.equal(original, after) is not seen


@pytest.mark.parametrize("name", ["smolvlm", "smolvlm500m", "qwen"])
def test_same_page_twice_gives_identical_text(native_page, name):
    pytest.importorskip("torch")
    reader = _gpu_reader(name)
    _, page = native_page(name)
    first, second = (
        readers.transcribe_full(reader, page, reader.cfg["prompt"], max_new_tokens=DETERMINISM_TOKENS)
        for _ in range(2)
    )
    assert first.text.strip()  # two empty texts would be equal too
    assert first.text == second.text
    assert first.new_tokens == second.new_tokens
