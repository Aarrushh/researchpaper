# PURPOSE: The no-resize test: the array the vision encoder receives is the page we rendered, byte for byte.
# WHAT GOES HERE: Each reader's native page through readers.processor_inputs, inverted and compared with the rendered bytes; unpatchify_qwen, tested first on a coordinate image; grey levels after the cast to the model dtype; the negative controls (D8; build guide 1B.6).
# FILLED IN AT STEP: W1.8.
# DO NOT: Load model weights; the processors are enough. Do not run in CI: every test here is marked model.
"""The no-resize test: what we render is what the encoder gets."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from conftest import READERS
from mrr import grid, readers

pytestmark = pytest.mark.model

NAMES = ("qwen", "smolvlm", "smolvlm500m")
# pixel_values for one native page (build guide 1B.6): Qwen's 32 x 32 patches of 3 x 2 x 14 x 14 numbers,
# SmolVLM's single frame.
PIXEL_VALUES = {"qwen": (1024, 1176), "smolvlm": (1, 1, 3, 384, 384), "smolvlm500m": (1, 1, 3, 512, 512)}


def unpatchify_qwen(pixel_values, grid_thw) -> np.ndarray:
    """Invert Qwen2VLImageProcessor.patchify for one image.

    patchify (transformers 5.19.0, models/qwen2_vl/image_processing_qwen2_vl.py, lines 177-210) cuts the
    (C, H, W) image into (C, H/(mP), m, P, W/(mP), m, P), permutes that to (H/(mP), W/(mP), m, m, C, P, P),
    copies it T times on a new axis after C, and flattens each patch to one row. Here P = 14, m = 2, T = 2.

    Args:
        pixel_values: tensor (H/P * W/P, C * T * P * P) for one image.
        grid_thw: tensor [[1, H/P, W/P]].

    Returns:
        Array (T, H, W, C): the image once per time copy.
    """
    patch, merge, channels = int(READERS["qwen"]["patch_size"]), int(READERS["qwen"]["merge_size"]), 3
    ((frames, grid_h, grid_w),) = grid_thw.tolist()
    rows, row_len = pixel_values.shape
    assert frames == 1 and rows == grid_h * grid_w, (grid_thw.tolist(), tuple(pixel_values.shape))
    copies = row_len // (channels * patch * patch)
    x = pixel_values.reshape(grid_h // merge, grid_w // merge, merge, merge, channels, copies, patch, patch)
    x = x.permute(5, 0, 2, 6, 1, 3, 7, 4)  # (T, H/(mP), m, P, W/(mP), m, P, C)
    return x.reshape(copies, grid_h * patch, grid_w * patch, channels).cpu().numpy()


def _encoder_view(name, inputs) -> np.ndarray:
    """The normalised page the encoder receives, (H, W, C) in float32."""
    if name == "qwen":
        copies = unpatchify_qwen(inputs["pixel_values"], inputs["image_grid_thw"])
        np.testing.assert_array_equal(copies[0], copies[1])  # the single frame, copied on the time axis
        return copies[0]
    return inputs["pixel_values"][0, 0].permute(1, 2, 0).cpu().numpy()


def _to_levels(normalised, image_processor) -> np.ndarray:
    """Undo the processor's rescale and normalisation: x = (p * std + mean) / rescale_factor, rounded half up."""
    mean = np.asarray(image_processor.image_mean, dtype=np.float64)
    std = np.asarray(image_processor.image_std, dtype=np.float64)
    return np.floor((normalised.astype(np.float64) * std + mean) / image_processor.rescale_factor + 0.5)


def _page(native_page, name, kind) -> np.ndarray:
    _, page = native_page(name)
    if kind == "noise":  # every level in every channel, so a swapped channel or a lost level cannot pass
        return np.random.default_rng(0).integers(0, 256, size=page.shape, dtype=np.uint8)
    return page


def _default_inputs(reader, page):
    """The processor's inputs with its own defaults, which readers.processor_inputs overrides."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": reader.cfg["prompt"]}]}]
    text = reader.processor.apply_chat_template(messages, add_generation_prompt=True)
    return reader.processor(text=[text], images=[Image.fromarray(page)], return_tensors="pt")


def test_unpatchify_qwen_inverts_patchify_on_a_coordinate_image(cpu_reader):
    torch = pytest.importorskip("torch")
    image_processor = cpu_reader("qwen").processor.image_processor
    height, width = 3 * 28, 5 * 28  # not square, so swapped axes cannot pass
    rows, cols = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    coords = torch.stack([rows, cols, rows * width + cols]).double()  # each pixel holds its own address
    flat, grid_h, grid_w = image_processor.patchify(
        coords[None],
        patch_size=image_processor.patch_size,
        merge_size=image_processor.merge_size,
        temporal_patch_size=image_processor.temporal_patch_size,
    )
    copies = unpatchify_qwen(flat[0], torch.tensor([[1, grid_h, grid_w]]))
    assert copies.shape == (image_processor.temporal_patch_size, height, width, 3)
    for copy in copies:
        np.testing.assert_array_equal(copy, coords.permute(1, 2, 0).numpy())


@pytest.mark.parametrize("kind", ["text", "noise"])
@pytest.mark.parametrize("name", NAMES)
def test_encoder_input_is_the_page(cpu_reader, native_page, name, kind):
    pytest.importorskip("torch")
    reader = cpu_reader(name)
    page = _page(native_page, name, kind)
    inputs = readers.processor_inputs(reader, page, reader.cfg["prompt"])
    assert tuple(inputs["pixel_values"].shape) == PIXEL_VALUES[name]
    if name == "qwen":
        assert inputs["image_grid_thw"].tolist() == [[1, 32, 32]]
    else:
        assert bool(inputs["pixel_attention_mask"].all())
    np.testing.assert_array_equal(_to_levels(_encoder_view(name, inputs), reader.processor.image_processor), page)


@pytest.mark.parametrize("name", NAMES)
def test_every_grey_level_stays_distinct_in_the_model_dtype(cpu_reader, name):
    """The model casts pixel_values to its dtype before the patch embedding; no two grey levels may merge there.

    Single channels can merge: in bfloat16, Qwen's blue channel gives levels 253 and 254 one value. The red
    and green channels still tell them apart, so a grey page loses nothing.
    """
    torch = pytest.importorskip("torch")
    reader = cpu_reader(name)
    height, width = grid.page_size_px(reader.cfg)
    grey = (np.arange(height * width) % 256).astype(np.uint8).reshape(height, width)
    inputs = readers.processor_inputs(reader, np.repeat(grey[:, :, None], 3, axis=2), reader.cfg["prompt"])
    view = torch.from_numpy(np.ascontiguousarray(_encoder_view(name, inputs)))
    cast = view.to(getattr(torch, reader.cfg["dtype"])).float().numpy()
    pairs = np.unique(np.column_stack([grey.reshape(-1), cast.reshape(-1, 3)]), axis=0)
    assert len(pairs) == 256  # one value triple per level...
    assert len(np.unique(pairs[:, 1:], axis=0)) == 256  # ...and no triple shared by two levels


def test_qwen_resizes_a_450_page_to_448_by_default(cpu_reader):
    """Negative control: with its defaults, Qwen's processor resamples a 450 px page to 448 px."""
    pytest.importorskip("torch")
    reader = cpu_reader("qwen")
    page = np.random.default_rng(1).integers(0, 256, size=(450, 450, 3), dtype=np.uint8)
    assert _default_inputs(reader, page)["image_grid_thw"].tolist() == [[1, 32, 32]]  # 32 * 14 = 448


def test_smolvlm_tiles_a_384_page_by_default(cpu_reader, native_page):
    """Negative control: with its defaults, SmolVLM's processor makes 4 x 4 tiles and a global image of the page."""
    pytest.importorskip("torch")
    reader = cpu_reader("smolvlm")
    inputs = _default_inputs(reader, native_page("smolvlm")[1])
    assert tuple(inputs["pixel_values"].shape) == (1, 17, 3, 384, 384)
    assert int((inputs["input_ids"] == reader.image_token_id).sum()) == 17 * 81 == 1377


@pytest.mark.parametrize(
    ("name", "size", "message"),
    [
        ("qwen", 450, "not whole 28 px token cells"),  # refused before patchify, which cannot cut 450 px into them
        ("smolvlm", 400, "pixel_values"),  # resized to 768, then 2 x 2 tiles and a global image
        ("smolvlm", 378, "pixel_values"),  # resized up to 384
        ("smolvlm500m", 520, "pixel_values"),
        ("smolvlm500m", 500, "pixel_values"),
    ],
)
def test_processor_inputs_refuses_a_page_that_is_not_native(cpu_reader, name, size, message):
    """SmolVLM's processor resizes even with do_resize=False: each side goes up to a multiple of the frame."""
    pytest.importorskip("torch")
    reader = cpu_reader(name)
    page = np.full((size, size, 3), 255, dtype=np.uint8)
    with pytest.raises(RuntimeError, match=message):
        readers.processor_inputs(reader, page, reader.cfg["prompt"])
