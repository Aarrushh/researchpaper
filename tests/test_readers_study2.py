# PURPOSE: The Study 2 reader file and the Study 2 additions to readers.py, without the models.
# WHAT GOES HERE: configs/readers_study2.yaml against configs/readers.yaml and PREREGISTRATION_2.md section 2 (Study 1's blocks unchanged, the validation readers pinned at their native page); readers._quantization_config; readers._check_vision_input on InternVL-shaped inputs.
# FILLED IN AT STEP: W5.1.
# DO NOT: Load a processor or a model here, or touch the GPU. The BitsAndBytesConfig test imports transformers, so it is marked model.
"""configs/readers_study2.yaml and the Study 2 additions to readers.py."""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from conftest import READERS, REPO_ROOT
from mrr import grid, readers

STUDY2 = yaml.safe_load((REPO_ROOT / "configs" / "readers_study2.yaml").read_text(encoding="utf-8"))["readers"]
BY_NAME = {reader["name"]: reader for reader in STUDY2}
VALIDATION = ("qwen7b", "internvl3_2b")
INTERNVL_IMAGE_TOKEN = 151667  # <IMG_CONTEXT> at the pinned revision (added_tokens.json)


def test_the_five_readers_in_order():
    assert [reader["name"] for reader in STUDY2] == ["qwen", "smolvlm", "smolvlm500m", "qwen7b", "internvl3_2b"]


@pytest.mark.parametrize("name", ["qwen", "smolvlm", "smolvlm500m"])
def test_study1_blocks_are_unchanged(name):
    assert BY_NAME[name] == READERS[name]


@pytest.mark.parametrize("name", VALIDATION)
def test_validation_readers_are_pinned_at_one_native_448_page(name):
    cfg = BY_NAME[name]
    assert re.fullmatch(r"[0-9a-f]{40}", cfg["revision"])
    assert grid.token_cell_px(cfg) == 28
    assert grid.page_size_px(cfg) == (448, 448)
    assert grid.token_count(cfg, 448, 448) == 256
    assert grid.blind_strip_px(cfg) == 0
    assert cfg["prompt"] == READERS["qwen"]["prompt"]
    assert cfg["dtype"] == "bfloat16"


def test_qwen7b_has_the_3b_geometry_and_4bit_weights():
    big, small = BY_NAME["qwen7b"], READERS["qwen"]
    for key in ("patch_size", "merge_size", "token_cell_px", "window_px", "page_px", "min_pixels", "max_pixels",
                "aligned_pitch_px", "image_token", "max_new_tokens", "prompt"):
        assert big[key] == small[key], key
    assert big["quantization"] == "nf4"


def test_only_qwen7b_is_quantised():
    assert [reader["name"] for reader in STUDY2 if "quantization" in reader] == ["qwen7b"]


def test_internvl_processor_has_a_model_class():
    assert readers.MODEL_CLASSES["InternVLProcessor"] == "InternVLForConditionalGeneration"


def test_quantization_absent_gives_no_config():
    assert readers._quantization_config(READERS["qwen"], dtype=None) is None


def test_quantization_refuses_an_unknown_scheme():
    with pytest.raises(ValueError, match="unknown quantization 'int8'"):
        readers._quantization_config({**BY_NAME["qwen7b"], "quantization": "int8"}, dtype=None)


@pytest.mark.model
def test_nf4_config_keeps_the_vision_tower_and_head_unquantised():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    q = readers._quantization_config(BY_NAME["qwen7b"], torch.bfloat16)
    assert q.load_in_4bit and q.bnb_4bit_quant_type == "nf4" and q.bnb_4bit_use_double_quant
    assert q.bnb_4bit_compute_dtype == torch.bfloat16
    assert {"visual", "lm_head"} <= set(q.llm_int8_skip_modules)


def _internvl_inputs(tiles=1, tokens=256):
    ids = np.concatenate([np.full(tokens, INTERNVL_IMAGE_TOKEN), np.arange(5)])[None, :]
    return {"pixel_values": np.zeros((tiles, 3, 448, 448), np.float32), "input_ids": ids}


def _internvl_stub():
    return SimpleNamespace(name="internvl3_2b", cfg=BY_NAME["internvl3_2b"], image_token_id=INTERNVL_IMAGE_TOKEN)


def test_internvl_one_tile_with_256_tokens_passes():
    readers._check_vision_input(_internvl_stub(), _internvl_inputs(), 448, 448)


def test_internvl_two_tiles_are_refused():
    with pytest.raises(RuntimeError, match="one tile"):
        readers._check_vision_input(_internvl_stub(), _internvl_inputs(tiles=2), 448, 448)


def test_internvl_wrong_token_count_is_refused():
    with pytest.raises(RuntimeError, match="255 image tokens"):
        readers._check_vision_input(_internvl_stub(), _internvl_inputs(tokens=255), 448, 448)
