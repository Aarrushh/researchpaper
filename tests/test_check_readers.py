"""CPU tests for the reader facts table of scripts/01_check_readers.py (W4.1; R07 and R08 of the spec).

The GPU part of the script (loading a reader, the blind-strip noise test) is exercised by tests/test_reader_facts.py;
here the pure row builder and the writer are tested with the values that part would hand them.
"""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import READERS, REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "01_check_readers.py"
DICTIONARY = REPO_ROOT / "docs" / "04_data_dictionary.md"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("check_readers_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves postponed annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _result(new_tokens=412, hit_limit=False, latency_s=21.3741):
    return SimpleNamespace(text="Sir Walter Elliot", new_tokens=new_tokens, hit_limit=hit_limit, latency_s=latency_s)


def test_facts_columns_match_the_data_dictionary(script):
    text = DICTIONARY.read_text(encoding="utf-8")
    section = re.search(r"^## reports/tables/readers/reader_facts\.csv.*?(?=^## |\Z)", text, flags=re.MULTILINE | re.DOTALL)
    assert section is not None
    documented = re.findall(r"^\| `([a-z_0-9]+)` \|", section.group(0), flags=re.MULTILINE)
    assert tuple(documented) == script.FACTS_COLUMNS


def test_encoder_input_px_reads_qwen_grid_and_smolvlm_frames(script):
    assert script.encoder_input_px(READERS["qwen"], {"image_grid_thw": np.array([[1, 32, 32]])}) == (1, 448, 448)
    assert script.encoder_input_px(READERS["smolvlm"], {"pixel_values": np.zeros((1, 1, 3, 384, 384))}) == (1, 384, 384)
    assert script.encoder_input_px(READERS["smolvlm"], {"pixel_values": np.zeros((1, 5, 3, 384, 384))}) == (5, 384, 384)


def test_facts_row_for_a_native_qwen_page(script):
    row = script.facts_row(READERS["qwen"], 448, 448, {"image_grid_thw": np.array([[1, 32, 32]])}, 256, 256, _result(),
                           5.0117, "hf_device_map {'': 0}")
    assert tuple(row) == script.FACTS_COLUMNS
    assert row["reader"] == "qwen" and row["hf_id"] == READERS["qwen"]["hf_id"]
    assert (row["patch_px"], row["merge"], row["token_cell_px"], row["window_px"]) == (14, 2, 28, 112)
    assert row["encoder_input_px"] == "1x448x448" and row["native_frame"] is True
    assert (row["vision_tokens"], row["grid_token_count"]) == (256, 256)
    assert (row["blind_from_px"], row["blind_strip_px"], row["blind_strip_invisible"], row["control_column_visible"]) == ("", 0, "", "")
    assert (row["max_new_tokens"], row["transcription_tokens"], row["hit_limit"]) == (591, 412, False)
    assert (row["latency_s"], row["peak_gib"]) == ("21.374", "5.012")


def test_facts_row_for_smolvlm_with_the_blind_strip_facts(script):
    blind = {"columns": False, "rows": False, "control": True}  # noise there changes the encoder's output: False, False, True
    row = script.facts_row(READERS["smolvlm"], 384, 384, {"pixel_values": np.zeros((1, 1, 3, 384, 384))}, 81, 81,
                           _result(hit_limit=True), 4.5, "hf_device_map {'': 0}", blind)
    assert row["encoder_input_px"] == "1x384x384" and row["native_frame"] is True
    assert (row["blind_from_px"], row["blind_strip_px"]) == (378, 6)
    assert (row["blind_strip_invisible"], row["control_column_visible"]) == (True, True)
    assert row["hit_limit"] is True
    assert row["encoder_input"].startswith("pixel_values (1, 1, 3, 384, 384): 27 x 27 patches of 14 px")


def test_native_frame_is_false_when_the_encoder_got_something_else(script):
    tiled = script.facts_row(READERS["smolvlm"], 384, 384, {"pixel_values": np.zeros((1, 5, 3, 384, 384))}, 405, 81,
                             _result(), 4.5, "")
    resized = script.facts_row(READERS["qwen"], 448, 448, {"image_grid_thw": np.array([[1, 28, 28]])}, 196, 256,
                               _result(), 5.0, "")
    assert tiled["native_frame"] is False and tiled["encoder_input_px"] == "5x384x384"
    assert resized["native_frame"] is False and resized["encoder_input_px"] == "1x392x392"


def test_write_facts_writes_the_columns_in_order(script, tmp_path):
    rows = [script.facts_row(READERS["qwen"], 448, 448, {"image_grid_thw": np.array([[1, 32, 32]])}, 256, 256, _result(),
                             5.0, "no hf_device_map")]
    path = tmp_path / "readers" / "reader_facts.csv"
    script.write_facts(path, rows)
    with open(path, encoding="utf-8", newline="") as fh:
        header, *data = list(csv.reader(fh))
    assert tuple(header) == script.FACTS_COLUMNS
    assert len(data) == 1 and data[0][0] == "qwen" and data[0][header.index("native_frame")] == "True"
