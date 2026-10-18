"""Tests for scripts/24_feasibility.py: the pass rule of PREREGISTRATION_2.md section 7.2 and the 4-bit layer count (W5.1)."""

from __future__ import annotations

import importlib.util
import sys

import pytest

from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "24_feasibility.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("feasibility", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["feasibility"] = module
    spec.loader.exec_module(module)
    return module


feasibility = _load_script()
GOOD = dict(devices=["cuda"], vision_4bit=0, blank=256, expected=256, texts=["ab cd", "The page"])


def test_a_reader_on_the_gpu_with_the_right_count_and_two_decoded_pages_passes():
    assert feasibility.passes(**GOOD)


@pytest.mark.parametrize(
    "change",
    [{"devices": ["cpu", "cuda"]}, {"vision_4bit": 3}, {"blank": 255}, {"texts": ["ab cd", "  "]}],
    ids=["cpu-offload", "quantised-tower", "token-count", "empty-page"],
)
def test_each_failed_condition_fails_the_reader(change):
    assert not feasibility.passes(**{**GOOD, **change})


def test_layer_counts_finds_4bit_layers_inside_and_outside_the_tower():
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes")
    model = torch.nn.Module()
    model.visual = torch.nn.Module()
    model.visual.proj = bnb.nn.Linear4bit(8, 8)
    model.visual.norm = torch.nn.Linear(8, 8)
    model.language_model = torch.nn.Module()
    model.language_model.q = bnb.nn.Linear4bit(8, 8)
    model.language_model.k = bnb.nn.Linear4bit(8, 8)
    model.lm_head = torch.nn.Linear(8, 8)
    assert feasibility.layer_counts(model) == (3, 1)


def test_the_validation_readers_get_the_qwen_allowances():
    assert feasibility.ALLOWANCES == (105, 26)
