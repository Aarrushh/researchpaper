# PURPOSE: readers.py without the models: the page checks before the processor, and transcribe_full on a fake model.
# WHAT GOES HERE: processor_inputs refusing a page that is not uint8 (H, W, 3) or, for Qwen, not whole token cells; the greedy arguments; transcribe_full's hit_limit, its max_new_tokens check and what it passes to generate (build guide 1B.4).
# FILLED IN AT STEP: W1.6 (Phase 1 review).
# DO NOT: Load a processor or a model here, or touch the GPU. The page checks run in CI; the transcribe_full tests import torch, so they are marked model.
"""readers.py without the models."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import READERS
from mrr import readers

PAGE = np.zeros((4, 4, 3), dtype=np.uint8)
PROMPT_IDS = [7, 8, 9]  # the fake prompt; the fake generate returns it followed by the new tokens


def _stub(name):
    """A reader with its config and no processor or model: enough for the checks before the processor."""
    return SimpleNamespace(name=name, cfg=READERS[name], processor=None, model=None)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 4, 3), dtype=np.float64),
        np.zeros((4, 4, 3), dtype=np.uint16),
        np.zeros((4, 4, 1), dtype=np.uint8),
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4, 4), dtype=np.uint8),
        [[[0, 0, 0]]],
    ],
    ids=["float64", "uint16", "one-channel", "grey", "rgba", "list"],
)
def test_processor_inputs_refuses_anything_but_a_uint8_rgb_page(image):
    with pytest.raises(ValueError, match=r"uint8 page of shape \(H, W, 3\)"):
        readers.processor_inputs(_stub("smolvlm"), image, "prompt")


@pytest.mark.parametrize("size", [450, 434])  # 434 = 31 x 14: whole patches, but not whole 2 x 2 merge cells
def test_processor_inputs_refuses_a_qwen_page_that_is_not_whole_token_cells(size):
    page = np.full((size, size, 3), 255, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="not whole 28 px token cells"):
        readers.processor_inputs(_stub("qwen"), page, "prompt")


def test_greedy_arguments_are_the_guides():
    """Build guide 1B.4: do_sample=False, num_beams=1 and repetition_penalty=1.0 on every call."""
    assert readers.GREEDY == {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0}


# transcribe_full on a fake model, which needs torch.


class _FakeModel:
    """Returns the prompt followed by fixed new tokens, cut at the limit, and records what generate got."""

    device = "cpu"

    def __init__(self, new_ids, eos):
        self.new_ids, self.calls = new_ids, []
        self.generation_config = SimpleNamespace(eos_token_id=eos)

    def generate(self, input_ids, **kwargs):
        import torch

        self.calls.append(kwargs)
        return torch.cat([input_ids, torch.tensor([self.new_ids[: kwargs["max_new_tokens"]]])], dim=1)


class _Inputs(dict):
    """The processor's output, which transcribe_full moves to the model's device."""

    def to(self, device):
        return self


def _fake_reader(monkeypatch, new_ids, eos=2, limit=4):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)  # no CUDA context
    inputs = _Inputs(input_ids=torch.tensor([PROMPT_IDS]))
    monkeypatch.setattr(readers, "processor_inputs", lambda reader, image, prompt: inputs)
    processor = SimpleNamespace(batch_decode=lambda ids, **kwargs: [" ".join(str(i) for i in ids[0].tolist())])
    return readers.Reader(
        name="fake", model=_FakeModel(new_ids, eos), processor=processor, cfg={"max_new_tokens": limit}, image_token_id=0
    )


@pytest.mark.model
@pytest.mark.parametrize(
    ("new_ids", "eos", "new_tokens", "hit_limit"),
    [
        ([5, 6, 3, 4], 2, 4, True),
        ([5, 6, 3, 2], 2, 4, False),
        ([5, 2], 2, 2, False),
        ([5, 6, 3, 1], [1, 2], 4, False),
        ([5, 6, 3, 2], None, 4, True),
    ],
    ids=["no-end-token", "end-token-last-allowed", "end-token-early", "one-of-two-end-tokens", "no-end-token-set"],
)
def test_transcribe_full_says_whether_the_limit_cut_the_text(monkeypatch, new_ids, eos, new_tokens, hit_limit):
    """hit_limit: max_new_tokens new tokens, the last of them not an end token (LAB_NOTEBOOK.md, W1.6)."""
    reader = _fake_reader(monkeypatch, new_ids, eos)
    result = readers.transcribe_full(reader, PAGE, "prompt")
    assert (result.text, result.new_tokens, result.hit_limit) == (" ".join(map(str, new_ids)), new_tokens, hit_limit)
    assert result.latency_s >= 0


@pytest.mark.model
def test_transcribe_full_gives_generate_the_greedy_arguments_and_the_limit(monkeypatch):
    reader = _fake_reader(monkeypatch, [5, 6, 3, 4], limit=4)
    readers.transcribe_full(reader, PAGE, "prompt")  # the config's limit
    readers.transcribe_full(reader, PAGE, "prompt", max_new_tokens=3)  # an explicit limit wins
    assert reader.model.calls == [{**readers.GREEDY, "max_new_tokens": 4}, {**readers.GREEDY, "max_new_tokens": 3}]


@pytest.mark.model
@pytest.mark.parametrize("limit", [True, 0, -1, 2.0, "512"])
def test_transcribe_full_refuses_a_limit_that_is_not_a_positive_integer(monkeypatch, limit):
    reader = _fake_reader(monkeypatch, [5], limit=limit)
    with pytest.raises(ValueError, match="max_new_tokens"):
        readers.transcribe_full(reader, PAGE, "prompt")
    assert reader.model.calls == []
