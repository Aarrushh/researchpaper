"""Tests for spec/RENDERING_SPEC.md (build guide section 4.1): its structure, what it must not contain, and its evidence.

The spec is the only input GLM-5.3 receives in H2, so it may name nothing of the reference code. The tests marked todo
are W4.1's "done when": they pass once every file a rule cites exists and no TODO[W4.1] remains, after the full run.
"""

from __future__ import annotations

import re

import pytest

from conftest import REPO_ROOT

SPEC = REPO_ROOT / "spec" / "RENDERING_SPEC.md"
TEXT = SPEC.read_text(encoding="utf-8")
RULES = [f"R{n:02d}" for n in range(1, 12)]
FIELDS = ("RULE", "WHY", "EVIDENCE", "PARAMETERS", "APPLIES WHEN", "TEST")
HEADING = re.compile(r"^### (R\d\d)\. (.+)$", re.MULTILINE)


def _sections() -> dict[str, str]:
    """Rule id -> the text of its section, up to the next heading."""
    parts = HEADING.split(TEXT)  # preamble, then (id, name, body) triples
    return {parts[i]: parts[i + 2] for i in range(1, len(parts), 3)}


def _evidence(body: str) -> list[str]:
    line = re.search(r"^- EVIDENCE: (.*)$", body, flags=re.MULTILINE)
    assert line is not None, "no EVIDENCE line"
    return re.findall(r"`([^`]+)`", line.group(1))


def test_rules_r01_to_r11_in_order_each_once():
    assert HEADING.findall(TEXT) and [rule for rule, _ in HEADING.findall(TEXT)] == RULES


@pytest.mark.parametrize("rule", RULES)
def test_every_rule_has_the_six_fields_in_the_template_order(rule):
    body = _sections()[rule]
    positions = [body.find(f"- {field}: ") for field in FIELDS]
    assert all(p >= 0 for p in positions), f"{rule} lacks {[f for f, p in zip(FIELDS, positions) if p < 0]}"
    assert positions == sorted(positions), f"{rule}: fields out of the template's order"
    assert all(body.count(f"- {field}: ") == 1 for field in FIELDS)


@pytest.mark.parametrize("rule", RULES)
def test_every_rule_cites_only_files_under_reports(rule):
    paths = _evidence(_sections()[rule])
    assert paths, f"{rule} cites no file"
    assert all(path.startswith("reports/") for path in paths), paths


def test_the_sections_of_the_template_are_present():
    for heading in ("## Scope", "## Definitions", "## Rule template", "## Kernel interface", "## Conformance"):
        assert TEXT.count(f"\n{heading}\n") == 1, heading


def test_names_nothing_of_the_reference_code():
    """Build guide 4.1: no function, module or file outside reports/; the kernel is specified from text alone."""
    for forbidden in ("src/", "scripts/", "tests/", "mrr.", "configs/", "results/"):
        assert forbidden not in TEXT, forbidden
    assert re.search(r"\.py\b", TEXT) is None
    assert re.search(r"\bdef \w+\(", TEXT.replace("def render(", "")) is None  # only the kernel's own signature


def test_suggests_no_kernel_organisation():
    """Build guide 4.1: never block sizes, programs per glyph or pixel, or atomics; the model decides those."""
    assert re.search(r"\b(atomic|atomics|block size|BLOCK_SIZE|num_warps|num_stages|programs? per|tl\.)\b", TEXT, re.IGNORECASE) is None


def test_rounds_halves_up_everywhere():
    """Every rounding is floor(v + 0.5); the words 'round to even' and 'nearest even' never describe a rule."""
    assert "⌊v + 0.5⌋" in TEXT
    assert re.search(r"round(s|ed|ing)? (halves )?to (the nearest )?even", TEXT) is None or "Nothing rounds halves to even" in TEXT


def test_kernel_interface_states_the_tolerances():
    table = TEXT[TEXT.index("## Conformance"):]
    for mode, e_max, f_diff in (("binary", "0", "0"), ("coverage_srgb", "0", "0"), ("coverage_linear", "1", "0.001")):
        assert re.search(rf"^\| `{mode}` \| {e_max} \| {f_diff} \|", table, flags=re.MULTILINE), mode


@pytest.mark.todo  # W4.1 done when: the full run has written reports/tables/full/ and the GPU the reader facts
@pytest.mark.parametrize("rule", RULES)
def test_every_cited_file_exists(rule):
    for path in _evidence(_sections()[rule]):
        assert (REPO_ROOT / path).exists(), path


@pytest.mark.todo  # W4.1 done when: R09 to R11 hold the full run's numbers
def test_no_w41_todo_remains():
    assert "TODO[W4.1]" not in TEXT
