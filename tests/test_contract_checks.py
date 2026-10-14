"""Tests for scripts/08_contract_checks.py (build guide section 4.1). CPU only.

The script is the evidence R01 to R06 of the spec cite: these tests run it once into a temporary folder and check its
table, its manifest and that a wrong expectation fails the run.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from conftest import PERSUASION, REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "08_contract_checks.py"
DEFAULTS = yaml.safe_load((REPO_ROOT / "configs" / "render_defaults.yaml").read_text(encoding="utf-8"))
FONT = REPO_ROOT / DEFAULTS["font_file"]
HAVE_DATA = FONT.exists() and PERSUASION.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="font or book not present (W1.4)")
RULES = ("R01", "R02", "R03", "R04", "R05", "R06", "Conformance")


def _module():
    spec = importlib.util.spec_from_file_location("contract_checks_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves postponed annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _args(out_dir: Path) -> list[str]:
    return ["--config", str(REPO_ROOT / "configs" / "readers.yaml"),
            "--render-config", str(REPO_ROOT / "configs" / "render_defaults.yaml"),
            "--out-dir", str(out_dir), "--root", str(REPO_ROOT)]


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One run of the script into a temporary folder."""
    if not HAVE_DATA:
        pytest.skip("font or book not present (W1.4)")
    out_dir = tmp_path_factory.mktemp("contract") / "contract"
    module = _module()
    code = module.main(_args(out_dir))
    return SimpleNamespace(module=module, code=code, out_dir=out_dir, rows=_rows(out_dir / "contract_checks.csv"))


def test_every_check_passes(run):
    failed = [row for row in run.rows if row["passed"] != "True"]
    assert run.code == 0 and not failed, failed


def test_the_five_columns_in_order(run):
    with open(run.out_dir / "contract_checks.csv", encoding="utf-8", newline="") as fh:
        assert tuple(next(csv.reader(fh))) == run.module.COLUMNS == ("rule", "check", "expected", "observed", "passed")


def test_every_rule_r01_to_r06_and_the_conformance_table_have_rows(run):
    assert {row["rule"] for row in run.rows} == set(RULES)
    assert all(sum(row["rule"] == rule for row in run.rows) >= 3 for rule in RULES)


def test_the_worked_examples_are_computed_through_the_renderer(run):
    """The observed cells of R05 come out of composite_coverage, not out of the spec."""
    examples = {(row["check"], row["observed"]) for row in run.rows if row["rule"] == "R05"}
    assert ("coverage_linear: bitmap [[0, 64], [128, 255]] of (0, 0, 0) on (255, 255, 255)", "[[255, 224], [187, 0]]") in examples
    assert ("coverage_srgb: byte 128 of (200, 30, 60) on (250, 245, 230)", "[225, 137, 145]") in examples
    assert ("binary: glyph bytes 27 and 100 (S = 127), black on white", "[255, 255, 255]") in examples


def test_the_conformance_rows_cover_every_triple(run):
    rows = {row["check"]: row for row in run.rows if row["rule"] == "Conformance"}
    gap = next(row for name, row in rows.items() if name.startswith("coverage_srgb: smallest distance"))
    assert abs(float(gap["observed"]) - 1 / 510) < 1e-12
    srgb = next(row for name, row in rows.items() if "single precision differs" in name)
    assert srgb["observed"] == "0"
    linear = [row for name, row in rows.items() if name.startswith("coverage_linear in single precision")]
    assert len(linear) == 2
    for row in linear:
        moved, largest = (int(v) for v in re.fullmatch(r"\((\d+), (\d+)\)", row["observed"]).groups())
        assert 0 < moved < 256**3 * 1e-4 and largest == 1


def test_the_manifest_names_the_run_and_the_table(run):
    manifest = json.loads((run.out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "_contract_checks_" in manifest["run_id"]
    assert manifest["config"]["checks"] == len(run.rows)
    assert manifest["config"]["failed"] == 0
    assert manifest["config"]["rules"] == list(RULES)
    assert manifest["config"]["render"]["font_file"] == DEFAULTS["font_file"]  # relative, as the config states it


@needs_data
def test_a_wrong_expectation_fails_the_run(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "FREETYPE_VERSION", (9, 9, 9))
    out_dir = tmp_path / "contract"
    assert module.main(_args(out_dir)) == 1
    rows = _rows(out_dir / "contract_checks.csv")
    failed = [row for row in rows if row["passed"] != "True"]
    assert [(row["rule"], row["check"]) for row in failed] == [("R01", "FreeType version")]
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["failed"] == 1


def test_the_script_never_rounds_with_a_library_call():
    """Build guide 4.1: floor(v + 0.5) everywhere; round(), np.round and torch.round would round halves to even."""
    source = SCRIPT.read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    assert re.search(r"(?<![\w.])round\(|np\.round\b|np\.rint\b|torch\.round\b", code) is None
