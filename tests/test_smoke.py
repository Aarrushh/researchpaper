# PURPOSE: Checks the scaffold is importable and the bookkeeping works.
# WHAT GOES HERE: Import tests for every mrr module; manifest and schema checks.
# FILLED IN AT STEP: Scaffolded at setup. Must pass on the untouched scaffold.
# DO NOT: Mark these todo. Do not require a GPU or network.
"""Smoke tests."""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from conftest import REPO_ROOT

MODULES = [
    "mrr",
    "mrr.color",
    "mrr.grid",
    "mrr.render",
    "mrr.text_source",
    "mrr.readers",
    "mrr.score",
    "mrr.stats",
    "mrr.plotting",
    "mrr.manifest",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_readers_import_without_torch():
    # A fresh interpreter, because torch may already be loaded in this one; readers.py imports it inside functions.
    code = "import sys, mrr.readers; print(sorted(m for m in ('torch', 'transformers') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_run_id_format():
    from mrr.manifest import new_run_id

    rid = new_run_id("pilot", "qwen", now=datetime(2026, 10, 5, 14, 32, tzinfo=timezone.utc))
    assert re.fullmatch(r"20261005-1432_pilot_qwen_([0-9a-f]{7}|nogit)", rid)


def test_write_manifest(tmp_path):
    from mrr.manifest import TRACKED_PACKAGES, write_manifest

    path = write_manifest(tmp_path / "run1", {"n_pages": 3})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == "run1"
    assert data["config"] == {"n_pages": 3}
    assert set(data["packages"]) == set(TRACKED_PACKAGES)
    for key in ("timestamp_utc", "git_commit", "git_dirty", "python_version", "gpu_name"):
        assert key in data


def test_results_columns_match_data_dictionary():
    from mrr.manifest import RESULTS_COLUMNS

    text = (REPO_ROOT / "docs" / "04_data_dictionary.md").read_text(encoding="utf-8")
    documented = re.findall(r"^\| `([a-z_]+)` \|", text, flags=re.MULTILINE)
    assert tuple(documented) == RESULTS_COLUMNS
