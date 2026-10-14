"""Tests for scripts/40_freeze.py (build guide section 3.8). CPU only.

One fake-backend run of the V and H blocks of the analysis sweep stands in for the results tree, beside a
manifest-only run folder like those of make readers, an equivariance CSV and a git-ignored images folder.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import READERS, REPO_ROOT
from mrr.sweep import run_sweep
from test_analyze import ANALYSIS_SWEEP, DEFAULTS, HAVE_DATA, DroppingBackend, _analyze, _write_sweep

SCRIPT = REPO_ROOT / "scripts" / "40_freeze.py"
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="font or books not present (W1.4)")


def _module():
    spec = importlib.util.spec_from_file_location("freeze_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _freeze(*args) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONUTF8": "1",
           "PYTHONPATH": os.pathsep.join(p for p in (str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")) if p)}
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, env=env,
                          cwd=REPO_ROOT)


def _data_rows(path: Path) -> int:
    with open(path, encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.reader(fh)) - 1


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """A results tree with one fake run, a manifest-only folder, an equivariance CSV and an images folder, analysed."""
    if not HAVE_DATA:
        pytest.skip("font or books not present (W1.4)")
    folder = tmp_path_factory.mktemp("freeze")
    sweep_path = _write_sweep(folder, blocks={k: ANALYSIS_SWEEP["blocks"][k] for k in ("V", "H")})
    results = folder / "results"
    dirs = run_sweep(sweep_path, READERS, DEFAULTS, DroppingBackend(), output_root=results / "runs", root=REPO_ROOT,
                     require_clean=False)
    run_dir = dirs[0]
    (run_dir / "images").mkdir(exist_ok=True)
    (run_dir / "images" / "page.png").write_bytes(b"not copied")
    only = results / "runs" / "20261013-1123_checkreaders_qwen_e4ec23b"
    only.mkdir()
    (only / "manifest.json").write_text(
        json.dumps({"run_id": only.name, "git_commit": "e4ec23b" + "0" * 33, "config": {"reader": "qwen"}}),
        encoding="utf-8")
    equivariance = results / "equivariance"
    equivariance.mkdir()
    (equivariance / "qwen_natural.csv").write_text("reader,text_type,layer,shift_px,e\nqwen,natural,0,0,0.0\n",
                                                   encoding="utf-8")
    (equivariance / "manifest.json").write_text("{}\n", encoding="utf-8")
    tables = folder / "tables"
    proc = _analyze(sweep_path, results / "runs", tables)
    assert proc.returncode == 0, proc.stderr
    return SimpleNamespace(folder=folder, sweep_path=sweep_path, results=results, run_dir=run_dir, tables=tables)


@needs_data
def test_freeze_copies_the_folders_without_images_and_lists_every_file_with_its_hash(results, tmp_path):
    m = _module()
    final = tmp_path / "final"
    assert m.freeze(results.results, final, root=REPO_ROOT) == ["runs", "equivariance"]
    copied = final / "runs" / results.run_dir.name
    assert (copied / "results.csv").exists() and (copied / "transcriptions").is_dir()
    assert (results.run_dir / "images").is_dir() and not (copied / "images").exists()
    readme = (final / "README.md").read_text(encoding="utf-8")
    manifest = json.loads((results.run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows, planned = _data_rows(results.run_dir / "results.csv"), manifest["config"]["rows_planned"]
    assert rows == planned > 0
    assert f"| `{results.run_dir.name}` | analysis | qwen | `{manifest['git_commit'][:7]}`" in readme
    assert f"| {rows} | {planned} | {manifest['timestamp_utc']} |" in readme
    assert "| `20261013-1123_checkreaders_qwen_e4ec23b` |  | qwen | `e4ec23b` | 0 |  |  |" in readme
    assert "- `runs/`:" in readme and "- `equivariance/`:" in readme and "`mtf/`" not in readme
    hashes = m.read_hashes(final / "README.md")
    files = [p for p in final.rglob("*") if p.is_file() and p.name != "README.md"]
    assert len(hashes) == len(files) > 10 and "README.md" not in hashes
    rel = f"runs/{results.run_dir.name}/results.csv"
    assert hashes[rel] == hashlib.sha256((final / rel).read_bytes()).hexdigest()
    assert "equivariance/qwen_natural.csv" in hashes
    assert m.check_hashes(final) == []


@needs_data
def test_a_second_freeze_is_refused_without_replace_and_redone_with_it(results, tmp_path):
    m = _module()
    final = tmp_path / "final"
    m.freeze(results.results, final, root=REPO_ROOT)
    with pytest.raises(FileExistsError, match="--replace"):
        m.freeze(results.results, final, root=REPO_ROOT)
    (final / "equivariance" / "qwen_natural.csv").write_text("damaged", encoding="utf-8")
    assert m.freeze(results.results, final, replace=True, root=REPO_ROOT) == ["runs", "equivariance"]
    assert m.check_hashes(final) == []
    assert (final / "equivariance" / "qwen_natural.csv").read_text(encoding="utf-8").startswith("reader,")


def test_freeze_with_nothing_to_copy_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing to freeze"):
        _module().freeze(tmp_path / "results", tmp_path / "final")


@needs_data
def test_check_hashes_names_a_changed_a_missing_and_an_unlisted_file(results, tmp_path):
    m = _module()
    final = tmp_path / "final"
    m.freeze(results.results, final, root=REPO_ROOT)
    rel = f"runs/{results.run_dir.name}/results.csv"
    (final / rel).write_bytes((final / rel).read_bytes() + b" ")
    (final / "equivariance" / "manifest.json").unlink()
    (final / "extra.txt").write_text("not frozen", encoding="utf-8")
    assert m.check_hashes(final) == ["missing: equivariance/manifest.json", "not in README: extra.txt", f"changed: {rel}"]
    assert m.check_hashes(tmp_path / "nowhere") == [f"{tmp_path / 'nowhere' / 'README.md'} is missing"]


@needs_data
def test_the_tables_regenerate_byte_for_byte_from_the_frozen_runs_and_a_changed_table_is_named(results, tmp_path):
    m = _module()
    final = tmp_path / "final"
    m.freeze(results.results, final, root=REPO_ROOT)
    assert m.check_tables(final, [results.sweep_path], results.tables, REPO_ROOT) == []
    tables = tmp_path / "tables"
    shutil.copytree(results.tables, tables)
    paired = tables / "analysis" / "paired.csv"
    paired.write_bytes(paired.read_bytes() + b"\n")
    (tables / "analysis" / "sizes.csv").unlink()
    problems = m.check_tables(final, [results.sweep_path], tables, REPO_ROOT)
    assert "analysis/paired.csv: differs from the regenerated table" in problems
    assert any(p.startswith("analysis/sizes.csv: regenerated but not in") for p in problems)
    assert len(problems) == 2
    problems = m.check_tables(final, [results.sweep_path], tmp_path / "none", REPO_ROOT)
    assert len(problems) == 1 and "make analyze" in problems[0]


@needs_data
def test_the_script_freezes_then_checks_and_exits_1_after_a_change(results, tmp_path):
    final = tmp_path / "final"
    proc = _freeze("--results", results.results, "--final", final)
    assert proc.returncode == 0, proc.stderr
    assert "runs, equivariance copied" in proc.stderr and "data-freeze-v1" in proc.stderr
    proc = _freeze("--results", results.results, "--final", final)
    assert proc.returncode == 1 and "--replace" in proc.stderr
    proc = _freeze("--check", "--final", final, "--sweep", results.sweep_path, "--tables", results.tables)
    assert proc.returncode == 0, proc.stderr
    assert "regenerate byte for byte" in proc.stderr and "0 problem(s)" in proc.stderr
    (final / "equivariance" / "qwen_natural.csv").write_text("edited", encoding="utf-8")
    proc = _freeze("--check", "--final", final)
    assert proc.returncode == 1 and "changed: equivariance/qwen_natural.csv" in proc.stderr
