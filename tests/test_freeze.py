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


def test_each_sweep_is_checked_with_the_script_that_wrote_its_tables():
    m = _module()
    assert m.analyzer(Path("configs/sweeps/study2.yaml")) == "22_analyze_study2.py"  # PREREGISTRATION_2.md, 9.7
    for name in ("full", "ladder", "analysis"):
        assert m.analyzer(Path(f"configs/sweeps/{name}.yaml")) == "20_analyze.py"
    for script in {m.ANALYZE, *m.ANALYZERS.values()}:
        assert (REPO_ROOT / "scripts" / script).is_file()


def test_the_stage0_fit_is_checked_over_the_sweeps_that_wrote_its_tables():
    # ANALYSIS_PLAN_3.md section 2.7: the check runs scripts/25_fit_stage0.py over its default sweeps, which wrote
    # reports/tables/s3_stage0/, with no other argument.
    m = _module()
    script, configs, extra = m.FITS["s3_stage0"]
    spec = importlib.util.spec_from_file_location("fit_stage0", REPO_ROOT / "scripts" / script)
    fit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit)
    assert configs == tuple(fit.DEFAULT_CONFIGS) and extra == ()
    assert all((REPO_ROOT / c).is_file() for c in configs)


def test_the_rescoring_fit_is_checked_with_the_sweep_and_name_that_wrote_its_tables(tmp_path, monkeypatch):
    # ANALYSIS_PLAN_3.md section 3.2: scripts/26_rescore_study2.py reads one sweep (--config, not --configs) and
    # wrote reports/tables/study2_rescored/ with its defaults.
    m = _module()
    script, configs, extra = m.FITS["study2_rescored"]
    spec = importlib.util.spec_from_file_location("rescore_study2", REPO_ROOT / "scripts" / script)
    rescore = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rescore)
    defaults = rescore.parse_args([])
    assert configs == () and extra == ("--config", defaults.config) and defaults.name == "study2_rescored"
    assert (REPO_ROOT / defaults.config).is_file()
    seen = []
    monkeypatch.setattr(m, "_regenerate_and_compare", lambda name, script, args, *rest: seen.append(list(args)) or [])
    (tmp_path / "study2_rescored").mkdir()
    assert m.check_fits(tmp_path / "final", ["study2_rescored"], tmp_path) == []
    assert seen == [["--name", "study2_rescored", "--runs-root", str(tmp_path / "final" / "runs"),
                     "--config", "configs/sweeps/study2.yaml"]]


def test_check_fits_names_an_unknown_fit_and_missing_tables(tmp_path):
    m = _module()
    assert m.check_fits(tmp_path, ["nothing"], tmp_path)[0].startswith("nothing: not a fit")
    problems = m.check_fits(tmp_path, ["s3_stage0"], tmp_path / "tables")
    assert len(problems) == 1 and "is missing: run scripts/25_fit_stage0.py first" in problems[0]


@needs_data
def test_a_fit_regenerates_byte_for_byte_and_a_changed_table_is_named(tmp_path, monkeypatch):
    # Study 1's ladder through scripts/25_fit_stage0.py, with few resamples, stands in for stage 0's three sweeps.
    m = _module()
    quick = ("--resamples", "20", "--loso-resamples", "3", "--subsets", "3", "--no-tokens")
    monkeypatch.setitem(m.FITS, "ladder_fit", ("25_fit_stage0.py", ("configs/sweeps/ladder.yaml",), quick))
    tables, runs = tmp_path / "tables", REPO_ROOT / "results" / "runs"
    env = {**os.environ, "PYTHONUTF8": "1",
           "PYTHONPATH": os.pathsep.join(p for p in (str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")) if p)}
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "25_fit_stage0.py"), "--configs",
                           "configs/sweeps/ladder.yaml", "--name", "ladder_fit", "--runs-root", str(runs),
                           "--out-dir", str(tables), *quick], capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr
    assert m.check_fits(runs.parent, ["ladder_fit"], tables, REPO_ROOT) == []
    cells = tables / "ladder_fit" / "cells.csv"
    cells.write_bytes(cells.read_bytes() + b"\n")
    assert m.check_fits(runs.parent, ["ladder_fit"], tables, REPO_ROOT) == [
        "ladder_fit/cells.csv: differs from the regenerated table"]


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


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
def test_progress_logs_under_results_are_not_ignored_by_git():
    # The freeze lists the SHA-256 of every file it copies, progress.log included; a file git ignores would be missing
    # from every clone and the check would fail there (pre-analysis audit, 15 October).
    inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=REPO_ROOT, capture_output=True, text=True)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        pytest.skip("not a git work tree (an unpacked archive): nothing to check")
    probe = "results/final/runs/20990101-0000_full_qwen_0000000/progress.log"
    done = subprocess.run(["git", "check-ignore", "-q", probe], cwd=REPO_ROOT, capture_output=True, text=True)
    assert done.returncode == 1, "git ignores progress.log under results/: .gitignore needs !results/**/progress.log"
    other = subprocess.run(["git", "check-ignore", "-q", "scratch.log"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert other.returncode == 0  # other logs stay ignored
