"""Tests for scripts/20_analyze.py and sweep.load_results (build guide section 3.2; PREREGISTRATION.md sections 8, 9).

CPU only. One run of a small sweep with a fake backend whose transcription drops characters by a rule of the rendered
page's bytes, so that CER varies across conditions and pages while identical pixels, (H, 0) and (S, 0), give identical
transcriptions. The script then runs through subprocess, as make analyze runs it.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from conftest import REPO_ROOT
from mrr import grid
from mrr.manifest import RESULTS_COLUMNS
from mrr.readers import Transcription
from mrr.sweep import load_results, run_sweep, sweep_run_dirs
from mrr.text_source import BOOKS, TEXTS_DIR

SCRIPT = REPO_ROOT / "scripts" / "20_analyze.py"
READERS = {
    reader["name"]: reader
    for reader in yaml.safe_load((REPO_ROOT / "configs" / "readers.yaml").read_text(encoding="utf-8"))["readers"]
}
DEFAULTS = yaml.safe_load((REPO_ROOT / "configs" / "render_defaults.yaml").read_text(encoding="utf-8"))
FONT = REPO_ROOT / "data" / "external" / "fonts" / "source-sans-3" / "SourceSans3-Regular.ttf"
HAVE_DATA = FONT.exists() and all((REPO_ROOT / TEXTS_DIR / book.file).exists() for book in BOOKS.values())
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="font or books not present (W1.4)")
TABLES = ("checks", "harmonics", "paired", "subpixel", "sizes", "coherence_summary")

# Qwen only: V over its 28 px cell in steps of 7 (4 levels: groups 28 px {1} and 14 px {2}), H over its 14 px patch in
# steps of 7 (2 levels: group 14 px {1}), C over its 112 px sweep in steps of 14 (8 levels: groups 112 px {1, 2, 3}
# and 28 px {4}), S at the four quarter pixels, A's two modes at phase 0, and a block Z of three sizes on its own pages.
ANALYSIS_SWEEP = {
    "readers": ["qwen"],
    "seed": 20261014,
    "pages": {"natural": {"books": ["pg105"], "per_text_type": 3}, "random62": {"per_text_type": 3}, "z_per_text_type": 2},
    "repeats": 1,
    "font_file": "data/external/fonts/source-sans-3/SourceSans3-Regular.ttf",
    "font_size_px": 16,
    "aa_mode": "coverage_srgb",
    "blocks": {
        "V": {"render_mode": "conventional", "line_pitch_mode": "aligned", "axis": "y", "range": "cell", "step_px": 7},
        "H": {"render_mode": "patch_aware", "line_pitch_mode": "unaligned", "axis": "x", "range": "patch", "step_px": 7},
        "C": {"render_mode": "conventional", "line_pitch_mode": "unaligned", "axis": "x", "range": "sweep", "step_px": 14},
        "S": {"render_mode": "patch_aware", "line_pitch_mode": "unaligned", "axis": "x", "phases_px": [0.0, 0.25, 0.5, 0.75]},
        "A": {"render_mode": "conventional", "line_pitch_mode": "unaligned", "axis": "x", "phases_px": [0.0],
              "aa_mode": ["binary", "coverage_linear"]},
        "Z": {"line_pitch_mode": "unaligned", "phases_px": [0.0], "render_mode": ["conventional", "patch_aware"],
              "font_size_px": [12, 20, 24], "paginate": "per_condition"},
    },
}
SHARED_CONDITIONS = 4 + 2 + 8 + 4 + 2  # V, H, C, S, A for Qwen
SHARED_PAGES = 6  # 3 natural + 3 random62
Z_ROWS = 6 * 2 * 2  # 3 sizes x 2 modes, each on 2 natural + 2 random62 pages of its own
ROWS = SHARED_CONDITIONS * SHARED_PAGES + Z_ROWS  # 144
HARMONIC_COLUMNS = {"V": 3, "H": 1, "C": 7}  # every identifiable bin of each block: n_levels - 1 columns
RANGE_PX = {"V": 28, "H": 14, "C": 112}


class DroppingBackend:
    """readers.py's five functions without a model: the transcription is the page's reference with characters dropped.

    The share dropped (2 to 32 percent) and the characters chosen come from the SHA-1 of the rendered page, so the
    CER differs between conditions and pages but identical pixels give identical text. About one row in ten claims to
    have hit max_new_tokens, so checks.csv has a hit_limit count to check.
    """

    def __init__(self):
        self.reference = ""

    def load_reader(self, cfg):
        return SimpleNamespace(name=cfg["name"], cfg=cfg, model="a model")

    def unload_reader(self, reader):
        reader.model = None

    def count_vision_tokens(self, reader, image):
        return grid.token_count(reader.cfg, image.shape[0], image.shape[1])

    def count_text_tokens(self, reader, text):
        self.reference = text  # the runner counts the page's tokens before it transcribes the page's rows
        return len(text.split())

    def transcribe_full(self, reader, image, prompt, max_new_tokens=None):
        assert reader.model is not None
        digest = hashlib.sha1(image.tobytes()).digest()
        rng = random.Random(digest)
        share = 0.02 + 0.30 * digest[0] / 255
        text = "".join(ch for ch in self.reference if rng.random() >= share)
        return Transcription(text=text, new_tokens=len(text.split()), hit_limit=digest[1] < 26,
                             latency_s=0.001 + digest[2] / 25500)


def _write_sweep(folder: Path, **changes) -> Path:
    cfg = copy.deepcopy(ANALYSIS_SWEEP)
    cfg.update(changes)
    path = folder / "analysis.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _analyze(sweep_path: Path, runs_root: Path, out_dir: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONUTF8": "1",
           "PYTHONPATH": os.pathsep.join(p for p in (str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")) if p)}
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(sweep_path), "--runs-root", str(runs_root), "--out-dir", str(out_dir)],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT,
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _header(path: Path) -> list[str]:
    with open(path, encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RESULTS_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _script_module():
    spec = importlib.util.spec_from_file_location("analyze_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    """One run of ANALYSIS_SWEEP with the dropping backend, analysed once by the script."""
    if not HAVE_DATA:
        pytest.skip("font or books not present (W1.4)")
    folder = tmp_path_factory.mktemp("analysis")
    sweep_path = _write_sweep(folder)
    runs_root = folder / "runs"
    dirs = run_sweep(sweep_path, READERS, DEFAULTS, DroppingBackend(), output_root=runs_root, root=REPO_ROOT,
                     require_clean=False)
    assert len(dirs) == 1
    out_root = folder / "tables"
    result = _analyze(sweep_path, runs_root, out_root)
    return SimpleNamespace(folder=folder, sweep_path=sweep_path, runs_root=runs_root, run_dir=dirs[0],
                           out_dir=out_root / sweep_path.stem, result=result)


def _copy_runs(analysis, tmp_path: Path) -> Path:
    """A private copy of the runs root, so that a test can damage it without touching the module fixture."""
    runs_root = tmp_path / "runs"
    shutil.copytree(analysis.runs_root, runs_root)
    return runs_root


# --------------------------------------------------------------------------- the script on the fake run


@needs_data
def test_the_script_exits_0_and_writes_every_table_with_status_and_family_and_a_manifest(analysis):
    assert analysis.result.returncode == 0, analysis.result.stderr
    for name in TABLES:
        header = _header(analysis.out_dir / f"{name}.csv")
        assert "status" in header and "family" in header, name
    manifest = json.loads((analysis.out_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    assert cfg["sweep_sha1"] == hashlib.sha1(analysis.sweep_path.read_bytes()).hexdigest()
    assert cfg["sweep_name"] == "analysis" and cfg["run_dirs"] == [analysis.run_dir.name]
    assert cfg["rows_loaded"] == cfg["rows_analysed"] == ROWS and cfg["pages_excluded"] == {}
    assert cfg["resamples"] == 2000 and cfg["seed"] == 20261014 and cfg["readers_without_a_run"] == []
    assert sorted(cfg["tables"]) == sorted(TABLES) and manifest["run_id"].split("_")[1:3] == ["analysis", "analysis"]
    for name in TABLES:
        rows = _rows(analysis.out_dir / f"{name}.csv")
        assert len(rows) == cfg["tables"][name], name
        assert all(r["status"] == "preregistered" for r in rows), name


@needs_data
def test_checks_counts_rows_hit_limits_determinism_pairs_and_no_exclusion(analysis):
    (row,) = _rows(analysis.out_dir / "checks.csv")
    results = _rows(analysis.run_dir / "results.csv")
    manifest = json.loads((analysis.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert row["run_dir"] == analysis.run_dir.name and row["reader"] == "qwen"
    assert int(row["rows_done"]) == len(results) == ROWS
    assert int(row["rows_planned"]) == manifest["config"]["rows_planned"] == ROWS
    assert int(row["repeated_keys"]) == 0 and int(row["missing_transcriptions"]) == 0
    hits = sum(r["hit_limit"] == "True" for r in results)
    assert int(row["hit_limit_count"]) == hits and math.isclose(float(row["hit_limit_share"]), hits / ROWS)
    assert int(row["determinism_pairs"]) == SHARED_PAGES and int(row["determinism_mismatches"]) == 0
    assert int(row["pages_seen"]) == SHARED_PAGES + Z_ROWS and int(row["pages_excluded"]) == 0
    assert row["excluded_page_ids"] == "" and float(row["median_latency_s"]) > 0
    assert row["family"] == "checks"


@needs_data
def test_harmonics_tests_the_preregistered_groups_with_holm_within_each_family(analysis):
    rows = _rows(analysis.out_dir / "harmonics.csv")
    seen = {(r["text_type"], r["block"], int(r["period_px"])): r for r in rows}
    assert len(rows) == len(seen) == 10
    for text_type in ("natural", "random62"):
        expected = {
            ("V", 28): ("1", 2, 4, f"a_{text_type}"), ("V", 14): ("2", 1, 4, f"a_{text_type}"),
            ("H", 14): ("1", 1, 2, f"a_{text_type}"),
            ("C", 112): ("1 2 3", 6, 8, "control_C"), ("C", 28): ("4", 1, 8, "control_C"),
        }
        for (block, period), (bins, q, n_levels, family) in expected.items():
            r = seen[(text_type, block, period)]
            assert r["bins"] == bins and int(r["q"]) == q and int(r["n_levels"]) == n_levels, (block, period)
            assert r["family"] == family and r["status"] == "preregistered"
            assert r["axis"] == ("y" if block == "V" else "x") and int(r["range_px"]) == RANGE_PX[block]
            assert int(r["n_pages"]) == 3 and int(r["n_rows"]) == 3 * n_levels
            assert int(r["df1"]) == q and int(r["df2"]) == 3 * n_levels - 3 - HARMONIC_COLUMNS[block]
            assert 0 <= float(r["p"]) <= 1 and float(r["p"]) <= float(r["p_holm"]) <= 1
            assert float(r["rms"]) >= 0 and float(r["rms_low"]) <= float(r["rms_high"])
            assert 0 <= float(r["share"]) <= 1 and float(r["sigma"]) > 0
        a_family = [seen[(text_type, "V", 28)], seen[(text_type, "V", 14)], seen[(text_type, "H", 14)]]
        assert math.isclose(sum(float(x["share"]) for x in a_family[:2]), 1.0)  # V's two groups share V's variance
        smallest = min(a_family, key=lambda x: float(x["p"]))
        assert math.isclose(float(smallest["p_holm"]), min(1.0, 3 * float(smallest["p"])))  # Holm's first step


@needs_data
def test_paired_compares_h_at_phase_0_with_c_pooled_and_each_aa_mode_with_c_at_phase_0(analysis):
    rows = _rows(analysis.out_dir / "paired.csv")
    by = {(r["text_type"], r["comparison"]): r for r in rows}
    assert len(rows) == len(by) == 6
    for text_type in ("natural", "random62"):
        c = by[(text_type, "H_phi0_vs_C_pooled")]
        assert c["family"] == f"c_{text_type}" and int(c["n"]) == 3
        assert c["method"] in {"exact", "permutation", "asymptotic"}
        assert 0 < float(c["p"]) <= float(c["p_holm"]) <= 1
        assert float(c["diff_low"]) <= float(c["median_diff"]) <= float(c["diff_high"])
        modes = [by[(text_type, f"A_{mode}_vs_C_phi0")] for mode in ("binary", "coverage_linear")]
        assert all(m["family"] == "A" and int(m["n"]) == 3 for m in modes)
        smallest = min(modes, key=lambda m: float(m["p"]))
        assert math.isclose(float(smallest["p_holm"]), min(1.0, 2 * float(smallest["p"])))


@needs_data
def test_subpixel_has_one_f_test_per_text_type_and_the_pooled_cer_at_the_four_phases(analysis):
    rows = _rows(analysis.out_dir / "subpixel.csv")
    assert [r["text_type"] for r in rows] == ["natural", "random62"]
    for r in rows:
        assert r["family"] == "S" and int(r["df1"]) == 3 and int(r["df2"]) == 12 - 3 - 3
        assert int(r["n_pages"]) == 3 and int(r["n_levels"]) == 4 and int(r["n_rows"]) == 12
        assert 0 <= float(r["p"]) <= 1 and float(r["p_holm"]) == float(r["p"])
        for phase in ("0", "0.25", "0.5", "0.75"):
            cer, low, high = (float(r[f"{k}_at_{phase}"]) for k in ("cer", "cer_low", "cer_high"))
            assert 0 < cer < 1 and low <= cer <= high


@needs_data
def test_sizes_has_block_z_with_tokens_characters_per_token_the_plateau_and_s_star(analysis):
    rows = _rows(analysis.out_dir / "sizes.csv")
    assert len(rows) == 2 * 2 * 3
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in rows:
        groups.setdefault((r["text_type"], r["render_mode"]), []).append(r)
    assert sorted(groups) == [(t, m) for t in ("natural", "random62") for m in ("conventional", "patch_aware")]
    for members in groups.values():
        assert [float(r["font_size_px"]) for r in members] == [12.0, 20.0, 24.0]
        for r in members:
            assert r["family"] == "description" and int(r["vision_tokens"]) == 256
            assert int(r["n_pages"]) == 2 and int(r["n_rows"]) == 2
            assert float(r["cer_low"]) <= float(r["cer"]) <= float(r["cer_high"])
            assert 0 < float(r["symbols_per_token"]) < float(r["chars_per_token"])
        plateau = {r["cer_plateau"] for r in members}
        assert len(plateau) == 1 and float(plateau.pop()) > 0
        s_star = {r["s_star_px"] for r in members}
        assert len(s_star) == 1
        s_star = s_star.pop()
        within = [float(r["font_size_px"]) for r in members if float(r["cer"]) <= float(r["cer_plateau"]) + 0.01]
        assert (s_star == "" and not within) or float(s_star) == within[0]


@needs_data
def test_coherence_summary_puts_the_matching_rms_beside_the_mean_r(analysis):
    rows = _rows(analysis.out_dir / "coherence_summary.csv")
    harmonics = {(r["text_type"], r["block"], int(r["period_px"])): float(r["rms"])
                 for r in _rows(analysis.out_dir / "harmonics.csv")}
    assert len(rows) == 2 * (5 + 2) * 3 * 3  # two text types; V, H, C, S, A and the two modes of Z; 3 periods; 3 harmonics
    for r in rows:
        assert r["family"] == "description" and 0 <= float(r["R_min"]) <= float(r["R_mean"]) <= float(r["R_max"]) <= 1
        key = (r["text_type"], r["block"], int(r["period_px"]))
        if key in harmonics:
            assert math.isclose(float(r["rms"]), harmonics[key])
        else:
            assert r["rms"] == ""
    v = [r for r in rows if r["block"] == "V" and r["text_type"] == "natural"]
    assert {r["axis"] for r in v} == {"y"} and all(int(r["n_pages"]) == 3 for r in v)
    assert all(int(r["n_pages"]) == 6 for r in rows if r["block"] == "Z")  # 3 sizes x 2 pages, no size column


@needs_data
def test_a_page_missing_one_planned_condition_is_counted_in_checks_and_leaves_every_other_table(analysis, tmp_path):
    runs_root = _copy_runs(analysis, tmp_path)
    results_path = runs_root / analysis.run_dir.name / "results.csv"
    rows = _rows(results_path)
    victim = next(r for r in rows if r["block"] == "H" and r["text_type"] == "random62")
    _write_rows(results_path, [r for r in rows if r is not victim])  # as a run stopped at its deadline would leave it
    result = _analyze(analysis.sweep_path, runs_root, tmp_path / "tables")
    assert result.returncode == 0, result.stderr
    out_dir = tmp_path / "tables" / "analysis"
    (check,) = _rows(out_dir / "checks.csv")
    assert int(check["rows_done"]) == ROWS - 1 and int(check["rows_planned"]) == ROWS
    assert int(check["pages_excluded"]) == 1 and check["excluded_page_ids"] == victim["page_id"]
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))["config"]
    assert manifest["pages_excluded"] == {analysis.run_dir.name: [victim["page_id"]]}
    assert manifest["rows_loaded"] == ROWS - 1 and manifest["rows_analysed"] == ROWS - SHARED_CONDITIONS
    for r in _rows(out_dir / "harmonics.csv"):
        assert int(r["n_pages"]) == (2 if r["text_type"] == "random62" else 3)
    for r in _rows(out_dir / "paired.csv"):
        assert int(r["n"]) == (2 if r["text_type"] == "random62" else 3)
    for r in _rows(out_dir / "subpixel.csv"):
        assert int(r["n_pages"]) == (2 if r["text_type"] == "random62" else 3)
    for r in _rows(out_dir / "coherence_summary.csv"):
        if r["block"] != "Z":
            assert int(r["n_pages"]) == (2 if r["text_type"] == "random62" else 3)


@needs_data
def test_a_second_run_folder_holding_the_same_key_raises_naming_it(analysis, tmp_path):
    runs_root = _copy_runs(analysis, tmp_path)
    shutil.copytree(runs_root / analysis.run_dir.name, runs_root / (analysis.run_dir.name + "_again"))
    with pytest.raises(ValueError, match="repeated key") as info:
        load_results(analysis.sweep_path, runs_root)
    message = str(info.value)
    assert "qwen" in message and analysis.run_dir.name in message and (analysis.run_dir.name + "_again") in message
    assert any(r["page_id"] in message for r in _rows(analysis.run_dir / "results.csv"))
    result = _analyze(analysis.sweep_path, runs_root, tmp_path / "tables")
    assert result.returncode != 0 and "repeated key" in result.stderr


@needs_data
def test_folders_of_another_sweep_or_without_a_manifest_are_ignored_and_an_edited_sweep_is_refused(analysis, tmp_path):
    runs_root = _copy_runs(analysis, tmp_path)
    other = runs_root / "20261014-0000_other_qwen_abcdef1"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({"config": {"sweep_name": "other", "sweep_sha1": "0" * 40}}), encoding="utf-8")
    (other / "results.csv").write_text("not,a,results,file\n1,2,3,4\n", encoding="utf-8")
    (runs_root / "no_manifest_here").mkdir()
    (runs_root / "a_file_not_a_folder.txt").write_text("x", encoding="utf-8")
    assert sweep_run_dirs(analysis.sweep_path, runs_root) == [runs_root / analysis.run_dir.name]
    assert len(load_results(analysis.sweep_path, runs_root)) == ROWS
    edited = _write_sweep(tmp_path, seed=20261015)  # the same stem, another SHA-1
    with pytest.raises(ValueError, match="frozen config"):
        sweep_run_dirs(edited, runs_root)
    with pytest.raises(ValueError, match="frozen config"):
        load_results(edited, runs_root)


@needs_data
def test_hash_1e5123456789_is_read_back_as_the_same_string(analysis, tmp_path):
    runs_root = _copy_runs(analysis, tmp_path)
    results_path = runs_root / analysis.run_dir.name / "results.csv"
    rows = _rows(results_path)
    rows[0]["condition_hash"] = "1e5123456789"
    rows[1]["condition_hash"] = "000000000042"
    _write_rows(results_path, rows)
    df = load_results(analysis.sweep_path, runs_root)
    hashes = df["condition_hash"].tolist()
    assert "1e5123456789" in hashes and "000000000042" in hashes
    assert all(isinstance(h, str) and len(h) == 12 for h in hashes)
    keys = list(zip(df["reader"], df["page_id"], df["block"], df["condition_hash"]))
    assert keys == sorted(keys)


# --------------------------------------------------------------------------- pieces of the script


def test_page_key_tells_a_block_z_page_from_a_shared_page():
    module = _script_module()
    page_key = module.RunFolder.page_key
    assert page_key("pg105_p0000") == "shared" and page_key("rnd20261014_p0003") == "shared"
    assert page_key("pg105_z12_conv_p0000") == "z:conventional:12"
    assert page_key("rnd20261014_z10.5_pa_p0001") == "z:patch_aware:10.5"


def test_write_table_puts_the_fixed_columns_first_and_nan_as_an_empty_cell(tmp_path):
    module = _script_module()
    path = tmp_path / "t.csv"
    n = module.write_table(path, [{"b": float("nan"), "a": 1, "extra": "x"}, {"a": 2, "b": 0.5}], ["a", "b"])
    assert n == 2 and path.read_text(encoding="utf-8") == "a,b,extra\n1,,x\n2,0.5,\n"


def test_the_script_without_a_run_folder_exits_1(tmp_path):
    sweep_path = _write_sweep(tmp_path)
    result = _analyze(sweep_path, tmp_path / "runs", tmp_path / "tables")
    assert result.returncode == 1 and "no run folder" in result.stderr
    assert not (tmp_path / "tables").exists()
