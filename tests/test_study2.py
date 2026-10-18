"""Tests for src/mrr/study2.py, scripts/22_analyze_study2.py and Study 2's figures (PREREGISTRATION_2.md sections 6, 9).

The end-to-end fixture runs a reduced copy of configs/sweeps/study2.yaml (qwen, smolvlm500m and the validation reader
qwen7b; 7 random and 3 prose pages) with a backend that returns a placeholder, then rewrites every transcription with
an error of known kind and rescored with mrr.score, as the runner scores: the conventional rows (S) change one O into
0 (or 0 into O), the slot rows (Q0) gain a space inside every seventh pair of neighbouring characters, the cut rows
(M at P / 2) swap one of I, l and 1, the density rows (block D) change one character, and the contained rows (M0, M3)
are exact. Each hypothesis of section 6 then has a known answer.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import REPO_ROOT
from mrr import grid, plotting, study2
from mrr.readers import Transcription
from mrr.score import cer, char_counts, char_counts_letters_only, spaces_inserted, wer
from mrr.sweep import run_sweep
from mrr.text_source import BOOKS, TEXTS_DIR

SCRIPT = REPO_ROOT / "scripts" / "22_analyze_study2.py"
FIGURES = REPO_ROOT / "scripts" / "30_make_figures.py"
STUDY2 = yaml.safe_load((REPO_ROOT / "configs" / "sweeps" / "study2.yaml").read_text(encoding="utf-8"))
READERS = {r["name"]: r for r in
           yaml.safe_load((REPO_ROOT / "configs" / "readers_study2.yaml").read_text(encoding="utf-8"))["readers"]}
DEFAULTS = yaml.safe_load((REPO_ROOT / "configs" / "render_defaults.yaml").read_text(encoding="utf-8"))
FONTS = [REPO_ROOT / STUDY2["font_file"], REPO_ROOT / STUDY2["blocks"]["M"]["font_file"]]
HAVE_DATA = all(f.exists() for f in FONTS) and all((REPO_ROOT / TEXTS_DIR / b.file).exists() for b in BOOKS.values())
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="fonts or books not present")

TEST_READERS = ("qwen", "smolvlm500m", "qwen7b")
N_RANDOM, N_NATURAL = 7, 3  # 7 pages: an all-negative exact Wilcoxon gives p = 2 / 2**7, under alpha after Holm x 2
D_SIZE = 28
ROWS = {"qwen": 8 * N_RANDOM + 6 * N_NATURAL, "smolvlm500m": 6 * N_NATURAL, "qwen7b": 8 * N_RANDOM + 6 * N_NATURAL}
SWAP_IL1 = {"I": "l", "l": "1", "1": "I"}


# --------------------------------------------------------------------------- the condition codes and groups


def test_condition_codes_follow_section_3_and_block_d_is_named_by_its_size():
    assert study2.condition_code("S", 5.0, 25) == "S5"
    assert study2.condition_code("Q", 0.0, 25) == "Q0"
    assert study2.condition_code("M", 7.0, 23.333) == "M7"
    assert study2.condition_code("D", 0.0, 28) == "S28"
    assert study2.condition_order(["M7", "S28", "Q0", "S10", "M0", "S0", "M3", "S5"]) == [
        "S0", "S5", "S10", "Q0", "M0", "M3", "M7", "S28"]


def test_groups_are_those_of_section_6_and_the_cut_phase_is_half_the_patch():
    assert study2.group_of("S", 10.0, 14) == study2.CONVENTIONAL
    assert study2.group_of("Q", 0.0, 14) == study2.SLOT
    assert study2.group_of("M", 0.0, 14) == study2.group_of("M", 3.0, 16) == study2.CONTAINED
    assert study2.group_of("M", 7.0, 14) == study2.group_of("M", 8.0, 16) == study2.CUT
    assert study2.group_of("D", 0.0, 14) == study2.DENSITY
    for phase, patch in ((8.0, 14), (7.0, 16), (5.0, 14)):
        with pytest.raises(ValueError, match="not a condition of Study 2"):
            study2.group_of("M", phase, patch)
    assert study2.role("qwen7b") == study2.role("internvl3_2b") == "validation"
    assert study2.role("qwen") == study2.role("smolvlm500m") == "development"


def test_letter_confusions_count_by_character_without_spaces():
    subs, dels, letters = study2.letter_confusions("O0Ixyl1 abc", "00lxyI1 aBc")
    assert subs == {("O", "0"): 1, ("I", "l"): 1, ("l", "I"): 1, ("b", "B"): 1} and not dels and letters == 10
    assert study2.confusion_totals(subs) == {"O/0": 1, "I/l/1": 2, "case_pair": 1}
    subs, dels, letters = study2.letter_confusions("ab cd", "abcd")  # a dropped space is no letter confusion
    assert not subs and not dels and letters == 4
    subs, dels, _ = study2.letter_confusions("abcd", "abd")
    assert not subs and dels == {"c": 1}
    with pytest.raises(ValueError, match="no letters"):
        study2.letter_confusions("   ", "x")


def _frame(rows):
    base = {"hits": 0, "substitutions": 0, "deletions": 0, "insertions": 0, "hits_ns": 0, "substitutions_ns": 0,
            "deletions_ns": 0, "insertions_ns": 0, "spaces_inserted": 0, "wer": 0.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_page_values_pool_the_counts_of_a_group_on_a_page():
    df = _frame([
        {"page_id": "p1", "group": "contained", "condition": "M0", "hits": 9, "substitutions": 1, "spaces_inserted": 2,
         "hits_ns": 8, "substitutions_ns": 0, "wer": 0.5},
        {"page_id": "p1", "group": "contained", "condition": "M3", "hits": 7, "substitutions": 2, "deletions": 1,
         "insertions": 3, "hits_ns": 6, "deletions_ns": 2, "wer": 0.25},
    ])
    assert study2.page_values(df, "cer") == {"p1": pytest.approx((1 + 2 + 1 + 3) / (9 + 1 + 7 + 2 + 1))}
    assert study2.page_values(df, "letters") == {"p1": pytest.approx(2 / 16)}
    assert study2.page_values(df, "spaces") == {"p1": pytest.approx(2 / 20)}
    assert study2.page_values(df, "wer") == {"p1": pytest.approx(0.375)}
    assert study2.side_rows(df, "contained").shape[0] == 2 and study2.side_rows(df, "M3").shape[0] == 1
    with pytest.raises(ValueError, match="no pooled form"):
        study2.page_values(df, "bleu")


# --------------------------------------------------------------------------- the paired rows and their verdicts


def _script():
    spec = importlib.util.spec_from_file_location("analyze_study2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_family_is_holm_adjusted_across_its_readers_and_supported_needs_the_predicted_sign():
    mod = _script()
    a = np.full(7, 0.1)
    lower, higher = a - np.arange(1, 8) / 100, a + np.arange(1, 8) / 100
    rows = [
        mod.paired_row("gap", "H3", "qwen", "random62", "slot", "contained", "spaces", a, lower, -1, None, True, "x"),
        mod.paired_row("gap", "H3", "smolvlm", "random62", "slot", "contained", "spaces", a, higher, -1, None, True, "x"),
        mod.paired_row("gap", "H3", "qwen", "random62", "conventional", "contained", "spaces", a, a, None, 0.005, False,
                       "x"),
    ]
    mod._fill_holm_and_verdicts(rows)
    down, up, parity = rows
    assert down["p"] == pytest.approx(2 / 2 ** 7) and down["p_holm"] == pytest.approx(2 * 2 / 2 ** 7)
    assert down["significant"] is True and down["supported"] is True and down["median_diff"] < 0
    assert up["significant"] is True and up["supported"] is False  # significant, but the wrong sign
    assert parity["method"].startswith("undefined") and math.isnan(parity["p_holm"]) and parity["significant"] == ""
    assert parity["median_diff"] == 0.0 and parity["ci_high"] == 0.0 and parity["passes"] is True
    assert down["contrast"] == "contained_vs_slot" and "_holm" not in down


def test_a_bound_passes_only_when_the_whole_interval_is_below_it():
    mod = _script()
    a = np.zeros(8)
    worse = np.linspace(0.005, 0.03, 8)
    row = mod.paired_row("prose", "H6", "smolvlm", "natural", "conventional", "contained", "cer", a, worse, None,
                         0.01, True, "x")
    assert row["ci_high"] >= 0.01 and row["passes"] is False
    row = mod.paired_row("prose", "H6", "qwen", "natural", "conventional", "contained", "cer", a, worse - 0.03, None,
                         0.01, True, "x")
    assert row["ci_high"] < 0.01 and row["passes"] is True


# --------------------------------------------------------------------------- the reduced Study 2 run


class PlaceholderBackend:
    """The five functions of readers.py without a model; every transcription is replaced by the fixture."""

    def load_reader(self, cfg):
        return SimpleNamespace(name=cfg["name"], cfg=cfg, model="a model")

    def unload_reader(self, reader):
        reader.model = None

    def count_vision_tokens(self, reader, image):
        return grid.token_count(reader.cfg, image.shape[0], image.shape[1])

    def count_text_tokens(self, reader, text):
        return len(text.split())

    def transcribe_full(self, reader, image, prompt, max_new_tokens=None):
        return Transcription(text="placeholder", new_tokens=1, hit_limit=False, latency_s=0.01)


def _first(text, chars=None):
    return next(i for i, c in enumerate(text) if (c in chars if chars else not c.isspace()))


def _swap(text, i, new):
    return text[:i] + new + text[i + 1:]


def _one_changed(ref):
    i = _first(ref)
    return _swap(ref, i, "y" if ref[i] == "x" else "x")


def _o0(ref):
    if not any(c in "O0" for c in ref):
        return _one_changed(ref)
    i = _first(ref, "O0")
    return _swap(ref, i, "0" if ref[i] == "O" else "O")


def _il1(ref):
    if not any(c in SWAP_IL1 for c in ref):
        return _one_changed(ref)
    i = _first(ref, "Il1")
    return _swap(ref, i, SWAP_IL1[ref[i]])


def _gaps(ref, every=7):
    out, pairs, inserted = [], 0, 0
    for here, after in zip(ref, ref[1:] + " "):
        out.append(here)
        if not here.isspace() and not after.isspace():
            pairs += 1
            if pairs % every == 0:
                out.append(" ")
                inserted += 1
    return "".join(out), inserted


def _hypothesis(group, ref):
    """The controlled transcription of a row of ``group``, and the spaces it inserts."""
    if group == study2.CONVENTIONAL:
        return _o0(ref), 0
    if group == study2.SLOT:
        return _gaps(ref)
    if group == study2.CUT:
        return _il1(ref), 0
    if group == study2.DENSITY:
        return _one_changed(ref), 0
    return ref, 0


def _rescore(run_dir: Path, patch: int) -> dict[str, int]:
    """Rewrite every transcription of ``run_dir`` with ``_hypothesis`` and its row with mrr.score's counts."""
    with open(run_dir / "results.csv", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header, rows = reader.fieldnames, list(reader)
    inserted = {}
    for row in rows:
        ref = (run_dir / "references" / f"{row['page_id']}.txt").read_text(encoding="utf-8")
        group = study2.group_of(row["block"], float(row["phase_x_px"]), patch)
        hyp, n = _hypothesis(group, ref)
        inserted[group] = inserted.get(group, 0) + n
        (run_dir / row["transcription_file"]).write_text(hyp, encoding="utf-8")
        counts, letters = char_counts(ref, hyp), char_counts_letters_only(ref, hyp)
        row.update(cer=repr(cer(ref, hyp)), wer=repr(wer(ref, hyp)), hits=counts.hits,
                   substitutions=counts.substitutions, deletions=counts.deletions, insertions=counts.insertions,
                   hits_ns=letters.hits, substitutions_ns=letters.substitutions, deletions_ns=letters.deletions,
                   insertions_ns=letters.insertions, spaces_inserted=spaces_inserted(ref, hyp))
    with open(run_dir / "results.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return inserted


def _reduced_sweep(folder: Path) -> Path:
    cfg = copy.deepcopy(STUDY2)
    cfg["readers"] = list(TEST_READERS)
    cfg["pages"] = {
        "natural": {"books": STUDY2["pages"]["natural"]["books"],
                    "per_text_type": {r: N_NATURAL for r in TEST_READERS}},
        "random62": {"per_text_type": {"qwen": N_RANDOM, "smolvlm500m": 0, "qwen7b": N_RANDOM}},
    }
    cfg["blocks"]["D"]["font_size_px"] = D_SIZE
    path = folder / "study2_reduced.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _subprocess(script: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONUTF8": "1",
           "PYTHONPATH": os.pathsep.join(p for p in (str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")) if p)}
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, env=env, cwd=REPO_ROOT)


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """One reduced Study 2 run, rewritten with controlled errors, analysed and drawn once."""
    if not HAVE_DATA:
        pytest.skip("fonts or books not present")
    folder = tmp_path_factory.mktemp("study2")
    sweep_path = _reduced_sweep(folder)
    runs_root = folder / "runs"
    dirs = run_sweep(sweep_path, READERS, DEFAULTS, PlaceholderBackend(), output_root=runs_root, root=REPO_ROOT,
                     require_clean=False, save_images="none")
    assert len(dirs) == len(TEST_READERS)
    inserted = {}
    for run_dir in dirs:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        name = manifest["config"]["reader"]["name"]
        inserted[name] = _rescore(run_dir, int(READERS[name]["patch_size"]))
    tables = folder / "tables"
    result = _subprocess(SCRIPT, "--config", str(sweep_path), "--runs-root", str(runs_root), "--out-dir", str(tables))
    figures = _subprocess(FIGURES, "--config", str(sweep_path), "--runs-root", str(runs_root), "--out-dir",
                          str(folder / "figures"), "--mtf-dir", str(folder / "no mtf"))
    return SimpleNamespace(folder=folder, sweep_path=sweep_path, runs_root=runs_root, dirs=dirs, inserted=inserted,
                           out_dir=tables / sweep_path.stem, result=result, figures=figures)


def _paired(study, **match):
    return [r for r in _rows(study.out_dir / "paired.csv") if all(r[k] == v for k, v in match.items())]


@needs_data
def test_the_script_writes_every_table_with_status_and_family_and_a_manifest(study):
    assert study.result.returncode == 0, study.result.stderr
    mod = _script()
    for name in mod.TABLES:
        with open(study.out_dir / f"{name}.csv", encoding="utf-8", newline="") as fh:
            header = next(csv.reader(fh))
        assert header == mod.COLUMNS[name], name
    cfg = json.loads((study.out_dir / "manifest.json").read_text(encoding="utf-8"))["config"]
    assert cfg["seed"] == 20261021 and cfg["resamples"] == 2000 and cfg["ni_spaces"] == 0.005 and cfg["delta"] == 0.01
    assert cfg["readers_analysed"] == sorted(TEST_READERS) and cfg["readers_without_a_run"] == []
    assert cfg["rows_loaded"] == cfg["rows_analysed"] == sum(ROWS.values()) and cfg["pages_excluded"] == {}
    for name in mod.TABLES:
        assert len(_rows(study.out_dir / f"{name}.csv")) == cfg["tables"][name], name


@needs_data
def test_checks_count_every_planned_row_and_expect_no_determinism_pairs(study):
    checks = {r["reader"]: r for r in _rows(study.out_dir / "checks.csv")}
    assert sorted(checks) == sorted(TEST_READERS)
    for name, row in checks.items():
        assert int(row["rows_done"]) == int(row["rows_planned"]) == ROWS[name]
        assert row["determinism_expected"] == "0" and row["pages_excluded"] == "0"
        assert row["missing_transcriptions"] == row["empty_transcriptions"] == "0"


@needs_data
def test_h3_the_slot_layout_inserts_spaces_and_contained_is_at_parity_with_conventional(study):
    gap = _paired(study, family="gap", contrast="contained_vs_slot")
    assert sorted(r["reader"] for r in gap) == ["qwen", "qwen7b"]  # smolvlm500m reads no random page
    for r in gap:
        assert r["measure"] == "spaces" and int(r["n"]) == N_RANDOM and float(r["median_diff"]) < 0
        assert float(r["p_holm"]) == pytest.approx(2 * 2 / 2 ** N_RANDOM) and r["supported"] == "True"
    parity = _paired(study, family="gap", contrast="contained_vs_conventional")
    assert len(parity) == 2
    for r in parity:
        assert r["bound"] == "0.005" and r["passes"] == "True" and r["p_holm"] == "" and r["supported"] == ""
    assert {r["role"] for r in gap} == {"development", "validation"}


@needs_data
def test_h4_h5_and_h6_have_their_known_answers(study):
    for family, contrast, measure, readers in (
        ("containment", "contained_vs_cut", "letters", ["qwen", "qwen7b"]),
        ("advantage", "contained_vs_conventional", "cer", ["qwen", "qwen7b"]),
    ):
        rows = _paired(study, family=family, contrast=contrast)
        assert sorted(r["reader"] for r in rows) == readers
        for r in rows:
            assert r["measure"] == measure and r["text_type"] == "random62" and r["supported"] == "True"
            assert float(r["ci_high"]) < 0 and float(r["mde"]) > 0
    prose = _paired(study, family="prose")
    assert sorted(r["reader"] for r in prose) == sorted(TEST_READERS)
    for r in prose:
        assert r["text_type"] == "natural" and r["passes"] == "True" and int(r["n"]) == N_NATURAL
        assert r["predicted_sign"] == "" and r["supported"] == ""


@needs_data
def test_secondary_and_wer_rows_are_present_and_wer_is_exploratory(study):
    secondary = _paired(study, family="secondary")
    names = {(r["text_type"], r["contrast"]) for r in secondary}
    assert names == {("random62", "M3_vs_M0"), ("natural", "M3_vs_M0"), ("random62", "contained_vs_density"),
                     ("natural", "contained_vs_cut"), ("natural", "contained_vs_conventional")}
    m3 = _paired(study, family="secondary", contrast="M3_vs_M0", reader="qwen", text_type="random62")[0]
    assert m3["method"].startswith("undefined") and m3["median_diff"] == "0.0"  # both exact: every difference zero
    density = _paired(study, family="secondary", contrast="contained_vs_density", reader="qwen")[0]
    assert float(density["median_diff"]) < 0 and density["p_holm"] == ""
    wer_rows = _paired(study, family="wer")
    assert wer_rows and all(r["status"] == "exploratory" and r["measure"] == "wer" for r in wer_rows)
    assert all(r["status"] == "preregistered" for r in _rows(study.out_dir / "paired.csv") if r["family"] != "wer")


@needs_data
def test_letters_and_spaces_describe_every_condition_with_its_band(study):
    letters = _rows(study.out_dir / "letters.csv")
    by = {(r["reader"], r["text_type"], r["condition"]): r for r in letters}
    assert {c for (reader, t, c) in by if reader == "qwen" and t == "random62"} == {
        "S0", "S5", "S10", "Q0", "M0", "M3", "M7", f"S{D_SIZE}"}
    assert {c for (reader, t, c) in by if reader == "smolvlm500m"} == {"S0", "S5", "S10", "M0", "M3", "M8"}
    for key in (("qwen", "random62", "M0"), ("qwen7b", "natural", "M3")):
        assert float(by[key]["cer"]) == float(by[key]["letters_cer"]) == 0.0
    cut = by[("qwen", "random62", "M7")]
    assert float(cut["letters_low"]) <= float(cut["letters_cer"]) <= float(cut["letters_high"])
    assert float(cut["letters_cer"]) > 0 and cut["group"] == "cut" and cut["family"] == "description"
    spaces = {(r["reader"], r["text_type"], r["condition"]): r for r in _rows(study.out_dir / "spaces.csv")}
    for reader in ("qwen", "qwen7b"):
        slot = spaces[(reader, "random62", "Q0")]
        assert int(slot["spaces_inserted"]) == study.inserted[reader]["slot"] > 0
        assert float(slot["spaces_per_char"]) == pytest.approx(
            int(slot["spaces_inserted"]) / int(slot["reference_chars"]))
        assert int(spaces[(reader, "random62", "M0")]["spaces_inserted"]) == 0


def _references(study, reader, text_type):
    run_dir = next(d for d in study.dirs if json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                   ["config"]["reader"]["name"] == reader)
    prefix = "rnd" if text_type == "random62" else "pg"
    return [p.read_text(encoding="utf-8") for p in sorted((run_dir / "references").glob(f"{prefix}*.txt"))]


@needs_data
def test_confusions_count_the_planted_o0_and_il1_swaps(study):
    rows = _rows(study.out_dir / "confusions.csv")
    total = {(r["reader"], r["text_type"], r["condition"], r["kind"]): int(r["count"]) for r in rows
             if r["kind"] in ("O/0", "I/l/1", "case_pair")}
    refs = _references(study, "qwen", "random62")
    assert len(refs) == N_RANDOM
    with_o0 = sum(any(c in "O0" for c in ref) for ref in refs)
    with_il1 = sum(any(c in "Il1" for c in ref) for ref in refs)
    for code in ("S0", "S5", "S10"):
        assert total[("qwen", "random62", code, "O/0")] == with_o0
        assert total[("qwen", "random62", code, "I/l/1")] == 0
    assert total[("qwen", "random62", "M7", "I/l/1")] == with_il1
    assert total[("qwen", "random62", "M0", "O/0")] == total[("qwen", "random62", "M0", "I/l/1")] == 0
    m0 = [r for r in rows if r["reader"] == "qwen" and r["condition"] == "M0" and r["kind"] == "substitution"]
    assert m0 == []
    s0 = [r for r in rows if r["reader"] == "qwen" and r["text_type"] == "random62" and r["condition"] == "S0"]
    assert sum(int(r["count"]) for r in s0 if r["kind"] == "substitution") == N_RANDOM
    assert all(r["group"] == "conventional" for r in s0)


@needs_data
def test_capacity_describes_each_layout_on_the_study_text(study):
    cap = _rows(study.out_dir / "capacity.csv")
    qwen = {(r["text_type"], r["render_mode"], Path(r["font_file"]).stem, float(r["font_size_px"])) for r in cap
            if r["reader"] == "qwen"}
    assert qwen == {
        ("random62", "conventional", "SourceSans3-Regular", 25.0), ("random62", "patch_aware", "SourceSans3-Regular", 25.0),
        ("random62", "patch_aware", "FiraMono-Regular", 23.333), ("random62", "conventional", "SourceSans3-Regular", D_SIZE),
        ("natural", "conventional", "SourceSans3-Regular", 25.0), ("natural", "patch_aware", "FiraMono-Regular", 23.333),
    }
    assert {r["text_type"] for r in cap if r["reader"] == "smolvlm500m"} == {"natural"}
    for r in cap:
        tokens = {"qwen": 256, "qwen7b": 256, "smolvlm500m": 64}[r["reader"]]
        assert int(r["vision_tokens"]) == tokens and int(r["n_pages"]) >= 1
        assert float(r["chars_per_token"]) == pytest.approx(float(r["chars_per_page"]) / tokens)
        assert float(r["glyphs_per_line"]) > 0 and float(r["lines_per_page"]) >= 1


@needs_data
def test_checks_only_writes_checks_alone(study, tmp_path):
    result = _subprocess(SCRIPT, "--config", str(study.sweep_path), "--runs-root", str(study.runs_root), "--out-dir",
                         str(tmp_path), "--checks-only")
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in (tmp_path / study.sweep_path.stem).iterdir()) == ["checks.csv", "manifest.json"]


def test_the_script_refuses_a_sweep_without_the_study2_schema(tmp_path):
    result = _subprocess(SCRIPT, "--config", str(REPO_ROOT / "configs" / "sweeps" / "full.yaml"), "--runs-root",
                         str(tmp_path / "runs"), "--out-dir", str(tmp_path))
    assert result.returncode == 1 and "results_schema: study2" in result.stderr


# --------------------------------------------------------------------------- Figures 3 and 4


@needs_data
def test_figures_3_and_4_are_drawn_from_the_run(study):
    assert study.figures.returncode == 0, study.figures.stderr
    names = sorted(p.name for p in (study.folder / "figures").iterdir())
    stem = study.sweep_path.stem
    assert names == [f"fig3_conditions_{stem}.pdf", f"fig4_confusions_{stem}.pdf"]


def _bands():
    rows = []
    for reader in ("qwen", "qwen7b"):
        for code, group in (("S0", "conventional"), ("Q0", "slot"), ("M0", "contained"), ("M7", "cut"),
                            ("S28", "density")):
            rows.append({"reader": reader, "text_type": "random62", "condition": code, "group": group,
                         "value": 0.1, "low": 0.05, "high": 0.2})
    rows.append({"reader": "smolvlm500m", "text_type": "natural", "condition": "M0", "group": "contained",
                 "value": 0.02, "low": 0.01, "high": 0.03})
    return rows


def test_the_study2_figures_give_the_same_bytes_twice_and_refuse_nothing_to_draw(tmp_path):
    rates = [{"reader": r, "text_type": "random62", "group": g, "kind": k, "count": 3, "letters": 1000,
              "per_1000": 3.0} for r in ("qwen", "qwen7b") for g in ("conventional", "contained")
             for k in ("O/0", "I/l/1", "case_pair")]
    for name, draw, data in (("fig3", plotting.plot_study2_conditions, _bands()),
                             ("fig4", plotting.plot_study2_confusions, rates)):
        draw(data, tmp_path / f"{name}_a.pdf")
        draw(data, tmp_path / f"{name}_b.pdf")
        assert (tmp_path / f"{name}_a.pdf").read_bytes() == (tmp_path / f"{name}_b.pdf").read_bytes()
        with pytest.raises(ValueError, match="to draw"):
            draw([], tmp_path / f"{name}_empty.pdf")


def test_pair_rates_pool_a_group_and_count_per_thousand_letters():
    cells = {
        ("qwen", "random62", "S0"): {"subs": {("O", "0"): 2}, "letters": 500, "group": "conventional"},
        ("qwen", "random62", "S5"): {"subs": {("0", "O"): 1, ("I", "l"): 1}, "letters": 500, "group": "conventional"},
    }
    rates = {r["kind"]: r for r in study2.pair_rates(cells)}
    assert rates["O/0"]["count"] == 3 and rates["O/0"]["letters"] == 1000 and rates["O/0"]["per_1000"] == 3.0
    assert rates["I/l/1"]["per_1000"] == 1.0 and rates["case_pair"]["count"] == 0
