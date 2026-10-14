"""Tests for src/mrr/plotting.py and scripts/30_make_figures.py (build guide section 3.3). CPU only.

Figures 1, 2 and 3 and Figure 0 are drawn from small synthetic frames: a cosine of the patch period plus a page
effect and noise, with the character counts a pooled CER needs. The script runs once through subprocess on a
fake-backend run of a four-block sweep (V, H, C, Z), as make figures runs it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import READERS, REPO_ROOT
from mrr import grid, plotting
from mrr.sweep import run_sweep
from test_analyze import ANALYSIS_SWEEP, DEFAULTS, HAVE_DATA, DroppingBackend, _write_sweep

SCRIPT = REPO_ROOT / "scripts" / "30_make_figures.py"
RESAMPLES = 100  # enough for a band in a test; the script uses 2,000
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="font or books not present (W1.4)")


def _is_pdf(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 2000 and path.read_bytes()[:5] == b"%PDF-"


def _counts(cer: float, ref: int = 1000) -> dict[str, int]:
    """Hits, substitutions, deletions and insertions with H + S + D = ref and (S + D + I) / ref close to cer."""
    errors = int(round(min(max(cer, 0.0), 0.9) * ref))
    s, i = errors // 2, errors // 4
    d = errors - s - i
    return {"hits": ref - s - d, "substitutions": s, "deletions": d, "insertions": i, "reference_chars": ref}


def _phase_rows(reader_cfg, blocks=("V", "H", "C"), pages: int = 3, seed: int = 0, amplitude: float = 0.03) -> list[dict]:
    """Rows of one reader: a cosine of the patch period on every block, plus a page effect and noise."""
    rng = np.random.default_rng(seed)
    patch, cell, sweep = int(reader_cfg["patch_size"]), grid.token_cell_px(reader_cfg), grid.sweep_period_px(reader_cfg)
    tokens = grid.token_count(reader_cfg, *grid.page_size_px(reader_cfg))
    ranges = {"V": (cell, "y", 4), "H": (patch, "x", 2), "C": (sweep, "x", 8)}
    rows = []
    for block in blocks:
        if block in ranges:
            L, axis, n = ranges[block]
            levels = [L * k / n for k in range(n)]
        elif block == "S":
            axis, levels = "x", [0.0, 0.25, 0.5, 0.75]
        else:
            axis, levels = "x", [0.0]
        mode = "patch_aware" if block in ("H", "S") else "conventional"
        for text_type in ("natural", "random62"):
            for p in range(pages):
                base = 0.05 + 0.02 * p + (0.10 if text_type == "random62" else 0.0)
                for phi in levels:
                    cer = base + amplitude * np.cos(2 * np.pi * phi / patch) + rng.normal(0.0, 0.004)
                    counts = _counts(cer)
                    rows.append({
                        "reader": reader_cfg["name"], "text_type": text_type, "block": block,
                        "page_id": f"{text_type[:3]}_p{p:04d}", "condition_hash": f"{block}{phi:g}".ljust(12, "0"),
                        "repeat": 0, "render_mode": mode, "font_size_px": 16.0, "vision_tokens": tokens,
                        "phase_x_px": phi if axis == "x" else 0.0, "phase_y_px": phi if axis == "y" else 0.0,
                        "cer": (counts["substitutions"] + counts["deletions"] + counts["insertions"]) / counts["reference_chars"],
                        **counts,
                    })
    return rows


def _z_rows(reader_cfg, sizes=(12.0, 16.0, 20.0, 24.0), pages: int = 2, seed: int = 1) -> list[dict]:
    """Block Z rows: CER falling with the font size, fewer characters per page as the glyphs grow."""
    rng = np.random.default_rng(seed)
    tokens = grid.token_count(reader_cfg, *grid.page_size_px(reader_cfg))
    rows = []
    for text_type in ("natural", "random62"):
        for mode, tag in (("conventional", "conv"), ("patch_aware", "pa")):
            for size in sizes:
                ref = int(tokens * 30 * (12.0 / size) ** 2)
                for p in range(pages):
                    cer = 0.3 * np.exp(-(size - 8.0) / 6.0) + (0.02 if mode == "conventional" else 0.0) + rng.normal(0, 0.003)
                    counts = _counts(cer, ref)
                    rows.append({
                        "reader": reader_cfg["name"], "text_type": text_type, "block": "Z",
                        "page_id": f"{text_type[:3]}_z{size:g}_{tag}_p{p:04d}", "condition_hash": f"Z{size:g}{tag}".ljust(12, "0"),
                        "repeat": 0, "render_mode": mode, "font_size_px": size, "vision_tokens": tokens,
                        "phase_x_px": 0.0, "phase_y_px": 0.0,
                        "cer": (counts["substitutions"] + counts["deletions"] + counts["insertions"]) / counts["reference_chars"],
                        **counts,
                    })
    return rows


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Figure 1


def test_plot_phase_curve_writes_a_pdf_for_three_readers_and_ignores_s_a_and_z_rows(tmp_path):
    rows = []
    for name in ("qwen", "smolvlm", "smolvlm500m"):
        rows += _phase_rows(READERS[name], blocks=("V", "H", "C", "S", "A")) + _z_rows(READERS[name])
    out = tmp_path / "fig1.pdf"
    plotting.plot_phase_curve(_frame(rows), out, n_resamples=RESAMPLES, seed=20261014, reader_cfgs=READERS)
    assert _is_pdf(out)


def test_plot_phase_curve_works_without_s_a_or_z_rows_and_for_one_reader(tmp_path):
    out = tmp_path / "fig1.pdf"
    plotting.plot_phase_curve(_frame(_phase_rows(READERS["qwen"])), out, n_resamples=RESAMPLES, seed=1,
                              reader_cfgs={"qwen": READERS["qwen"]})
    assert _is_pdf(out)


def test_plot_phase_curve_draws_a_panel_without_rows_and_a_block_with_one_page(tmp_path):
    rows = [r for r in _phase_rows(READERS["qwen"]) if r["block"] != "H"]  # no H rows: an empty panel
    rows = [r for r in rows if not (r["block"] == "V" and r["page_id"] != "nat_p0000" and r["text_type"] == "natural")]
    out = tmp_path / "fig1.pdf"
    plotting.plot_phase_curve(_frame(rows), out, n_resamples=RESAMPLES, seed=1, reader_cfgs=READERS)
    assert _is_pdf(out)


def test_plot_phase_curve_gives_the_same_bytes_for_the_same_seed(tmp_path):
    df = _frame(_phase_rows(READERS["qwen"]))
    a, b, c = (tmp_path / f"{k}.pdf" for k in "abc")
    plotting.plot_phase_curve(df, a, n_resamples=RESAMPLES, seed=20261014, reader_cfgs=READERS)
    plotting.plot_phase_curve(df, b, n_resamples=RESAMPLES, seed=20261014, reader_cfgs=READERS)
    plotting.plot_phase_curve(df, c, n_resamples=RESAMPLES, seed=7, reader_cfgs=READERS)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()


def test_plot_phase_curve_reads_the_reader_configs_from_the_run_manifests(tmp_path):
    run_dir = tmp_path / "20261014-1200_pilot_qwen_abcdef1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"config": {"reader": READERS["qwen"]}}), encoding="utf-8")
    df = _frame(_phase_rows(READERS["qwen"]))
    df["run_dir"] = str(run_dir)
    out = tmp_path / "fig1.pdf"
    plotting.plot_phase_curve(df, out, n_resamples=RESAMPLES, seed=1)
    assert _is_pdf(out)
    with pytest.raises(ValueError, match="reader_cfgs"):
        plotting.plot_phase_curve(df.drop(columns=["run_dir"]), tmp_path / "no.pdf", n_resamples=RESAMPLES)
    with pytest.raises(ValueError, match="no reader config"):
        plotting.plot_phase_curve(df, tmp_path / "no.pdf", n_resamples=RESAMPLES, reader_cfgs={"smolvlm": READERS["smolvlm"]})


def test_plot_phase_curve_refuses_a_frame_without_v_h_or_c_rows(tmp_path):
    with pytest.raises(ValueError, match="V, H or C"):
        plotting.plot_phase_curve(_frame(_z_rows(READERS["qwen"])), tmp_path / "no.pdf", reader_cfgs=READERS)


def test_block_range_and_phase_axis_read_the_levels():
    assert plotting._block_range(np.array([0.0, 7.0, 14.0, 21.0])) == 28.0
    assert plotting._block_range(np.array([0.0, 10.5, 21.0, 31.5])) == 42.0
    assert plotting._block_range(np.array([0.0, 0.25, 0.5, 0.75])) == 1.0
    assert plotting._block_range(np.array([3.0, 7.0, 14.0])) == 18.0  # not from 0: the span plus a step
    assert plotting._block_range(np.array([0.0])) == 1.0
    rows = _frame(_phase_rows(READERS["qwen"]))
    assert plotting._phase_axis(rows[rows["block"] == "V"]) == "y"
    assert plotting._phase_axis(rows[rows["block"] == "C"]) == "x"


# --------------------------------------------------------------------------- Figure 2


def test_plot_tradeoff_writes_a_pdf_from_block_z_rows_and_refuses_a_frame_without_them(tmp_path):
    rows = _z_rows(READERS["qwen"]) + _z_rows(READERS["smolvlm"]) + _phase_rows(READERS["qwen"])
    out = tmp_path / "fig2.pdf"
    plotting.plot_tradeoff(_frame(rows), out)
    assert _is_pdf(out)
    plotting.plot_tradeoff(_frame(rows), tmp_path / "fig2b.pdf", pe_assumed=(0.05,))
    assert _is_pdf(tmp_path / "fig2b.pdf")
    with pytest.raises(ValueError, match="block Z"):
        plotting.plot_tradeoff(_frame(_phase_rows(READERS["qwen"])), tmp_path / "no.pdf")


# --------------------------------------------------------------------------- Figure 3


def test_plot_mtf_writes_a_pdf_with_and_without_a_degree_axis_and_checks_its_input(tmp_path):
    f = np.linspace(0.02, 0.5, 13)
    curves = {"qwen": {"frequency": f, "gain": np.exp(-f * 4)}, "smolvlm": {"frequency": f[::-1], "gain": np.exp(-f[::-1] * 6)}}
    plotting.plot_mtf(curves, tmp_path / "fig3.pdf")
    plotting.plot_mtf(curves, tmp_path / "fig3_deg.pdf", px_per_degree=60.0)
    assert _is_pdf(tmp_path / "fig3.pdf") and _is_pdf(tmp_path / "fig3_deg.pdf")
    with pytest.raises(ValueError, match="no curves"):
        plotting.plot_mtf({}, tmp_path / "no.pdf")
    with pytest.raises(ValueError, match="one length"):
        plotting.plot_mtf({"bad": {"frequency": [0.1, 0.2], "gain": [1.0]}}, tmp_path / "no.pdf")
    with pytest.raises(ValueError, match="positive"):
        plotting.plot_mtf(curves, tmp_path / "no.pdf", px_per_degree=0.0)


# --------------------------------------------------------------------------- Figure 0


def test_plot_equivariance_writes_a_pdf_from_a_synthetic_frame(tmp_path):
    cfgs = [READERS[name] for name in ("qwen", "smolvlm", "smolvlm500m")]
    rows = []
    for cfg in cfgs:
        shifts = np.arange(0, 2 * int(cfg["patch_size"]) + 1)
        for layer in (0, 4):
            for text_type in ("natural", "random62"):
                for s in shifts:
                    e = 0.0 if s % int(cfg["patch_size"]) == 0 else 0.3 + 0.1 * layer
                    rows.append({"reader": cfg["name"], "text_type": text_type, "layer": layer, "shift_px": int(s), "e": e})
    out = tmp_path / "fig0.pdf"
    plotting.plot_equivariance(_frame(rows), cfgs, out)
    assert _is_pdf(out)


# --------------------------------------------------------------------------- the script


def _figures(sweep_path: Path, runs_root: Path, out_dir: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONUTF8": "1",
           "PYTHONPATH": os.pathsep.join(p for p in (str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")) if p)}
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(sweep_path), "--runs-root", str(runs_root), "--out-dir", str(out_dir)],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT,
    )


@pytest.fixture(scope="module")
def figures_run(tmp_path_factory):
    """One fake-backend run of the V, H, C and Z blocks of the analysis sweep."""
    if not HAVE_DATA:
        pytest.skip("font or books not present (W1.4)")
    folder = tmp_path_factory.mktemp("figures")
    sweep_path = _write_sweep(folder, blocks={k: ANALYSIS_SWEEP["blocks"][k] for k in ("V", "H", "C", "Z")})
    runs_root = folder / "runs"
    dirs = run_sweep(sweep_path, READERS, DEFAULTS, DroppingBackend(), output_root=runs_root, root=REPO_ROOT,
                     require_clean=False)
    assert len(dirs) == 1
    return SimpleNamespace(folder=folder, sweep_path=sweep_path, runs_root=runs_root, run_dir=dirs[0])


@needs_data
def test_the_script_writes_figure_1_and_figure_2_from_the_run_folder(figures_run, tmp_path):
    out_dir = tmp_path / "figures"
    result = _figures(figures_run.sweep_path, figures_run.runs_root, out_dir)
    assert result.returncode == 0, result.stderr
    assert _is_pdf(out_dir / "fig1_phase_analysis.pdf") and _is_pdf(out_dir / "fig2_tradeoff_analysis.pdf")
    assert "0 page(s) excluded" in result.stderr and "Figure 1" in result.stderr and "Figure 2" in result.stderr


def test_the_script_without_a_run_folder_exits_1(tmp_path):
    sweep_path = _write_sweep(tmp_path)
    result = _figures(sweep_path, tmp_path / "runs", tmp_path / "figures")
    assert result.returncode == 1 and "no run folder" in result.stderr
    assert not (tmp_path / "figures").exists()
