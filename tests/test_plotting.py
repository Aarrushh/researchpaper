"""Tests for src/mrr/plotting.py, scripts/30_make_figures.py (build guide section 3.3) and scripts/31_stage0_figures.py.
CPU only.

Figures 1, 2 and 3 and Figure 0 are drawn from small synthetic frames: a cosine of the patch period plus a page
effect and noise, with the character counts a pooled CER needs. The script runs once through subprocess on a
fake-backend run of a four-block sweep (V, H, C, Z), as make figures runs it.
"""

from __future__ import annotations

import importlib.util
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
    reference = {"name": "human CSF (assumed)", "frequency": np.linspace(0, 0.5, 50), "gain": np.exp(-np.linspace(0, 0.5, 50))}
    plotting.plot_mtf(curves, tmp_path / "fig3_ref.pdf", px_per_degree=40.0, reference=reference)
    assert _is_pdf(tmp_path / "fig3_ref.pdf")
    with pytest.raises(ValueError, match="no curves"):
        plotting.plot_mtf({}, tmp_path / "no.pdf")
    with pytest.raises(ValueError, match="one length"):
        plotting.plot_mtf({"bad": {"frequency": [0.1, 0.2], "gain": [1.0]}}, tmp_path / "no.pdf")
    with pytest.raises(ValueError, match="reference"):
        plotting.plot_mtf(curves, tmp_path / "no.pdf", reference={"frequency": [0.1], "gain": [1.0, 2.0]})
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


def _figures(sweep_path, runs_root: Path, out_dir: Path, mtf_dir=None) -> subprocess.CompletedProcess:
    """Run the script; without mtf_dir an empty folder stands in, so the repository's results/mtf stays out of the test."""
    env = {**os.environ, "PYTHONUTF8": "1",
           "PYTHONPATH": os.pathsep.join(p for p in (str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")) if p)}
    args = [sys.executable, str(SCRIPT), "--runs-root", str(runs_root), "--out-dir", str(out_dir),
            "--mtf-dir", str(mtf_dir if mtf_dir is not None else Path(out_dir).parent / "no_mtf")]
    if sweep_path is not None:
        args += ["--config", str(sweep_path)]
    return subprocess.run(args, capture_output=True, text=True, env=env, cwd=REPO_ROOT)


def _layer0_csv(folder: Path, reader: str, decay: float) -> Path:
    """A fake results/mtf/layer0_<reader>.csv with the four profiles scripts/04_mtf.py writes."""
    ring_f = (np.arange(32) + 0.5) / 64
    lines = ["profile,f_cyc_per_px,value"]
    lines += [f"ring,{f:.10g},{np.exp(-decay * f):.12g}" for f in ring_f]
    lines += [f"horizontal,{f:.10g},{np.exp(-decay * f):.12g}" for f in np.arange(64) / 128]
    lines += [f"m0_0deg,{f:.10g},{1.0:.12g}" for f in ring_f] + [f"m0_45deg,{f:.10g},{1.0:.12g}" for f in ring_f]
    path = folder / f"layer0_{reader}.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


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


def test_the_script_draws_figure_3a_from_the_mtf_csvs_without_a_sweep(tmp_path):
    mtf_dir = tmp_path / "mtf"
    mtf_dir.mkdir()
    _layer0_csv(mtf_dir, "qwen", 2.0)
    _layer0_csv(mtf_dir, "smolvlm", 4.0)
    out_dir = tmp_path / "figures"
    result = _figures(None, tmp_path / "runs", out_dir, mtf_dir=mtf_dir)
    assert result.returncode == 0, result.stderr
    assert _is_pdf(out_dir / "fig3a_layer0_mtf.pdf") and "Figure 3a" in result.stderr and "qwen, smolvlm" in result.stderr
    assert sorted(p.name for p in out_dir.iterdir()) == ["fig3a_layer0_mtf.pdf"]
    first = (out_dir / "fig3a_layer0_mtf.pdf").read_bytes()
    assert _figures(None, tmp_path / "runs", out_dir, mtf_dir=mtf_dir).returncode == 0
    assert (out_dir / "fig3a_layer0_mtf.pdf").read_bytes() == first  # the same CSVs give the same bytes
    sweep_path = _write_sweep(tmp_path)  # a sweep without a run folder still exits 1, with Figure 3a drawn
    result = _figures(sweep_path, tmp_path / "runs", tmp_path / "figures2", mtf_dir=mtf_dir)
    assert result.returncode == 1 and "no run folder" in result.stderr and _is_pdf(tmp_path / "figures2" / "fig3a_layer0_mtf.pdf")


def test_the_script_with_neither_a_sweep_nor_mtf_csvs_exits_1(tmp_path):
    result = _figures(None, tmp_path / "runs", tmp_path / "figures", mtf_dir=tmp_path / "mtf")
    assert result.returncode == 1 and "nothing to draw" in result.stderr
    assert not (tmp_path / "figures").exists()


# --------------------------------------------------------------------------- Study 3 stage 0, Figures 5 and 6

STAGE0_TABLES = REPO_ROOT / "reports" / "tables" / "s3_stage0"


def _stage0_script():
    spec = importlib.util.spec_from_file_location("stage0_figures", REPO_ROOT / "scripts" / "31_stage0_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage0_rows() -> tuple[list[dict], list[dict], list[dict]]:
    """cells, acuity and behaviour rows as csv reads scripts/25_fit_stage0.py's tables back (strings): qwen with a fit
    per text type, its random s01 just past the measured sizes and one reading interval empty, qwen7b's prose cells
    without a fit, a cell interval reaching 0, and format on one size."""
    from mrr.acuity import cer_curve

    cells, fits, behaviour = [], [], []
    for reader, text_type, a, beta in (("qwen", "natural", 8.5, 12.0), ("qwen", "random62", 13.0, 5.6),
                                       ("qwen7b", "natural", 7.8, 11.0)):
        for s in (6, 9, 13, 20, 29):
            cer = float(cer_curve(s, a, beta))
            cells.append({"reader": reader, "text_type": text_type, "font_size_px": str(s), "cer": f"{cer + 0.002}",
                          "cer_lo": "0" if s == 29 else f"{cer / 2}", "cer_hi": f"{1.5 * cer + 0.004}",
                          "reading_cer": f"{cer + 0.001}", "reading_cer_lo": f"{cer / 2}",
                          "reading_cer_hi": f"{1.5 * cer + 0.003}", "behaviour_rate_lo": "0.001",
                          "behaviour_rate_hi": "0.012"})
            behaviour.append({"reader": reader, "text_type": text_type, "font_size_px": str(s),
                              "omitted_head_rate": "0", "omitted_interior_rate": f"{0.001 * (s % 3)}",
                              "omitted_tail_rate": "0.002", "runaway_rate": "0.001" if s == 9 else "0",
                              "case_normalised_rate": "0.003" if text_type == "random62" else "0",
                              "format_rate": "0.004" if s == 29 else "0"})
        if reader == "qwen":
            s01 = a * 99 ** (1 / beta)  # CER(s01) = 0.01
            fits.append({"reader": reader, "text_type": text_type, "patch_size": "14", "a": f"{a}", "beta": f"{beta}",
                         "s01": f"{s01}", "s01_lo": f"{0.9 * s01}", "s01_hi": f"{1.1 * s01}",
                         "s01_in_range": str(s01 <= 29), "reading_a": f"{0.97 * a}", "reading_beta": f"{beta}",
                         "reading_s01": f"{0.97 * s01}", "reading_s01_lo": "" if text_type == "random62" else "6",
                         "reading_s01_hi": "" if text_type == "random62" else "40", "reading_s01_in_range": True})
    return cells, fits, behaviour


def test_plot_stage0_curves_and_behaviour_write_the_same_bytes_twice(tmp_path):
    cells, fits, behaviour = _stage0_rows()
    assert fits[1]["s01_in_range"] == "False"  # the random s01 lies past 29 px: drawn faint
    for name in ("first", "second"):
        plotting.plot_stage0_curves(cells, fits, tmp_path / name / "fig5.pdf")
        plotting.plot_stage0_behaviour(behaviour, cells, tmp_path / name / "fig6.pdf")
    for fig in ("fig5.pdf", "fig6.pdf"):
        assert _is_pdf(tmp_path / "first" / fig)
        assert (tmp_path / "first" / fig).read_bytes() == (tmp_path / "second" / fig).read_bytes()


def test_plot_stage0_figures_refuse_empty_tables(tmp_path):
    with pytest.raises(ValueError, match="Figure 5"):
        plotting.plot_stage0_curves([], [], tmp_path / "fig5.pdf")
    with pytest.raises(ValueError, match="Figure 6"):
        plotting.plot_stage0_behaviour([], [], tmp_path / "fig6.pdf")
    assert not any(tmp_path.iterdir())


def test_the_stage0_script_draws_figures_5_and_6_and_exits_1_without_a_table(tmp_path):
    import csv

    tables = tmp_path / "s3_stage0"
    tables.mkdir()
    for name, rows in zip(("cells", "acuity", "behaviour"), _stage0_rows()):
        columns = list(dict.fromkeys(k for r in rows for k in r))
        with open(tables / f"{name}.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    script = _stage0_script()
    out_dir = tmp_path / "figures"
    assert script.main(["--tables-dir", str(tables), "--out-dir", str(out_dir)]) == 0
    assert sorted(p.name for p in out_dir.iterdir()) == ["fig5_acuity_s3_stage0.pdf", "fig6_behaviour_s3_stage0.pdf"]
    assert all(_is_pdf(p) for p in out_dir.iterdir())
    (tables / "behaviour.csv").unlink()
    assert script.main(["--tables-dir", str(tables), "--out-dir", str(tmp_path / "none")]) == 1
    assert not (tmp_path / "none").exists()


@pytest.mark.skipif(not (STAGE0_TABLES / "behaviour.csv").exists(), reason="stage-0 tables not present")
def test_figure_6s_stack_adds_up_to_the_committed_behaviour_rate():
    """Figure 6 stacks BEHAVIOUR_PARTS and puts the behaviour rate's interval on top: the parts must be that rate."""
    behaviour = pd.read_csv(STAGE0_TABLES / "behaviour.csv")
    cells = pd.read_csv(STAGE0_TABLES / "cells.csv")
    keys = ["reader", "text_type", "font_size_px"]
    both = behaviour.merge(cells[keys + ["behaviour_rate"]], on=keys, validate="one_to_one")
    assert len(both) == len(behaviour) == len(cells)
    parts = both[[column for column, _, _ in plotting.BEHAVIOUR_PARTS]].sum(axis=1)
    assert np.allclose(parts, both["behaviour_rate"], rtol=0, atol=1e-9)
