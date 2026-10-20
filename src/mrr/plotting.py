# PURPOSE: The paper's figures.
# WHAT GOES HERE: Figure 0 (shift-equivariance error), Figure 1 (phase curve), Figure 2 (the trade-off: CER against characters per token, and the Fano floor) and Figure 3 (the encoder's modulation transfer); Study 2's Figures 3 and 4; Study 3 stage 0's Figures 5 (acuity curves) and 6 (behaviour rates).
# FILLED IN AT STEP: W1.9 (Figure 0), W2.5 (Figures 1 and 2, the MTF axes), W3.4 (Figure 2 on the full run); Figures 5 and 6 on 20 October 2026, after the tag data-freeze-s3-stage0.
# DO NOT: Hand-edit output images. Do not import matplotlib at module top level. Do not smooth the points: the line is the fitted model, the points are the data.
"""Figures.

Learn B8 (plotting) before implementing. Every function imports matplotlib inside itself, builds a ``Figure`` without
pyplot (so no global backend is set) and writes a PDF with TrueType fonts (``pdf.fonttype`` 42) and no creation date,
so the same rows and the same seed give the same bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mrr import grid, stats
from mrr.sweep import reader_periods

PHASE_BLOCKS = ("V", "H", "C")  # Figure 1's columns: the blocks the harmonic model is fitted to (section 9, item 2)
DEFAULT_AXIS = {"V": "y", "H": "x", "C": "x"}  # the phase each block sweeps, when a panel has no rows to tell
TEXT_COLOURS = {"natural": "#1f77b4", "random62": "#d62728"}  # blue and red, the same in every figure
MODE_STYLES = {"conventional": "-", "patch_aware": "--"}
_RC = {
    "pdf.fonttype": 42, "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 6,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
}
_NO_DATE = {"CreationDate": None}  # the one field that would change between two runs on the same rows


def _save(fig: Any, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", metadata=_NO_DATE)


# --------------------------------------------------------------------------- Figure 0 (W1.9)


def plot_equivariance(df: Any, reader_cfgs: Sequence[Mapping[str, Any]], out_path: str | Path) -> None:
    """Figure 0: shift-equivariance error E(s) against the shift, one panel per reader, one line per layer.

    Natural pages are solid lines and random62 pages dashed. Light vertical lines mark the multiples of the
    reader's patch size P, darker ones the multiples of its attention window where it has one (window_px:
    Qwen's 112 px). Each panel's legend sits below it, so it covers no curve. The matplotlib figure is built
    without pyplot, so no global backend is set.

    Args:
        df: pandas DataFrame with columns reader, text_type, layer, shift_px and e, as
            scripts/03_equivariance.py reads them back from results/equivariance/<reader>_<text_type>.csv.
        reader_cfgs: reader blocks of configs/readers.yaml, in panel order.
        out_path: output PDF path under reports/figures/.

    Raises:
        ValueError: if a reader has no rows, or a text type other than natural and random62 appears.
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    styles = {"natural": "-", "random62": "--"}
    unknown = set(df["text_type"]) - set(styles)
    if unknown:
        raise ValueError(f"unknown text types {sorted(unknown)}; want {sorted(styles)}")
    colormap = matplotlib.colormaps["viridis"]
    with matplotlib.rc_context({"pdf.fonttype": 42, "font.size": 7, "axes.titlesize": 8, "legend.fontsize": 6}):
        fig = Figure(figsize=(6.75, 2.6))
        axes = fig.subplots(1, len(reader_cfgs), sharey=True, squeeze=False)[0]
        for ax, cfg in zip(axes, reader_cfgs):
            rows = df[df["reader"] == cfg["name"]]
            if rows.empty:
                raise ValueError(f"no rows for reader {cfg['name']}")
            s_max = float(rows["shift_px"].max())
            for period, grey, width in ((cfg["patch_size"], "0.85", 0.4), (cfg.get("window_px"), "0.45", 0.8)):
                if period:
                    for x in range(int(period), int(s_max) + 1, int(period)):
                        ax.axvline(x, color=grey, linewidth=width, zorder=0)
            layers = sorted(rows["layer"].unique())
            for i, layer in enumerate(layers):
                colour = colormap(0.9 * i / max(1, len(layers) - 1))
                for text_type, style in styles.items():
                    line = rows[(rows["layer"] == layer) & (rows["text_type"] == text_type)].sort_values("shift_px")
                    if not line.empty:
                        label = f"layer {layer}" if text_type == "natural" else None
                        ax.plot(line["shift_px"], line["e"], style, color=colour, linewidth=0.8, label=label)
            ax.set_title(cfg["name"])
            ax.set_xlabel("shift s (px)")
            ax.set_xlim(0, s_max)
            handles, labels = ax.get_legend_handles_labels()
            if ax is axes[0]:
                handles += [Line2D([], [], color="black", linestyle=style, linewidth=0.8) for style in styles.values()]
                labels += list(styles)
            ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False,
                      handlelength=1.2, columnspacing=0.8, handletextpad=0.4)
        axes[0].set_ylabel("E(s)")
        axes[0].set_ylim(bottom=0)
        _save(fig, out_path)


# --------------------------------------------------------------------------- Figure 1 (W2.5)


def _reader_cfgs_from_runs(df: Any) -> dict[str, Mapping[str, Any]]:
    """Each reader's config from the manifest of a run folder named in the frame's ``run_dir`` column."""
    if "run_dir" not in df.columns:
        raise ValueError("pass reader_cfgs, or a frame from sweep.load_results, whose run_dir column names the manifests")
    cfgs: dict[str, Mapping[str, Any]] = {}
    for reader, run_dir in df[["reader", "run_dir"]].drop_duplicates().itertuples(index=False):
        path = Path(str(run_dir)) / "manifest.json"
        if not path.exists():
            raise ValueError(f"{path} is missing; pass reader_cfgs instead")
        cfgs.setdefault(str(reader), json.loads(path.read_text(encoding="utf-8"))["config"]["reader"])
    return cfgs


def _phase_axis(rows: Any) -> str:
    """The axis a block sweeps: the one whose phase varies across its rows; x when neither does."""
    varies_y = rows["phase_y_px"].nunique() > 1
    varies_x = rows["phase_x_px"].nunique() > 1
    return "y" if varies_y and not varies_x else "x"


def _block_range(levels: np.ndarray) -> float:
    """The sweep length the levels sample: n * step for levels 0, step, ..., (n - 1) step; else the span plus a step."""
    levels = np.sort(np.unique(np.asarray(levels, dtype=np.float64)))
    if levels.size < 2:
        return float(levels[0]) if levels.size and levels[0] > 0 else 1.0
    step = float(levels[1] - levels[0])
    if np.allclose(levels, step * np.arange(levels.size)):
        return step * levels.size
    return float(levels[-1] + step)


def _errors_and_ref(rows: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per row: errors = S + D + I and reference characters = H + S + D, the two sums a pooled CER needs (D13)."""
    s, d, i, h = (rows[c].to_numpy(dtype=np.float64) for c in ("substitutions", "deletions", "insertions", "hits"))
    return s + d + i, h + s + d


def _draw_phase_series(ax: Any, rows: Any, axis: str, L: float, periods: Sequence[int], n_resamples: int, seed: int,
                       colour: str) -> None:
    """One text type in one panel: the pooled CER per level with its band, and the fitted harmonic line."""
    phase = rows[f"phase_{axis}_px"].to_numpy(dtype=np.float64)
    pages = rows["page_id"].to_numpy()
    errors, ref = _errors_and_ref(rows)
    band = stats.pooled_cer_by_level(errors, ref, pages, phase, n_resamples, seed)
    yerr = np.vstack([np.maximum(band["cer"] - band["low"], 0.0), np.maximum(band["high"] - band["cer"], 0.0)])
    ax.errorbar(band["levels"], band["cer"], yerr=yerr, fmt="o", color=colour, markersize=2.5, linewidth=0.6,
                capsize=1.5, zorder=3)
    levels = band["levels"]
    sweep_px = int(round(L))
    tested = [int(T) for T in periods if sweep_px % int(T) == 0]
    if levels.size < 2 or not tested or not np.isclose(L, sweep_px):
        return  # one level, or a sweep no tested period divides: the points stand alone
    try:
        groups = stats.bin_groups(sweep_px, tested, int(levels.size))
        fit = stats.harmonic_fit(rows["cer"].to_numpy(dtype=np.float64), pages, phase, float(sweep_px), groups, levels)
    except ValueError:
        return  # too few pages for a residual degree of freedom: the points stand alone
    fine = np.linspace(0.0, float(sweep_px), 241)
    line = errors.sum() / ref.sum() + stats.harmonic_component(fit["beta_harm"], fit["names"], float(sweep_px), fine)
    ax.plot(fine, line, "-", color=colour, linewidth=0.9, zorder=2)


def plot_phase_curve(df: Any, out_path: str | Path, n_resamples: int = 2000, seed: int = 0, *,
                     reader_cfgs: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    """Figure 1: CER against glyph phase, readers in rows, blocks V, H and C in columns.

    Points: the pooled CER at each level (D13) with its page-bootstrap band from ``stats.pooled_cer_by_level``,
    drawn as error bars, natural pages in blue and random62 in red. Line: the harmonic component fitted by
    ``stats.harmonic_fit`` with page fixed effects, evaluated on a fine grid with ``stats.harmonic_component`` and
    added to the block's pooled mean CER; it is drawn only where the model is defined (two or more levels, a
    tested period dividing the range, a residual degree of freedom). Light vertical lines mark the multiples of
    the reader's patch, darker ones the multiples of its token cell; the x axis runs from 0 to the block's range in
    pixels. Rows of other blocks (S, A, Z) are ignored, so the pilot and the full sweep draw alike.

    Args:
        df: results rows as ``sweep.load_results`` returns them: reader, text_type, block, page_id, phase_x_px,
            phase_y_px, hits, substitutions, deletions, insertions and cer, plus run_dir when ``reader_cfgs`` is None.
        out_path: the PDF to write, under reports/figures/.
        n_resamples: page resamples for the bands (PREREGISTRATION.md section 9: 2,000).
        seed: the bands' seed; pass the preregistered one so they reproduce.
        reader_cfgs: reader name -> its block of configs/readers.yaml (patch_size, merge_size, token_cell_px and
            window_px where it has one), in row order. None reads them from the run folders' manifests through
            the frame's run_dir column.

    Raises:
        ValueError: when no row of block V, H or C exists, or a reader's config cannot be found.
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    rows_all = df[df["block"].isin(PHASE_BLOCKS)]
    if rows_all.empty:
        raise ValueError("no rows of block V, H or C: Figure 1 has nothing to draw")
    cfgs = dict(reader_cfgs) if reader_cfgs is not None else _reader_cfgs_from_runs(rows_all)
    present = set(rows_all["reader"].tolist())
    missing = sorted(present - set(cfgs))
    if missing:
        raise ValueError(f"no reader config for {missing}; pass reader_cfgs")
    readers = [name for name in cfgs if name in present]
    text_types = sorted(set(rows_all["text_type"].tolist()), key=lambda t: (t not in TEXT_COLOURS, t))
    colours = {t: TEXT_COLOURS.get(t, f"C{i + 2}") for i, t in enumerate(text_types)}

    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(6.75, 0.9 + 1.7 * len(readers)))
        axes = fig.subplots(len(readers), len(PHASE_BLOCKS), sharey="row", squeeze=False)
        for i, (row_axes, reader) in enumerate(zip(axes, readers)):
            first, last = i == 0, i == len(readers) - 1
            cfg = cfgs[reader]
            patch, cell = int(cfg["patch_size"]), grid.token_cell_px(cfg)
            periods = reader_periods(cfg)
            for ax, block in zip(row_axes, PHASE_BLOCKS):
                rows = rows_all[(rows_all["reader"] == reader) & (rows_all["block"] == block)]
                if first:
                    axis = _phase_axis(rows) if not rows.empty else DEFAULT_AXIS[block]
                    ax.set_title(f"block {block} ({axis} phase)")
                if last:
                    ax.set_xlabel("phase (px)")
                if rows.empty:
                    ax.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax.transAxes, color="0.5")
                    ax.set_xticks([])
                    continue
                axis = _phase_axis(rows)
                L = _block_range(rows[f"phase_{axis}_px"].to_numpy(dtype=np.float64))
                for x in range(patch, int(round(L)) + 1, patch):
                    ax.axvline(x, color="0.85", linewidth=0.4, zorder=0)
                for x in range(cell, int(round(L)) + 1, cell):
                    ax.axvline(x, color="0.55", linewidth=0.6, zorder=0)
                for text_type in text_types:
                    part = rows[rows["text_type"] == text_type]
                    if not part.empty:
                        _draw_phase_series(ax, part, axis, L, periods, n_resamples, seed, colours[text_type])
                ax.set_xlim(0, L)
            row_axes[0].set_ylabel(f"{reader}\nCER")
            row_axes[0].set_ylim(bottom=0)
            row_axes[0].text(0.02, 0.97, f"patch {patch} px, cell {cell} px", transform=row_axes[0].transAxes,
                             ha="left", va="top", fontsize=6, color="0.3")
        handles = [Line2D([], [], color=colours[t], marker="o", linestyle="", markersize=3) for t in text_types]
        handles += [Line2D([], [], color="0.3", linewidth=0.9), Line2D([], [], color="0.85", linewidth=0.8),
                    Line2D([], [], color="0.55", linewidth=0.8)]
        labels = [*text_types, "fitted harmonic model", "patch multiples", "token-cell multiples"]
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.0),
                   handlelength=1.4, columnspacing=1.0, handletextpad=0.4)
        _save(fig, out_path)


# --------------------------------------------------------------------------- Figure 2 (W2.5, W3.4)


def plot_tradeoff(df: Any, out_path: str | Path, pe_assumed: Sequence[float] = (0.0, 0.03, 0.1)) -> None:
    """Figure 2: the trade-off. Left: block Z's pooled CER against characters per vision token. Right: the Fano floor.

    Left panels, one per text type: the pooled CER (D13) of every page of a font size against the pages' mean
    reference characters per vision token, one line per reader (colour) and render mode (solid conventional,
    dashed patch-aware), one point per font size labelled with the size in pixels. Each reader has one token count
    under D8 (256, 81, 64), so the x axis is characters per token, not tokens.

    Right panel: ``stats.fano_floor_bits`` (equation (4) of build guide 3.1, M = 62) times the symbols per token,
    at each assumed symbol-error probability of ``pe_assumed``, labelled as assumed: the bits a token must carry
    to name its symbols that well. No measured CER enters it: CER counts edits per reference character, not the
    probability that a symbol is wrong (learning notes A9).

    Args:
        df: results rows (reader, text_type, block, page_id, render_mode, font_size_px, vision_tokens,
            reference_chars, hits, substitutions, deletions, insertions); only block Z is drawn.
        out_path: the PDF to write, under reports/figures/.
        pe_assumed: the symbol-error probabilities of the Fano lines.

    Raises:
        ValueError: when the frame has no block Z rows.
    """
    import matplotlib
    from matplotlib.figure import Figure

    z = df[df["block"] == "Z"]
    if z.empty:
        raise ValueError("no rows of block Z: Figure 2 needs the size sweep (the pilot has none)")
    text_types = sorted(set(z["text_type"].tolist()), key=lambda t: (t not in TEXT_COLOURS, t))
    readers = sorted(set(z["reader"].tolist()))
    colours = {r: f"C{i}" for i, r in enumerate(readers)}

    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(6.75, 2.5))
        axes = fig.subplots(1, len(text_types) + 1, squeeze=False)[0]
        x_max = 1.0
        for ax, text_type in zip(axes, text_types):
            part = z[z["text_type"] == text_type]
            for (reader, mode), group in part.groupby(["reader", "render_mode"], sort=True):
                xs, ys, sizes = [], [], []
                for size, g in group.groupby("font_size_px", sort=True):
                    errors, ref = _errors_and_ref(g)
                    chars_per_page = float(g.groupby("page_id")["reference_chars"].first().astype(float).mean())
                    tokens = float(g["vision_tokens"].astype(float).mean())
                    xs.append(chars_per_page / tokens)
                    ys.append(float(errors.sum() / ref.sum()))
                    sizes.append(float(size))
                ax.plot(xs, ys, MODE_STYLES.get(str(mode), ":"), marker="o", color=colours[str(reader)], markersize=3,
                        linewidth=0.9, label=f"{reader}, {mode}")
                for x, y, size in zip(xs, ys, sizes):
                    ax.annotate(f"{size:g}", (x, y), textcoords="offset points", xytext=(3, 3), fontsize=5,
                                color=colours[str(reader)])
                x_max = max(x_max, max(xs))
            ax.set_title(f"{text_type}: block Z")
            ax.set_xlabel("characters per vision token")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
        axes[0].set_ylabel("pooled CER")
        handles, labels = axes[0].get_legend_handles_labels()
        fano = axes[-1]
        fine = np.linspace(0.0, x_max * 1.05, 101)
        for pe in pe_assumed:
            fano.plot(fine, fine * stats.fano_floor_bits(float(pe)), "-", linewidth=0.9, label=f"P_e = {pe:g} (assumed)")
        fano.set_title("bits a token must carry (M = 62)")
        fano.set_xlabel("symbols per vision token")
        fano.set_ylabel("Fano floor (bits per token)")
        fano.set_xlim(left=0)
        fano.set_ylim(bottom=0)
        fano.legend(frameon=False, loc="upper left")
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.legend(handles, labels, loc="lower center", ncol=min(3, max(1, len(labels))), frameon=False,
                   bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.0, handletextpad=0.4)
        _save(fig, out_path)


# --------------------------------------------------------------------------- Figure 3 (section 3S.4)


def _curve(name: str, curve: Mapping[str, Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(curve["frequency"], dtype=np.float64)
    g = np.asarray(curve["gain"], dtype=np.float64)
    if f.ndim != 1 or f.shape != g.shape or f.size == 0:
        raise ValueError(f"{name}: frequency and gain must be one-dimensional sequences of one length")
    order = np.argsort(f)
    return f[order], g[order]


def plot_mtf(curves: Mapping[str, Mapping[str, Sequence[float]]], out_path: str | Path,
             px_per_degree: float | None = None, *, reference: Mapping[str, Any] | None = None) -> None:
    """Figure 3: the encoder's modulation transfer, gain against spatial frequency, one line per curve.

    Args:
        curves: name -> ``{"frequency": cycles per pixel, "gain": the transfer at each frequency}``, one entry per
            reader or per layer (results/mtf/, build guide 3S.4). A light line marks the Nyquist frequency, half a
            cycle per pixel.
        out_path: the PDF to write, under reports/figures/.
        px_per_degree: when given, a second x axis on top in cycles per degree at that many pixels per degree, so
            the curve can be read against a human contrast-sensitivity function at one viewing distance; the axis
            label names the value as an assumption, since it is one (a 96 dpi screen at 60 cm gives 39.6).
        reference: an optional ``{"name", "frequency", "gain"}`` drawn as a plain black line without markers, for
            a curve that is not a measurement of the encoder, such as the Mannos-Sakrison CSF of ``mtf.csf_mannos_sakrison``
            placed on the pixel axis by f_px = f_deg / px_per_degree.

    Raises:
        ValueError: on an empty mapping, a curve or reference whose two sequences differ in length or are empty, or
            a px_per_degree that is not positive.
    """
    import matplotlib
    from matplotlib.figure import Figure

    if not curves:
        raise ValueError("no curves to draw")
    if px_per_degree is not None and not px_per_degree > 0:
        raise ValueError("px_per_degree must be positive")
    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(3.3, 2.5))
        ax = fig.subplots()
        f_max = 0.0
        for name, curve in curves.items():
            f, g = _curve(str(name), curve)
            ax.plot(f, g, "-o", markersize=2.5, linewidth=0.9, label=str(name))
            f_max = max(f_max, float(f.max()))
        if reference is not None:
            f, g = _curve("reference", reference)
            ax.plot(f, g, "-", color="black", linewidth=0.9, label=str(reference.get("name", "reference")))
            f_max = max(f_max, float(f.max()))
        ax.axvline(0.5, color="0.7", linewidth=0.6, zorder=0)
        ax.set_xlabel("spatial frequency (cycles per pixel)")
        ax.set_ylabel("gain")
        ax.set_xlim(0, max(0.5, f_max) * 1.02)
        ax.set_ylim(bottom=0)
        if px_per_degree is not None:
            scale = float(px_per_degree)
            top = ax.secondary_xaxis("top", functions=(lambda f: f * scale, lambda c: c / scale))
            top.set_xlabel(f"cycles per degree at ρ = {scale:g} px per degree (assumed)")
        ax.legend(frameon=False, fontsize=6, loc="lower right")  # every curve rises from the left; the right stays free
        _save(fig, out_path)


# --------------------------------------------------------------------------- Study 2, Figures 3 and 4 (PREREGISTRATION_2.md section 7)

GROUP_COLOURS = {  # Study 2's comparison groups (mrr.study2.GROUPS), the same in both figures
    "conventional": "#7f7f7f", "slot": "#9467bd", "contained": "#2ca02c", "cut": "#ff7f0e", "density": "#8c564b",
}
_STUDY2_TEXT_ORDER = ("random62", "natural")


def _study2_grid(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """The text types (random first) and readers (Study 1's, then the validation readers) that ``rows`` hold."""
    from mrr.study2 import VALIDATION_READERS

    text_types = sorted({str(r["text_type"]) for r in rows},
                        key=lambda t: (_STUDY2_TEXT_ORDER.index(t) if t in _STUDY2_TEXT_ORDER else 99, t))
    readers = sorted({str(r["reader"]) for r in rows}, key=lambda r: (r in VALIDATION_READERS, r))
    return text_types, readers


def plot_study2_conditions(bands: Sequence[Mapping[str, Any]], out_path: str | Path) -> None:
    """Study 2's Figure 3: pooled CER by condition, one panel per text type (rows) and reader (columns).

    Each condition is a point at its pooled CER (D13) with its page-bootstrap interval (``study2.condition_bands``:
    one draw of pages serves every condition of a panel), coloured by its group, so the conventional, contained and
    cut conditions section 7 names are marked. A reader that reads no page of a text type (smolvlm500m, random) has
    its panel left blank with a note.

    Args:
        bands: rows of ``study2.condition_bands(df, "cer", ...)``: reader, text_type, condition, group, value, low
            and high.
        out_path: the PDF to write, under reports/figures/.

    Raises:
        ValueError: when ``bands`` is empty.
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from mrr.study2 import GROUPS, condition_order, role

    if not bands:
        raise ValueError("no condition rows to draw: Study 2's Figure 3 needs complete pages")
    text_types, readers = _study2_grid(bands)
    cell: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for r in bands:
        cell.setdefault((str(r["reader"]), str(r["text_type"])), []).append(r)
    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(6.75, 1.5 + 1.35 * len(text_types)))
        axes = fig.subplots(len(text_types), len(readers), squeeze=False)
        for i, text_type in enumerate(text_types):
            for j, reader in enumerate(readers):
                ax = axes[i][j]
                rows = cell.get((reader, text_type))
                ax.set_title(f"{reader} ({role(reader)}), {text_type}")
                if not rows:
                    ax.text(0.5, 0.5, f"no {text_type} pages", ha="center", va="center", transform=ax.transAxes,
                            fontsize=6, color="0.5")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                by_code = {str(r["condition"]): r for r in rows}
                codes = condition_order(by_code)
                for x, code in enumerate(codes):
                    r = by_code[code]
                    value, low, high = float(r["value"]), float(r["low"]), float(r["high"])
                    ax.errorbar([x], [value], yerr=[[value - low], [high - value]], fmt="o", markersize=3,
                                color=GROUP_COLOURS.get(str(r["group"]), "black"), elinewidth=0.8, capsize=1.5)
                ax.set_xticks(range(len(codes)))
                ax.set_xticklabels(codes, rotation=90)
                ax.set_xlim(-0.6, len(codes) - 0.4)
                ax.set_ylim(bottom=0)
                if j == 0:
                    ax.set_ylabel("pooled CER")
        present = {str(r["group"]) for r in bands}
        handles = [Line2D([], [], marker="o", linestyle="", color=GROUP_COLOURS[g], markersize=3, label=g)
                   for g in GROUPS if g in present]
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.0))
        _save(fig, out_path)


def plot_study2_confusions(rates: Sequence[Mapping[str, Any]], out_path: str | Path) -> None:
    """Study 2's Figure 4: the O/0 and I/l/1 letter confusions per 1,000 reference letters, by font and layout.

    One panel per pair kind (columns) and text type (rows); in each, a group of bars per reader and one bar per
    comparison group (conventional and density are Source Sans 3 laid out conventionally, slot is Source Sans 3 in
    the patch-aware layout, contained and cut are Fira Mono in it), each labelled with its count of confusions.

    Args:
        rates: rows of ``study2.pair_rates``: reader, text_type, group, kind, count, letters and per_1000.
        out_path: the PDF to write, under reports/figures/.

    Raises:
        ValueError: when ``rates`` holds no O/0 or I/l/1 row.
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    from mrr.study2 import GROUPS

    kinds = ("O/0", "I/l/1")
    rows = [r for r in rates if str(r["kind"]) in kinds]
    if not rows:
        raise ValueError("no O/0 or I/l/1 rows to draw: Study 2's Figure 4 needs complete pages")
    text_types, readers = _study2_grid(rows)
    groups = [g for g in GROUPS if any(str(r["group"]) == g for r in rows)]
    width = 0.8 / len(groups)
    value = {(str(r["reader"]), str(r["text_type"]), str(r["group"]), str(r["kind"])): r for r in rows}
    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(6.75, 1.3 + 1.5 * len(text_types)))
        axes = fig.subplots(len(text_types), len(kinds), squeeze=False)
        for i, text_type in enumerate(text_types):
            for j, kind in enumerate(kinds):
                ax = axes[i][j]
                for k, group in enumerate(groups):
                    xs, ys, counts = [], [], []
                    for x, reader in enumerate(readers):
                        r = value.get((reader, text_type, group, kind))
                        if r is None:
                            continue
                        xs.append(x - 0.4 + width * (k + 0.5))
                        ys.append(float(r["per_1000"]))
                        counts.append(int(r["count"]))
                    ax.bar(xs, ys, width=width * 0.95, color=GROUP_COLOURS[group])
                    for x, y, n in zip(xs, ys, counts):
                        ax.annotate(f"{n}", (x, y), textcoords="offset points", xytext=(0, 1), ha="center", fontsize=4)
                ax.set_xticks(range(len(readers)))
                ax.set_xticklabels(readers, rotation=30, ha="right")
                ax.set_ylim(bottom=0)
                ax.set_title(f"{kind}, {text_type}")
                if j == 0:
                    ax.set_ylabel("confusions per 1,000 letters")
        handles = [Patch(color=GROUP_COLOURS[g], label=g) for g in groups]
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.0))
        _save(fig, out_path)


# --------------------------------------------------------------------------- Study 3 stage 0, Figures 5 and 6 (ANALYSIS_PLAN_3.md section 2.7)

BEHAVIOUR_PARTS = (  # Figure 6's stack, bottom up: mrr.behaviour's behaviour errors, which add up to behaviour_rate
    ("omitted_head_rate", "omitted, head", "#c5b0d5"),
    ("omitted_interior_rate", "omitted, interior", "#9467bd"),
    ("omitted_tail_rate", "omitted, tail", "#5e3c99"),
    ("runaway_rate", "runaway inserted", "#ff7f0e"),
    ("case_normalised_rate", "case normalised", "#8c564b"),
)
FORMAT_HATCH = "//////"  # format characters (R1) are counted apart from the alignment, so they are hatched, not filled
_FAINT = 0.35  # the alpha of a curve or a 1% size beyond the measured sizes
_CER_FLOOR = 5e-4  # Figure 5's log axis starts here; an interval that reaches 0 is drawn down to it
_PX_TICKS = (6, 8, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200)
_PATCH_TICKS = (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 15)


def _num(value: Any) -> float:
    """A table value as a float: an empty CSV field is NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _flag(value: Any) -> bool:
    """A table's True/False column, read by pandas (bool) or csv (str)."""
    return str(value).strip().lower() in ("true", "1")


def _panels(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    """(reader, text type) -> its rows, smallest size first."""
    panels: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for r in rows:
        panels.setdefault((str(r["reader"]), str(r["text_type"])), []).append(r)
    return {k: sorted(v, key=lambda r: _num(r["font_size_px"])) for k, v in panels.items()}


def _log_ticks(axis: Any, ticks: Sequence[float], low: float, high: float) -> None:
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    axis.set_major_locator(FixedLocator([t for t in ticks if low <= t <= high]))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def plot_stage0_curves(cells: Sequence[Mapping[str, Any]], fits: Sequence[Mapping[str, Any]],
                       out_path: str | Path) -> None:
    """Study 3 stage 0's Figure 5: each reader's acuity curves, one panel per reader (rows) and text type (columns).

    The primary and the reading CER are drawn against the font size in pixels (bottom axis) and in patches of the
    reader's P (top axis), both logarithmic. The cells are points with their passage-bootstrap intervals (filled:
    primary; open: reading, set 4% to the right so neither hides the other). The lines are the fits of
    ``mrr.acuity``, CER(s) = 1 / (1 + (s / a) ** beta) (solid: primary; dashed: reading), faint beyond the measured
    sizes. The dotted line is CER 0.01; each fit's s01 sits on it as a diamond with its interval (filled: primary;
    open: reading, just below), faint when it lies beyond the measured sizes. An interval that reaches 0 is drawn
    down to the axis. Nothing is refitted: the curves and intervals are the tables'.

    Args:
        cells: rows of cells.csv (scripts/25_fit_stage0.py): reader, text_type, font_size_px, cer, cer_lo, cer_hi,
            reading_cer, reading_cer_lo and reading_cer_hi.
        fits: rows of acuity.csv: reader, text_type, patch_size, a, beta, s01, s01_lo, s01_hi, s01_in_range and the
            same reading_* columns. A panel without a fit shows its cells only.
        out_path: the PDF to write, under reports/figures/.

    Raises:
        ValueError: when ``cells`` is empty.
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from mrr.acuity import cer_curve

    if not cells:
        raise ValueError("no cells to draw: Figure 5 needs cells.csv from scripts/25_fit_stage0.py")
    text_types, readers = _study2_grid(cells)
    panels = _panels(cells)
    fit_of = {(str(f["reader"]), str(f["text_type"])): f for f in fits}
    measures = (("", "o", "-", 1.0, 0.01), ("reading_", "s", "--", 1.04, 0.01 / 1.45))  # prefix, marker, line, x, s01 y
    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(6.75, 1.0 + 1.8 * len(readers)))
        axes = fig.subplots(len(readers), len(text_types), squeeze=False, sharey=True)
        for i, reader in enumerate(readers):
            for j, text_type in enumerate(text_types):
                ax = axes[i][j]
                rows, fit = panels.get((reader, text_type)), fit_of.get((reader, text_type))
                ax.set_title(f"{reader}, {text_type}")
                if not rows:
                    ax.text(0.5, 0.5, f"no {text_type} cells", ha="center", va="center", transform=ax.transAxes,
                            fontsize=6, color="0.5")
                    ax.set_xticks([])
                    continue
                colour = TEXT_COLOURS.get(text_type, "black")
                sizes = np.array([_num(r["font_size_px"]) for r in rows])
                first, last = float(sizes.min()), float(sizes.max())
                reach = [last] + ([_num(fit[f"{p}s01_hi"]) for p, *_ in measures] if fit else [])
                left, right = first / 1.12, max(v for v in reach if np.isfinite(v)) * 1.12
                ax.axhline(0.01, color="0.6", linewidth=0.6, linestyle=":", zorder=1)
                for prefix, marker, line, shift, s01_y in measures:
                    face = colour if not prefix else "white"
                    y = np.array([_num(r[f"{prefix}cer"]) for r in rows])
                    low = np.maximum([_num(r[f"{prefix}cer_lo"]) for r in rows], _CER_FLOOR)
                    high = np.array([_num(r[f"{prefix}cer_hi"]) for r in rows])
                    ax.errorbar(sizes * shift, y, yerr=[np.clip(y - low, 0, None), np.clip(high - y, 0, None)],
                                fmt=marker, markersize=3, color=colour, markerfacecolor=face, elinewidth=0.6,
                                capsize=1.2, zorder=3)
                    a, beta = (_num(fit[f"{prefix}a"]), _num(fit[f"{prefix}beta"])) if fit else (np.nan, np.nan)
                    if not (np.isfinite(a) and np.isfinite(beta)):
                        continue
                    for lo, hi, alpha in ((left, first, _FAINT), (first, last, 1.0), (last, right, _FAINT)):
                        if hi > lo:
                            s = np.geomspace(lo, hi, 80)
                            ax.plot(s, cer_curve(s, a, beta), line, color=colour, linewidth=0.9, alpha=alpha, zorder=2)
                    s01, s01_lo, s01_hi = (_num(fit[f"{prefix}s01{k}"]) for k in ("", "_lo", "_hi"))
                    if np.isfinite(s01):
                        banded = np.isfinite(s01_lo) and np.isfinite(s01_hi)
                        xerr = [[max(s01 - s01_lo, 0)], [max(s01_hi - s01, 0)]] if banded else None
                        ax.errorbar([s01], [s01_y], xerr=xerr, fmt="D", markersize=3.2, color=colour,
                                    markerfacecolor=face, elinewidth=0.8, capsize=1.5,
                                    alpha=1.0 if _flag(fit[f"{prefix}s01_in_range"]) else _FAINT, zorder=4)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(left, right)
                ax.set_ylim(_CER_FLOOR, 1.5)
                _log_ticks(ax.xaxis, _PX_TICKS, left, right)
                patch = _num(fit["patch_size"]) if fit else np.nan
                if np.isfinite(patch):
                    top = ax.secondary_xaxis("top", functions=(lambda v, p=patch: v / p, lambda v, p=patch: v * p))
                    _log_ticks(top.xaxis, _PATCH_TICKS, left / patch, right / patch)
                    top.set_xlabel(f"patches (P = {patch:g} px)", fontsize=6, labelpad=1)
                if j == 0:
                    ax.set_ylabel("CER")
                if i == len(readers) - 1:
                    ax.set_xlabel("font size (px)")
        grey = "0.3"
        handles = [
            Line2D([], [], marker="o", markersize=3, color=grey, linestyle="-", label="primary CER: cells and fit"),
            Line2D([], [], marker="s", markersize=3, color=grey, markerfacecolor="white", linestyle="--",
                   label="reading CER: cells and fit"),
            Line2D([], [], color="0.6", linestyle=":", linewidth=0.6, label="CER 0.01"),
            Line2D([], [], marker="D", markersize=3.2, color=grey, linestyle="", label="s01 and its interval"),
            Line2D([], [], color=grey, alpha=_FAINT, linewidth=0.9, label="faint: beyond the measured sizes"),
        ]
        fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.6)
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.0))
        _save(fig, out_path)


def plot_stage0_behaviour(behaviour: Sequence[Mapping[str, Any]], cells: Sequence[Mapping[str, Any]],
                          out_path: str | Path) -> None:
    """Study 3 stage 0's Figure 6: the behaviour rates against size, one panel per reader (rows) and text type (columns).

    Each size is a bar of behaviour errors per 100 reference characters (ANALYSIS_PLAN_3.md section 1.2), stacked
    bottom up as omitted (head, interior, tail), runaway inserted and case normalised. These add up to the cell's
    behaviour rate, whose passage-bootstrap interval from cells.csv is the black bar. Format characters (R1) are
    counted apart from the alignment and sit hatched on top. Each panel has its own scale, since SmolVLM's prose
    rates are tens of times Qwen's.

    Args:
        behaviour: rows of behaviour.csv (scripts/25_fit_stage0.py): reader, text_type, font_size_px and the rates
            of ``BEHAVIOUR_PARTS`` and format_rate.
        cells: rows of cells.csv, for behaviour_rate_lo and behaviour_rate_hi; a size without a cell has no interval.
        out_path: the PDF to write, under reports/figures/.

    Raises:
        ValueError: when ``behaviour`` is empty.
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    if not behaviour:
        raise ValueError("no behaviour rows to draw: Figure 6 needs behaviour.csv from scripts/25_fit_stage0.py")
    text_types, readers = _study2_grid(behaviour)
    panels = _panels(behaviour)
    band = {(str(c["reader"]), str(c["text_type"]), _num(c["font_size_px"])):
            (_num(c["behaviour_rate_lo"]), _num(c["behaviour_rate_hi"])) for c in cells}
    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(6.75, 1.0 + 1.45 * len(readers)))
        axes = fig.subplots(len(readers), len(text_types), squeeze=False)
        for i, reader in enumerate(readers):
            for j, text_type in enumerate(text_types):
                ax = axes[i][j]
                rows = panels.get((reader, text_type))
                ax.set_title(f"{reader}, {text_type}")
                if not rows:
                    ax.text(0.5, 0.5, f"no {text_type} cells", ha="center", va="center", transform=ax.transAxes,
                            fontsize=6, color="0.5")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                x = np.arange(len(rows))
                stack = np.zeros(len(rows))
                for column, _, colour in BEHAVIOUR_PARTS:
                    height = 100 * np.array([_num(r[column]) for r in rows])
                    ax.bar(x, height, bottom=stack, width=0.7, color=colour, linewidth=0)
                    stack += height
                fmt = 100 * np.array([_num(r["format_rate"]) for r in rows])
                ax.bar(x, fmt, bottom=stack, width=0.7, facecolor="white", edgecolor="0.4", hatch=FORMAT_HATCH,
                       linewidth=0.4)
                tallest = float((stack + fmt).max())
                for k, r in enumerate(rows):
                    lo, hi = band.get((reader, text_type, _num(r["font_size_px"])), (np.nan, np.nan))
                    if np.isfinite(lo) and np.isfinite(hi):
                        below, above = max(stack[k] - 100 * lo, 0), max(100 * hi - stack[k], 0)
                        ax.errorbar([x[k]], [stack[k]], yerr=[[below], [above]], fmt="none", ecolor="black",
                                    elinewidth=0.6, capsize=1.5, zorder=3)
                        tallest = max(tallest, 100 * hi)
                ax.set_xticks(x)
                ax.set_xticklabels([f"{_num(r['font_size_px']):g}" for r in rows])
                ax.set_xlim(-0.6, len(rows) - 0.4)
                ax.set_ylim(0, max(tallest * 1.12, 0.05))
                if j == 0:
                    ax.set_ylabel("% of reference chars")
                if i == len(readers) - 1:
                    ax.set_xlabel("font size (px)")
        handles = [Patch(color=colour, label=label) for _, label, colour in BEHAVIOUR_PARTS]
        handles.append(Patch(facecolor="white", edgecolor="0.4", hatch=FORMAT_HATCH, label="format (counted apart)"))
        fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.6)
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.0))
        _save(fig, out_path)
