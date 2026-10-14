# PURPOSE: The paper's figures.
# WHAT GOES HERE: Figure 0 (shift-equivariance error), Figure 1 (phase curve) and Figure 2 (accuracy versus tokens trade-off).
# FILLED IN AT STEP: W1.9 (Figure 0), W2.5 (Figure 1), W3.4 (Figure 2).
# DO NOT: Hand-edit output images. Do not import matplotlib at module top level.
"""Figures.

Learn B5 (plotting) before implementing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


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
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")


def plot_phase_curve(df: Any, out_path: str | Path) -> None:
    """Figure 1: CER versus glyph phase, one line per reader, bootstrap band.

    Args:
        df: pandas DataFrame of results rows.
        out_path: output image path under reports/figures/.
    """
    raise NotImplementedError("TODO[W2.5]: plot CER against phase per reader with a bootstrap band")


def plot_tradeoff(df: Any, out_path: str | Path) -> None:
    """Figure 2: CER versus vision tokens, conventional versus patch-aware.

    Args:
        df: pandas DataFrame of results rows.
        out_path: output image path under reports/figures/.
    """
    raise NotImplementedError("TODO[W3.4]: plot CER against vision tokens for each render mode")
