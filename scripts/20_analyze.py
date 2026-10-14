# PURPOSE: Statistics over run CSVs.
# WHAT GOES HERE: Load results, pair conditions by page, run preregistered tests, write tables.
# FILLED IN AT STEP: W2.4 (pilot), W3.4 (full).
# DO NOT: Run analyses that are not in PREREGISTRATION.md without labelling them exploratory.
"""Steps W2.4 and W3.4: turn the run folders of one sweep into the tables of PREREGISTRATION.md section 9.

    python scripts/20_analyze.py --config configs/sweeps/pilot.yaml [--runs-root results/runs] [--out-dir reports/tables]

The script reads only the run folders whose manifest names the sweep file (``sweep.load_results``), applies the one
exclusion of section 8 (a page that misses a planned condition because its run stopped at the deadline leaves every
table but ``checks.csv``), runs the tests of section 9 family by family and writes one CSV per kind of result into
``<out-dir>/<sweep_name>/`` with a manifest:

    checks.csv             one row per run folder: rows done and planned, repeated keys, missing transcription files,
                           the hit_limit count and share, determinism mismatches, pages excluded, median latency
    harmonics.csv          one row per reader, text type, block (V, H, C) and period group: the F-test, Holm, the RMS
                           amplitude with its page-bootstrap interval, the share, sigma, pages and levels
    paired.csv             one row per reader, text type and comparison: (H, phi = 0) against C pooled (clause (c));
                           each anti-aliasing mode of block A against (C, phi = 0)
    subpixel.csv           one row per reader and text type: the F-test of block S and the pooled CER at each phase
    sizes.csv              one row per reader, text type, render mode and size of block Z: pooled CER with its
                           interval, vision tokens, characters and symbols per token, the plateau and s*
    coherence_summary.csv  one row per reader, text type, block layout, period and harmonic: the mean and range of
                           R_k over pages beside the RMS amplitude of the matching harmonic group

Every table carries ``status`` (``preregistered`` or ``exploratory``) and ``family``, and every preregistered p-value
has its Holm-adjusted value beside it. Every resampling is by page, never by row, with 2,000 resamples and seed
20261014 (section 9). Nothing here is exploratory yet: anything added that section 9 does not name must be written
with ``status = exploratory``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from mrr import grid, stats
from mrr.manifest import new_run_id, write_manifest
from mrr.score import CharCounts, pooled_cer
from mrr.sweep import (
    COHERENCE_FILE,
    DETERMINISM_FILE,
    SHARED,
    Z_MODE_TAGS,
    condition_hash,
    expand_conditions,
    load_results,
    pagination_key,
    reader_periods,
    sweep_run_dirs,
)

RESAMPLES = 2000  # PREREGISTRATION.md section 9: every resampling is by page, with 2,000 resamples
SEED = 20261014  # and this seed
ALPHA = 0.05  # section 6
DELTA = 0.01  # section 9, item 5: s* = min{s : CER(s) <= CER_inf + delta}, in absolute CER
PLATEAU_SIZES = (20.0, 24.0)  # section 9, item 5: CER_inf is the pooled CER of the rows at 20 and 24 px
PREREGISTERED, EXPLORATORY = "preregistered", "exploratory"
HARMONIC_BLOCKS = ("V", "H", "C")  # section 9, item 2
TABLES = ("checks", "harmonics", "paired", "subpixel", "sizes", "coherence_summary")
COLUMNS = {
    "checks": [
        "run_dir", "reader", "rows_done", "rows_planned", "repeated_keys", "missing_transcriptions", "hit_limit_count",
        "hit_limit_share", "determinism_pairs", "determinism_mismatches", "pages_seen", "pages_excluded",
        "excluded_page_ids", "median_latency_s", "status", "family",
    ],
    "harmonics": [
        "reader", "text_type", "block", "axis", "range_px", "period_px", "bins", "q", "F", "df1", "df2", "p", "p_holm",
        "rms", "rms_low", "rms_high", "share", "sigma", "n_pages", "n_levels", "n_rows", "status", "family",
    ],
    "paired": [
        "reader", "text_type", "comparison", "n", "n_zero", "statistic", "p", "method", "p_holm", "median_diff",
        "diff_low", "diff_high", "status", "family",
    ],
    "subpixel": ["reader", "text_type", "F", "df1", "df2", "p", "p_holm", "n_pages", "n_levels", "n_rows", "status", "family"],
    "sizes": [
        "reader", "text_type", "render_mode", "font_size_px", "cer", "cer_low", "cer_high", "n_pages", "n_rows",
        "vision_tokens", "chars_per_token", "symbols_per_token", "cer_plateau", "s_star_px", "status", "family",
    ],
    "coherence_summary": [
        "reader", "text_type", "block", "render_mode", "line_pitch_mode", "axis", "period_px", "harmonic", "R_mean",
        "R_min", "R_max", "n_pages", "rms", "status", "family",
    ],
}
Z_PAGE = re.compile(r"_z(?P<size>[0-9.]+)_(?P<mode>conv|pa)_p\d{4}$")  # <book>_z<size>_<conv|pa>_p<NNNN>
Z_MODES = {tag: mode for mode, tag in Z_MODE_TAGS.items()}
NAN = float("nan")

_log = logging.getLogger("analyze")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="configs/sweeps/<name>.yaml; the tables go to <out-dir>/<name>/")
    p.add_argument("--runs-root", default="results/runs", help="where the run folders are (default results/runs)")
    p.add_argument("--out-dir", default="reports/tables", help="the parent of the sweep's table folder")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- the run folders and their side files


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Every row of a small CSV as strings; an empty list without the file."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class RunFolder:
    """One run folder: its manifest, the conditions its reader was to run, and its coherence and determinism rows."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.manifest = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        cfg = self.manifest["config"]
        self.reader_cfg: Mapping[str, Any] = cfg["reader"]
        self.reader = str(self.reader_cfg["name"])
        self.sweep: Mapping[str, Any] = cfg["sweep"]
        self.blocks: dict[str, Mapping[str, Any]] = {str(k): v for k, v in self.sweep["blocks"].items()}
        self.repeats = int(cfg.get("repeats", 1))
        self.rows_planned = int(cfg.get("rows_planned", 0))
        self.conditions = expand_conditions(self.sweep, self.reader_cfg)
        self.hashes_by_key: dict[str, list[str]] = {}
        for c in self.conditions:
            self.hashes_by_key.setdefault(pagination_key(c), []).append(condition_hash(c))
        self.coherence = _read_csv_rows(self.run_dir / COHERENCE_FILE)
        self.determinism = _read_csv_rows(self.run_dir / DETERMINISM_FILE)

    @staticmethod
    def page_key(page_id: str) -> str:
        """The pagination key a page id belongs to: ``shared`` or ``z:<render_mode>:<size>`` (sweep.pagination_key)."""
        m = Z_PAGE.search(page_id)
        if m is None:
            return SHARED
        return f"z:{Z_MODES[m['mode']]}:{float(m['size']):g}"

    def planned_rows(self, page_id: str) -> set[tuple[str, int]]:
        """The (condition hash, repeat) pairs the runner was to write for this page."""
        return {(h, r) for h in self.hashes_by_key.get(self.page_key(page_id), []) for r in range(self.repeats)}

    def reference_symbols(self, page_id: str) -> float:
        """The page's reference length without whitespace, from references/<page_id>.txt; NaN without the file."""
        path = self.run_dir / "references" / f"{page_id}.txt"
        if not path.exists():
            return NAN
        return float(len("".join(path.read_text(encoding="utf-8").split())))


def excluded_pages(df, folders: Sequence[RunFolder]) -> dict[str, set[str]]:
    """Section 8: the pages of each run folder that miss a planned (condition, repeat) row, by run folder path."""
    out: dict[str, set[str]] = {}
    for folder in folders:
        rows = df[df["run_dir"] == str(folder.run_dir)]
        missing: set[str] = set()
        for page_id, group in rows.groupby("page_id", sort=True):
            done = set(zip(group["condition_hash"].tolist(), (int(r) for r in group["repeat"].tolist())))
            if folder.planned_rows(str(page_id)) - done:
                missing.add(str(page_id))
        out[str(folder.run_dir)] = missing
    return out


# --------------------------------------------------------------------------- pooled counts (D13)


def _pooled(rows) -> float:
    """(S + D + I) / (H + S + D) with the counts summed over ``rows``: never a mean of row CERs (D13)."""
    counts = CharCounts(
        hits=int(rows["hits"].sum()), substitutions=int(rows["substitutions"].sum()),
        deletions=int(rows["deletions"].sum()), insertions=int(rows["insertions"].sum()),
    )
    if counts.hits + counts.substitutions + counts.deletions == 0:
        return NAN
    return pooled_cer([counts])


def _errors_and_ref(rows) -> tuple[np.ndarray, np.ndarray]:
    """Per row: errors = S + D + I and reference characters = H + S + D, the two sums a pooled CER needs."""
    s, d, i, h = (rows[c].to_numpy(dtype=np.float64) for c in ("substitutions", "deletions", "insertions", "hits"))
    return s + d + i, h + s + d


def _page_cer(rows) -> dict[str, float]:
    """Each page's CER pooled over its rows (every row of one page reads the same reference)."""
    return {str(page_id): _pooled(group) for page_id, group in rows.groupby("page_id", sort=True)}


def _fill_holm(rows: list[dict[str, Any]]) -> None:
    """Holm within each ``_holm`` group (section 6), then drop the group key. Undefined p-values stay NaN."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.pop("_holm"), []).append(row)
    for members in groups.values():
        tested = [r for r in members if not math.isnan(r["p"])]
        if tested:
            for r, adjusted in zip(tested, stats.holm([r["p"] for r in tested])):
                r["p_holm"] = float(adjusted)


# --------------------------------------------------------------------------- the tables


def checks_table(df_all, folders: Sequence[RunFolder], excluded: Mapping[str, set[str]]) -> list[dict[str, Any]]:
    """Section 9, item 1: one row per run folder."""
    out = []
    for folder in folders:
        key = str(folder.run_dir)
        rows = df_all[df_all["run_dir"] == key]
        n = int(len(rows))
        missing = sum(1 for f in rows["transcription_file"].tolist() if not (folder.run_dir / str(f)).exists())
        hit = int(rows["hit_limit"].astype(bool).sum()) if n else 0
        mismatches = sum(1 for r in folder.determinism if str(r.get("identical")) != "True")
        out.append({
            "run_dir": folder.run_dir.name, "reader": folder.reader, "rows_done": n, "rows_planned": folder.rows_planned,
            "repeated_keys": 0,  # load_results raises on a repeated key, so a table exists only when there is none
            "missing_transcriptions": int(missing), "hit_limit_count": hit, "hit_limit_share": hit / n if n else NAN,
            "determinism_pairs": len(folder.determinism), "determinism_mismatches": int(mismatches),
            "pages_seen": int(rows["page_id"].nunique()) if n else 0, "pages_excluded": len(excluded.get(key, ())),
            "excluded_page_ids": ";".join(sorted(excluded.get(key, ()))),
            "median_latency_s": float(rows["latency_s"].median()) if n else NAN,
            "status": PREREGISTERED, "family": "checks",
        })
    return out


def _range_px(spec: Mapping[str, Any], reader_cfg: Mapping[str, Any]) -> int:
    name = spec["range"]
    if name == "patch":
        return int(reader_cfg["patch_size"])
    if name == "cell":
        return grid.token_cell_px(reader_cfg)
    return grid.sweep_period_px(reader_cfg)


def harmonics_table(df, folders_by_reader: Mapping[str, RunFolder]) -> list[dict[str, Any]]:
    """Section 9, item 2: the saturated harmonic model per reader, text type and block V, H and C."""
    out: list[dict[str, Any]] = []
    for reader, folder in sorted(folders_by_reader.items()):
        reader_cfg = folder.reader_cfg
        patch, cell = int(reader_cfg["patch_size"]), grid.token_cell_px(reader_cfg)
        for text_type in sorted(df["text_type"].unique().tolist()):
            for block in HARMONIC_BLOCKS:
                spec = folder.blocks.get(block)
                if spec is None:
                    continue
                if "range" not in spec:
                    _log.warning("%s %s block %s lists its phases instead of a range; no harmonic model for it",
                                 reader, text_type, block)
                    continue
                rows = df[(df["reader"] == reader) & (df["text_type"] == text_type) & (df["block"] == block)]
                if rows.empty:
                    continue
                levels = grid.phase_levels(spec, reader_cfg, block=block)
                L = _range_px(spec, reader_cfg)
                periods = [T for T in reader_periods(reader_cfg) if L % T == 0]
                groups = stats.bin_groups(L, periods, len(levels))
                axis = str(spec.get("axis", "x"))
                phase = rows["phase_x_px" if axis == "x" else "phase_y_px"].to_numpy(dtype=np.float64)
                y, pages = rows["cer"].to_numpy(dtype=np.float64), rows["page_id"].to_numpy()
                fit = stats.harmonic_fit(y, pages, phase, L, groups, levels)
                for T, g in sorted(fit["groups"].items()):
                    low, high = stats.page_bootstrap_rms(y, pages, phase, L, groups, levels, T, RESAMPLES, SEED, ALPHA)
                    if block == "C":
                        status, family = PREREGISTERED, "control_C"
                    elif (block == "H" and T == patch) or (block == "V" and T in (patch, cell)):
                        status, family = PREREGISTERED, f"a_{text_type}"
                    else:  # a period that is neither the patch nor the cell in V or H: section 9 names no test for it
                        status, family = EXPLORATORY, f"other_{block}"
                    out.append({
                        "reader": reader, "text_type": text_type, "block": block, "axis": axis, "range_px": L,
                        "period_px": int(T), "bins": " ".join(str(k) for k in g["bins"]), "q": g["q"], "F": g["F"],
                        "df1": g["df1"], "df2": g["df2"], "p": g["p"], "p_holm": NAN, "rms": g["rms"], "rms_low": low,
                        "rms_high": high, "share": g["share"], "sigma": fit["sigma"], "n_pages": fit["n_pages"],
                        "n_levels": len(levels), "n_rows": fit["n_rows"], "status": status, "family": family,
                        "_holm": f"{family}|{reader}|{text_type}",
                    })
    _fill_holm(out)
    return out


def _paired_row(reader: str, text_type: str, comparison: str, a: np.ndarray, b: np.ndarray, family: str,
                holm_group: str) -> dict[str, Any]:
    """Section 9, item 3: the paired Wilcoxon test, the median of b - a and its bootstrap interval."""
    row: dict[str, Any] = {
        "reader": reader, "text_type": text_type, "comparison": comparison, "n": int(a.size), "n_zero": NAN,
        "statistic": NAN, "p": NAN, "method": "", "p_holm": NAN, "median_diff": NAN, "diff_low": NAN, "diff_high": NAN,
        "status": PREREGISTERED, "family": family, "_holm": holm_group,
    }
    try:
        w = stats.paired_wilcoxon(a, b)
        row.update(n_zero=int(w["n_zero"]), statistic=float(w["statistic"]), p=float(w["pvalue"]), method=w["method"])
    except ValueError as exc:  # every difference zero, or too few pages: the test is undefined and the table says so
        row["method"] = f"undefined: {exc}"
    row["median_diff"] = stats.median_paired_difference(a, b)
    row["diff_low"], row["diff_high"] = stats.bootstrap_ci(b - a, np.median, RESAMPLES, ALPHA, SEED)
    return row


def paired_table(df) -> list[dict[str, Any]]:
    """Section 9, item 3: clause (c) per reader and text type, and each mode of block A against (C, phi = 0)."""
    out: list[dict[str, Any]] = []
    for (reader, text_type), sub in df.groupby(["reader", "text_type"], sort=True):
        reader, text_type = str(reader), str(text_type)
        at_zero = (sub["phase_x_px"] == 0.0) & (sub["phase_y_px"] == 0.0)
        c_rows = sub[sub["block"] == "C"]
        c_pooled = _page_cer(c_rows)
        h_zero = _page_cer(sub[(sub["block"] == "H") & at_zero])
        pages = sorted(set(c_pooled) & set(h_zero))
        if pages:
            a = np.array([c_pooled[p] for p in pages])
            b = np.array([h_zero[p] for p in pages])
            out.append(_paired_row(reader, text_type, "H_phi0_vs_C_pooled", a, b, f"c_{text_type}", f"c_{text_type}"))
        c_zero = _page_cer(c_rows[(c_rows["phase_x_px"] == 0.0) & (c_rows["phase_y_px"] == 0.0)])
        a_rows = sub[sub["block"] == "A"]
        for mode in sorted(a_rows["aa_mode"].unique().tolist()):
            b_pages = _page_cer(a_rows[a_rows["aa_mode"] == mode])
            pages = sorted(set(c_zero) & set(b_pages))
            if not pages:
                continue
            a = np.array([c_zero[p] for p in pages])
            b = np.array([b_pages[p] for p in pages])
            out.append(_paired_row(reader, text_type, f"A_{mode}_vs_C_phi0", a, b, "A", f"A|{reader}|{text_type}"))
    _fill_holm(out)
    return out


def subpixel_table(df, folders_by_reader: Mapping[str, RunFolder]) -> list[dict[str, Any]]:
    """Section 9, item 4: block S, phase as a factor with page fixed effects, and the pooled CER at each phase."""
    out: list[dict[str, Any]] = []
    s_rows = df[df["block"] == "S"]
    for (reader, text_type), sub in s_rows.groupby(["reader", "text_type"], sort=True):
        reader, text_type = str(reader), str(text_type)
        spec = folders_by_reader[reader].blocks.get("S", {})
        axis = str(spec.get("axis", "x"))
        phase = sub["phase_x_px" if axis == "x" else "phase_y_px"].to_numpy(dtype=np.float64)
        y, pages = sub["cer"].to_numpy(dtype=np.float64), sub["page_id"].to_numpy()
        X, names = stats.harmonic_design(phase, 1.0, [1, 2])  # cos1, sin1, cos2; sin2 vanishes at quarter pixels
        test = stats.group_f_test(y, pages, X, names, names)
        errors, ref = _errors_and_ref(sub)
        pooled = stats.pooled_cer_by_level(errors, ref, pages, phase, RESAMPLES, SEED, ALPHA)
        row: dict[str, Any] = {
            "reader": reader, "text_type": text_type, "F": test["F"], "df1": test["df1"], "df2": test["df2"],
            "p": test["p"], "p_holm": NAN, "n_pages": int(len(np.unique(pages))), "n_levels": int(len(pooled["levels"])),
            "n_rows": int(y.size), "status": PREREGISTERED, "family": "S", "_holm": f"S|{reader}|{text_type}",
        }
        for level, cer, low, high in zip(pooled["levels"], pooled["cer"], pooled["low"], pooled["high"]):
            row[f"cer_at_{float(level):g}"] = float(cer)
            row[f"cer_low_at_{float(level):g}"] = float(low)
            row[f"cer_high_at_{float(level):g}"] = float(high)
        out.append(row)
    _fill_holm(out)  # a family of one test per reader and text type: Holm leaves p unchanged
    return out


def sizes_table(df, folders: Sequence[RunFolder]) -> list[dict[str, Any]]:
    """Section 9, item 5: block Z per reader, text type, render mode and size, with the plateau and s*."""
    out: list[dict[str, Any]] = []
    by_path = {str(f.run_dir): f for f in folders}
    z_rows = df[df["block"] == "Z"]
    for (reader, text_type, mode), group in z_rows.groupby(["reader", "text_type", "render_mode"], sort=True):
        per_size: dict[float, dict[str, Any]] = {}
        for size, rows in group.groupby("font_size_px", sort=True):
            errors, ref = _errors_and_ref(rows)
            pages = rows["page_id"].to_numpy()
            band = stats.pooled_cer_by_level(errors, ref, pages, np.zeros(len(rows)), RESAMPLES, SEED, ALPHA)
            tokens = int(rows["vision_tokens"].iloc[0])
            per_page = rows.drop_duplicates("page_id")
            chars = float(per_page["reference_chars"].mean())
            symbols = float(np.mean([
                by_path[str(d)].reference_symbols(str(p)) for d, p in zip(per_page["run_dir"], per_page["page_id"])
            ]))
            per_size[float(size)] = {
                "reader": str(reader), "text_type": str(text_type), "render_mode": str(mode), "font_size_px": float(size),
                "cer": float(band["cer"][0]), "cer_low": float(band["low"][0]), "cer_high": float(band["high"][0]),
                "n_pages": int(len(per_page)), "n_rows": int(len(rows)), "vision_tokens": tokens,
                "chars_per_token": chars / tokens, "symbols_per_token": symbols / tokens,
            }
        plateau_rows = group[group["font_size_px"].isin(PLATEAU_SIZES)]
        plateau = _pooled(plateau_rows) if not plateau_rows.empty else NAN
        s_star = NAN
        if not math.isnan(plateau):
            within = [s for s in sorted(per_size) if per_size[s]["cer"] <= plateau + DELTA]
            s_star = within[0] if within else NAN  # missing when no size reaches the plateau, never the largest size
        for size in sorted(per_size):
            out.append({**per_size[size], "cer_plateau": plateau, "s_star_px": s_star,
                        "status": PREREGISTERED, "family": "description"})
    return out


def coherence_table(df, folders: Sequence[RunFolder], harmonics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Section 9, item 6: the mean and range of R_k over the analysed pages, beside the matching RMS amplitude."""
    text_type_of = {(str(r), str(p)): str(t) for r, p, t in
                    df.drop_duplicates(["reader", "page_id"])[["reader", "page_id", "text_type"]].itertuples(index=False)}
    rms_of = {(h["reader"], h["text_type"], h["block"], int(h["period_px"])): float(h["rms"]) for h in harmonics}
    groups: dict[tuple, list[float]] = {}
    pages_of: dict[tuple, set[str]] = {}
    for folder in folders:
        for r in folder.coherence:
            text_type = text_type_of.get((folder.reader, r["page_id"]))
            if text_type is None:
                continue  # a page without analysed rows (excluded, or its run stopped before its first row)
            key = (folder.reader, text_type, r["block"], r["render_mode"], r["line_pitch_mode"], r["axis"],
                   int(float(r["period_px"])), int(r["harmonic"]))
            groups.setdefault(key, []).append(float(r["R"]))
            pages_of.setdefault(key, set()).add(r["page_id"])
    out = []
    for key in sorted(groups):
        reader, text_type, block, render_mode, pitch_mode, axis, period, harmonic = key
        values = np.asarray(groups[key], dtype=np.float64)
        out.append({
            "reader": reader, "text_type": text_type, "block": block, "render_mode": render_mode,
            "line_pitch_mode": pitch_mode, "axis": axis, "period_px": period, "harmonic": harmonic,
            "R_mean": float(values.mean()), "R_min": float(values.min()), "R_max": float(values.max()),
            "n_pages": len(pages_of[key]), "rms": rms_of.get((reader, text_type, block, period), NAN),
            "status": PREREGISTERED, "family": "description",
        })
    return out


# --------------------------------------------------------------------------- writing


def write_table(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> int:
    """Write ``rows`` as CSV with ``columns`` first and any further key in order of appearance; NaN is an empty cell."""
    names = list(columns)
    for row in rows:
        names.extend(k for k in row if k not in names)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if isinstance(v, float) and math.isnan(v) else v for k, v in row.items()})
    return len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    sweep_path, runs_root = Path(args.config), Path(args.runs_root)
    sweep_cfg = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    sweep_name = sweep_path.stem
    run_dirs = sweep_run_dirs(sweep_path, runs_root)
    if not run_dirs:
        _log.error("no run folder of sweep %s under %s: run the sweep first (scripts/10_run_sweep.py)", sweep_name, runs_root)
        return 1
    folders = [RunFolder(d) for d in run_dirs]
    folders_by_reader: dict[str, RunFolder] = {}
    for folder in folders:
        folders_by_reader.setdefault(folder.reader, folder)

    df_all = load_results(sweep_path, runs_root)  # raises on a repeated key or an edited sweep file
    excluded = excluded_pages(df_all, folders)
    keep = np.ones(len(df_all), dtype=bool)
    for run_dir, pages in excluded.items():
        if pages:
            keep &= ~((df_all["run_dir"] == run_dir) & df_all["page_id"].isin(pages)).to_numpy()
    df = df_all[keep].reset_index(drop=True)
    _log.info("%s: %d rows in %d run folder(s); %d page(s) excluded under section 8; %d rows analysed",
              sweep_name, len(df_all), len(folders), sum(len(p) for p in excluded.values()), len(df))

    out_dir = Path(args.out_dir) / sweep_name
    out_dir.mkdir(parents=True, exist_ok=True)
    harmonics = harmonics_table(df, folders_by_reader)
    tables = {
        "checks": checks_table(df_all, folders, excluded),
        "harmonics": harmonics,
        "paired": paired_table(df),
        "subpixel": subpixel_table(df, folders_by_reader),
        "sizes": sizes_table(df, folders),
        "coherence_summary": coherence_table(df, folders, harmonics),
    }
    counts = {name: write_table(out_dir / f"{name}.csv", tables[name], COLUMNS[name]) for name in TABLES}
    for name in TABLES:
        _log.info("  %-18s %5d row(s)  %s", name + ".csv", counts[name], out_dir / f"{name}.csv")

    planned_readers = [str(r) for r in (sweep_cfg.get("readers") or [])]
    cfg = {
        "sweep_file": str(sweep_path), "sweep_sha1": hashlib.sha1(sweep_path.read_bytes()).hexdigest(),
        "sweep_name": sweep_name, "runs_root": str(runs_root), "run_dirs": [d.name for d in run_dirs],
        "readers_analysed": sorted(folders_by_reader), "readers_planned": planned_readers,
        "readers_without_a_run": [r for r in planned_readers if r not in folders_by_reader],
        "rows_loaded": int(len(df_all)), "rows_analysed": int(len(df)),
        "pages_excluded": {Path(k).name: sorted(v) for k, v in excluded.items() if v},
        "resamples": RESAMPLES, "seed": SEED, "alpha": ALPHA, "delta": DELTA, "plateau_sizes_px": list(PLATEAU_SIZES),
        "tables": counts, "analysis_plan": "PREREGISTRATION.md section 9",
    }
    write_manifest(out_dir, cfg, run_id=new_run_id(sweep_name, "analysis"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
