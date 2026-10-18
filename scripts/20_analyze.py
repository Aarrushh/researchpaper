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
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# One BLAS thread, set before NumPy loads: the freeze check (scripts/40_freeze.py --check) regenerates every table and
# compares it byte for byte, and least-squares results can differ in the last bit between thread counts.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"  # forced, not defaulted: the analysis and its check must run the same arithmetic

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from mrr import grid, render, stats  # noqa: E402
from mrr import sweep as sweep_mod  # noqa: E402
from mrr.manifest import new_run_id, write_manifest  # noqa: E402
from mrr.score import CharCounts, pooled_cer  # noqa: E402
from mrr.sweep import RunFolder, complete_rows, load_results, reader_periods, sweep_run_dirs  # noqa: E402

RESAMPLES = 2000  # PREREGISTRATION.md section 9: every resampling is by page, with 2,000 resamples
SEED = 20261014  # and this seed
ALPHA = 0.05  # section 6
DELTA = 0.01  # section 9, item 5: s* = min{s : CER(s) <= CER_inf + delta}, in absolute CER
PLATEAU_SIZES = (20.0, 24.0)  # section 9, item 5: CER_inf is the pooled CER of the rows at 20 and 24 px
PREREGISTERED, EXPLORATORY = "preregistered", "exploratory"
HARMONIC_BLOCKS = ("V", "H", "C")  # section 9, item 2
TABLES = ("checks", "harmonics", "paired", "subpixel", "sizes", "coherence_summary", "wer", "pages", "density")
COLUMNS = {
    "checks": [
        "run_dir", "reader", "rows_done", "rows_planned", "repeated_keys", "missing_transcriptions",
        "empty_transcriptions", "hit_limit_count", "hit_limit_share", "median_output_tokens", "determinism_pairs",
        "determinism_expected", "determinism_mismatches", "coherence_malformed", "pages_seen", "pages_excluded",
        "excluded_page_ids", "median_latency_s", "status", "family",
    ],
    "harmonics": [
        "reader", "text_type", "block", "axis", "range_px", "period_px", "bins", "q", "F", "df1", "df2", "p", "p_holm",
        "rms", "rms_low", "rms_high", "share", "sigma", "n_pages", "n_levels", "n_rows", "mde_rms", "status", "family",
        "note",
    ],
    "paired": [
        "reader", "text_type", "comparison", "n", "n_zero", "statistic", "p", "method", "p_holm", "median_diff",
        "diff_low", "diff_high", "sd_diff", "mde_median_diff", "status", "family",
    ],
    "subpixel": ["reader", "text_type", "F", "df1", "df2", "p", "p_holm", "n_pages", "n_levels", "n_rows", "status", "family"],
    "sizes": [
        "reader", "text_type", "render_mode", "font_size_px", "cer", "cer_low", "cer_high", "n_pages", "n_rows",
        "vision_tokens", "chars_per_token", "symbols_per_token", "cer_plateau", "s_star_px", "status", "family",
    ],
    "coherence_summary": [
        "reader", "text_type", "block", "render_mode", "line_pitch_mode", "axis", "period_px", "harmonic", "R_mean",
        "R_min", "R_max", "n_pages", "rms_bin", "rms", "status", "family",
    ],
    "wer": [  # section 4: WER is secondary and descriptive, the mean of row WERs over a cell
        "reader", "text_type", "block", "render_mode", "line_pitch_mode", "aa_mode", "font_size_px", "phase_x_px",
        "phase_y_px", "wer_mean", "n_pages", "n_rows", "status", "family",
    ],
    "pages": [  # sections 4 and 5: every page's characters and symbols per vision token, and its place in its source
        "reader", "run_dir", "page_id", "pagination", "text_type", "book_id", "start_word", "n_words",
        "reference_chars", "reference_symbols", "vision_tokens", "chars_per_token", "symbols_per_token", "rows_done",
        "rows_planned", "complete", "reference_matches_plan", "status", "family",
    ],
    "density": [  # section 3: the density cost of each shared layout, beside R_k
        "reader", "text_type", "render_mode", "line_pitch_mode", "n_pages", "glyphs_per_page", "lines_per_page",
        "glyphs_per_line", "glyphs_per_full_line", "status", "family",
    ],
}
NAN = float("nan")

_log = logging.getLogger("analyze")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="configs/sweeps/<name>.yaml; the tables go to <out-dir>/<name>/")
    p.add_argument("--runs-root", default="results/runs", help="where the run folders are (default results/runs)")
    p.add_argument("--out-dir", default="reports/tables", help="the parent of the sweep's table folder")
    p.add_argument("--readers-config", default="configs/readers.yaml",
                   help="every reader config by name, to replan the pages (pages.csv, density.csv)")
    p.add_argument("--render-config", default="configs/render_defaults.yaml", help="the render defaults the run used")
    p.add_argument("--checks-only", action="store_true",
                   help="write checks.csv alone (and its manifest): the pipeline check allowed while a run is going "
                        "(PREREGISTRATION.md section 7), which computes no test of H1")
    return p.parse_args(argv)


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
    """Holm within each ``_holm`` group (section 6), then drop the group key.

    An undefined test stays in its family as p = 1, so the family keeps the size m section 6 sets (and _fill_mde
    assumes); its own p and p_holm stay NaN.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.pop("_holm"), []).append(row)
    for members in groups.values():
        if all(math.isnan(r["p"]) for r in members):
            continue
        adjusted = stats.holm([1.0 if math.isnan(r["p"]) else r["p"] for r in members])
        for r, a in zip(members, adjusted):
            if not math.isnan(r["p"]):
                r["p_holm"] = float(a)


# --------------------------------------------------------------------------- the tables


def _coherence_row_ok(r: Mapping[str, Any]) -> bool:
    """A coherence.csv row with its eight fields, a numeric R and period: a torn line merged with the next is not."""
    if None in r or any(v is None for v in r.values()):
        return False
    try:
        float(r["R"]), float(r["period_px"]), int(r["harmonic"])
    except (TypeError, ValueError, KeyError):
        return False
    return True


def _torn_transcription(path: Path, scored_chars: int) -> bool:
    """A transcription file that exists but is empty or NUL-filled although the scored text had ``scored_chars`` > 0
    characters (hits + substitutions + insertions): what an OS crash between a row's fsync and its file leaves."""
    return scored_chars > 0 and path.exists() and (path.stat().st_size == 0 or b"\x00" in path.read_bytes())


def _has_twins(folder: RunFolder) -> bool:
    """Whether the run's shared conditions hold a pair that renders the same pixels ((H, 0) and (S, 0))."""
    keys = [sweep_mod._twin_key(c) for c in folder.conditions if c.paginate == sweep_mod.SHARED]
    return len(keys) != len(set(keys))


def checks_table(df_all, folders: Sequence[RunFolder], excluded: Mapping[str, set[str]]) -> list[dict[str, Any]]:
    """Section 9, item 1: one row per run folder.

    Beyond the list of section 9: transcription files that exist but are empty or NUL-filled while the scored text was
    not (an OS crash between a row's fsync and its file; a reply of the end token alone is a real empty text and does
    not count), the median output tokens (section 4), the number of complete
    shared pages that should each hold one determinism row, and coherence rows that do not parse (a torn line).
    """
    out = []
    for folder in folders:
        key = str(folder.run_dir)
        rows = df_all[df_all["run_dir"] == key]
        n = int(len(rows))
        files = [folder.run_dir / str(f) for f in rows["transcription_file"].tolist()]
        missing = sum(1 for f in files if not f.exists())
        scored = (rows["hits"] + rows["substitutions"] + rows["insertions"]).astype(int).tolist()  # > 0: text not empty
        empty = sum(1 for f, h in zip(files, scored) if _torn_transcription(f, h))
        hit = int(rows["hit_limit"].astype(bool).sum()) if n else 0
        mismatches = sum(1 for r in folder.determinism if str(r.get("identical")) != "True")
        expected = 0
        if n and _has_twins(folder):
            for page_id, group in rows.groupby("page_id", sort=True):
                if folder.page_key(str(page_id)) != sweep_mod.SHARED:
                    continue
                done = set(zip(group["condition_hash"].tolist(), (int(r) for r in group["repeat"].tolist())))
                expected += int(not (folder.planned_rows(str(page_id), str(group["text_type"].iloc[0])) - done))
        out.append({
            "run_dir": folder.run_dir.name, "reader": folder.reader, "rows_done": n, "rows_planned": folder.rows_planned,
            "repeated_keys": 0,  # load_results raises on a repeated key, so a table exists only when there is none
            "missing_transcriptions": int(missing), "empty_transcriptions": int(empty), "hit_limit_count": hit,
            "hit_limit_share": hit / n if n else NAN,
            "median_output_tokens": float(rows["output_tokens"].median()) if n else NAN,
            "determinism_pairs": len(folder.determinism), "determinism_expected": expected,
            "determinism_mismatches": int(mismatches),
            "coherence_malformed": sum(1 for r in folder.coherence if not _coherence_row_ok(r)),
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
                why = ""
                try:
                    fit = stats.harmonic_fit(y, pages, phase, L, groups, levels)
                except ValueError as exc:
                    # No within-page variation (a reader at its floor reads every page alike at every phase), or a
                    # single complete page (no residual degree of freedom): every group's test is undefined. The rows
                    # are written with empty statistics and the cause, so the table says so.
                    _log.warning("%s %s block %s: harmonic model undefined (%s)", reader, text_type, block, exc)
                    fit, why = None, getattr(exc, "cause", str(exc))
                for T in sorted(groups):
                    if fit is not None:
                        g = fit["groups"][int(T)]
                        low, high = stats.page_bootstrap_rms(y, pages, phase, L, groups, levels, T, RESAMPLES, SEED,
                                                             ALPHA)
                    else:
                        g = {"bins": sorted(int(k) for k in groups[T]), "q": 0, "F": NAN, "df1": 0, "df2": 0, "p": NAN,
                             "rms": NAN, "share": NAN}
                        low = high = NAN
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
                        "rms_high": high, "share": g["share"], "sigma": fit["sigma"] if fit else NAN,
                        "n_pages": fit["n_pages"] if fit else int(len(np.unique(pages))),
                        "n_levels": len(levels), "n_rows": fit["n_rows"] if fit else int(y.size), "mde_rms": NAN,
                        "status": status, "family": family, "_holm": f"{family}|{reader}|{text_type}",
                        "note": "" if fit else f"undefined: {why}",
                    })
    _fill_mde(out)
    _fill_holm(out)
    return out


def _fill_mde(rows: list[dict[str, Any]]) -> None:
    """PREREGISTRATION.md section 5 (D26): the smallest RMS amplitude each F-test detects with 80% power at the pages
    obtained, s_g = sigma sqrt(lambda_q / (N n)), N n being the rows of the fit and lambda_q the noncentrality at
    alpha / m, m the size of the row's Holm family (3 in a primary family)."""
    sizes: dict[str, int] = {}
    for row in rows:
        sizes[row["_holm"]] = sizes.get(row["_holm"], 0) + 1
    for row in rows:
        q, dfd, sigma, n_rows = int(row["q"]), int(row["df2"]), float(row["sigma"]), int(row["n_rows"])
        if q >= 1 and dfd >= 1 and sigma > 0 and n_rows > 0:
            try:
                lam = stats.power_noncentrality(q, dfd, ALPHA / sizes[row["_holm"]])
            except ValueError:  # so few residual degrees of freedom that 80% power needs lambda > 400: left undefined
                _log.warning("%s %s %s %s px: no minimum detectable amplitude at df2 = %d", row["reader"],
                             row["text_type"], row["block"], row["period_px"], dfd)
                continue
            row["mde_rms"] = sigma * math.sqrt(lam / n_rows)


def _paired_row(reader: str, text_type: str, comparison: str, a: np.ndarray, b: np.ndarray, family: str,
                holm_group: str) -> dict[str, Any]:
    """Section 9, item 3: the paired Wilcoxon test, the median of b - a and its bootstrap interval."""
    row: dict[str, Any] = {
        "reader": reader, "text_type": text_type, "comparison": comparison, "n": int(a.size), "n_zero": NAN,
        "statistic": NAN, "p": NAN, "method": "", "p_holm": NAN, "median_diff": NAN, "diff_low": NAN, "diff_high": NAN,
        "sd_diff": NAN, "mde_median_diff": NAN, "status": PREREGISTERED, "family": family, "_holm": holm_group,
    }
    try:
        w = stats.paired_wilcoxon(a, b)
        row.update(n_zero=int(w["n_zero"]), statistic=float(w["statistic"]), p=float(w["pvalue"]), method=w["method"])
    except ValueError as exc:  # every difference zero, or too few pages: the test is undefined and the table says so
        row["method"] = f"undefined: {exc}"
    row["median_diff"] = stats.median_paired_difference(a, b)
    row["diff_low"], row["diff_high"] = stats.bootstrap_ci(b - a, np.median, RESAMPLES, ALPHA, SEED)
    if a.size >= 2:  # PREREGISTRATION.md section 5 (D26): the smallest detectable median difference at these pages
        row["sd_diff"] = float(np.std(b - a, ddof=1))
        if row["sd_diff"] > 0:
            row["mde_median_diff"] = stats.paired_mde(row["sd_diff"], int(a.size), ALPHA)
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
        try:
            test = stats.group_f_test(y, pages, X, names, names)
        except ValueError as exc:  # no within-page variation: the test is undefined, written with empty statistics
            _log.warning("%s %s block S: F-test undefined (%s)", reader, text_type, exc)
            test = {"F": NAN, "df1": 0, "df2": 0, "p": NAN}
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
    # R_k(T) is the coherence at harmonic k of period T, i.e. at sweep bin k * L / T of the block's model, and that
    # bin belongs to whichever period group stats.bin_groups gave it (R_2(28) is the 14 px patch bin), so the RMS beside
    # it is the RMS of the group holding that bin; a bin the levels cannot identify (above n / 2) gets none.
    range_of: dict[tuple, int] = {}
    bin_rms: dict[tuple, float] = {}
    for h in harmonics:
        block_key = (h["reader"], h["text_type"], h["block"])
        range_of[block_key] = int(h["range_px"])
        for b in str(h["bins"]).split():
            bin_rms[block_key + (int(b),)] = float(h["rms"])
    groups: dict[tuple, list[float]] = {}
    pages_of: dict[tuple, set[str]] = {}
    for folder in folders:
        for r in folder.coherence:
            if not _coherence_row_ok(r):
                continue  # a torn line; counted in checks.csv (coherence_malformed)
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
        L = range_of.get((reader, text_type, block))
        k_bin = harmonic * L // period if L is not None and (harmonic * L) % period == 0 else None
        rms = bin_rms.get((reader, text_type, block, k_bin), NAN) if k_bin is not None else NAN
        out.append({
            "reader": reader, "text_type": text_type, "block": block, "render_mode": render_mode,
            "line_pitch_mode": pitch_mode, "axis": axis, "period_px": period, "harmonic": harmonic,
            "R_mean": float(values.mean()), "R_min": float(values.min()), "R_max": float(values.max()),
            "n_pages": len(pages_of[key]), "rms_bin": k_bin if not math.isnan(rms) else "", "rms": rms,
            "status": PREREGISTERED, "family": "description",
        })
    return out


def wer_table(df) -> list[dict[str, Any]]:
    """Section 4: WER, secondary and descriptive, as the mean of row WERs over each cell (no test uses it)."""
    keys = COLUMNS["wer"][:9]
    out = []
    for values, group in df.groupby(keys, sort=True, dropna=False):
        out.append({**dict(zip(keys, values)), "wer_mean": float(group["wer"].mean()),
                    "n_pages": int(group["page_id"].nunique()), "n_rows": int(len(group)),
                    "status": PREREGISTERED, "family": "description"})
    return out


def _replan(folder: RunFolder, readers_by_name: Mapping[str, Mapping[str, Any]], render_cfg: Mapping[str, Any],
            root: Path) -> dict[str, list]:
    """The run's pages, replanned from its manifest with the runner's own plan_pages (section 5: start_word)."""
    readers = {**readers_by_name, folder.reader: folder.reader_cfg}
    return sweep_mod.plan_pages(folder.sweep, folder.reader_cfg, render_cfg, root=root, readers=readers)


def pages_table(df_all, folders: Sequence[RunFolder], plans: Mapping[str, dict[str, list]]) -> list[dict[str, Any]]:
    """Sections 4 and 5: one row per page with at least one row on disk, in each run folder.

    The page's source and start word come from replanning the run (``plans``), and the replanned text is compared
    with the references/<page_id>.txt the runner wrote: ``reference_matches_plan`` false would mean the pages were
    not the ones this code plans. Characters and symbols per vision token are the reference's, as section 4 lists.
    """
    out = []
    for folder in folders:
        plan = plans.get(str(folder.run_dir))
        if plan is None:
            continue
        rows = df_all[df_all["run_dir"] == str(folder.run_dir)]
        rows_of = {str(p): g for p, g in rows.groupby("page_id", sort=True)}
        for key, pages in plan.items():
            for page in pages:
                group = rows_of.get(page.page_id)
                if group is None:
                    continue  # never started: the clock stopped the run before it (D26)
                done = set(zip(group["condition_hash"].tolist(), (int(r) for r in group["repeat"].tolist())))
                planned = folder.planned_rows(page.page_id, page.text_type)
                ref_path = folder.run_dir / "references" / f"{page.page_id}.txt"
                reference = ref_path.read_text(encoding="utf-8") if ref_path.exists() else None
                tokens = int(group["vision_tokens"].iloc[0])
                chars = int(group["reference_chars"].iloc[0])
                symbols = folder.reference_symbols(page.page_id)
                out.append({
                    "reader": folder.reader, "run_dir": folder.run_dir.name, "page_id": page.page_id,
                    "pagination": key, "text_type": page.text_type, "book_id": page.book_id,
                    "start_word": int(page.start_word), "n_words": len(page.text.split()), "reference_chars": chars,
                    "reference_symbols": symbols, "vision_tokens": tokens, "chars_per_token": chars / tokens,
                    "symbols_per_token": symbols / tokens, "rows_done": len(done), "rows_planned": len(planned),
                    "complete": not (planned - done), "reference_matches_plan": reference == page.text,
                    "status": PREREGISTERED, "family": "description",
                })
    return out


def density_table(df, folders: Sequence[RunFolder], plans: Mapping[str, dict[str, list]],
                  render_cfg: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    """Section 3: the density cost of each shared layout, glyphs per line and per page, on the analysed pages.

    Each layout is the runner's own at phase 0 under the run's phase allowances (sweep._layout_cfg), laid out by
    render.layout_page; spaces are not placed glyphs, so they do not count.
    """
    totals: dict[tuple, list[int]] = {}
    for folder in folders:
        plan = plans.get(str(folder.run_dir))
        if plan is None or sweep_mod.SHARED not in plan:
            continue
        shared = [c for c in folder.conditions if c.paginate == sweep_mod.SHARED]
        allowances = sweep_mod.shared_allowances(folder.sweep, folder.reader, shared)
        font_paths = sweep_mod._font_paths(folder.sweep, root)
        layouts: dict[tuple, tuple[str, str, dict]] = {}
        for c in shared:
            lid = (c.render_mode, c.line_pitch_mode, sweep_mod._layout_id(c))
            if lid not in layouts:
                layouts[lid] = (c.render_mode, c.line_pitch_mode,
                                sweep_mod._layout_cfg(c, folder.reader_cfg, render_cfg, font_paths, allowances))
        analysed = set(df.loc[df["run_dir"] == str(folder.run_dir), "page_id"].astype(str))
        for page in plan[sweep_mod.SHARED]:
            if page.page_id not in analysed:
                continue
            for mode, pitch_mode, cfg in layouts.values():
                placed = render.layout_page(page.text, cfg)
                lines = (max(g.line for g in placed) + 1) if placed else 0
                t = totals.setdefault((folder.reader, page.text_type, mode, pitch_mode), [0, 0, 0, 0, 0])
                t[0] += 1
                t[1] += len(placed)
                t[2] += lines
                # Full lines only: a shared page is cut to fit the tightest layout, so every looser layout ends
                # partway through its last line; the density cost of a layout (as W1.10 measures it) leaves that out.
                t[3] += sum(1 for g in placed if g.line < lines - 1)
                t[4] += max(lines - 1, 0)
    out = []
    for (reader, text_type, mode, pitch_mode), (n, glyphs, lines, full_glyphs, full_lines) in sorted(totals.items()):
        out.append({
            "reader": reader, "text_type": text_type, "render_mode": mode, "line_pitch_mode": pitch_mode,
            "n_pages": n, "glyphs_per_page": glyphs / n, "lines_per_page": lines / n,
            "glyphs_per_line": glyphs / lines if lines else NAN,
            "glyphs_per_full_line": full_glyphs / full_lines if full_lines else NAN,
            "status": PREREGISTERED, "family": "description",
        })
    return out


# --------------------------------------------------------------------------- writing


FLOAT_DIGITS = 12  # significant digits written: the last bits of a least-squares fit depend on the BLAS kernel


def _cell(v: Any) -> Any:
    """NaN as an empty cell; any other float rounded to FLOAT_DIGITS significant digits, so the tables regenerate
    alike on machines whose BLAS picks another kernel (they then differ only at a rare rounding boundary)."""
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return "" if math.isnan(v) else float(f"{v:.{FLOAT_DIGITS}g}")
    return v


def write_table(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> int:
    """Write ``rows`` as CSV with ``columns`` first and any further key in order of appearance; NaN is an empty cell
    and floats carry FLOAT_DIGITS significant digits."""
    names = list(columns)
    for row in rows:
        names.extend(k for k in row if k not in names)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(v) for k, v in row.items()})
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
    df, excluded = complete_rows(df_all, folders)  # section 8: a page missing a planned row leaves every table but checks
    _log.info("%s: %d rows in %d run folder(s); %d page(s) excluded under section 8; %d rows analysed",
              sweep_name, len(df_all), len(folders), sum(len(p) for p in excluded.values()), len(df))

    out_dir = Path(args.out_dir) / sweep_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.checks_only:  # section 7: while a run is going, its pages are looked at for the pipeline only
        names = ("checks",)
        tables = {"checks": checks_table(df_all, folders, excluded)}
        replan_errors: dict[str, str] = {}
    else:
        root = Path(".")
        readers_cfg = yaml.safe_load(Path(args.readers_config).read_text(encoding="utf-8"))
        readers_by_name = {str(r["name"]): r for r in readers_cfg["readers"]}
        render_cfg = yaml.safe_load(Path(args.render_config).read_text(encoding="utf-8"))
        plans: dict[str, dict[str, list]] = {}
        replan_errors = {}
        for folder in folders:
            try:
                plans[str(folder.run_dir)] = _replan(folder, readers_by_name, render_cfg, root)
            except (ValueError, OSError, KeyError) as exc:  # descriptive tables only; the tests do not need it
                _log.error("%s: could not replan its pages (%s); pages.csv and density.csv leave it out",
                           folder.run_dir.name, exc)
                replan_errors[folder.run_dir.name] = str(exc)
        harmonics = harmonics_table(df, folders_by_reader)
        names = TABLES
        tables = {
            "checks": checks_table(df_all, folders, excluded),
            "harmonics": harmonics,
            "paired": paired_table(df),
            "subpixel": subpixel_table(df, folders_by_reader),
            "sizes": sizes_table(df, folders),
            "coherence_summary": coherence_table(df, folders, harmonics),
            "wer": wer_table(df),
            "pages": pages_table(df_all, folders, plans),
            "density": density_table(df, folders, plans, render_cfg, root),
        }
        mismatched = [r["page_id"] for r in tables["pages"] if not r["reference_matches_plan"]]
        if mismatched:
            _log.error("%d page(s) whose reference differs from the replanned text: %s", len(mismatched),
                       ", ".join(mismatched[:10]))
    counts = {name: write_table(out_dir / f"{name}.csv", tables[name], COLUMNS[name]) for name in names}
    for name in names:
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
        "tables": counts, "analysis_plan": "PREREGISTRATION.md section 9", "checks_only": bool(args.checks_only),
        "blas_threads": 1, "replan_errors": replan_errors,
    }
    write_manifest(out_dir, cfg, run_id=new_run_id(sweep_name, "analysis"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
