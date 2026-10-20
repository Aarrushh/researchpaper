# PURPOSE: Fit Study 3 stage 0 and write its tables: each reader's acuity curve per text type on the primary and the reading CER, every row scored by the rules of ANALYSIS_PLAN_3.md section 1, behaviour, text tokens, cost, latency and the checks of the curve's form (ANALYSIS_PLAN_3.md section 2; design note R7, section 3).
# WHAT GOES HERE: Load the stage-0 sweeps' rows, keep block Z's conventional rows, score every row with mrr.behaviour (pages.csv), fit mrr.acuity per reader and text type with passage-bootstrap intervals, and write the tables of section 2 with a manifest.
# FILLED IN AT STEP: Study 3 stage 0 (19 October 2026); the tables of ANALYSIS_PLAN_3.md section 2 added on 20 October 2026, after that plan's commit.
# DO NOT: Pool stage-0 rows with Study 1 or Study 2 in a reported table; --configs with Study 1's ladder and full sweep is a check of the fit only. Change a scoring rule or its parameters here (mrr.behaviour; ANALYSIS_PLAN_3.md section 1.3). Read reports/tables/.
"""Study 3 stage 0: turn the run folders of the stage-0 sweeps into each reader's acuity curve and the tables of
ANALYSIS_PLAN_3.md section 2. Every table but the primary columns of cells.csv and acuity.csv is exploratory.

    python scripts/25_fit_stage0.py [--configs configs/sweeps/s3_stage0_random.yaml configs/sweeps/s3_stage0_prose.yaml
                                               configs/sweeps/s3_stage0_prose_internvl3.yaml]
                                    [--runs-root results/runs] [--out-dir reports/tables] [--name s3_stage0]
                                    [--resamples 2000] [--loso-resamples 500] [--subsets 200] [--seed 20261025]
                                    [--no-tokens]

Reads each sweep's run folders through ``sweep.load_results``, drops the pages ``sweep.complete_rows`` excludes, keeps
the rows of block Z in the conventional layout, scores every row's stored transcription with ``mrr.behaviour``, and
writes into ``<out-dir>/<name>/``:

    pages.csv        one row per results row: section 1's counts (format, omissions by place, runaway insertions, case
                     errors by shape, reading errors), coverage, reading CER, the primary count recomputed beside the
                     stored one, and whether the row's latency is clean
    cells.csv        one row per reader, text type and size: the primary cell as before (pages, reference characters,
                     errors, CER, characters a page and a vision token, token-limit share, output tokens, latency), then
                     its 95% passage-bootstrap interval, the median and 90th-percentile page CER, the reading CER
                     (also case-folded), coverage and the behaviour rate with their intervals, and on the rows R1
                     touched, how far the primary count lies from reading plus behaviour errors
    acuity.csv       one row per reader and text type: the two-parameter fit of CER(s) = 1 / (1 + (s / a) ** beta) to
                     the primary CER as before (a, beta, s05, s01, kappa05, kappa01 with intervals, a / P, a over the
                     token cell, flags, the free-floor check), then the sizes read off the cells without a model and
                     whether they disagree with the fit, coverage at those sizes, the leave-one-size-out share, and the
                     same fit to the reading CER (reading_*)
    behaviour.csv    per reader, text type and size: each behaviour per reference character and the share of pages
                     with it, and the token-limit share
    sensitivity.csv  per variant of section 1.3 (L = 10 and 30, anchors of 20 and 50 characters), reader, text type and
                     size: coverage and reading CER beside the reported rules' (sensitivity_fits.csv: the reading-CER
                     fit's point values per variant)
    form.csv         per measure, reader, text type and size: observed (capped at 1), fitted, the logit residual (empty
                     on a capped cell, where the logit of 1 is infinite), and the leave-one-size-out prediction with the
                     95% interval of observed minus predicted (section 2.2)
    subsample.csv    per measure, reader and text type: how far s01 moves on every other size, 4 of the passages, or
                     both (section 2.3)
    tokens.csv       per reader and text type: characters per text token from the reader's own tokenizer on its
                     stage-0 references, and the token ratio kappa / (characters per text token) at s05 and s01
    cost.csv         per reader: rows, GPU minutes of inference and from first to last row, and the inference minutes
                     with each row whose latency my own work inflated (LAB_NOTEBOOK.md, 20 October) replaced by the
                     mean of the clean rows of its sweep and size
    latency.csv      per reader and text type: median latency a page and an output token, those rows left out

Every resampling draws passages with replacement and keeps each drawn passage's page at every size (mrr.acuity), seed
20261025: 2,000 resamples for the fits and the cells, 500 for the leave-one-size-out intervals (each refits every
other size), 200 subsets for section 2.3.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# One BLAS thread before NumPy loads, as scripts/20_analyze.py, so the tables regenerate byte for byte.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mrr import acuity, behaviour  # noqa: E402
from mrr.manifest import new_run_id, write_manifest  # noqa: E402
from mrr.sweep import RunFolder, complete_rows, load_results, sweep_run_dirs  # noqa: E402
from mrr.text_source import normalise_text  # noqa: E402

_spec = importlib.util.spec_from_file_location("analyze_study1", Path(__file__).with_name("20_analyze.py"))
analyze1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze1)

_log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = ("configs/sweeps/s3_stage0_random.yaml", "configs/sweeps/s3_stage0_prose.yaml",
                   "configs/sweeps/s3_stage0_prose_internvl3.yaml")  # InternVL3's prose above 20 px, pooled with the second
RESAMPLES = 2000
LOSO_RESAMPLES = 500  # each resample refits the other sizes of every size, so fewer than the fit's
SUBSETS = 200  # ANALYSIS_PLAN_3.md section 2.3
SUBSET_PASSAGES = 4  # section 2.3: 4 passages instead of 8 or 16
SEED = 20261025  # new: 20261012 to 20261024 are taken (sweeps, the bootstrap of Study 2, the font pre-test, stage 0)
SENSITIVITY = (("L10", 10, 30), ("L30", 30, 30), ("anchor20", 15, 20), ("anchor50", 15, 50))  # section 1.3
INFLATED = {("s3_stage0_random", "qwen"): frozenset(range(4, 8))}  # passages with inflated latency (notebook, 20 Oct)
STATUS = "exploratory"  # every table but the primary columns of cells.csv and acuity.csv (ANALYSIS_PLAN_3.md)
RATIOS = {"cer": ("primary_errors", "ref_chars"), "reading_cer": ("reading_errors", "read_chars"),
          "omitted_rate": ("omitted", "ref_chars"), "behaviour_rate": ("behaviour_errors", "ref_chars")}
COUNTS = ("primary_errors", "ref_chars", "reading_errors", "read_chars", "reading_casefold_errors", "omitted",
          "omitted_head", "omitted_interior", "omitted_tail", "runaway_inserted", "format_chars", "case_normalised",
          "case_alike", "behaviour_errors")
EVENTS = {"omission": "omissions", "runaway": "runaways", "format": "format_touched",
          "case_normalised": "case_normalised"}

CELL_COLUMNS = ("sweep", "reader", "text_type", "font_size_px", "pages", "reference_chars", "errors", "cer",
                "chars_per_page", "vision_tokens", "chars_per_vision_token", "hit_limit_share", "output_tokens_mean",
                "latency_s_mean",
                "cer_lo", "cer_hi", "page_cer_median", "page_cer_p90", "read_chars", "reading_errors", "reading_cer",
                "reading_cer_lo", "reading_cer_hi", "reading_cer_casefold", "coverage", "coverage_lo", "coverage_hi",
                "behaviour_rate", "behaviour_rate_lo", "behaviour_rate_hi", "format_rate", "format_rows",
                "format_difference", "rows_not_adding_up", "analysis_status")
_FITTED = ("a", "beta", "s05", "kappa05", "s01", "kappa01")
ACUITY_COLUMNS = (("sweep", "reader", "text_type", "status", "patch_size", "token_cell_px", "sizes_px", "passages",
                   "pages_min", "vision_tokens")
                  + tuple(c for name in _FITTED for c in (name, f"{name}_lo", f"{name}_hi"))
                  + ("a_over_patch", "a_over_cell", "s05_in_range", "s01_in_range", "chars_c", "chars_gamma",
                     "deviance", "converged", "resamples", "resamples_fitted", "floor_e0", "floor_a", "floor_beta",
                     "floor_s01", "floor_deviance")
                  + ("readoff_s05", "readoff_s05_how", "readoff_s01", "readoff_s01_how", "s01_disagrees",
                     "coverage_at_s01", "loso_inside", "loso_cells", "reading_status")
                  + tuple(f"reading_{c}" for name in _FITTED for c in (name, f"{name}_lo", f"{name}_hi"))
                  + ("reading_s05_in_range", "reading_s01_in_range", "reading_deviance", "reading_converged",
                     "reading_resamples_fitted", "reading_readoff_s05", "reading_readoff_s05_how",
                     "reading_readoff_s01", "reading_readoff_s01_how", "reading_s01_disagrees",
                     "coverage_at_reading_s05", "coverage_at_reading_s01", "reading_loso_inside", "reading_loso_cells",
                     "analysis_status"))
PAGE_COLUMNS = ("sweep", "run", "reader", "text_type", "font_size_px", "page_id", "passage", "font_file",
                "shape_alike", "reference_chars", "errors", "cer", "results_errors", "format_touched", "format_chars",
                "omitted_head", "omitted_interior", "omitted_tail", "omitted", "omissions", "runaway_inserted",
                "runaways", "case_alike", "case_normalised", "reading_errors", "read_chars", "reading_cer",
                "coverage", "behaviour_errors", "parts_add_up", "hit_limit", "output_tokens", "latency_s",
                "latency_clean", "analysis_status")
BEHAVIOUR_COLUMNS = ("sweep", "reader", "text_type", "font_size_px", "pages", "reference_chars", "omitted_head_rate",
                     "omitted_interior_rate", "omitted_tail_rate", "omitted_rate", "runaway_rate", "format_rate",
                     "case_normalised_rate", "case_alike_rate", "pages_with_omission", "pages_with_runaway",
                     "pages_with_format", "pages_with_case_normalised", "hit_limit_share", "analysis_status")
SENSITIVITY_COLUMNS = ("variant", "min_run", "anchor_chars", "reader", "text_type", "font_size_px", "pages",
                       "omitted_rate", "coverage", "reading_cer", "base_coverage", "base_reading_cer",
                       "analysis_status")
SENSITIVITY_FIT_COLUMNS = ("variant", "min_run", "anchor_chars", "reader", "text_type", "reading_a", "reading_beta",
                           "reading_s05", "reading_s01", "base_reading_s01", "analysis_status")
FORM_COLUMNS = ("measure", "reader", "text_type", "font_size_px", "pages", "observed", "capped", "fitted",
                "residual_logit", "loso_predicted", "loso_difference", "loso_difference_lo", "loso_difference_hi",
                "loso_inside", "analysis_status")
SUBSAMPLE_COLUMNS = ("measure", "reader", "text_type", "design", "sizes_used", "passages_used", "subsets",
                     "s01_full", "fitted", "not_reached", "failed", "s01_median", "rel_dev_median", "rel_dev_p90",
                     "within_10pct", "analysis_status")
TOKEN_COLUMNS = (("reader", "text_type", "tokenizer", "tokenizer_revision", "pages", "chars", "text_tokens",
                  "chars_per_text_token")
                 + tuple(f"{p}{q}{t}{b}" for p in ("", "reading_") for t in ("05", "01")
                         for q in ("kappa", "token_ratio") for b in ("", "_lo", "_hi"))
                 + ("analysis_status",))
COST_COLUMNS = ("reader", "sweeps", "run_folders", "rows", "rows_inflated", "inference_minutes", "wall_minutes",
                "inference_minutes_imputed", "seconds_per_page_imputed", "analysis_status")
LATENCY_COLUMNS = ("reader", "text_type", "rows", "rows_left_out", "latency_s_median",
                   "latency_s_per_output_token_median", "output_tokens_median", "analysis_status")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS), help="the sweep files to fit")
    p.add_argument("--runs-root", default="results/runs")
    p.add_argument("--out-dir", default="reports/tables")
    p.add_argument("--name", default="s3_stage0", help="the folder under --out-dir")
    p.add_argument("--resamples", type=int, default=RESAMPLES)
    p.add_argument("--loso-resamples", type=int, default=LOSO_RESAMPLES)
    p.add_argument("--subsets", type=int, default=SUBSETS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-tokens", action="store_true",
                   help="skip tokens.csv (it loads each reader's tokenizer from the local Hugging Face cache)")
    return p.parse_args(argv)


def load_rows(configs: Sequence[str], runs_root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, RunFolder]]:
    """Block Z's conventional rows of every sweep in ``configs``, a ``sweep`` column naming each; the sweeps' facts for
    the manifest; one run folder per reader (for its patch and token cell)."""
    frames, facts, folders_by_reader = [], {}, {}
    for config in configs:
        sweep_path = Path(config)
        run_dirs = sweep_run_dirs(sweep_path, runs_root)
        folders = [RunFolder(d) for d in run_dirs]
        df_all = load_results(sweep_path, runs_root)  # raises on a repeated key, an edited sweep file or another schema
        df, excluded = complete_rows(df_all, folders)
        keep = (df["block"] == "Z") & (df["render_mode"] == "conventional")
        df = df[keep].copy()
        df["sweep"] = sweep_path.stem
        frames.append(df)
        for f in folders:
            folders_by_reader.setdefault(f.reader, f)
        facts[sweep_path.stem] = {
            "sweep_file": str(sweep_path), "sweep_sha1": hashlib.sha1(sweep_path.read_bytes()).hexdigest(),
            "run_dirs": [d.name for d in run_dirs], "rows_loaded": int(len(df_all)), "rows_fitted": int(len(df)),
            "pages_excluded": {Path(k).name: sorted(v) for k, v in excluded.items() if v},
        }
        _log.info("%s: %d rows in %d run folder(s), %d fitted", sweep_path.stem, len(df_all), len(folders), len(df))
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return rows, facts, folders_by_reader


# --------------------------------------------------------------------------- section 1: every row scored


def _flag(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def score_pages(rows: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, pd.DataFrame],
                                                          list[str]]:
    """Every row scored by ``behaviour.score_row``: the rows with section 1's counts added (one per row, in order), the
    pages.csv records, each SENSITIVITY variant's omitted and reading-error counts, and the normalised references."""
    records: list[dict[str, Any]] = []
    extra: list[dict[str, Any]] = []
    variants: dict[str, list[tuple[int, int]]] = {name: [] for name, _, _ in SENSITIVITY}
    references: list[str] = []
    for r in rows.itertuples(index=False):
        reference, hypothesis = behaviour.read_texts(r.run_dir, r.page_id, r.transcription_file)
        alike = behaviour.shape_alike_letters(str(behaviour.font_path(r.font_file, root)))
        s = behaviour.score_row(reference, hypothesis, alike)
        book, index = acuity.passage_key(r.page_id)
        clean = index not in INFLATED.get((str(r.sweep), str(r.reader)), frozenset())
        stored = int(float(r.substitutions)) + int(float(r.deletions)) + int(float(r.insertions))
        records.append({
            "sweep": str(r.sweep), "run": Path(str(r.run_dir)).name, "reader": str(r.reader),
            "text_type": str(r.text_type), "font_size_px": float(r.font_size_px), "page_id": str(r.page_id),
            "passage": f"{book}:{index}", "font_file": str(r.font_file), "shape_alike": "".join(sorted(alike)),
            "reference_chars": s.reference_chars, "errors": s.errors, "cer": s.errors / s.reference_chars,
            "results_errors": stored, "format_touched": s.format_touched, "format_chars": s.format_chars,
            "omitted_head": s.omitted_head, "omitted_interior": s.omitted_interior, "omitted_tail": s.omitted_tail,
            "omitted": s.omitted, "omissions": s.omissions, "runaway_inserted": s.runaway_inserted,
            "runaways": s.runaways, "case_alike": s.case_alike, "case_normalised": s.case_normalised,
            "reading_errors": s.reading_errors, "read_chars": s.reference_chars - s.omitted,
            "reading_cer": s.reading_cer, "coverage": s.coverage, "behaviour_errors": s.behaviour_errors,
            "parts_add_up": s.parts_add_up, "hit_limit": _flag(r.hit_limit),
            "output_tokens": int(float(r.output_tokens)), "latency_s": float(r.latency_s), "latency_clean": clean,
            "analysis_status": STATUS,
        })
        extra.append({
            "primary_errors": s.errors, "ref_chars": s.reference_chars, "reading_errors": s.reading_errors,
            "read_chars": s.reference_chars - s.omitted, "reading_casefold_errors": s.reading_errors - s.case_alike,
            "omitted": s.omitted, "omitted_head": s.omitted_head, "omitted_interior": s.omitted_interior,
            "omitted_tail": s.omitted_tail, "runaway_inserted": s.runaway_inserted, "format_chars": s.format_chars,
            "case_normalised": s.case_normalised, "case_alike": s.case_alike, "behaviour_errors": s.behaviour_errors,
            "omissions": s.omissions, "runaways": s.runaways, "format_touched": int(s.format_touched),
            "parts_add_up": s.parts_add_up, "latency_clean": clean,
        })
        for name, min_run, anchor in SENSITIVITY:
            v = behaviour.score_row(reference, hypothesis, alike, min_run=min_run, anchor_chars=anchor)
            variants[name].append((v.omitted, v.reading_errors))
        references.append(normalise_text(reference))
    scored = pd.concat([rows.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    variant_frames = {name: pd.DataFrame(v, columns=["omitted", "reading_errors"]) for name, v in variants.items()}
    return scored, records, variant_frames, references


# --------------------------------------------------------------------------- section 2: one reader and text type


def _disagrees(readoff: float, how: str, lo: float, hi: float, in_range: bool) -> bool:
    """Whether the size read off the cells disagrees with the fitted one beyond the fit's interval."""
    if how == "interpolated":
        return not (lo <= readoff <= hi)
    if how == "not_reached":
        return bool(in_range)  # the fit puts the size inside the ladder, and no cell gets there
    return bool(lo > readoff)  # at_or_below_first: the cells are there at the smallest size


def _logit(p: float, n: float) -> float:
    q = min(max(p, 0.5 / n), 1.0 - 0.5 / n)  # half a character from 0 and 1
    return math.log(q / (1.0 - q))


def analyse_group(group: pd.DataFrame, primary_cells: list[dict[str, Any]], fit: dict[str, Any],
                  rfit: dict[str, Any], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    """Section 2's rows for one reader and text type: the cells' extra columns (added to ``primary_cells``), the fit
    row's extra columns (added to ``fit``), and the behaviour, form and subsample rows."""
    reader, text_type = str(group["reader"].iloc[0]), str(group["text_type"].iloc[0])
    passages, sizes, mats, present = acuity.passage_matrices(group, COUNTS)
    bands = acuity.ratio_intervals(mats, RATIOS, resamples=args.resamples,
                                   rng=np.random.default_rng(acuity.group_seed(args.seed, reader, text_type, "cells")))
    total = {c: mats[c].sum(axis=0) for c in COUNTS}
    pages_at = present.sum(axis=0)

    def ratio(num: str, den: str) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(total[den] > 0, total[num] / total[den], np.nan)

    cer, reading = ratio("primary_errors", "ref_chars"), ratio("reading_errors", "read_chars")
    coverage = 1.0 - ratio("omitted", "ref_chars")
    out: dict[str, list[dict[str, Any]]] = {"behaviour": [], "form": [], "subsample": []}
    for k, s in enumerate(sizes):
        g = group[group["font_size_px"].astype(float) == s]
        page_cer = (g["primary_errors"] / g["ref_chars"]).to_numpy(dtype=np.float64)
        untouched, touched = g[g["format_touched"] == 0], g[g["format_touched"] != 0]
        cell = next(c for c in primary_cells if c["font_size_px"] == float(s))
        ref = float(total["ref_chars"][k])
        cell.update({
            "cer_lo": bands["cer"][0][k], "cer_hi": bands["cer"][1][k],
            "page_cer_median": float(np.median(page_cer)), "page_cer_p90": float(np.quantile(page_cer, 0.9)),
            "read_chars": int(total["read_chars"][k]), "reading_errors": int(total["reading_errors"][k]),
            "reading_cer": float(reading[k]), "reading_cer_lo": bands["reading_cer"][0][k],
            "reading_cer_hi": bands["reading_cer"][1][k],
            "reading_cer_casefold": float(total["reading_casefold_errors"][k] / total["read_chars"][k]),
            "coverage": float(coverage[k]), "coverage_lo": 1.0 - bands["omitted_rate"][1][k],
            "coverage_hi": 1.0 - bands["omitted_rate"][0][k],
            "behaviour_rate": float(total["behaviour_errors"][k] / ref),
            "behaviour_rate_lo": bands["behaviour_rate"][0][k], "behaviour_rate_hi": bands["behaviour_rate"][1][k],
            "format_rate": float(total["format_chars"][k] / ref), "format_rows": int(len(touched)),
            "format_difference": int((touched["primary_errors"] - touched["reading_errors"]
                                      - touched["behaviour_errors"]).sum()),  # section 1.2: the two alignments apart
            "rows_not_adding_up": int((~untouched["parts_add_up"].astype(bool)).sum()), "analysis_status": STATUS,
        })
        out["behaviour"].append({
            "sweep": cell["sweep"], "reader": reader, "text_type": text_type, "font_size_px": float(s),
            "pages": int(pages_at[k]), "reference_chars": int(ref),
            "omitted_head_rate": total["omitted_head"][k] / ref,
            "omitted_interior_rate": total["omitted_interior"][k] / ref,
            "omitted_tail_rate": total["omitted_tail"][k] / ref, "omitted_rate": total["omitted"][k] / ref,
            "runaway_rate": total["runaway_inserted"][k] / ref, "format_rate": total["format_chars"][k] / ref,
            "case_normalised_rate": total["case_normalised"][k] / ref, "case_alike_rate": total["case_alike"][k] / ref,
            **{f"pages_with_{e}": float(np.mean(g[col].astype(float) > 0)) for e, col in EVENTS.items()},
            "hit_limit_share": cell["hit_limit_share"], "analysis_status": STATUS,
        })

    for prefix, f, values, num, den in (("", fit, cer, "primary_errors", "ref_chars"),
                                        ("reading_", rfit, reading, "reading_errors", "read_chars")):
        measure = "reading" if prefix else "primary"
        for eps, tag in ((0.05, "05"), (0.01, "01")):
            readoff, how = acuity.readoff_size(sizes, values, eps)
            fit[f"{prefix}readoff_s{tag}"], fit[f"{prefix}readoff_s{tag}_how"] = readoff, how
        fit[f"{prefix}s01_disagrees"] = _disagrees(fit[f"{prefix}readoff_s01"], fit[f"{prefix}readoff_s01_how"],
                                                   f["s01_lo"], f["s01_hi"], f["s01_in_range"])
        loso = acuity.leave_one_size_out(
            sizes, mats[num], mats[den], present, resamples=args.loso_resamples,
            rng=np.random.default_rng(acuity.group_seed(args.seed, reader, text_type, f"loso_{measure}")))
        predicted = [r for r in loso if math.isfinite(r["predicted"])]
        fit[f"{prefix}loso_inside"] = float(np.mean([r["inside"] for r in predicted])) if predicted else math.nan
        fit[f"{prefix}loso_cells"] = len(predicted)
        for k, (s, r) in enumerate(zip(sizes, loso)):
            fitted = float(acuity.cer_curve(s, f["a"], f["beta"]))
            capped = float(values[k]) >= 1.0
            observed = min(float(values[k]), 1.0)
            n = float(total[den][k])
            residual = math.nan if capped else _logit(observed, n) - _logit(fitted, n)
            out["form"].append({
                "measure": measure, "reader": reader, "text_type": text_type, "font_size_px": float(s),
                "pages": int(pages_at[k]), "observed": observed, "capped": capped, "fitted": fitted,
                "residual_logit": residual,
                "loso_predicted": r["predicted"], "loso_difference": r["difference"],
                "loso_difference_lo": r["difference_lo"], "loso_difference_hi": r["difference_hi"],
                "loso_inside": r["inside"], "analysis_status": STATUS,
            })
        every_other = np.zeros(sizes.size, dtype=bool)
        every_other[::2] = True
        starts = [(f["a"], f["beta"], 0.0), (float(np.median(sizes)), 5.0, 0.0)]
        for design, keep, n_passages in (("passages4", np.ones(sizes.size, dtype=bool), SUBSET_PASSAGES),
                                         ("sizes_every_other", every_other, None),
                                         ("both", every_other, SUBSET_PASSAGES)):
            seed = acuity.group_seed(args.seed, reader, text_type, f"subsample_{measure}_{design}")
            rng = np.random.default_rng(seed)
            found = acuity.subsample_fits(sizes, mats[num], mats[den], present, size_keep=keep,
                                          n_passages=n_passages, subsets=args.subsets, rng=rng, starts=starts)
            finite = found[np.isfinite(found)]
            full = float(f["s01"])
            rel = np.abs(finite - full) / full if math.isfinite(full) and finite.size else np.array([])
            out["subsample"].append({
                "measure": measure, "reader": reader, "text_type": text_type, "design": design,
                "sizes_used": int(keep.sum()), "passages_used": len(passages) if n_passages is None else n_passages,
                "subsets": int(found.size), "s01_full": full, "fitted": int(finite.size),
                "not_reached": int(np.isinf(found).sum()), "failed": int(np.isnan(found).sum()),
                "s01_median": float(np.median(finite)) if finite.size else math.nan,
                "rel_dev_median": float(np.median(rel)) if rel.size else math.nan,
                "rel_dev_p90": float(np.quantile(rel, 0.9)) if rel.size else math.nan,
                "within_10pct": float(np.sum(rel <= 0.10) / found.size) if found.size else math.nan,
                "analysis_status": STATUS,
            })
    fit["coverage_at_s01"] = acuity.interp_log(sizes, coverage, float(fit["s01"]))
    fit["coverage_at_reading_s05"] = acuity.interp_log(sizes, coverage, float(rfit["s05"]))
    fit["coverage_at_reading_s01"] = acuity.interp_log(sizes, coverage, float(rfit["s01"]))
    fit["reading_status"] = rfit["status"]
    for name in _FITTED:
        for c in (name, f"{name}_lo", f"{name}_hi"):
            fit[f"reading_{c}"] = rfit[c]
    for c in ("s05_in_range", "s01_in_range", "deviance", "converged", "resamples_fitted"):
        fit[f"reading_{c}"] = rfit[c]
    fit["analysis_status"] = STATUS
    return out


def sensitivity_rows(scored: pd.DataFrame, variants: Mapping[str, pd.DataFrame], fits: Mapping[tuple[str, str], Any],
                     cells: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Section 1.3: coverage and reading CER per cell under each variant of the rules, and the reading-CER fit's point
    values per variant, beside the reported rules'."""
    base = {(c["reader"], c["text_type"], c["font_size_px"]): c for c in cells}
    rows_out, fits_out = [], []
    for name, min_run, anchor in SENSITIVITY:
        v = scored[["reader", "text_type", "font_size_px", "ref_chars"]].copy()
        v["omitted"] = variants[name]["omitted"].to_numpy()
        v["reading_errors"] = variants[name]["reading_errors"].to_numpy()
        v["font_size_px"] = v["font_size_px"].astype(float)
        for (reader, text_type), g in v.groupby(["reader", "text_type"], sort=True):
            if (reader, text_type) not in fits:
                continue
            sizes, ys, ws = [], [], []
            for s, h in g.groupby("font_size_px", sort=True):
                ref, omitted, reading = h["ref_chars"].sum(), h["omitted"].sum(), h["reading_errors"].sum()
                b = base[(reader, text_type, float(s))]
                cer = reading / (ref - omitted) if ref > omitted else math.nan
                rows_out.append({"variant": name, "min_run": min_run, "anchor_chars": anchor, "reader": reader,
                                 "text_type": text_type, "font_size_px": float(s), "pages": int(len(h)),
                                 "omitted_rate": omitted / ref, "coverage": 1.0 - omitted / ref, "reading_cer": cer,
                                 "base_coverage": b["coverage"], "base_reading_cer": b["reading_cer"],
                                 "analysis_status": STATUS})
                if math.isfinite(cer):
                    sizes.append(float(s))
                    ys.append(cer)
                    ws.append(float(len(h)))
            point = acuity.fit_curve(sizes, ys, ws)
            fits_out.append({"variant": name, "min_run": min_run, "anchor_chars": anchor, "reader": reader,
                             "text_type": text_type, "reading_a": point.a, "reading_beta": point.beta,
                             "reading_s05": point.size_at(0.05), "reading_s01": point.size_at(0.01),
                             "base_reading_s01": fits[(reader, text_type)]["reading_s01"], "analysis_status": STATUS})
    return rows_out, fits_out


# --------------------------------------------------------------------------- tokens, cost and latency


def token_rows(scored: pd.DataFrame, references: Sequence[str], folders: Mapping[str, RunFolder],
               fits: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Characters per text token of each reader's own tokenizer on its stage-0 references, and the token ratio
    kappa / (characters per text token) with kappa's interval: above 1 the page image needs fewer tokens than the
    text. The tokenizers come from the local Hugging Face cache at the revisions the run folders froze."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer  # a tokenizer only, no model

    out = []
    for reader in sorted({k[0] for k in fits}):
        cfg = folders[reader].reader_cfg
        tok = AutoTokenizer.from_pretrained(cfg["hf_id"], revision=cfg["revision"], local_files_only=True)
        for text_type in sorted({k[1] for k in fits if k[0] == reader}):
            idx = np.flatnonzero(((scored["reader"] == reader) & (scored["text_type"] == text_type)).to_numpy())
            chars = sum(len(references[i]) for i in idx)
            tokens = sum(len(tok(references[i], add_special_tokens=False)["input_ids"]) for i in idx)
            tau = chars / tokens
            row: dict[str, Any] = {"reader": reader, "text_type": text_type, "tokenizer": cfg["hf_id"],
                                   "tokenizer_revision": cfg["revision"], "pages": int(idx.size), "chars": int(chars),
                                   "text_tokens": int(tokens), "chars_per_text_token": tau}
            fit = fits[(reader, text_type)]
            for prefix in ("", "reading_"):
                for tag in ("05", "01"):
                    for b in ("", "_lo", "_hi"):
                        kappa = float(fit[f"{prefix}kappa{tag}{b}"])
                        row[f"{prefix}kappa{tag}{b}"] = kappa
                        row[f"{prefix}token_ratio{tag}{b}"] = kappa / tau
            row["analysis_status"] = STATUS
            out.append(row)
            _log.info("  %-13s %-9s %.2f characters a text token (%d pages)", reader, text_type, tau, idx.size)
    return out


def cost_rows(scored: pd.DataFrame) -> list[dict[str, Any]]:
    """GPU minutes per reader: the sum of the rows' latencies; each run folder's first to last row (the first row's
    start is its timestamp less its latency; model load apart); and the sum with each inflated row's latency replaced
    by the mean of the clean rows of its sweep and size, the cost of the method without my concurrent load."""
    out = []
    stamps = pd.to_datetime(scored["timestamp_utc"], utc=True)
    latency = scored["latency_s"].astype(float)
    clean = scored["latency_clean"].astype(bool)
    keys = [scored["sweep"], scored["reader"], scored["font_size_px"].astype(float)]
    imputed = latency.where(clean, latency.where(clean).groupby(keys).transform("mean"))
    for reader, g in scored.groupby("reader", sort=True):
        wall = 0.0
        for _, h in g.groupby("run_dir", sort=True):
            t, lat = stamps[h.index], latency[h.index]
            first = t.idxmin()
            wall += (t.max() - (t[first] - pd.Timedelta(seconds=float(lat[first])))).total_seconds() / 60.0
        minutes = float(imputed[g.index].sum() / 60.0)
        out.append({"reader": reader, "sweeps": ";".join(sorted(set(g["sweep"]))),
                    "run_folders": int(g["run_dir"].nunique()), "rows": int(len(g)),
                    "rows_inflated": int((~clean[g.index]).sum()),
                    "inference_minutes": float(latency[g.index].sum() / 60.0), "wall_minutes": wall,
                    "inference_minutes_imputed": minutes, "seconds_per_page_imputed": minutes * 60.0 / len(g),
                    "analysis_status": STATUS})
    return out


def latency_rows(scored: pd.DataFrame) -> list[dict[str, Any]]:
    """Median latency a page and an output token per reader and text type, without the inflated rows."""
    out = []
    for (reader, text_type), g in scored.groupby(["reader", "text_type"], sort=True):
        clean = g[g["latency_clean"].astype(bool)]
        lat, tokens = clean["latency_s"].astype(float), clean["output_tokens"].astype(float)
        out.append({"reader": reader, "text_type": text_type, "rows": int(len(clean)),
                    "rows_left_out": int(len(g) - len(clean)), "latency_s_median": float(lat.median()),
                    "latency_s_per_output_token_median": float((lat / tokens.where(tokens > 0)).median()),
                    "output_tokens_median": float(tokens.median()), "analysis_status": STATUS})
    return out


# --------------------------------------------------------------------------- the command


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    runs_root = Path(args.runs_root)
    rows, facts, folders = load_rows(args.configs, runs_root)
    if rows.empty:
        _log.error("no block Z row in %s under %s: run the sweeps first (scripts/11_run_schedule.py)",
                   ", ".join(args.configs), runs_root)
        return 1
    scored, pages, variants, references = score_pages(rows, ROOT)
    mismatched = [p["page_id"] for p in pages if p["errors"] != p["results_errors"]]
    touched = sorted(f"{p['run']}/{p['page_id']}" for p in pages if p["format_touched"])
    _log.info("scored %d rows: %d with format removed (R1), %d whose recomputed primary count differs from results.csv",
              len(pages), len(touched), len(mismatched))
    for name in touched:
        _log.info("  R1 touched: %s", name)

    cells, fits, by_group = [], [], {}
    extra: dict[str, list[dict[str, Any]]] = {"behaviour": [], "form": [], "subsample": []}
    for (reader, text_type), group in scored.groupby(["reader", "text_type"], sort=True):
        if group["font_size_px"].nunique() < 3:
            _log.info("  %-13s %-9s %d size(s) so far: not fitted", reader, text_type, group["font_size_px"].nunique())
            continue
        group_cells, fit = acuity.fit_group(group, resamples=args.resamples, seed=args.seed)
        _, rfit = acuity.fit_group(group, resamples=args.resamples, seed=args.seed, errors_column="reading_errors",
                                   chars_column="read_chars")
        reader_cfg = folders[reader].reader_cfg
        fit.update({"patch_size": int(reader_cfg["patch_size"]), "token_cell_px": int(reader_cfg["token_cell_px"]),
                    "a_over_patch": fit["a"] / int(reader_cfg["patch_size"]),
                    "a_over_cell": fit["a"] / int(reader_cfg["token_cell_px"])})
        for name, rows_ in analyse_group(group, group_cells, fit, rfit, args).items():
            extra[name].extend(rows_)
        cells.extend(group_cells)
        fits.append(fit)
        by_group[(reader, text_type)] = fit
        _log.info("  %-13s %-9s a %.1f px [%.1f, %.1f]  beta %.1f  s05 %.1f  s01 %.1f px  kappa01 %.2f  %s"
                  "  (free floor %.4f)", reader, text_type, fit["a"], fit["a_lo"], fit["a_hi"], fit["beta"],
                  fit["s05"], fit["s01"], fit["kappa01"], fit["status"], fit["floor_e0"])
        _log.info("  %-23s reading: a %.1f px  s01 %.1f px [%.1f, %.1f]  read off %.1f px (%s)  coverage there %.3f",
                  "", fit["reading_a"], fit["reading_s01"], fit["reading_s01_lo"], fit["reading_s01_hi"],
                  fit["reading_readoff_s01"], fit["reading_readoff_s01_how"], fit["coverage_at_reading_s01"])
    sensitivity, sensitivity_fits = sensitivity_rows(scored, variants, by_group, cells)
    tokens = [] if args.no_tokens else token_rows(scored, references, folders, by_group)

    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_sorted = sorted(pages, key=lambda p: (p["sweep"], p["reader"], p["text_type"], p["font_size_px"],
                                                p["page_id"]))
    tables = {"pages": (pages_sorted, PAGE_COLUMNS), "cells": (cells, CELL_COLUMNS), "acuity": (fits, ACUITY_COLUMNS),
              "behaviour": (extra["behaviour"], BEHAVIOUR_COLUMNS),
              "sensitivity": (sensitivity, SENSITIVITY_COLUMNS),
              "sensitivity_fits": (sensitivity_fits, SENSITIVITY_FIT_COLUMNS),
              "form": (extra["form"], FORM_COLUMNS), "subsample": (extra["subsample"], SUBSAMPLE_COLUMNS),
              "cost": (cost_rows(scored), COST_COLUMNS), "latency": (latency_rows(scored), LATENCY_COLUMNS)}
    if not args.no_tokens:
        tables["tokens"] = (tokens, TOKEN_COLUMNS)
    counts = {name: analyze1.write_table(out_dir / f"{name}.csv", rows_, columns)
              for name, (rows_, columns) in tables.items()}
    for name, n in counts.items():
        _log.info("  %-21s %4d row(s)  %s", name + ".csv", n, out_dir / f"{name}.csv")
    write_manifest(out_dir, {
        "sweeps": facts, "runs_root": str(runs_root), "resamples": args.resamples,
        "loso_resamples": args.loso_resamples, "subsets": args.subsets, "subset_passages": SUBSET_PASSAGES,
        "seed": args.seed, "alpha": acuity.ALPHA, "epsilons": list(acuity.EPSILONS), "tables": counts,
        "curve": "CER(s) = 1 / (1 + (s / a) ** beta), binomial deviance of capped cell CERs, pages as weights; "
                 "check fit with a free floor e0",
        "rules": {"min_run": behaviour.MIN_RUN, "anchor_chars": behaviour.ANCHOR_CHARS,
                  "anchor_errors": behaviour.ANCHOR_ERRORS, "shape_size_px": behaviour.SHAPE_SIZE_PX,
                  "shape_min_iou": behaviour.SHAPE_MIN_IOU,
                  "shape_rasteriser": "mrr.render.load_glyph (unhinted FreeType, as the pages)",
                  "shape_alike": {p["font_file"]: p["shape_alike"] for p in pages_sorted},
                  "preamble": behaviour.PREAMBLE.pattern, "fence": behaviour.FENCE.pattern,
                  "sensitivity": [{"variant": n, "min_run": m, "anchor_chars": a} for n, m, a in SENSITIVITY]},
        "rows_scored": len(pages), "format_touched": touched, "primary_count_mismatches": mismatched,
        "latency_inflated": {f"{s}/{r}": sorted(v) for (s, r), v in INFLATED.items()},
        "tokens": not args.no_tokens, "status": STATUS,
        "analysis_plan": "ANALYSIS_PLAN_3.md section 2; design note R7, section 3 (stage 0)", "blas_threads": 1,
    }, run_id=new_run_id(args.name, "analysis"))
    return 0 if not mismatched else 1


if __name__ == "__main__":
    sys.exit(main())
