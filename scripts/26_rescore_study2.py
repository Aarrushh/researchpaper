# PURPOSE: Study 2's exploratory re-scoring (ANALYSIS_PLAN_3.md section 3.2): every row's stored transcription scored by the rules of section 1, the measures per reader and condition, families 2 to 4 of PREREGISTRATION_2.md section 6 on reading CER, and the tables that answer Q1 to Q3.
# WHAT GOES HERE: Load the study2 sweep's rows as scripts/22_analyze_study2.py does, score each row with mrr.behaviour.score_row on its normalised strings and on its letters-only strings, pool counts by page and condition (D13), run the paired contrasts with Study 2's own method, and write the tables into <out-dir>/study2_rescored/ with a manifest.
# FILLED IN AT STEP: ANALYSIS_PLAN_3.md section 3.2 (20 October 2026), after data-freeze-v2 and data-freeze-s3-stage0.
# DO NOT: Change a scoring rule or its parameters (mrr.behaviour; ANALYSIS_PLAN_3.md section 1.3). Write into reports/tables/study2/, whose confirmatory tables stand as PREREGISTRATION_2.md produced them. Label any table here anything but exploratory.
"""Study 2 re-scored by the rules of ANALYSIS_PLAN_3.md section 1. Every table is exploratory.

    python scripts/26_rescore_study2.py [--config configs/sweeps/study2.yaml] [--runs-root results/runs]
                                        [--out-dir reports/tables] [--name study2_rescored]

Reads the sweep's run folders through ``sweep.load_results``, drops the pages section 8 excludes
(``sweep.complete_rows``), codes each row's condition and group (``study2.with_codes``) and scores its stored
transcription with ``behaviour.score_row``, unchanged, twice: on the normalised strings, and on the letters-only
strings of Study 2's letters-only CER (``score.char_counts_letters_only``: every space removed, after R1 has removed
the format lines, so L and the anchor count letters there). The letters-only alignment also gives the O/0 and I/l/1
substitutions (``study2.letter_confusions``, ``study2.confusion_totals``). Writes into ``<out-dir>/<name>/``:

    pages.csv          one row per results row: section 1's counts on both strings, the O/0 and I/l/1 substitutions,
                       and the primary counts recomputed beside the stored ones
    conditions.csv     per reader, text type and condition: the primary CER, reading CER, coverage and behaviour rate
                       with page-bootstrap intervals; the case-folded reading CER; each behaviour per reference
                       character and the share of pages with it; the token-limit share
    paired.csv         families 2 to 4 of PREREGISTRATION_2.md section 6 on reading CER, with Study 2's method and Holm
                       across each family's readers: containment (cut against contained) on letters-only reading CER,
                       advantage (conventional against contained) and prose (H6's bound) on reading CER. Secondary to
                       reports/tables/study2/paired.csv, never in its place
    q1_prose.csv       Q1, per reader on prose, contained against conventional: the primary CER, coverage, reading CER
                       and the behaviour rates, paired by page
    q2_letterform.csv  Q2, per reader on random text, contained against cut: letters-only CER as stored, letters-only
                       reading CER, and the same without the O/0, I/l/1 and case classes, paired by page
    q2_classes.csv     per reader, text type and group: the letters-only errors split into omitted, runaway, case
                       (shape-alike and case-normalised), O/0, I/l/1 and the rest
    q3_case.csv        Q3, per reader, text type and condition, and per group: shape-alike and case-normalised errors
                       per reference character and as a share of the primary errors

A page's value for a group pools the counts of the group's rows on that page, and a cell's value pools its pages'
counts (D13). Every contrast is ``<treatment>_vs_<baseline>`` and its median difference is treatment minus baseline,
so a negative value favours the contained monospace (for coverage, a negative value means it covers less). Every
resampling is by page, 2,000 resamples, seed 20261021, as Study 2's.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

# One BLAS thread before NumPy loads, as scripts/22_analyze_study2.py, so the tables regenerate byte for byte.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from mrr import behaviour, study2  # noqa: E402
from mrr.manifest import new_run_id, write_manifest  # noqa: E402
from mrr.study2 import CONTAINED, CONVENTIONAL, CUT  # noqa: E402
from mrr.sweep import RunFolder, complete_rows, load_results, sweep_run_dirs  # noqa: E402
from mrr.text_source import normalise_text  # noqa: E402


def _script(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze1 = _script("analyze_study1", "20_analyze.py")
analyze2 = _script("analyze_study2", "22_analyze_study2.py")  # Study 2's paired method: paired_row, Holm, verdicts

_log = logging.getLogger("rescore_study2")

ROOT = Path(__file__).resolve().parents[1]
NAME = "study2_rescored"
RESAMPLES, SEED, ALPHA, DELTA = analyze2.RESAMPLES, analyze2.SEED, analyze2.ALPHA, analyze2.DELTA
RANDOM, NATURAL = analyze2.RANDOM, analyze2.NATURAL
STATUS = "exploratory"  # ANALYSIS_PLAN_3.md section 3.2: every table here

# A measure pooled from the scored rows: (numerator column, denominator column). "cer" and "letters" are Study 2's
# own, pooled from results.csv by mrr.study2; "coverage" is 1 - omitted_rate.
MEASURES = {
    "reading_cer": ("reading_errors", "read_chars"),
    "reading_cer_casefold": ("reading_casefold_errors", "read_chars"),
    "omitted_rate": ("omitted", "ref_chars"),
    "runaway_rate": ("runaway_inserted", "ref_chars"),
    "case_normalised_rate": ("case_normalised", "ref_chars"),
    "behaviour_rate": ("behaviour_errors", "ref_chars"),
    "letters_reading_cer": ("letters_reading_errors", "letters_read_chars"),
    "letters_reading_cer_noclass": ("letters_reading_noclass_errors", "letters_read_chars"),
}
STORED = ("cer", "letters")

# (family, hypothesis, text type, baseline, treatment, measure, predicted sign, bound, Holm): families 2 to 4 of
# PREREGISTRATION_2.md section 6, the reading CER in place of the CER. Containment keeps H4's letters-only strings.
FAMILIES: tuple[tuple, ...] = (
    ("containment", "H4", RANDOM, CUT, CONTAINED, "letters_reading_cer", -1, None, True),
    ("advantage", "H5", RANDOM, CONVENTIONAL, CONTAINED, "reading_cer", -1, None, True),
    ("prose", "H6", NATURAL, CONVENTIONAL, CONTAINED, "reading_cer", None, DELTA, True),
)
Q1 = tuple(("q1", "H6", NATURAL, CONVENTIONAL, CONTAINED, m, None, None, False)
           for m in ("cer", "coverage", "reading_cer", "omitted_rate", "runaway_rate", "case_normalised_rate",
                     "behaviour_rate"))
Q2 = tuple(("q2", "H4", RANDOM, CUT, CONTAINED, m, None, None, False)
           for m in ("letters", "letters_reading_cer", "letters_reading_cer_noclass"))

PAGE_COLUMNS = (
    "run", "reader", "role", "text_type", "condition", "group", "page_id", "font_file", "shape_alike",
    "reference_chars", "errors", "results_errors", "format_touched", "format_chars", "omitted_head",
    "omitted_interior", "omitted_tail", "omitted", "omissions", "runaway_inserted", "runaways", "case_alike",
    "case_normalised", "reading_errors", "read_chars", "reading_cer", "coverage", "behaviour_errors", "parts_add_up",
    "hit_limit", "letters_chars", "letters_errors", "letters_results_errors", "letters_omitted", "letters_runaway",
    "letters_case_alike", "letters_case_normalised", "letters_o0", "letters_il1", "letters_reading_errors",
    "letters_reading_noclass_errors", "letters_read_chars", "letters_format_touched", "letters_parts_add_up",
    "analysis_status",
)
CONDITION_COLUMNS = (
    "reader", "role", "text_type", "condition", "group", "block", "phase_x_px", "font_file", "font_size_px",
    "render_mode", "shape_alike", "n_pages", "n_rows", "reference_chars", "errors", "cer", "cer_low", "cer_high",
    "read_chars", "reading_errors", "reading_cer", "reading_cer_low", "reading_cer_high", "reading_cer_casefold",
    "coverage", "coverage_low", "coverage_high", "behaviour_rate", "behaviour_rate_low", "behaviour_rate_high",
    "omitted_head_rate", "omitted_interior_rate", "omitted_tail_rate", "omitted_rate", "runaway_rate", "format_rate",
    "case_normalised_rate", "case_alike_rate", "pages_with_omission", "pages_with_runaway", "pages_with_format",
    "pages_with_case_normalised", "hit_limit_share", "format_rows", "format_difference", "rows_not_adding_up",
    "analysis_status",
)
PAIRED_COLUMNS = tuple(analyze2.COLUMNS["paired"]) + ("baseline_pooled", "treatment_pooled", "analysis_status")
CLASS_COLUMNS = (
    "reader", "role", "text_type", "group", "font_file", "shape_alike", "n_pages", "n_rows", "letters",
    "letters_errors", "letters_cer", "letters_omitted", "letters_runaway", "letters_case_normalised",
    "letters_case_alike", "letters_o0", "letters_il1", "letters_reading_other", "letters_read_chars",
    "letters_reading_cer", "letters_reading_cer_noclass", "case_alike_rate", "o0_rate", "il1_rate",
    "reading_other_rate", "analysis_status",
)
CASE_COLUMNS = (
    "reader", "role", "text_type", "level", "condition", "group", "font_file", "shape_alike", "n_rows",
    "reference_chars", "errors", "cer", "case_alike", "case_normalised", "case_errors", "case_alike_rate",
    "case_normalised_rate", "case_rate", "case_share_of_errors", "reading_cer", "reading_cer_casefold",
    "analysis_status",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/sweeps/study2.yaml", help="the Study 2 sweep")
    p.add_argument("--runs-root", default="results/runs", help="where the run folders are (default results/runs)")
    p.add_argument("--out-dir", default="reports/tables", help="the parent of the table folder")
    p.add_argument("--name", default=NAME, help=f"the table folder under --out-dir (default {NAME}); never the "
                                                "sweep's own, which holds the confirmatory tables")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- section 1: every row scored


def _flag(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _alike_from_font(root: Path) -> Callable[[str], frozenset[str]]:
    return lambda font_file: behaviour.shape_alike_letters(str(behaviour.font_path(font_file, root)))


def score_rows(rows: pd.DataFrame, root: Path = ROOT,
               alike_for: Callable[[str], frozenset[str]] | None = None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Every row of ``rows`` (coded by ``study2.with_codes``) scored by ``behaviour.score_row``: the rows with the
    counts added (one per row, in order) and the pages.csv records.

    The letters-only score is ``score_row`` on the letters-only strings of the transcription R1 leaves; its O/0 and
    I/l/1 substitutions come from ``study2.letter_confusions`` on the same strings, so from the same alignment.
    ``alike_for`` maps a row's ``font_file`` to its shape-alike letters (default: measured from the font under
    ``root``).
    """
    alike_for = alike_for or _alike_from_font(root)
    extra: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for r in rows.itertuples(index=False):
        reference, hypothesis = behaviour.read_texts(r.run_dir, r.page_id, r.transcription_file)
        alike = frozenset(alike_for(str(r.font_file)))
        s = behaviour.score_row(reference, hypothesis, alike)
        kept, _ = behaviour.strip_format(hypothesis)
        ref_l, hyp_l = normalise_text(reference).replace(" ", ""), normalise_text(kept).replace(" ", "")
        lt = behaviour.score_row(ref_l, hyp_l, alike)
        subs, _, letters = study2.letter_confusions(reference, kept)
        pairs = study2.confusion_totals(subs)
        if letters != lt.reference_chars or pairs["case_pair"] != lt.case_alike + lt.case_normalised:
            raise RuntimeError(f"{r.run_dir}/{r.page_id}: the letters-only score and its confusions disagree")
        noclass = lt.reading_errors - lt.case_alike - pairs["O/0"] - pairs["I/l/1"]
        stored = int(float(r.substitutions)) + int(float(r.deletions)) + int(float(r.insertions))
        stored_l = int(float(r.substitutions_ns)) + int(float(r.deletions_ns)) + int(float(r.insertions_ns))
        counts = {
            "primary_errors": s.errors, "ref_chars": s.reference_chars, "reading_errors": s.reading_errors,
            "read_chars": s.reference_chars - s.omitted, "reading_casefold_errors": s.reading_errors - s.case_alike,
            "omitted": s.omitted, "omitted_head": s.omitted_head, "omitted_interior": s.omitted_interior,
            "omitted_tail": s.omitted_tail, "runaway_inserted": s.runaway_inserted, "format_chars": s.format_chars,
            "case_normalised": s.case_normalised, "case_alike": s.case_alike, "behaviour_errors": s.behaviour_errors,
            "omissions": s.omissions, "runaways": s.runaways, "format_touched": int(s.format_touched),
            "parts_add_up": s.parts_add_up, "hit_limit_flag": _flag(r.hit_limit), "results_errors": stored,
            "letters_chars": lt.reference_chars, "letters_errors": lt.errors, "letters_results_errors": stored_l,
            "letters_omitted": lt.omitted, "letters_runaway": lt.runaway_inserted,
            "letters_case_alike": lt.case_alike, "letters_case_normalised": lt.case_normalised,
            "letters_o0": pairs["O/0"], "letters_il1": pairs["I/l/1"], "letters_reading_errors": lt.reading_errors,
            "letters_reading_noclass_errors": noclass, "letters_read_chars": lt.reference_chars - lt.omitted,
            "letters_format_touched": int(lt.format_touched), "letters_parts_add_up": lt.parts_add_up,
        }
        extra.append(counts)
        records.append({
            "run": Path(str(r.run_dir)).name, "reader": str(r.reader), "role": study2.role(str(r.reader)),
            "text_type": str(r.text_type), "condition": str(r.condition), "group": str(r.group),
            "page_id": str(r.page_id), "font_file": str(r.font_file), "shape_alike": "".join(sorted(alike)),
            "reference_chars": s.reference_chars, "errors": s.errors, "results_errors": stored,
            "format_touched": s.format_touched, "format_chars": s.format_chars, "omitted_head": s.omitted_head,
            "omitted_interior": s.omitted_interior, "omitted_tail": s.omitted_tail, "omitted": s.omitted,
            "omissions": s.omissions, "runaway_inserted": s.runaway_inserted, "runaways": s.runaways,
            "case_alike": s.case_alike, "case_normalised": s.case_normalised, "reading_errors": s.reading_errors,
            "read_chars": s.reference_chars - s.omitted, "reading_cer": s.reading_cer, "coverage": s.coverage,
            "behaviour_errors": s.behaviour_errors, "parts_add_up": s.parts_add_up, "hit_limit": _flag(r.hit_limit),
            **{k: counts[k] for k in PAGE_COLUMNS if k.startswith("letters_")}, "analysis_status": STATUS,
        })
    scored = pd.concat([rows.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    return scored, records


def score_checks(scored: pd.DataFrame) -> dict[str, list[str]]:
    """Rows where a recomputed primary count differs from results.csv (the letters-only one only where R1 removed
    nothing, since it is recomputed on what R1 leaves), and rows where R1 touched either string."""
    key = [f"{Path(str(d)).name}/{p}" for d, p in zip(scored["run_dir"], scored["page_id"])]
    untouched = scored["format_touched"] == 0
    return {
        "primary_mismatches": [k for k, bad in zip(key, scored["primary_errors"] != scored["results_errors"]) if bad],
        "letters_mismatches": [k for k, bad in zip(key, untouched & (scored["letters_errors"]
                                                                     != scored["letters_results_errors"])) if bad],
        "format_touched": sorted(k for k, t in zip(key, scored["format_touched"]) if t),
        "letters_format_touched": sorted(k for k, t in zip(key, scored["letters_format_touched"]) if t),
    }


# --------------------------------------------------------------------------- pooling through mrr.study2


def counts_view(rows: pd.DataFrame, measure: str) -> pd.DataFrame:
    """``rows`` with the four counts that ``study2``'s pooled ``cer`` reads set to ``measure``'s numerator and
    denominator (``substitutions`` the errors, ``hits`` the characters less the errors), so that ``study2.page_values``,
    ``paired_pages`` and ``condition_bands`` pool that measure (D13). Study 2's own measures come back as they are."""
    if measure in STORED:
        return rows
    num, den = MEASURES[measure]
    out = rows.copy()
    out["substitutions"] = out[num].astype(np.float64)
    out["deletions"] = 0.0
    out["insertions"] = 0.0
    out["hits"] = out[den].astype(np.float64) - out[num].astype(np.float64)
    return out


def _pooled_as(measure: str) -> str:
    return "letters" if measure == "letters" else "cer"


def _base(measure: str) -> str:
    return "omitted_rate" if measure == "coverage" else measure


def paired_values(sub: pd.DataFrame, baseline: str, treatment: str,
                  measure: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The pages both sides read and each side's page values of ``measure``; coverage is 1 - the omitted rate."""
    pages, a, b = study2.paired_pages(counts_view(sub, _base(measure)), baseline, treatment,
                                      _pooled_as(_base(measure)))
    if measure == "coverage":
        a, b = 1.0 - a, 1.0 - b
    return pages, a, b


def pooled(rows: pd.DataFrame, measure: str) -> float:
    """``measure`` over ``rows``, the counts summed over every row (D13)."""
    if rows.empty:
        return float("nan")
    if measure == "letters":
        num = rows[["substitutions_ns", "deletions_ns", "insertions_ns"]].astype(np.float64).to_numpy().sum()
        den = rows[["hits_ns", "substitutions_ns", "deletions_ns"]].astype(np.float64).to_numpy().sum()
    elif measure == "cer":
        num = rows[["substitutions", "deletions", "insertions"]].astype(np.float64).to_numpy().sum()
        den = rows[["hits", "substitutions", "deletions"]].astype(np.float64).to_numpy().sum()
    else:
        num_col, den_col = MEASURES[_base(measure)]
        num, den = float(rows[num_col].sum()), float(rows[den_col].sum())
    value = num / den if den > 0 else float("nan")
    return 1.0 - value if measure == "coverage" else float(value)


def paired_table(scored: pd.DataFrame, contrasts: Sequence[tuple]) -> list[dict[str, Any]]:
    """Every contrast for every reader with rows on both sides, by ``22_analyze_study2.paired_row`` (the paired
    Wilcoxon test, the median of treatment minus baseline and its page-bootstrap interval), Holm across the readers
    of each family where the contrast asks for it, and each side's value pooled over the paired pages."""
    out: list[dict[str, Any]] = []
    for family, hypothesis, text_type, baseline, treatment, measure, predicted, bound, holm in contrasts:
        for reader, sub in scored[scored["text_type"] == text_type].groupby("reader", sort=True):
            if study2.side_rows(sub, baseline).empty or study2.side_rows(sub, treatment).empty:
                continue
            pages, a, b = paired_values(sub, baseline, treatment, measure)
            row = analyze2.paired_row(family, hypothesis, str(reader), text_type, baseline, treatment, measure, a, b,
                                      predicted, bound, holm, STATUS)
            for side, name in ((baseline, "baseline_pooled"), (treatment, "treatment_pooled")):
                rows = study2.side_rows(sub, side)
                row[name] = pooled(rows[rows["page_id"].isin(pages)], measure)
            row["analysis_status"] = STATUS
            out.append(row)
    analyze2._fill_holm_and_verdicts(out)
    return out


# --------------------------------------------------------------------------- the descriptive tables


def _alike_of(records: Sequence[dict[str, Any]]) -> dict[str, str]:
    return {str(p["font_file"]): str(p["shape_alike"]) for p in records}


def _fonts(g: pd.DataFrame, alike: dict[str, str]) -> tuple[str, str]:
    fonts = sorted(set(map(str, g["font_file"])))
    return ";".join(fonts), ";".join(alike.get(f, "") for f in fonts)


def conditions_table(scored: pd.DataFrame, alike: dict[str, str]) -> list[dict[str, Any]]:
    """Per reader, text type and condition: the primary CER as Study 2 pools it, and section 1's measures, each with
    the page-bootstrap interval of ``study2.condition_bands`` (one draw of pages per reader and text type)."""
    bands = {m: {(r["reader"], r["text_type"], r["condition"]): r
                 for r in study2.condition_bands(counts_view(scored, m), _pooled_as(m), RESAMPLES, SEED, ALPHA)}
             for m in ("cer", "reading_cer", "omitted_rate", "behaviour_rate")}
    out = []
    for key, cer in bands["cer"].items():
        reader, text_type, condition = key
        g = scored[(scored["reader"] == reader) & (scored["text_type"] == text_type)
                   & (scored["condition"] == condition)]
        t = {c: float(g[c].sum()) for c in ("ref_chars", "read_chars", "reading_errors", "reading_casefold_errors",
                                            "omitted_head", "omitted_interior", "omitted_tail", "omitted",
                                            "runaway_inserted", "format_chars", "case_normalised", "case_alike",
                                            "behaviour_errors")}
        reading, omitted, beh = (bands[m][key] for m in ("reading_cer", "omitted_rate", "behaviour_rate"))
        touched, untouched = g[g["format_touched"] != 0], g[g["format_touched"] == 0]
        ref = t["ref_chars"]
        out.append({
            **{k: cer[k] for k in CONDITION_COLUMNS[:10]}, "shape_alike": alike.get(cer["font_file"], ""),
            "n_pages": cer["n_pages"], "n_rows": cer["n_rows"], "reference_chars": int(cer["denominator"]),
            "errors": int(cer["numerator"]), "cer": cer["value"], "cer_low": cer["low"], "cer_high": cer["high"],
            "read_chars": int(t["read_chars"]), "reading_errors": int(t["reading_errors"]),
            "reading_cer": reading["value"], "reading_cer_low": reading["low"], "reading_cer_high": reading["high"],
            "reading_cer_casefold": t["reading_casefold_errors"] / t["read_chars"],
            "coverage": 1.0 - omitted["value"], "coverage_low": 1.0 - omitted["high"],
            "coverage_high": 1.0 - omitted["low"], "behaviour_rate": beh["value"], "behaviour_rate_low": beh["low"],
            "behaviour_rate_high": beh["high"],
            **{f"{c}_rate": t[c] / ref for c in ("omitted_head", "omitted_interior", "omitted_tail", "omitted")},
            "runaway_rate": t["runaway_inserted"] / ref, "format_rate": t["format_chars"] / ref,
            "case_normalised_rate": t["case_normalised"] / ref, "case_alike_rate": t["case_alike"] / ref,
            "pages_with_omission": float(np.mean(g["omissions"] > 0)),
            "pages_with_runaway": float(np.mean(g["runaways"] > 0)),
            "pages_with_format": float(np.mean(g["format_touched"] > 0)),
            "pages_with_case_normalised": float(np.mean(g["case_normalised"] > 0)),
            "hit_limit_share": float(np.mean(g["hit_limit_flag"].astype(bool))), "format_rows": int(len(touched)),
            "format_difference": int((touched["primary_errors"] - touched["reading_errors"]
                                      - touched["behaviour_errors"]).sum()),  # section 1.2: the two alignments apart
            "rows_not_adding_up": int((~untouched["parts_add_up"].astype(bool)).sum()), "analysis_status": STATUS,
        })
    return out


def classes_table(scored: pd.DataFrame, alike: dict[str, str]) -> list[dict[str, Any]]:
    """Per reader, text type and group: the letters-only errors by class, counts summed over the group's rows."""
    out = []
    for (reader, text_type, group), g in scored.groupby(["reader", "text_type", "group"], sort=True):
        t = {c: int(g[c].sum()) for c in ("letters_chars", "letters_omitted", "letters_runaway",
                                          "letters_case_normalised", "letters_case_alike", "letters_o0",
                                          "letters_il1", "letters_reading_errors", "letters_reading_noclass_errors",
                                          "letters_read_chars")}
        letters, read = t["letters_chars"], t["letters_read_chars"]
        font_file, shape_alike = _fonts(g, alike)
        out.append({
            "reader": reader, "role": study2.role(str(reader)), "text_type": text_type, "group": group,
            "font_file": font_file, "shape_alike": shape_alike, "n_pages": int(g["page_id"].nunique()),
            "n_rows": int(len(g)), "letters": letters,
            "letters_errors": int(g[["substitutions_ns", "deletions_ns", "insertions_ns"]].astype(np.float64)
                                  .to_numpy().sum()),
            "letters_cer": pooled(g, "letters"), "letters_omitted": t["letters_omitted"],
            "letters_runaway": t["letters_runaway"], "letters_case_normalised": t["letters_case_normalised"],
            "letters_case_alike": t["letters_case_alike"], "letters_o0": t["letters_o0"],
            "letters_il1": t["letters_il1"], "letters_reading_other": t["letters_reading_noclass_errors"],
            "letters_read_chars": read, "letters_reading_cer": t["letters_reading_errors"] / read,
            "letters_reading_cer_noclass": t["letters_reading_noclass_errors"] / read,
            "case_alike_rate": t["letters_case_alike"] / letters, "o0_rate": t["letters_o0"] / letters,
            "il1_rate": t["letters_il1"] / letters, "reading_other_rate": t["letters_reading_noclass_errors"] / read,
            "analysis_status": STATUS,
        })
    return out


def case_table(scored: pd.DataFrame, alike: dict[str, str]) -> list[dict[str, Any]]:
    """Per reader, text type and condition, then per group: the case errors of section 1 (R4) on the normalised
    strings, per reference character and as a share of the primary errors."""
    out = []
    for level in ("condition", "group"):
        for (reader, text_type, name), g in scored.groupby(["reader", "text_type", level], sort=True):
            ref, errors = int(g["ref_chars"].sum()), int(g["primary_errors"].sum())
            a, n = int(g["case_alike"].sum()), int(g["case_normalised"].sum())
            read = float(g["read_chars"].sum())
            font_file, shape_alike = _fonts(g, alike)
            out.append({
                "reader": reader, "role": study2.role(str(reader)), "text_type": text_type, "level": level,
                "condition": name if level == "condition" else "", "group": str(g["group"].iloc[0]),
                "font_file": font_file, "shape_alike": shape_alike, "n_rows": int(len(g)), "reference_chars": ref,
                "errors": errors, "cer": errors / ref, "case_alike": a, "case_normalised": n, "case_errors": a + n,
                "case_alike_rate": a / ref, "case_normalised_rate": n / ref, "case_rate": (a + n) / ref,
                "case_share_of_errors": (a + n) / errors if errors else float("nan"),
                "reading_cer": float(g["reading_errors"].sum()) / read,
                "reading_cer_casefold": float(g["reading_casefold_errors"].sum()) / read, "analysis_status": STATUS,
            })
    order = {c: i for i, c in enumerate(study2.condition_order(scored["condition"]))}
    groups = {g: i for i, g in enumerate(study2.GROUPS)}
    return sorted(out, key=lambda r: (r["reader"], r["text_type"], r["level"] != "condition",
                                      order.get(r["condition"], -1), groups.get(r["group"], -1)))


def build_tables(scored: pd.DataFrame, records: Sequence[dict[str, Any]]) -> dict[str, tuple[list, Sequence[str]]]:
    """Every table of section 3.2 with its columns, from the scored rows and their pages.csv records."""
    alike = _alike_of(records)
    order = {c: i for i, c in enumerate(study2.condition_order(scored["condition"]))}
    pages = sorted(records, key=lambda p: (p["reader"], p["text_type"], order[p["condition"]], p["page_id"]))
    return {
        "pages": (pages, PAGE_COLUMNS),
        "conditions": (conditions_table(scored, alike), CONDITION_COLUMNS),
        "paired": (paired_table(scored, FAMILIES), PAIRED_COLUMNS),
        "q1_prose": (paired_table(scored, Q1), PAIRED_COLUMNS),
        "q2_letterform": (paired_table(scored, Q2), PAIRED_COLUMNS),
        "q2_classes": (classes_table(scored, alike), CLASS_COLUMNS),
        "q3_case": (case_table(scored, alike), CASE_COLUMNS),
    }


# --------------------------------------------------------------------------- the command


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    sweep_path, runs_root = Path(args.config), Path(args.runs_root)
    sweep_cfg = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    if sweep_cfg.get("results_schema") != "study2":
        _log.error("%s has no results_schema: study2; this script re-scores Study 2 only", sweep_path)
        return 1
    sweep_name = sweep_path.stem
    if args.name == sweep_name:
        _log.error("--name %s is the folder of Study 2's confirmatory tables; the re-scoring never writes there",
                   args.name)
        return 1
    run_dirs = sweep_run_dirs(sweep_path, runs_root)
    if not run_dirs:
        _log.error("no run folder of sweep %s under %s", sweep_name, runs_root)
        return 1
    folders = [RunFolder(d) for d in run_dirs]
    patch = {f.reader: int(f.reader_cfg["patch_size"]) for f in folders}
    df_all = load_results(sweep_path, runs_root)  # raises on a repeated key, an edited sweep file or another schema
    df, excluded = complete_rows(df_all, folders)  # PREREGISTRATION_2.md section 8
    coded = study2.with_codes(df, patch)
    scored, records = score_rows(coded, ROOT)
    checks = score_checks(scored)
    _log.info("%s: %d rows in %d run folder(s), %d scored: %d with format removed (R1), %d whose recomputed primary "
              "count differs from results.csv, %d whose letters-only count does", sweep_name, len(df_all),
              len(folders), len(scored), len(checks["format_touched"]), len(checks["primary_mismatches"]),
              len(checks["letters_mismatches"]))
    for name in checks["format_touched"]:
        _log.info("  R1 touched: %s", name)

    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = build_tables(scored, records)
    counts = {name: analyze1.write_table(out_dir / f"{name}.csv", rows, columns)
              for name, (rows, columns) in tables.items()}
    for name, n in counts.items():
        _log.info("  %-19s %5d row(s)  %s", name + ".csv", n, out_dir / f"{name}.csv")
    write_manifest(out_dir, {
        "sweep_file": str(sweep_path), "sweep_sha1": hashlib.sha1(sweep_path.read_bytes()).hexdigest(),
        "sweep_name": sweep_name, "runs_root": str(runs_root), "run_dirs": [d.name for d in run_dirs],
        "rows_loaded": int(len(df_all)), "rows_scored": int(len(scored)),
        "pages_excluded": {Path(k).name: sorted(v) for k, v in excluded.items() if v},
        "resamples": RESAMPLES, "seed": SEED, "alpha": ALPHA, "delta": DELTA, "tables": counts,
        "rules": {"min_run": behaviour.MIN_RUN, "anchor_chars": behaviour.ANCHOR_CHARS,
                  "anchor_errors": behaviour.ANCHOR_ERRORS, "shape_size_px": behaviour.SHAPE_SIZE_PX,
                  "shape_min_iou": behaviour.SHAPE_MIN_IOU,
                  "shape_rasteriser": "mrr.render.load_glyph (unhinted FreeType, as the pages)",
                  "shape_alike": _alike_of(records), "preamble": behaviour.PREAMBLE.pattern,
                  "fence": behaviour.FENCE.pattern,
                  "letters_only": "score_row on the letters-only strings of the transcription R1 leaves; L and the "
                                  "anchor count letters there"},
        **checks, "families": [list(f[:6]) for f in FAMILIES], "status": STATUS,
        "secondary_to": "reports/tables/study2/paired.csv (confirmatory, unchanged)",
        "analysis_plan": "ANALYSIS_PLAN_3.md section 3.2", "blas_threads": 1,
    }, run_id=new_run_id(args.name, "analysis"))
    return 0 if not (checks["primary_mismatches"] or checks["letters_mismatches"]) else 1


if __name__ == "__main__":
    sys.exit(main())
