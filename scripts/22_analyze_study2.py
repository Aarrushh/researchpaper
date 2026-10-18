# PURPOSE: Statistics over the Study 2 run folders (PREREGISTRATION_2.md section 9).
# WHAT GOES HERE: Load the study2 sweep's rows, apply section 8, run the families of section 6 and the secondary contrasts, write the tables of section 9 into <out-dir>/study2/ with a manifest.
# FILLED IN AT STEP: W5.1 (written and tested before prereg-v2); run after the Study 2 run.
# DO NOT: Run analyses that PREREGISTRATION_2.md does not name without status exploratory. Pool Study 2 rows with Study 1, the ladder or the pilots.
"""Study 2: turn the run folders of configs/sweeps/study2.yaml into the tables of PREREGISTRATION_2.md section 9.

    python scripts/22_analyze_study2.py [--config configs/sweeps/study2.yaml] [--runs-root results/runs]
                                        [--out-dir reports/tables] [--checks-only]

Reads the sweep's run folders through ``sweep.load_results`` (the 38 columns of ``results_schema: study2``), drops the
pages section 8 excludes (``sweep.complete_rows``) and writes into ``<out-dir>/<sweep_name>/``:

    checks.csv      one row per run folder, as Study 1's (scripts/20_analyze.py checks_table)
    paired.csv      one row per family, reader, text type and contrast: the paired Wilcoxon test, Holm across the
                    readers of a family, the median of treatment minus baseline with its page-bootstrap interval, the
                    smallest detectable median difference, and the verdict columns of section 6
    letters.csv     per reader, text type and condition: pooled CER and letters-only CER with page-bootstrap intervals,
                    and the mean row WER
    spaces.csv      per reader, text type and condition: inserted spaces per reference character, with its interval
    confusions.csv  per reader, text type and condition: the letters-only substitutions by (reference, hypothesis)
                    character, deletions by character, and the O/0, I/l/1 and case-pair totals
    capacity.csv    per reader, text type and layout: characters, glyphs and lines per page at capacity on the study's
                    own text, glyphs per line, characters per vision token

A page's value for a group pools the counts of the group's rows on that page (D13): conventional is S0, S5 and S10;
contained is M0 and M3; cut is M at P / 2; slot is Q0; density is block D (S18). Every contrast is written
``<treatment>_vs_<baseline>`` and its median difference is treatment minus baseline, so a negative value means the
treatment (the contained monospace layout) has the lower error. Every resampling is by page, 2,000 resamples, seed
20261021 (section 6).
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
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

# One BLAS thread before NumPy loads, as scripts/20_analyze.py, so a freeze check regenerates the tables byte for byte.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from mrr import stats, study2  # noqa: E402
from mrr.manifest import new_run_id, write_manifest  # noqa: E402
from mrr.study2 import CONTAINED, CONVENTIONAL, CUT, DENSITY, SLOT  # noqa: E402
from mrr.sweep import RunFolder, complete_rows, load_results, sweep_run_dirs  # noqa: E402

_spec = importlib.util.spec_from_file_location("analyze_study1", Path(__file__).with_name("20_analyze.py"))
analyze1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze1)

RESAMPLES = 2000  # PREREGISTRATION_2.md section 6
SEED = 20261021
ALPHA = 0.05
NI_SPACES = 0.005  # H3: contained may exceed conventional by less than this many inserted spaces per character
DELTA = 0.01  # H6: contained may exceed conventional on prose by less than this CER
PREREGISTERED, EXPLORATORY = "preregistered", "exploratory"
RANDOM, NATURAL = "random62", "natural"
NAN = float("nan")

# (family, hypothesis, text type, baseline, treatment, measure, predicted sign, bound, Holm, status). Section 6: the four
# families with Holm across their readers (H7 is read from the validation readers' rows of gap and containment); the
# non-inferiority half of H3 is an interval, not a test; then the secondary contrasts, then WER, exploratory.
CONTRASTS: tuple[tuple, ...] = (
    ("gap", "H3", RANDOM, SLOT, CONTAINED, "spaces", -1, None, True, PREREGISTERED),
    ("gap", "H3", RANDOM, CONVENTIONAL, CONTAINED, "spaces", None, NI_SPACES, False, PREREGISTERED),
    ("containment", "H4", RANDOM, CUT, CONTAINED, "letters", -1, None, True, PREREGISTERED),
    ("advantage", "H5", RANDOM, CONVENTIONAL, CONTAINED, "cer", -1, None, True, PREREGISTERED),
    ("prose", "H6", NATURAL, CONVENTIONAL, CONTAINED, "cer", None, DELTA, True, PREREGISTERED),
    ("secondary", "", RANDOM, "M0", "M3", "cer", None, None, False, PREREGISTERED),
    ("secondary", "", NATURAL, "M0", "M3", "cer", None, None, False, PREREGISTERED),
    ("secondary", "", RANDOM, DENSITY, CONTAINED, "cer", None, None, False, PREREGISTERED),
    ("secondary", "", NATURAL, CUT, CONTAINED, "letters", None, None, False, PREREGISTERED),
    ("secondary", "", NATURAL, CONVENTIONAL, CONTAINED, "spaces", None, None, False, PREREGISTERED),
    ("wer", "", RANDOM, SLOT, CONTAINED, "wer", None, None, False, EXPLORATORY),
    ("wer", "", RANDOM, CUT, CONTAINED, "wer", None, None, False, EXPLORATORY),
    ("wer", "", RANDOM, CONVENTIONAL, CONTAINED, "wer", None, None, False, EXPLORATORY),
    ("wer", "", NATURAL, CONVENTIONAL, CONTAINED, "wer", None, None, False, EXPLORATORY),
)
TABLES = ("checks", "paired", "letters", "spaces", "confusions", "capacity")
COLUMNS = {
    "checks": analyze1.COLUMNS["checks"],
    "paired": [
        "family", "hypothesis", "reader", "role", "text_type", "contrast", "baseline", "treatment", "measure", "n",
        "n_zero", "statistic", "p", "method", "p_holm", "median_diff", "ci_low", "ci_high", "sd_diff", "mde",
        "predicted_sign", "significant", "supported", "bound", "passes", "status",
    ],
    "letters": [
        "reader", "role", "text_type", "condition", "group", "block", "phase_x_px", "font_file", "font_size_px",
        "render_mode", "n_pages", "n_rows", "cer", "cer_low", "cer_high", "letters_cer", "letters_low", "letters_high",
        "wer_mean", "status", "family",
    ],
    "spaces": [
        "reader", "role", "text_type", "condition", "group", "n_pages", "n_rows", "spaces_inserted", "reference_chars",
        "spaces_per_char", "low", "high", "status", "family",
    ],
    "confusions": [
        "reader", "role", "text_type", "condition", "group", "font_file", "render_mode", "kind", "ref_char", "hyp_char",
        "count", "letters", "n_rows", "status", "family",
    ],
    "capacity": [
        "reader", "role", "text_type", "layout", "render_mode", "font_file", "font_size_px", "line_pitch_px", "n_pages",
        "chars_per_page", "glyphs_per_page", "lines_per_page", "glyphs_per_line", "vision_tokens", "chars_per_token",
        "shared_chars_per_page", "status", "family",
    ],
}

_log = logging.getLogger("analyze_study2")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/sweeps/study2.yaml", help="the Study 2 sweep; tables go to <out-dir>/<name>/")
    p.add_argument("--runs-root", default="results/runs", help="where the run folders are (default results/runs)")
    p.add_argument("--out-dir", default="reports/tables", help="the parent of the sweep's table folder")
    p.add_argument("--render-config", default="configs/render_defaults.yaml", help="the render defaults the run used")
    p.add_argument("--checks-only", action="store_true",
                   help="write checks.csv alone: the pipeline check allowed while a run is going, which tests nothing")
    return p.parse_args(argv)


def contrast_name(baseline: str, treatment: str) -> str:
    return f"{treatment}_vs_{baseline}"


def paired_row(family: str, hypothesis: str, reader: str, text_type: str, baseline: str, treatment: str, measure: str,
               a: np.ndarray, b: np.ndarray, predicted: int | None, bound: float | None, holm: bool,
               status: str) -> dict[str, Any]:
    """One paired contrast: the Wilcoxon test of b against a, the median of b - a and its page-bootstrap interval."""
    row: dict[str, Any] = {
        "family": family, "hypothesis": hypothesis, "reader": reader, "role": study2.role(reader),
        "text_type": text_type, "contrast": contrast_name(baseline, treatment), "baseline": baseline,
        "treatment": treatment, "measure": measure, "n": int(a.size), "n_zero": NAN, "statistic": NAN, "p": NAN,
        "method": "", "p_holm": NAN, "median_diff": NAN, "ci_low": NAN, "ci_high": NAN, "sd_diff": NAN, "mde": NAN,
        "predicted_sign": "" if predicted is None else predicted, "significant": "", "supported": "",
        "bound": "" if bound is None else bound, "passes": "", "status": status,
        "_holm": f"{family}|{measure}|{text_type}" if holm else None,
    }
    if a.size == 0:
        row["method"] = "undefined: no page holds both sides"
        return row
    if np.all(b == a):  # nothing left once zeros are dropped; SciPy gives p = 1 up to 13 pages and NaN above 50
        row.update(n_zero=int(a.size), method="undefined: every difference is zero")
    else:
        try:
            w = stats.paired_wilcoxon(a, b)
            row.update(n_zero=int(w["n_zero"]), statistic=float(w["statistic"]), p=float(w["pvalue"]),
                       method=w["method"])
        except ValueError as exc:  # NaN in a page value: the test is undefined and the table says so
            row["method"] = f"undefined: {exc}"
    row["median_diff"] = stats.median_paired_difference(a, b)
    row["ci_low"], row["ci_high"] = stats.bootstrap_ci(b - a, np.median, RESAMPLES, ALPHA, SEED)
    if a.size >= 2:
        row["sd_diff"] = float(np.std(b - a, ddof=1))
        if row["sd_diff"] > 0:
            row["mde"] = stats.paired_mde(row["sd_diff"], int(a.size), ALPHA)
    if bound is not None:
        row["passes"] = bool(row["ci_high"] < bound)
    return row


def _fill_holm_and_verdicts(rows: list[dict[str, Any]]) -> None:
    """Holm within each family's group of readers (section 6); then significant and supported where they apply.

    An undefined test stays in its family as p = 1, so the family keeps its size; its own p and p_holm stay NaN. A
    row with a predicted sign is supported when its Holm-adjusted p is below alpha and its median has that sign.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row.pop("_holm")
        if key is not None:
            groups.setdefault(key, []).append(row)
    for members in groups.values():
        if all(math.isnan(r["p"]) for r in members):
            continue
        adjusted = stats.holm([1.0 if math.isnan(r["p"]) else r["p"] for r in members])
        for r, adj in zip(members, adjusted):
            if not math.isnan(r["p"]):
                r["p_holm"] = float(adj)
    for row in rows:
        if math.isnan(row["p_holm"]):
            continue
        row["significant"] = bool(row["p_holm"] < ALPHA)
        if row["predicted_sign"] != "":
            row["supported"] = bool(row["significant"] and np.sign(row["median_diff"]) == row["predicted_sign"])


def paired_table(df) -> list[dict[str, Any]]:
    """Section 6: every contrast of CONTRASTS for every reader that has rows on both sides."""
    out: list[dict[str, Any]] = []
    for family, hypothesis, text_type, baseline, treatment, measure, predicted, bound, holm, status in CONTRASTS:
        for reader, sub in df[df["text_type"] == text_type].groupby("reader", sort=True):
            if study2.side_rows(sub, baseline).empty or study2.side_rows(sub, treatment).empty:
                continue  # smolvlm500m reads no random page; a reader without block D has no density row
            _, a, b = study2.paired_pages(sub, baseline, treatment, measure)
            out.append(paired_row(family, hypothesis, str(reader), text_type, baseline, treatment, measure, a, b,
                                  predicted, bound, holm, status))
    _fill_holm_and_verdicts(out)
    return out


def letters_table(df) -> list[dict[str, Any]]:
    """Section 9, item 3: pooled CER and letters-only CER per condition, with page-bootstrap intervals."""
    cer = study2.condition_bands(df, "cer", RESAMPLES, SEED, ALPHA)
    letters = {(r["reader"], r["text_type"], r["condition"]): r
               for r in study2.condition_bands(df, "letters", RESAMPLES, SEED, ALPHA)}
    wer = df.groupby(["reader", "text_type", "condition"])["wer"].mean()
    out = []
    for r in cer:
        key = (r["reader"], r["text_type"], r["condition"])
        out.append({
            **{k: r[k] for k in COLUMNS["letters"][:12]},
            "cer": r["value"], "cer_low": r["low"], "cer_high": r["high"], "letters_cer": letters[key]["value"],
            "letters_low": letters[key]["low"], "letters_high": letters[key]["high"], "wer_mean": float(wer[key]),
            "status": PREREGISTERED, "family": "description",
        })
    return out


def spaces_table(df) -> list[dict[str, Any]]:
    """Section 9, item 3: inserted spaces per reference character per condition, with page-bootstrap intervals."""
    return [{
        **{k: r[k] for k in ("reader", "role", "text_type", "condition", "group", "n_pages", "n_rows")},
        "spaces_inserted": int(r["numerator"]), "reference_chars": int(r["denominator"]),
        "spaces_per_char": r["value"], "low": r["low"], "high": r["high"], "status": PREREGISTERED,
        "family": "description",
    } for r in study2.condition_bands(df, "spaces", RESAMPLES, SEED, ALPHA)]


def confusions_table(df, folders: Sequence[RunFolder]) -> list[dict[str, Any]]:
    """Section 9, item 4: letters-only substitutions and deletions by character, and the three pair totals."""
    cells = study2.run_confusions(df, {str(f.run_dir): f for f in folders})
    out = []
    for (reader, text_type, condition) in sorted(cells):
        cell = cells[(reader, text_type, condition)]
        base = {"reader": reader, "role": study2.role(reader), "text_type": text_type, "condition": condition,
                "group": cell["group"], "font_file": cell["font_file"], "render_mode": cell["render_mode"], "letters": cell["letters"],
                "n_rows": cell["rows"], "status": PREREGISTERED, "family": "description"}
        for kind, count in study2.confusion_totals(cell["subs"]).items():
            out.append({**base, "kind": kind, "ref_char": "", "hyp_char": "", "count": count})
        for (r, h), count in sorted(cell["subs"].items()):
            out.append({**base, "kind": "substitution", "ref_char": r, "hyp_char": h, "count": count})
        for r, count in sorted(cell["dels"].items()):
            out.append({**base, "kind": "deletion", "ref_char": r, "hyp_char": "", "count": count})
    return out


def _planned_pages(folder: RunFolder) -> list[SimpleNamespace]:
    """The run's planned pages from its references/, in plan order within each text type (random ids are rnd...)."""
    pages = []
    for path in (folder.run_dir / "references").glob("*.txt"):
        page_id = path.stem
        kind = RANDOM if page_id.startswith("rnd") else NATURAL
        index = int(page_id.rsplit("_p", 1)[1])
        pages.append((kind, index, SimpleNamespace(page_id=page_id, text_type=kind,
                                                    text=path.read_text(encoding="utf-8"))))
    return [p for _, _, p in sorted(pages, key=lambda t: (t[0], t[1]))]


def capacity_table(folders: Sequence[RunFolder], render_cfg: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    """Section 9, item 5: the geometry of each layout on the study's own text, one reader at a time."""
    out = []
    seen = set()
    for folder in folders:
        if folder.reader in seen:
            continue
        seen.add(folder.reader)
        for row in study2.capacity(folder, _planned_pages(folder), render_cfg, root):
            out.append({**row, "status": PREREGISTERED, "family": "description"})
    return sorted(out, key=lambda r: (r["reader"], r["text_type"] != RANDOM, r["render_mode"], r["font_file"],
                                      r["font_size_px"]))


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    sweep_path, runs_root = Path(args.config), Path(args.runs_root)
    sweep_cfg = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    if sweep_cfg.get("results_schema") != "study2":
        _log.error("%s has no results_schema: study2; this script analyses Study 2 only", sweep_path)
        return 1
    sweep_name = sweep_path.stem
    run_dirs = sweep_run_dirs(sweep_path, runs_root)
    if not run_dirs:
        _log.error("no run folder of sweep %s under %s: run the sweep first (scripts/10_run_sweep.py)", sweep_name,
                   runs_root)
        return 1
    folders = [RunFolder(d) for d in run_dirs]
    patch = {f.reader: int(f.reader_cfg["patch_size"]) for f in folders}
    df_all = load_results(sweep_path, runs_root)  # raises on a repeated key, an edited sweep file or another schema
    df, excluded = complete_rows(df_all, folders)  # section 8
    _log.info("%s: %d rows in %d run folder(s); %d page(s) excluded under section 8; %d rows analysed", sweep_name,
              len(df_all), len(folders), sum(len(p) for p in excluded.values()), len(df))
    out_dir = Path(args.out_dir) / sweep_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[dict[str, Any]]] = {"checks": analyze1.checks_table(df_all, folders, excluded)}
    names = ("checks",) if args.checks_only else TABLES
    if not args.checks_only:
        coded = study2.with_codes(df, patch)
        render_cfg = yaml.safe_load(Path(args.render_config).read_text(encoding="utf-8"))
        tables.update({
            "paired": paired_table(coded), "letters": letters_table(coded) if len(coded) else [],
            "spaces": spaces_table(coded) if len(coded) else [], "confusions": confusions_table(coded, folders),
            "capacity": capacity_table(folders, render_cfg, Path(".")),
        })
    counts = {name: analyze1.write_table(out_dir / f"{name}.csv", tables[name], COLUMNS[name]) for name in names}
    for name in names:
        _log.info("  %-15s %6d row(s)  %s", name + ".csv", counts[name], out_dir / f"{name}.csv")
    planned = [str(r) for r in (sweep_cfg.get("readers") or [])]
    analysed = sorted({f.reader for f in folders})
    write_manifest(out_dir, {
        "sweep_file": str(sweep_path), "sweep_sha1": hashlib.sha1(sweep_path.read_bytes()).hexdigest(),
        "sweep_name": sweep_name, "runs_root": str(runs_root), "run_dirs": [d.name for d in run_dirs],
        "readers_analysed": analysed, "readers_planned": planned,
        "readers_without_a_run": [r for r in planned if r not in analysed],
        "rows_loaded": int(len(df_all)), "rows_analysed": int(len(df)),
        "pages_excluded": {Path(k).name: sorted(v) for k, v in excluded.items() if v},
        "resamples": RESAMPLES, "seed": SEED, "alpha": ALPHA, "ni_spaces": NI_SPACES, "delta": DELTA,
        "tables": counts, "analysis_plan": "PREREGISTRATION_2.md section 9", "checks_only": bool(args.checks_only),
        "blas_threads": 1,
    }, run_id=new_run_id(sweep_name, "analysis"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
