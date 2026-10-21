# PURPOSE: The exploratory check of glyph size against text load (ANALYSIS_PLAN_3.md, Amendment 2): stage 0's full pages set against Study 2's partial pages, where on the page the errors fall, and the 1% token ratios at the read-off sizes.
# WHAT GOES HERE: Load Study 2's and stage 0's rows as scripts/25_fit_stage0.py does, pool the primary counts by page (D13), re-align each row with jiwer to place its errors on the page, read acuity.csv and tokens.csv for the curve and the token ratios, and write four tables into <out-dir>/load/ with a manifest.
# FILLED IN AT STEP: ANALYSIS_PLAN_3.md Amendment 2 (21 October 2026), after analysis-freeze-v1; ported from the scratch checks of research-paper-notes/research/exploratory_2026-10-21/.
# DO NOT: Load a model or read a row that is not frozen. Write into another table folder, or change a confirmatory table or its verdict. Label any table here anything but exploratory.
"""Glyph size against text load, from the frozen rows (ANALYSIS_PLAN_3.md, Amendment 2). Every table is exploratory.

    python scripts/27_load_check.py [--configs <Study 2's sweep and stage 0's three>] [--runs-root results/runs]
                                    [--stage0-tables reports/tables/s3_stage0] [--out-dir reports/tables] [--name load]

Stage 0 fills every page, so its characters per page fall roughly as s^-2 as the size s rises, and its curve
CER(s) = 1 / (1 + (s / a)^beta) moves size and load together. Study 2's pages hold a passage of fixed length, so set
beside stage 0 they part the two. "Study 2" here is its S0 condition (conventional Source Sans 3, phase 0) and, for
the size contrast, S28 (block D); "stage 0" is every row of its three sweeps. Writes into ``<out-dir>/<name>/``:

    contrasts.csv       per reader on random text, three ratios of pooled primary CER (D13), each with a page-bootstrap
                        interval and beside what the stage-0 curve predicts for the same change of size:
                          size   Study 2's S28 (28 px) over S0 (25 px): the same text on the same pages (paired)
                          load   stage 0 at 24 px (23 px for smolvlm), full pages, over S0: about the same size
                          equal  stage 0 at 36 px (34 px for smolvlm), full pages about as long as S0's, over S0:
                                 about the same load, glyphs about 1.4 times larger
    region.csv          per sweep, reader, text type and size: substitutions and deletions per reference character by
                        where its glyph sits: inside the region a Study 2 page of that reader and text type covers (x
                        below its block's right edge, line below its mean line count) and outside it, and by quarter
                        of the page width and of the page height
    position.csv        per sweep, reader, text type and size: the CER by quarter of the reference and over its first k
                        characters, k being Study 2's mean page length for that reader and text type
    readoff_tokens.csv  per reader and text type: acuity.csv's full-page capacity curve,
                        kappa(s) = chars_c * s^-chars_gamma / vision_tokens, evaluated at the read-off 1% size and
                        divided by tokens.csv's characters per text token, beside tokens.csv's ratio at the fitted s01;
                        where the read-off is not reached, the ratio at the largest size measured, an upper bound

Region and position re-align each row's normalised strings with jiwer: a substitution or deletion is charged to its
own reference characters, an insertion to the reference character it precedes (the last slot holds those after the
page's end). A row whose re-aligned counts differ from results.csv is named in the manifest, and the exit status is 1.
Region places each reference character at the pen position ``render.layout_page`` gives its glyph (a space takes the
glyph before it), with the render config of the row: Study 2's phase allowances for its pages, none for stage 0's.
The bootstrap resamples pages, 2,000 times, seed 20261021, as Study 2's; it is paired where the pages are shared.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

# One BLAS thread before NumPy loads, as scripts/22_analyze_study2.py, so the tables regenerate byte for byte.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import jiwer  # noqa: E402
import yaml  # noqa: E402

from mrr import grid, render  # noqa: E402
from mrr.behaviour import font_path, read_texts  # noqa: E402
from mrr.manifest import new_run_id, write_manifest  # noqa: E402
from mrr.sweep import RunFolder, complete_rows, load_results, sweep_run_dirs  # noqa: E402
from mrr.text_source import normalise_text  # noqa: E402


def _script(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze1 = _script("analyze_study1", "20_analyze.py")  # write_table: 12 significant digits, NaN an empty cell

_log = logging.getLogger("load_check")

ROOT = Path(__file__).resolve().parents[1]
NAME = "load"
STATUS = "exploratory"  # ANALYSIS_PLAN_3.md, Amendment 2: every table here
DEFAULT_CONFIGS = ("configs/sweeps/study2.yaml", "configs/sweeps/s3_stage0_random.yaml",
                   "configs/sweeps/s3_stage0_prose.yaml", "configs/sweeps/s3_stage0_prose_internvl3.yaml")
RESAMPLES, SEED, ALPHA = 2000, 20261021, 0.05  # Study 2's
RANDOM = "random62"
STAGE0, S0, S28 = "stage0", "study2_S0", "study2_S28"
STUDY2_CONDITIONS = {("S", 0.0): S0, ("D", 0.0): S28}  # (block, phase_x_px): conventional sans at 25 px, and 28 px
CONTRAST_READERS = ("qwen", "qwen7b", "internvl3_2b", "smolvlm")  # the readers that read random text in both studies
# The stage-0 sizes of the load and equal contrasts. 36 px holds about 211 characters, S0 about 201; smolvlm's ladder
# has 23 and 34 px, and 34 px holds about 147, S0 about 151.
FULL_SIZES = {"smolvlm": (23.0, 34.0)}
FULL_SIZES_DEFAULT = (24.0, 36.0)
CHECK_TOLERANCE = 1e-9  # readoff_tokens: the formula against acuity.csv's kappa01 and tokens.csv's ratio

CONTRAST_COLUMNS = (
    "reader", "text_type", "contrast", "baseline", "treatment", "baseline_size_px", "treatment_size_px", "paired",
    "baseline_pages", "treatment_pages", "baseline_chars_per_page", "treatment_chars_per_page", "baseline_cer",
    "treatment_cer", "ratio", "ratio_low", "ratio_high", "curve_ratio", "a", "beta", "resamples", "analysis_status",
)
REGION_COLUMNS = (
    "sweep", "reader", "text_type", "size_px", "pages", "lines_per_page", "x_edge", "s2_lines", "sd_all", "sd_inside",
    "chars_inside", "sd_outside", "chars_outside", "sd_x1", "sd_x2", "sd_x3", "sd_x4", "sd_y1", "sd_y2", "sd_y3",
    "sd_y4", "analysis_status",
)
POSITION_COLUMNS = (
    "sweep", "reader", "text_type", "size_px", "pages", "chars_per_page", "cer", "sd_rate", "sd_q1", "sd_q2", "sd_q3",
    "sd_q4", "ins_inside_rate", "ins_after_end_rate", "k", "sd_first_k", "cer_first_k", "analysis_status",
)
READOFF_COLUMNS = (
    "reader", "text_type", "s01_disagrees", "s01", "token_ratio01", "readoff_s01_how", "readoff_s01", "size_px",
    "kappa", "token_ratio_readoff", "upper_bound", "chars_per_text_token", "analysis_status",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS),
                   help="Study 2's sweep and stage 0's three (default: those that wrote the frozen rows)")
    p.add_argument("--runs-root", default="results/runs", help="where the run folders are (default results/runs)")
    p.add_argument("--stage0-tables", default="reports/tables/s3_stage0",
                   help="the stage-0 tables whose acuity.csv and tokens.csv give the curve and the token ratios")
    p.add_argument("--out-dir", default="reports/tables", help="the parent of the table folder")
    p.add_argument("--name", default=NAME, help=f"the table folder under --out-dir (default {NAME})")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- the rows


def load_rows(configs: Sequence[Path], runs_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    """Study 2's S0 and S28 rows and every stage-0 row, as records, and the run folders read.

    Each sweep's rows come through ``sweep.load_results`` (sorted by reader, page_id, block and condition hash) less
    the pages ``sweep.complete_rows`` excludes; the reader config of each record is its run folder's manifest's.
    """
    records: list[dict[str, Any]] = []
    run_dirs: list[Path] = []
    for sweep_path in map(Path, configs):
        sweep_cfg = yaml.safe_load(sweep_path.read_text(encoding="utf-8")) or {}
        study2 = not sweep_path.stem.startswith("s3_stage0")  # stage 0's sweeps share Study 2's results schema
        if study2 and sweep_cfg.get("results_schema") != "study2":
            raise ValueError(f"{sweep_path}: neither Study 2's sweep nor a stage-0 sweep")
        dirs = sweep_run_dirs(sweep_path, runs_root)
        if not dirs:
            raise FileNotFoundError(f"no run folder of sweep {sweep_path.stem} under {runs_root}")
        run_dirs += dirs
        folders = [RunFolder(d) for d in dirs]
        reader_cfgs = {f.reader: f.reader_cfg for f in folders}
        df, _ = complete_rows(load_results(sweep_path, runs_root), folders)
        for r in df.itertuples(index=False):
            reader = str(r.reader)
            if study2:
                source = STUDY2_CONDITIONS.get((str(r.block), float(r.phase_x_px)))
                if source is None:
                    continue
                allowances = tuple(int(v) for v in sweep_cfg["phase_allowances_px"][reader])
            else:
                source, allowances = STAGE0, (0, 0)
            records.append({
                "source": source, "reader": reader, "text_type": str(r.text_type), "size": float(r.font_size_px),
                "page_id": str(r.page_id), "run_dir": Path(str(r.run_dir)),
                "transcription_file": str(r.transcription_file), "font_file": str(r.font_file),
                "line_pitch_px": int(float(r.line_pitch_px)), "aa_mode": str(r.aa_mode),
                "reader_cfg": reader_cfgs[reader], "allowances": allowances,
                "counts": tuple(int(float(getattr(r, k))) for k in ("hits", "substitutions", "deletions", "insertions")),
            })
    return records, run_dirs


def errors_and_chars(counts: tuple[int, int, int, int]) -> tuple[int, int]:
    """(S + D + I, H + S + D) of (hits, substitutions, deletions, insertions): the primary CER's two counts."""
    h, s, d, i = counts
    return s + d + i, h + s + d


def pooled(pages: Sequence[tuple[int, int]]) -> float:
    """The pooled CER of (errors, characters) pairs (D13)."""
    return sum(e for e, _ in pages) / sum(n for _, n in pages)


def curve(s: float, a: float, beta: float) -> float:
    return 1 / (1 + (s / a) ** beta)


# --------------------------------------------------------------------------- contrasts.csv


def _interval(boot: list[float]) -> tuple[float, float]:
    boot = sorted(boot)
    return boot[int(ALPHA / 2 * len(boot)) - 1], boot[int((1 - ALPHA / 2) * len(boot)) - 1]


def contrasts_table(records: Sequence[dict[str, Any]], fits: Mapping[tuple[str, str], Mapping[str, str]],
                    resamples: int = RESAMPLES, seed: int = SEED) -> list[dict[str, Any]]:
    """The size, load and equal contrasts of every reader of ``CONTRAST_READERS``, from one generator in that order."""
    study2: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)  # (reader, source) -> page -> record
    stage0: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)  # (reader, size) -> records
    for rec in records:
        if rec["text_type"] != RANDOM:
            continue
        if rec["source"] == STAGE0:
            stage0[(rec["reader"], rec["size"])].append(rec)
        else:
            pages = study2[(rec["reader"], rec["source"])]
            if rec["page_id"] in pages:
                raise ValueError(f"{rec['reader']} {rec['source']}: page {rec['page_id']} has two rows")
            pages[rec["page_id"]] = rec
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for reader in CONTRAST_READERS:
        fit = fits[(reader, RANDOM)]
        a, beta = float(fit["a"]), float(fit["beta"])
        base, larger = study2[(reader, S0)], study2[(reader, S28)]
        pages = sorted(set(base) & set(larger))
        (base_size,) = {base[p]["size"] for p in pages}
        (s28_size,) = {larger[p]["size"] for p in pages}
        x = [errors_and_chars(base[p]["counts"]) for p in pages]

        def row(contrast: str, treatment: str, size: float, y: list[tuple[int, int]], paired: bool,
                boot: list[float]) -> dict[str, Any]:
            low, high = _interval(boot)
            return {
                "reader": reader, "text_type": RANDOM, "contrast": contrast, "baseline": S0, "treatment": treatment,
                "baseline_size_px": base_size, "treatment_size_px": size, "paired": paired,
                "baseline_pages": len(x), "treatment_pages": len(y),
                "baseline_chars_per_page": sum(n for _, n in x) / len(x),
                "treatment_chars_per_page": sum(n for _, n in y) / len(y), "baseline_cer": pooled(x),
                "treatment_cer": pooled(y), "ratio": pooled(y) / pooled(x), "ratio_low": low, "ratio_high": high,
                "curve_ratio": curve(size, a, beta) / curve(base_size, a, beta), "a": a, "beta": beta,
                "resamples": resamples, "analysis_status": STATUS,
            }

        y = [errors_and_chars(larger[p]["counts"]) for p in pages]
        boot = []
        for _ in range(resamples):
            idx = [rng.randrange(len(pages)) for _ in pages]
            boot.append(pooled([y[i] for i in idx]) / pooled([x[i] for i in idx]))
        out.append(row("size", S28, s28_size, y, True, boot))
        for contrast, size in zip(("load", "equal"), FULL_SIZES.get(reader, FULL_SIZES_DEFAULT)):
            full = [errors_and_chars(rec["counts"]) for rec in stage0[(reader, size)]]
            if not full:
                raise ValueError(f"{reader}: no stage-0 random-text row at {size:g} px")
            boot = []
            for _ in range(resamples):
                fb = [full[rng.randrange(len(full))] for _ in full]
                xb = [x[rng.randrange(len(x))] for _ in x]
                boot.append(pooled(fb) / pooled(xb))
            out.append(row(contrast, STAGE0, size, full, False, boot))
    return out


# --------------------------------------------------------------------------- re-alignment and layout


def align(reference: str, hypothesis: str) -> tuple[list[int], list[int], tuple[int, int, int, int]]:
    """Substitutions and deletions at each reference position; insertions before each position (index len(reference)
    holds those after the page's last character); and jiwer's (hits, substitutions, deletions, insertions)."""
    out = jiwer.process_characters(reference, hypothesis)
    sd = [0] * len(reference)
    ins = [0] * (len(reference) + 1)
    for c in out.alignments[0]:
        if c.type in ("substitute", "delete"):
            for i in range(c.ref_start_idx, c.ref_end_idx):
                sd[i] = 1
        elif c.type == "insert":
            ins[c.ref_start_idx] += c.hyp_end_idx - c.hyp_start_idx
    return sd, ins, (out.hits, out.substitutions, out.deletions, out.insertions)


def render_cfg(rec: Mapping[str, Any], defaults: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """The render config of a row: the conventional layout at phase 0, the row's font, size, pitch and anti-aliasing,
    and its sweep's phase allowances (which set the block for Study 2's shared pages)."""
    ax, ay = rec["allowances"]
    return grid.resolve_render_cfg(
        defaults, rec["reader_cfg"], font_file=str(font_path(rec["font_file"], root)), font_size_px=rec["size"],
        line_pitch_px=rec["line_pitch_px"], render_mode="conventional", aa_mode=rec["aa_mode"], phase_x_px=0.0,
        phase_y_px=0.0, max_phase_x_px=ax, max_phase_y_px=ay)


def placed(reference: str, cfg: Mapping[str, Any]) -> list[tuple[float, float, int]]:
    """(x px, baseline px, line) of every reference character; a space takes the glyph before it."""
    glyphs = render.layout_page(reference, cfg)
    out: list[tuple[float, float, int]] = []
    g, last = 0, None
    for c in reference:
        if c == " ":
            out.append(last)
            continue
        pg = glyphs[g]
        if pg.char != c:
            raise ValueError(f"layout mismatch: {pg.char!r} against {c!r}")
        last = (pg.x_64 / 64, pg.baseline_64 / 64, pg.line)
        out.append(last)
        g += 1
    if g != len(glyphs):
        raise ValueError("layout has glyphs left over")
    return out


def page_items(records: Sequence[dict[str, Any]], defaults: Mapping[str, Any],
               root: Path) -> tuple[dict[tuple, list[dict[str, Any]]], list[str]]:
    """Every S0 and stage-0 row re-aligned and laid out, by (sweep, reader, text type, size), and the rows whose
    re-aligned counts differ from results.csv."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    mismatches: list[str] = []
    for rec in records:
        if rec["source"] == S28:
            continue
        reference, hypothesis = read_texts(rec["run_dir"], rec["page_id"], rec["transcription_file"])
        reference, hypothesis = normalise_text(reference), normalise_text(hypothesis)
        sd, ins, counts = align(reference, hypothesis)
        if counts != rec["counts"]:
            mismatches.append(f"{rec['run_dir'].name}/{rec['page_id']}")
        cfg = render_cfg(rec, defaults, root)
        groups[(rec["source"], rec["reader"], rec["text_type"], rec["size"])].append(
            {"sd": sd, "ins": ins, "pos": placed(reference, cfg), "cfg": cfg})
    return groups, sorted(mismatches)


def _lines(item: Mapping[str, Any]) -> int:
    return max(p[2] for p in item["pos"]) + 1


# --------------------------------------------------------------------------- region.csv and position.csv


def region_table(groups: Mapping[tuple, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Substitutions and deletions per reference character by where its glyph sits (the module docstring)."""
    region: dict[tuple[str, str], tuple[float, float]] = {}  # (reader, text type) -> (Study 2's x edge, mean lines)
    for (source, reader, text_type, _), items in groups.items():
        if source == S0:
            cfg = items[0]["cfg"]
            region[(reader, text_type)] = (cfg["margin_px"] + render.block_size_px(cfg)[0],
                                           sum(map(_lines, items)) / len(items))
    out = []
    for key in sorted(groups):
        source, reader, text_type, size = key
        items = groups[key]
        x_edge, s2_lines = region.get((reader, text_type), (float("nan"), float("nan")))
        acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for item in items:
            width, height = item["cfg"]["page_width_px"], item["cfg"]["page_height_px"]
            for (x, y, line), e in zip(item["pos"], item["sd"]):
                sides = ["all", f"x{min(int(4 * x / width), 3) + 1}", f"y{min(int(4 * y / height), 3) + 1}"]
                if (reader, text_type) in region:
                    sides.append("inside" if (x < x_edge and line < s2_lines) else "outside")
                for side in sides:
                    acc[side][0] += e
                    acc[side][1] += 1

        def rate(k: str) -> float:
            return acc[k][0] / acc[k][1] if acc[k][1] else float("nan")

        out.append({
            "sweep": source, "reader": reader, "text_type": text_type, "size_px": size, "pages": len(items),
            "lines_per_page": sum(map(_lines, items)) / len(items), "x_edge": x_edge, "s2_lines": s2_lines,
            "sd_all": rate("all"), "sd_inside": rate("inside"), "chars_inside": acc["inside"][1],
            "sd_outside": rate("outside"), "chars_outside": acc["outside"][1],
            **{f"sd_x{i}": rate(f"x{i}") for i in range(1, 5)}, **{f"sd_y{i}": rate(f"y{i}") for i in range(1, 5)},
            "analysis_status": STATUS,
        })
    return out


def position_table(groups: Mapping[tuple, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """The CER by quarter of the reference and over its first k characters (the module docstring)."""
    k_of = {(reader, text_type): round(sum(len(i["sd"]) for i in items) / len(items))
            for (source, reader, text_type, _), items in groups.items() if source == S0}
    out = []
    for key in sorted(groups):
        source, reader, text_type, size = key
        items = groups[key]
        n_ref = sum(len(i["sd"]) for i in items)
        n_sd = sum(sum(i["sd"]) for i in items)
        n_in = sum(sum(i["ins"][:-1]) for i in items)
        n_end = sum(i["ins"][-1] for i in items)
        q_err, q_n = [0] * 4, [0] * 4
        k = k_of.get((reader, text_type))
        fk_sd = fk_in = fk_n = 0
        for item in items:
            sd, ins = item["sd"], item["ins"]
            for j, v in enumerate(sd):
                q = min(4 * j // len(sd), 3)
                q_err[q] += v
                q_n[q] += 1
                if k is not None and j < k:
                    fk_sd += v
                    fk_in += ins[j]
                    fk_n += 1
        out.append({
            "sweep": source, "reader": reader, "text_type": text_type, "size_px": size, "pages": len(items),
            "chars_per_page": round(n_ref / len(items)), "cer": (n_sd + n_in + n_end) / n_ref, "sd_rate": n_sd / n_ref,
            **{f"sd_q{q + 1}": q_err[q] / q_n[q] if q_n[q] else float("nan") for q in range(4)},
            "ins_inside_rate": n_in / n_ref, "ins_after_end_rate": n_end / n_ref,
            "k": "" if k is None else k, "sd_first_k": fk_sd / fk_n if fk_n else float("nan"),
            "cer_first_k": (fk_sd + fk_in) / fk_n if fk_n else float("nan"), "analysis_status": STATUS,
        })
    return out


# --------------------------------------------------------------------------- readoff_tokens.csv


def kappa(s: float, fit: Mapping[str, str]) -> float:
    """acuity.csv's full-page capacity curve: characters per vision token at size s."""
    return float(fit["chars_c"]) * s ** -float(fit["chars_gamma"]) / int(fit["vision_tokens"])


def readoff_table(acuity: Sequence[Mapping[str, str]],
                  tokens: Mapping[tuple[str, str], Mapping[str, str]]) -> tuple[list[dict[str, Any]], float]:
    """The 1% token ratio at the read-off size, per acuity.csv row, and the largest relative disagreement of the two
    checks (the formula at the fitted s01 against kappa01; kappa01 over characters per text token against
    token_ratio01)."""
    out, worst = [], 0.0
    for fit in acuity:
        t = tokens[(fit["reader"], fit["text_type"])]
        cpt = float(t["chars_per_text_token"])
        s01, k01, r01 = float(fit["s01"]), float(fit["kappa01"]), float(t["token_ratio01"])
        worst = max(worst, abs(kappa(s01, fit) / k01 - 1), abs(k01 / cpt / r01 - 1))
        reached = fit["readoff_s01_how"] != "not_reached"
        size = float(fit["readoff_s01"]) if reached else max(float(s) for s in fit["sizes_px"].split(";"))
        out.append({
            "reader": fit["reader"], "text_type": fit["text_type"], "s01_disagrees": fit["s01_disagrees"] == "True",
            "s01": s01, "token_ratio01": r01, "readoff_s01_how": fit["readoff_s01_how"],
            "readoff_s01": float(fit["readoff_s01"]) if reached else float("nan"), "size_px": size,
            "kappa": kappa(size, fit), "token_ratio_readoff": kappa(size, fit) / cpt, "upper_bound": not reached,
            "chars_per_text_token": cpt, "analysis_status": STATUS,
        })
    return out, worst


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- the command


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    configs, runs_root, stage0_tables = [Path(c) for c in args.configs], Path(args.runs_root), Path(args.stage0_tables)
    if args.name in {c.stem for c in configs} | {stage0_tables.name}:
        _log.error("--name %s is another analysis's table folder; this check writes only its own", args.name)
        return 1
    acuity = _read_csv(stage0_tables / "acuity.csv")
    tokens = {(r["reader"], r["text_type"]): r for r in _read_csv(stage0_tables / "tokens.csv")}
    fits = {(r["reader"], r["text_type"]): r for r in acuity}
    defaults = yaml.safe_load((ROOT / "configs" / "render_defaults.yaml").read_text(encoding="utf-8"))

    records, run_dirs = load_rows(configs, runs_root)
    groups, mismatches = page_items(records, defaults, ROOT)
    readoff, worst = readoff_table(acuity, tokens)
    _log.info("%d rows from %d run folder(s): %d re-aligned, %d whose re-aligned counts differ from results.csv; "
              "token-ratio checks agree to %.1e", len(records), len(run_dirs), sum(map(len, groups.values())),
              len(mismatches), worst)
    tables = {
        "contrasts": (contrasts_table(records, fits), CONTRAST_COLUMNS),
        "region": (region_table(groups), REGION_COLUMNS),
        "position": (position_table(groups), POSITION_COLUMNS),
        "readoff_tokens": (readoff, READOFF_COLUMNS),
    }
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {name: analyze1.write_table(out_dir / f"{name}.csv", rows, columns)
              for name, (rows, columns) in tables.items()}
    for name, n in counts.items():
        _log.info("  %-19s %5d row(s)  %s", name + ".csv", n, out_dir / f"{name}.csv")
    write_manifest(out_dir, {
        "sweep_files": [str(c) for c in configs],
        "sweep_sha1": {c.stem: hashlib.sha1(c.read_bytes()).hexdigest() for c in configs},
        "runs_root": str(runs_root), "run_dirs": [d.name for d in run_dirs], "rows": len(records),
        "stage0_tables": str(stage0_tables),
        "stage0_tables_sha256": {n: hashlib.sha256((stage0_tables / n).read_bytes()).hexdigest()
                                 for n in ("acuity.csv", "tokens.csv")},
        "resamples": RESAMPLES, "seed": SEED, "alpha": ALPHA, "contrast_readers": list(CONTRAST_READERS),
        "full_sizes": {r: list(FULL_SIZES.get(r, FULL_SIZES_DEFAULT)) for r in CONTRAST_READERS},
        "realigned_mismatches": mismatches, "token_ratio_check_max": worst, "tables": counts, "status": STATUS,
        "analysis_plan": "ANALYSIS_PLAN_3.md, Amendment 2", "blas_threads": 1,
        "ported_from": "research-paper-notes/research/exploratory_2026-10-21/ (load_vs_size_check.py, "
                       "page_region_check.py, page_position_check.py, readoff_tokens_check.py)",
    }, run_id=new_run_id(args.name, "analysis"))
    if worst > CHECK_TOLERANCE:
        _log.error("the token-ratio checks disagree by %.1e, more than %.0e", worst, CHECK_TOLERANCE)
    return 0 if not mismatches and worst <= CHECK_TOLERANCE else 1


if __name__ == "__main__":
    sys.exit(main())
