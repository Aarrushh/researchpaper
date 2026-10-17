# PURPOSE: Exploratory, after data-freeze-v1: split the pooled CER of every condition of a sweep into its parts.
# WHAT GOES HERE: Per reader, text type, block and level: rows, pages, reference characters per row, the pooled CER and its substitution, deletion and insertion parts, the spaces the readers added and where they inserted them, the CER without spaces, and the hit_limit share.
# FILLED IN AT STEP: W4.1, for the numbers of R09 to R11 that the preregistered tables do not hold (LAB_NOTEBOOK.md, 2026-10-17).
# DO NOT: Test anything here or feed these rows into a preregistered table; every row is status exploratory. Never write under results/final/.
"""Step W4.1: the exploratory per-level table of a sweep.

    python scripts/21_levels.py [--config configs/sweeps/full.yaml] [--runs-root results/final/runs] [--out-dir reports/tables/exploratory]

The rows are those scripts/20_analyze.py analyses: ``sweep.load_results``, then the exclusion of section 8
(``sweep.complete_rows``). A level is one condition of one block for one reader and text type: its render mode, line
pitch, anti-aliasing mode, font size and phase. With N = H + S + D the reference characters of the level's rows, and
every count pooled over the rows (D13), never averaged over pages:

    cer                         (S + D + I) / N from the counts in results.csv, as every preregistered table pools them
    substitution_rate           S / N; deletion_rate D / N; insertion_rate I / N; the three add up to cer
    added_spaces_per_char       (spaces in the transcriptions - spaces in the references) / N, both normalised as
                                for scoring (text_source.normalise_text); negative when the reader dropped more
                                spaces than it added
    spaces_inserted_in_words_per_char
                                the spaces the alignment inserts between two characters of one word of the
                                reference (a word is a run between spaces, R03), per reference character
    inserted_beside_wide_share  the share of those spaces next to a wide character: one whose linear advance at the
                                level's font size exceeds the reader's patch, so that it takes two slots in the
                                patch-aware layout (R03)
    gaps_beside_wide_share      the share of the gaps between two characters of one word that are next to a wide
                                character: the share the inserted spaces would have if they fell anywhere in a word
    cer_without_spaces          the pooled CER of the same strings with every space removed: the errors on every
                                character but the space
    hit_limit_share             the share of rows whose output stopped at max_new_tokens
    ref_chars_per_row           N / rows: how much text a page of the level holds

Each row's counts are recomputed from its reference and transcription as ``score.char_counts`` computes them (both
normalised, then jiwer's character alignment), and a difference from results.csv stops the script: the table
describes the frozen rows and nothing else. Writes ``<out-dir>/levels.csv`` after a manifest. Every row has status
``exploratory``: no test is run and no p-value is written.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import jiwer
import yaml

from mrr import render
from mrr.manifest import new_run_id, write_manifest
from mrr.score import CharCounts
from mrr.sweep import RunFolder, complete_rows, load_results, sweep_run_dirs
from mrr.text_source import normalise_text

KEY = ("reader", "text_type", "block", "render_mode", "line_pitch_mode", "aa_mode", "font_size_px", "phase_x_px",
       "phase_y_px")
NUMERIC_KEYS = ("font_size_px", "phase_x_px", "phase_y_px")
COLUMNS = (*KEY, "rows", "pages", "ref_chars_per_row", "cer", "substitution_rate", "deletion_rate", "insertion_rate",
           "added_spaces_per_char", "spaces_inserted_in_words_per_char", "inserted_beside_wide_share",
           "gaps_beside_wide_share", "cer_without_spaces", "hit_limit_share", "status", "family")
FLOAT_DIGITS = 12  # significant digits written, as scripts/20_analyze.py writes its tables
NAN = float("nan")

Wide = Callable[[str, float, int], bool]  # (character, font size in px, patch in px) -> wider than the patch

_log = logging.getLogger("levels")


def _spaces_in_words(ref: str, hyp: str, chunks, is_wide: Callable[[str], bool]) -> tuple[int, int]:
    """Spaces inserted between two characters of one word of ``ref``, and how many of them are next to a wide one."""
    inside = beside_wide = 0
    for chunk in chunks:
        if chunk.type != "insert":
            continue
        n = hyp[chunk.hyp_start_idx:chunk.hyp_end_idx].count(" ")
        i = chunk.ref_start_idx  # the insertion stands before ref[i]
        if n and 0 < i < len(ref) and ref[i - 1] != " " and ref[i] != " ":
            inside += n
            beside_wide += n * (is_wide(ref[i - 1]) or is_wide(ref[i]))
    return inside, beside_wide


def level_rows(records: Iterable[Mapping[str, Any]], wide: Wide | None = None) -> list[dict[str, Any]]:
    """One row per level of ``records``, sorted by KEY.

    Each record holds the KEY columns, ``page_id``, ``patch_px``, ``hits``, ``substitutions``, ``deletions``,
    ``insertions``, ``hit_limit`` and the raw ``reference`` and ``transcription`` texts. ``wide`` says whether a
    character is wider than the patch at a font size; without it the two wide shares are NaN. Raises ValueError when a
    record's counts are not the ones its texts give.
    """
    sums: dict[tuple, dict[str, Any]] = {}
    for rec in records:
        stored = CharCounts(int(rec["hits"]), int(rec["substitutions"]), int(rec["deletions"]), int(rec["insertions"]))
        ref, hyp = normalise_text(rec["reference"]), normalise_text(rec["transcription"])
        if not ref:
            raise ValueError(f"{rec['reader']} {rec['page_id']}: the normalised reference is empty")
        aligned = jiwer.process_characters(ref, hyp)  # what score.char_counts runs on the same normalised strings
        counts = CharCounts(aligned.hits, aligned.substitutions, aligned.deletions, aligned.insertions)
        if counts != stored:
            raise ValueError(f"{rec['reader']} {rec['page_id']} block {rec['block']}: results.csv holds {stored}, "
                             f"its texts give {counts}")
        letters = jiwer.process_characters(ref.replace(" ", ""), hyp.replace(" ", ""))
        size, patch = float(rec["font_size_px"]), int(rec["patch_px"])

        def is_wide(char: str) -> bool:
            return bool(wide and wide(char, size, patch))

        inside, beside_wide = _spaces_in_words(ref, hyp, aligned.alignments[0], is_wide)
        gaps = [(a, b) for a, b in zip(ref, ref[1:]) if a != " " and b != " "]
        key = tuple(float(rec[k]) if k in NUMERIC_KEYS else str(rec[k]) for k in KEY)
        s = sums.setdefault(key, {"rows": 0, "pages": set(), "h": 0, "s": 0, "d": 0, "i": 0, "added": 0,
                                  "inside": 0, "beside_wide": 0, "gaps": 0, "gaps_wide": 0,
                                  "lh": 0, "ls": 0, "ld": 0, "li": 0, "hit": 0})
        s["rows"] += 1
        s["pages"].add(str(rec["page_id"]))
        s["h"] += stored.hits
        s["s"] += stored.substitutions
        s["d"] += stored.deletions
        s["i"] += stored.insertions
        s["added"] += hyp.count(" ") - ref.count(" ")
        s["inside"] += inside
        s["beside_wide"] += beside_wide
        s["gaps"] += len(gaps)
        s["gaps_wide"] += sum(is_wide(a) or is_wide(b) for a, b in gaps)
        s["lh"] += letters.hits
        s["ls"] += letters.substitutions
        s["ld"] += letters.deletions
        s["li"] += letters.insertions
        s["hit"] += bool(rec["hit_limit"])
    out = []
    for key in sorted(sums):
        s = sums[key]
        n = s["h"] + s["s"] + s["d"]
        n_letters = s["lh"] + s["ls"] + s["ld"]
        out.append({
            **dict(zip(KEY, key)), "rows": s["rows"], "pages": len(s["pages"]), "ref_chars_per_row": n / s["rows"],
            "cer": (s["s"] + s["d"] + s["i"]) / n, "substitution_rate": s["s"] / n, "deletion_rate": s["d"] / n,
            "insertion_rate": s["i"] / n, "added_spaces_per_char": s["added"] / n,
            "spaces_inserted_in_words_per_char": s["inside"] / n,
            "inserted_beside_wide_share": s["beside_wide"] / s["inside"] if wide and s["inside"] else NAN,
            "gaps_beside_wide_share": s["gaps_wide"] / s["gaps"] if wide and s["gaps"] else NAN,
            "cer_without_spaces": (s["ls"] + s["ld"] + s["li"]) / n_letters, "hit_limit_share": s["hit"] / s["rows"],
            "status": "exploratory", "family": "description",
        })
    return out


def font_wide(font_path: str) -> Wide:
    """``wide`` for ``level_rows`` from the font: the character's linear advance (R01) exceeds the patch."""
    cache: dict[tuple[str, float, int], bool] = {}

    def wide(char: str, size: float, patch: int) -> bool:
        key = (char, size, patch)
        if key not in cache:
            cache[key] = render.load_glyph(font_path, size, char).advance_px > patch
        return cache[key]

    return wide


def records(df, patches: Mapping[str, int]) -> Iterator[dict[str, Any]]:
    """The records ``level_rows`` reads, one per row of ``df``, with the texts from the row's run folder and the patch
    of its reader (``patches``, by run folder path)."""
    references: dict[tuple[str, str], str] = {}
    for row in df.itertuples(index=False):
        run_dir = Path(row.run_dir)
        where = (row.run_dir, row.page_id)
        if where not in references:
            references[where] = (run_dir / "references" / f"{row.page_id}.txt").read_text(encoding="utf-8")
        yield {
            **{k: getattr(row, k) for k in KEY}, "page_id": row.page_id, "patch_px": patches[row.run_dir],
            "hits": row.hits, "substitutions": row.substitutions, "deletions": row.deletions,
            "insertions": row.insertions, "hit_limit": bool(row.hit_limit), "reference": references[where],
            "transcription": (run_dir / row.transcription_file).read_text(encoding="utf-8"),
        }


def _cell(v: Any) -> Any:
    """NaN as an empty cell; any other float to FLOAT_DIGITS significant digits."""
    if isinstance(v, float):
        return "" if math.isnan(v) else float(f"{v:.{FLOAT_DIGITS}g}")
    return v


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/sweeps/full.yaml")
    p.add_argument("--runs-root", default="results/final/runs", help="the frozen run folders (default results/final/runs)")
    p.add_argument("--out-dir", default="reports/tables/exploratory")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    sweep_path, runs_root, out = Path(args.config), Path(args.runs_root), Path(args.out_dir)
    font_path = str(yaml.safe_load(sweep_path.read_text(encoding="utf-8"))["font_file"])
    run_dirs = sweep_run_dirs(sweep_path, runs_root)
    if not run_dirs:
        _log.error("no run folder of sweep %s under %s", sweep_path.stem, runs_root)
        return 1
    folders = [RunFolder(d) for d in run_dirs]
    df_all = load_results(sweep_path, runs_root)  # raises on a repeated key or an edited sweep file
    df, excluded = complete_rows(df_all, folders)  # the rows scripts/20_analyze.py analyses
    patches = {str(f.run_dir): int(f.reader_cfg["patch_size"]) for f in folders}
    rows = level_rows(records(df, patches), font_wide(font_path))
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out, {
        "sweep_file": sweep_path.as_posix(), "sweep_name": sweep_path.stem, "runs_root": runs_root.as_posix(),
        "run_dirs": [d.name for d in run_dirs], "rows_loaded": int(len(df_all)), "rows_analysed": int(len(df)),
        "pages_excluded": {Path(k).name: sorted(v) for k, v in excluded.items() if v}, "font_file": font_path,
        "patch_px": {f.reader: int(f.reader_cfg["patch_size"]) for f in folders}, "levels": len(rows),
        "status": "exploratory",
    }, run_id=new_run_id(sweep_path.stem, "levels"))  # before the table, so that its git_dirty describes the code
    with open(out / "levels.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(v) for k, v in row.items()})
    _log.info("%d level(s) from %d row(s): %s", len(rows), len(df), out / "levels.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
