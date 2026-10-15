# PURPOSE: One Markdown digest of a sweep's analysis tables, with the preregistered verdicts applied mechanically.
# WHAT GOES HERE: Reading reports/tables/<sweep>/*.csv, the checks, the tests by family with their minimum detectable effects, the verdicts of PREREGISTRATION.md section 10, the sizes and coherence summaries, each number beside its file and column.
# FILLED IN AT STEP: W3.4, under D26 (the digest the interpretation and the number audit start from).
# DO NOT: Compute any statistic here; every number is read from a table that scripts/20_analyze.py wrote. Do not word an interpretation: the verdict lines restate section 10's criteria and nothing else.
"""Step W3.4: summarise a sweep's tables in one file.

    python scripts/50_summary.py --tables reports/tables/full [--out reports/summary_full.md]

Reads ``checks.csv``, ``harmonics.csv``, ``paired.csv``, ``subpixel.csv``, ``sizes.csv`` and ``coherence_summary.csv``
from the table folder and writes a Markdown file with one section per table. Every number is copied from its table
(rounded for display only; the table holds the full value), and the section names the file it came from. The verdict
lines apply section 10 of PREREGISTRATION.md as written: clause (a) holds for a reader when a group of its family
``a_random62`` has a Holm-adjusted p below 0.05; clause (b) names those groups' periods against the patch and the
cell; clause (c) holds when ``H_phi0_vs_C_pooled`` on random62 pages has a Holm-adjusted p below 0.05 and a negative
median difference; a significant group of the control block C with no significant V or H group of either text type (families a_random62 and a_natural) for the same reader contradicts D10. Rows whose
``status`` is ``exploratory`` are listed apart.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

ALPHA = 0.05  # PREREGISTRATION.md section 6
S_TARGET = 0.01  # section 5: the smallest RMS amplitude worth reporting
PATCH = {"qwen": 14, "smolvlm": 14, "smolvlm500m": 16}  # section 2
CELL = {"qwen": 28, "smolvlm": 42, "smolvlm500m": 64}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tables", required=True, help="reports/tables/<sweep_name>")
    p.add_argument("--out", default=None, help="default: reports/summary_<sweep_name>.md beside reports/tables")
    return p.parse_args(argv)


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def num(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: str, digits: int = 3) -> str:
    """A table cell for display: an empty cell is a dash, a p-value below 0.001 is written as such."""
    x = num(value)
    if math.isnan(x):
        return "–"
    if abs(x) >= 1000 or x == int(x) and abs(x) >= 1:
        return f"{x:.0f}" if x == int(x) else f"{x:.1f}"
    return f"{x:.{digits}f}"


def fmt_p(value: str) -> str:
    x = num(value)
    if math.isnan(x):
        return "–"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def significant(row: Mapping[str, str]) -> bool:
    return num(row.get("p_holm", "")) < ALPHA


def verdicts(harmonics: list[dict[str, str]], paired: list[dict[str, str]]) -> list[str]:
    """Section 10 of the preregistration, applied as written, one line per reader."""
    lines = []
    for reader in sorted({r["reader"] for r in harmonics} | {r["reader"] for r in paired}):
        prim = [r for r in harmonics if r["reader"] == reader and r["family"] == "a_random62"]
        sig = [r for r in prim if significant(r)]
        if not prim:
            a = "clause (a): no a_random62 test (no rows)"
        elif sig:
            a = "clause (a): SUPPORTED by " + ", ".join(f"{r['block']} {r['period_px']} px" for r in sig)
        else:
            a = "clause (a): not supported (no a_random62 group with Holm p < 0.05)"
        periods = sorted({int(num(r["period_px"])) for r in sig})
        kinds = ["patch" if p == PATCH.get(reader) else "cell" if p == CELL.get(reader) else f"{p} px" for p in periods]
        b = "clause (b): " + (", ".join(kinds) if kinds else "nothing to read (no significant group)")
        c_rows = [r for r in paired if r["reader"] == reader and r["comparison"] == "H_phi0_vs_C_pooled"
                  and r["text_type"] == "random62"]
        if not c_rows:
            c = "clause (c): no test (no rows)"
        else:
            r = c_rows[0]
            ok = significant(r) and num(r["median_diff"]) < 0
            c = ("clause (c): SUPPORTED" if ok else "clause (c): not supported") + (
                f" (median b - a {fmt(r['median_diff'])}, Holm p {fmt_p(r['p_holm'])}, n {r['n']})")
        # Section 10, literally: a significant C group with no significant V or H group for the same reader, on either
        # text type (the preregistered V and H groups are the families a_random62 and a_natural).
        control = [r for r in harmonics if r["reader"] == reader and r["family"] == "control_C" and significant(r)]
        coherent = [r for r in harmonics if r["reader"] == reader and r["family"] in ("a_random62", "a_natural")
                    and significant(r)]
        where_c = ", ".join(sorted({r["text_type"] for r in control}))
        where_vh = ", ".join(sorted({r["text_type"] for r in coherent}))
        if control and not coherent:
            d10 = f"control C: CONTRADICTS D10 (a significant C group on {where_c}, none in V or H on either text type)"
        elif control:
            d10 = f"control C: significant group(s) on {where_c}, beside significant V or H on {where_vh}"
        else:
            d10 = "control C: no significant group"
        lines.append(f"- **{reader}**: {a}; {b}; {c}; {d10}.")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    folder = Path(args.tables)
    name = folder.name
    out = Path(args.out) if args.out else folder.parent.parent / f"summary_{name}.md"
    checks, harmonics, paired = read(folder / "checks.csv"), read(folder / "harmonics.csv"), read(folder / "paired.csv")
    subpixel, sizes, coherence = read(folder / "subpixel.csv"), read(folder / "sizes.csv"), read(folder / "coherence_summary.csv")
    if not checks:
        print(f"no checks.csv in {folder}: run scripts/20_analyze.py first", file=sys.stderr)
        return 1
    lines = [f"# Summary of `{folder.as_posix()}`", "",
             "Generated by `scripts/50_summary.py` from the CSV files named in each section; every number is copied "
             "from its table and only rounded for display. The verdict lines restate PREREGISTRATION.md section 10.",
             "", "## Verdicts (section 10, applied mechanically)", ""]
    lines += verdicts(harmonics, paired) or ["- no tests"]
    lines += ["", "## Checks (`checks.csv`)", ""]
    lines += table(["run", "reader", "rows done", "pages seen", "excluded", "hit_limit share", "determinism pairs",
                    "expected", "mismatches", "missing / empty transcriptions", "malformed coherence rows",
                    "median output tokens", "median latency s"],
                   [[r["run_dir"], r["reader"], r["rows_done"], r["pages_seen"], r["pages_excluded"],
                     fmt(r["hit_limit_share"]), r["determinism_pairs"], r.get("determinism_expected", "–"),
                     r["determinism_mismatches"],
                     f"{r['missing_transcriptions']} / {r.get('empty_transcriptions', '–')}",
                     r.get("coherence_malformed", "–"), fmt(r.get("median_output_tokens", ""), 0),
                     fmt(r["median_latency_s"], 1)] for r in checks])
    for family_title, keep in (("Clause (a), random62 (`harmonics.csv`, family a_random62)", "a_random62"),
                               ("Clause (a), natural (`harmonics.csv`, family a_natural)", "a_natural"),
                               ("Control block C (`harmonics.csv`, family control_C)", "control_C")):
        rows = [r for r in harmonics if r["family"] == keep]
        lines += ["", f"## {family_title}", ""]
        lines += table(["reader", "text", "block", "period px", "q", "F", "df", "p", "Holm p", "RMS [95% CI]",
                        "s_g (80%)", "share", "sigma", "pages", "levels"],
                       [[r["reader"], r["text_type"], r["block"], r["period_px"], r["q"], fmt(r["F"], 2),
                         f"{r['df1']}, {r['df2']}", fmt_p(r["p"]), fmt_p(r["p_holm"]),
                         f"{fmt(r['rms'], 4)} [{fmt(r['rms_low'], 4)}, {fmt(r['rms_high'], 4)}]",
                         fmt(r.get("mde_rms", ""), 4), fmt(r["share"]), fmt(r["sigma"], 4), r["n_pages"],
                         r["n_levels"]] for r in rows]) if rows else ["(none)"]
    other = [r for r in harmonics if r["family"] not in ("a_random62", "a_natural", "control_C")]
    if other:
        lines += ["", "## Other harmonic groups (exploratory)", ""]
        lines += table(["reader", "text", "block", "period px", "F", "p", "RMS", "status", "family"],
                       [[r["reader"], r["text_type"], r["block"], r["period_px"], fmt(r["F"], 2), fmt_p(r["p"]),
                         fmt(r["rms"], 4), r["status"], r["family"]] for r in other])
    lines += ["", "## Paired tests (`paired.csv`): clause (c) and block A", "",
              "a = conventional (C pooled over its levels, or C at phase 0 for block A); b = the compared condition; "
              "median b - a < 0 means b has the lower CER.", ""]
    lines += table(["reader", "text", "comparison", "n", "zeros", "median b - a [95% CI]", "p", "Holm p", "method",
                    "MDE (80%)", "family"],
                   [[r["reader"], r["text_type"], r["comparison"], r["n"], fmt(r["n_zero"]),
                     f"{fmt(r['median_diff'], 4)} [{fmt(r['diff_low'], 4)}, {fmt(r['diff_high'], 4)}]",
                     fmt_p(r["p"]), fmt_p(r["p_holm"]), r["method"], fmt(r.get("mde_median_diff", ""), 4),
                     r["family"]] for r in paired]) if paired else ["(none)"]
    lines += ["", "## Block S, sub-pixel phase (`subpixel.csv`)", ""]
    if subpixel:
        cer_cols = [c for c in subpixel[0] if c.startswith("cer_at_")]
        lines += table(["reader", "text", "F", "df", "p", *[c.replace("cer_at_", "CER at ") for c in cer_cols], "pages"],
                       [[r["reader"], r["text_type"], fmt(r["F"], 2), f"{r['df1']}, {r['df2']}", fmt_p(r["p"]),
                         *[fmt(r[c], 4) for c in cer_cols], r["n_pages"]] for r in subpixel])
    else:
        lines += ["(none)"]
    lines += ["", "## Block Z, sizes (`sizes.csv`; description, no test)", ""]
    if sizes:
        lines += table(["reader", "text", "mode", "size px", "CER [95% CI]", "chars/token", "plateau", "s* px", "pages"],
                       [[r["reader"], r["text_type"], r["render_mode"], fmt(r["font_size_px"]),
                         f"{fmt(r['cer'])} [{fmt(r['cer_low'])}, {fmt(r['cer_high'])}]", fmt(r["chars_per_token"], 2),
                         fmt(r["cer_plateau"]), fmt(r["s_star_px"]), r["n_pages"]] for r in sizes])
    else:
        lines += ["(none)"]
    lines += ["", "## Coherence (`coherence_summary.csv`), harmonic 1", ""]
    first = [r for r in coherence if r.get("harmonic") == "1"]
    lines += table(["reader", "text", "block", "axis", "period px", "R mean", "R min", "R max", "RMS of the group",
                    "pages"],
                   [[r["reader"], r["text_type"], r["block"], r["axis"], r["period_px"], fmt(r["R_mean"]),
                     fmt(r["R_min"]), fmt(r["R_max"]), fmt(r["rms"], 4), r["n_pages"]] for r in first]) if first else ["(none)"]
    pages, density = read(folder / "pages.csv"), read(folder / "density.csv")
    lines += ["", "## Pages (`pages.csv`)", ""]
    if pages:
        keys = sorted({(r["reader"], r["text_type"], r["pagination"] == "shared") for r in pages})
        rows_out = []
        for reader, text_type, shared in keys:
            sel = [r for r in pages if r["reader"] == reader and r["text_type"] == text_type
                   and (r["pagination"] == "shared") == shared]
            complete = sum(r["complete"] == "True" for r in sel)
            matches = sum(r["reference_matches_plan"] == "True" for r in sel)
            cpt = [num(r["chars_per_token"]) for r in sel]
            rows_out.append([reader, text_type, "shared" if shared else "block Z", len(sel), complete,
                             f"{matches} of {len(sel)}", fmt(str(sum(cpt) / len(cpt)), 2)])
        lines += table(["reader", "text", "pagination", "pages seen", "complete", "reference = plan",
                        "mean chars/token"], rows_out)
    else:
        lines += ["(none)"]
    lines += ["", "## Density of the shared layouts (`density.csv`)", ""]
    lines += table(["reader", "text", "mode", "line pitch", "pages", "glyphs/page", "lines/page", "glyphs/line",
                    "glyphs/full line (density cost)"],
                   [[r["reader"], r["text_type"], r["render_mode"], r["line_pitch_mode"], r["n_pages"],
                     fmt(r["glyphs_per_page"], 1), fmt(r["lines_per_page"], 2), fmt(r["glyphs_per_line"], 2),
                     fmt(r.get("glyphs_per_full_line", ""), 2)]
                    for r in density]) if density else ["(none)"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
