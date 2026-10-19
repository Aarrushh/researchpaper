# PURPOSE: Study 2's computations, shared by its analysis and its figures (PREREGISTRATION_2.md sections 3, 4, 6 and 9).
# WHAT GOES HERE: The condition codes of section 3 and their groups; page values per group (CER, letters-only CER, inserted spaces per character, WER); pooled values per condition with page-bootstrap intervals; the letters-only confusion counts; the capacity of each layout on the study's own text.
# FILLED IN AT STEP: W5.1 (18 October 2026).
# DO NOT: Pool Study 2 rows with Study 1, the ladder or the pilots (section 8). Average row CERs: pool the counts (D13). Read reports/tables/.
"""Study 2's shared computations.

The sweep (``configs/sweeps/study2.yaml``) writes four blocks: S (sans, conventional, phases 0, 5, 10), Q (sans,
patch_aware, phase 0, random text only), M (mono, patch_aware, phases 0, 3 and P / 2) and D (sans, conventional at the
density size, random text only). ``condition_code`` names each row as section 3 does (S0, S5, S10, Q0, M0, M3, M7 or
M8, S28) and ``group_of`` puts it in the groups section 6 compares: conventional (S pooled over its phases), contained
(M0 and M3 pooled), cut (M at P / 2), slot (Q0) and density (D). A page's value for a group pools the counts of every
row of that group on the page.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import jiwer
import numpy as np

from mrr import grid, render, stats, text_source
from mrr import sweep as sweep_mod
from mrr.text_source import normalise_text

VALIDATION_READERS = ("qwen7b", "internvl3_2b")  # section 2; the rest are the development readers of Study 1
CONVENTIONAL, CONTAINED, CUT, SLOT, DENSITY = "conventional", "contained", "cut", "slot", "density"
GROUPS = (CONVENTIONAL, SLOT, CONTAINED, CUT, DENSITY)
CONTAINED_PHASES = (0.0, 3.0)  # M0 and M3 (section 1)
MEASURES = ("cer", "letters", "spaces", "wer")
BLOCK_ORDER = ("S", "Q", "M", "D")
S_PHASE_CODES = frozenset({"S0", "S5", "S10"})
O0_PAIRS = (("O", "0"), ("0", "O"))
IL1 = "Il1"


def role(reader: str) -> str:
    """``validation`` for the readers of section 2 chosen before the study, ``development`` for Study 1's."""
    return "validation" if reader in VALIDATION_READERS else "development"


def condition_code(block: str, phase_x_px: float, font_size_px: float) -> str:
    """Section 3's code of a row: the block and its phase (S0, S5, S10, Q0, M0, M3, M7), block D by its size (S28)."""
    if block == "D":
        return f"S{float(font_size_px):g}"
    return f"{block}{float(phase_x_px):g}"


def group_of(block: str, phase_x_px: float, patch_px: int) -> str:
    """The group of section 6 a row belongs to; ValueError for a block or an M phase the design does not have."""
    if block == "S":
        return CONVENTIONAL
    if block == "Q":
        return SLOT
    if block == "D":
        return DENSITY
    if block == "M":
        if float(phase_x_px) == patch_px / 2:
            return CUT
        if float(phase_x_px) in CONTAINED_PHASES:
            return CONTAINED
    raise ValueError(f"block {block} at phase {phase_x_px} px is not a condition of Study 2 (P = {patch_px})")


def with_codes(df, patch_by_reader: Mapping[str, int]):
    """A copy of ``df`` with ``condition`` and ``group`` columns, the patch of each row's reader from the run."""
    out = df.copy()
    out["condition"] = [condition_code(b, x, s) for b, x, s in
                        zip(out["block"], out["phase_x_px"], out["font_size_px"])]
    out["group"] = [group_of(b, x, int(patch_by_reader[r])) for b, x, r in
                    zip(out["block"], out["phase_x_px"], out["reader"])]
    return out


def condition_order(codes: Iterable[str]) -> list[str]:
    """Codes in the order of section 3's table: S by phase, Q0, M by phase, then the density size."""
    def key(code: str) -> tuple[int, float]:
        block = code[0] if code[0] != "S" or code in S_PHASE_CODES else "D"
        return BLOCK_ORDER.index(block), float(code[1:])
    return sorted(set(codes), key=key)


def _sums(rows, measure: str) -> tuple[np.ndarray, np.ndarray]:
    """Per row the numerator and denominator a pooled value needs: errors and reference characters (D13)."""
    def col(name: str) -> np.ndarray:
        return rows[name].to_numpy(dtype=np.float64)

    if measure == "cer":
        return (col("substitutions") + col("deletions") + col("insertions"),
                col("hits") + col("substitutions") + col("deletions"))
    if measure == "letters":
        return (col("substitutions_ns") + col("deletions_ns") + col("insertions_ns"),
                col("hits_ns") + col("substitutions_ns") + col("deletions_ns"))
    if measure == "spaces":  # inserted spaces per character of the normalised reference (H + S + D)
        return col("spaces_inserted"), col("hits") + col("substitutions") + col("deletions")
    raise ValueError(f"no pooled form for measure {measure!r}; one of cer, letters, spaces")


def page_values(rows, measure: str) -> dict[str, float]:
    """Each page's value over ``rows``: the pooled ratio for cer, letters and spaces, the mean row WER for wer."""
    out: dict[str, float] = {}
    for page_id, group in rows.groupby("page_id", sort=True):
        if measure == "wer":
            out[str(page_id)] = float(group["wer"].mean())
            continue
        num, den = _sums(group, measure)
        out[str(page_id)] = float(num.sum() / den.sum()) if den.sum() > 0 else math.nan
    return out


def side_rows(sub, side: str):
    """The rows of one side of a contrast: a group name (contained, cut, ...) or a condition code (M0, M3, ...)."""
    if side in GROUPS:
        return sub[sub["group"] == side]
    return sub[sub["condition"] == side]


def paired_pages(sub, baseline: str, treatment: str, measure: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The pages both sides read, with the baseline's and the treatment's page values, pages in sorted order."""
    a, b = page_values(side_rows(sub, baseline), measure), page_values(side_rows(sub, treatment), measure)
    pages = sorted(p for p in set(a) & set(b) if not (math.isnan(a[p]) or math.isnan(b[p])))
    return pages, np.array([a[p] for p in pages]), np.array([b[p] for p in pages])


def condition_bands(df, measure: str, n_resamples: int, seed: int, alpha: float) -> list[dict[str, Any]]:
    """Per reader, text type and condition: the pooled value with its page-bootstrap interval (one draw of pages
    serves every condition of a reader and text type, as ``stats.pooled_cer_by_level`` does for levels)."""
    out = []
    for (reader, text_type), sub in df.groupby(["reader", "text_type"], sort=True):
        codes = condition_order(sub["condition"])
        index = {code: i for i, code in enumerate(codes)}
        num, den = _sums(sub, measure)
        band = stats.pooled_cer_by_level(num, den, sub["page_id"].to_numpy(),
                                         np.array([index[c] for c in sub["condition"]], dtype=np.float64),
                                         n_resamples, seed, alpha)
        for i, code in enumerate(codes):
            rows = sub[sub["condition"] == code]
            first = rows.iloc[0]
            row_num, row_den = _sums(rows, measure)
            out.append({
                "reader": str(reader), "role": role(str(reader)), "text_type": str(text_type), "condition": code,
                "group": str(first["group"]), "block": str(first["block"]), "phase_x_px": float(first["phase_x_px"]),
                "font_file": str(first["font_file"]), "font_size_px": float(first["font_size_px"]),
                "render_mode": str(first["render_mode"]), "n_pages": int(rows["page_id"].nunique()),
                "n_rows": int(len(rows)), "value": float(band["cer"][i]), "low": float(band["low"][i]),
                "high": float(band["high"][i]), "numerator": float(row_num.sum()), "denominator": float(row_den.sum()),
            })
    return out


def letter_confusions(reference: str, hypothesis: str) -> tuple[Counter, Counter, int]:
    """Substitutions by (reference, hypothesis) character and deletions by character, on the letters-only strings.

    Both strings are normalised and stripped of spaces, as ``score.char_counts_letters_only``, so a misread space
    never counts as a letter confusion. Returns (substitutions, deletions, reference letters).
    """
    ref, hyp = normalise_text(reference).replace(" ", ""), normalise_text(hypothesis).replace(" ", "")
    if not ref:
        raise ValueError("letter_confusions: the normalised reference has no letters")
    subs: Counter = Counter()
    dels: Counter = Counter()
    for chunk in jiwer.process_characters(ref, hyp).alignments[0]:
        if chunk.type == "substitute":
            for r, h in zip(ref[chunk.ref_start_idx:chunk.ref_end_idx], hyp[chunk.hyp_start_idx:chunk.hyp_end_idx]):
                subs[(r, h)] += 1
        elif chunk.type == "delete":
            dels.update(ref[chunk.ref_start_idx:chunk.ref_end_idx])
    return subs, dels, len(ref)


def confusion_totals(subs: Mapping[tuple[str, str], int]) -> dict[str, int]:
    """The O/0, I/l/1 and case-pair (c/C, k/K, ...) substitution totals of section 9, item 4."""
    return {
        "O/0": sum(subs.get(pair, 0) for pair in O0_PAIRS),
        "I/l/1": sum(n for (r, h), n in subs.items() if r in IL1 and h in IL1 and r != h),
        "case_pair": sum(n for (r, h), n in subs.items() if r.isalpha() and r != h and r.lower() == h.lower()),
    }


def run_confusions(df, folders_by_path: Mapping[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Per (reader, text type, condition): the substitution and deletion counts over every row's stored texts."""
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        folder = folders_by_path[str(row.run_dir)]
        reference = (folder.run_dir / "references" / f"{row.page_id}.txt").read_text(encoding="utf-8")
        hypothesis = (folder.run_dir / str(row.transcription_file)).read_text(encoding="utf-8")
        subs, dels, letters = letter_confusions(reference, hypothesis)
        cell = out.setdefault((str(row.reader), str(row.text_type), str(row.condition)),
                              {"subs": Counter(), "dels": Counter(), "letters": 0, "rows": 0,
                               "font_file": str(row.font_file), "render_mode": str(row.render_mode)})
        cell["subs"].update(subs)
        cell["dels"].update(dels)
        cell["letters"] += letters
        cell["rows"] += 1
        cell["group"] = str(row.group)
    return out


def pair_rates(cells: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per (reader, text type, group) and pair kind: the confusions of ``confusion_totals`` pooled over the group's
    conditions, the reference letters, and the rate per 1,000 letters (Figure 4)."""
    pooled: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (reader, text_type, _), cell in cells.items():
        entry = pooled.setdefault((reader, text_type, cell["group"]), {"letters": 0, "counts": Counter()})
        entry["letters"] += cell["letters"]
        entry["counts"].update(confusion_totals(cell["subs"]))
    out = []
    for (reader, text_type, group), entry in sorted(pooled.items()):
        for kind in ("O/0", "I/l/1", "case_pair"):
            count = int(entry["counts"][kind])
            out.append({"reader": reader, "text_type": text_type, "group": group, "kind": kind, "count": count,
                        "letters": int(entry["letters"]), "per_1000": 1000.0 * count / entry["letters"]})
    return out


def capacity(folder: Any, plan_pages: Sequence[Any], render_cfg: Mapping[str, Any], root: Path,
             max_pages: int = 20) -> list[dict[str, Any]]:
    """Section 9, item 5: each shared layout of the run alone, on the study's own text of each type it reads.

    The text is the run's planned pages of that type, joined in plan order; it is paginated under the one layout
    (at phase 0, with the run's allowances) for up to ``max_pages`` pages, and a last page the text ran out on is
    dropped. Glyphs are placed characters, spaces not.
    """
    reader_cfg = folder.reader_cfg
    shared = [c for c in folder.conditions if c.paginate == sweep_mod.SHARED]
    allowances = sweep_mod.shared_allowances(folder.sweep, folder.reader, shared)
    font_paths = sweep_mod._font_paths(folder.sweep, root)
    width, height = grid.page_size_px(reader_cfg)
    tokens = grid.token_count(reader_cfg, height, width)
    out = []
    seen = set()
    for c in shared:
        for kind in c.text_types:
            key = (sweep_mod._layout_id(c), kind)
            if key in seen:
                continue
            seen.add(key)
            pages_of_kind = [p for p in plan_pages if p.text_type == kind]
            if not pages_of_kind:
                continue
            cfg = sweep_mod._layout_cfg(c, reader_cfg, render_cfg, font_paths, allowances)
            stream = " ".join(p.text for p in pages_of_kind)
            texts = text_source.paginate(stream, lambda t, cfg=cfg: render.fits_page(t, cfg), max_pages)
            if len(texts) > 1 and sum(len(t.split()) for t in texts) == len(stream.split()):
                texts = texts[:-1]  # the text ran out on the last page, which is not at capacity
            glyphs = [len(t.replace(" ", "")) for t in texts]
            lines = [max(g.line for g in render.layout_page(t, cfg)) + 1 for t in texts]
            chars = float(np.mean([len(t) for t in texts]))
            out.append({
                "reader": folder.reader, "role": role(folder.reader), "text_type": kind,
                "layout": f"{c.render_mode} {Path(c.font_file).stem} {c.font_size_px:g} px",
                "render_mode": c.render_mode, "font_file": c.font_file, "font_size_px": c.font_size_px,
                "line_pitch_px": c.line_pitch_px, "n_pages": len(texts), "chars_per_page": chars,
                "glyphs_per_page": float(np.mean(glyphs)), "lines_per_page": float(np.mean(lines)),
                "glyphs_per_line": float(np.sum(glyphs) / np.sum(lines)), "vision_tokens": tokens,
                "chars_per_token": chars / tokens,
                "shared_chars_per_page": float(np.mean([len(p.text) for p in pages_of_kind])),
            })
    return out
