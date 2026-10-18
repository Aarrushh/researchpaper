# PURPOSE: Study 2 pre-run step 7.1: screen candidate monospace faces by the measured criteria of PREREGISTRATION_2.md and run the declared 10-page pre-test on one reader.
# WHAT GOES HERE: The screening measures per face at the size where its advance equals one patch; the pre-test's confusion counts per face; both as CSV with a manifest under reports/tables/study2_prerun/.
# FILLED IN AT STEP: W5.1.
# DO NOT: Pool the pre-test pages with Study 2 (they are excluded by section 8). Do not choose the face by anything but the criteria and counts written here.
"""Study 2, pre-run step 7.1: font screening and the font pre-test.

    python scripts/23_font_pretest.py --fonts a.ttf b.ttf ... [--patch 14] [--out reports/tables/study2_prerun]
    python scripts/23_font_pretest.py --fonts ... --pretest [--reader qwen] [--pages 10] [--seed 20261022]

Screening (section 7.1 (a) to (d)), at the size where the face's advance equals the patch P:
  (a) every one of the 62 alphanumerics has its ink inside its own cell (bitmap left >= 0 and left + width <= P)
      and at least MIN_INK px wide;
  (b) the x-height is 11 to 13 pixel rows;
  (c) the O and 0 bitmaps differ in at least SHAPE_DIFF of their pixels after centring, and 0 carries a mark
      (ink inside its counter);
  (d) I, l and 1 are pairwise distinct by the same measure.
Pre-test: the faces that pass render the same PAGES pages of random characters (SEED) in the slot layout at phase 0;
the reader transcribes them; the O/0 and I/l/1 confusions and the letter errors (spaces removed) are counted. The
face with the fewest O/0 + I/l/1 confusions is chosen, ties broken by the fewest letter errors; the choice is written
into PREREGISTRATION_2.md section 11 by hand, with the page hashes this script records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from mrr import grid, render, text_source
from mrr.manifest import new_run_id, write_manifest
from mrr.render import load_glyph
from mrr.text_source import normalise_text

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
MIN_INK = 7          # px, criterion (a)
X_ROWS = (11, 13)    # criterion (b)
SHAPE_DIFF = 0.15    # criteria (c) and (d)
SCREEN_COLUMNS = ("font", "file", "version", "advance_per_em", "size_px", "advance_px", "x_rows", "cap_rows", "min_ink",
                  "max_overhang", "not_contained", "o0_diff", "zero_marked", "il1_min_diff", "pass_a", "pass_b", "pass_c",
                  "pass_d", "passes")
PRETEST_COLUMNS = ("font", "pages", "letters", "o0_confusions", "il1_confusions", "letter_errors", "letters_only_cer",
                   "page_hashes")


def font_version(path: str) -> str:
    """The face's version string (name table id 5), or '' if the font has none."""
    import freetype

    face = freetype.Face(open(path, "rb"))
    for i in range(face.sfnt_name_count):
        n = face.get_sfnt_name(i)
        if n.name_id == 5:
            try:
                return n.string.decode("utf-16-be") if n.platform_id == 3 else n.string.decode("latin-1")
            except UnicodeDecodeError:
                return repr(n.string)
    return ""


def size_for_advance(path: str, patch: int) -> tuple[float, float]:
    """The largest font size (to 1/64 px) at which the face's advance is at most `patch` px, and that advance."""
    per_em = load_glyph(path, 100.0, "x").advance_px / 100.0
    size = patch / per_em
    for _ in range(256):
        adv = load_glyph(path, size, "x").advance_px
        if adv <= patch + 1e-9:
            return size, adv
        size -= 1 / 64
    raise ValueError(f"{path}: no size with advance <= {patch} px near {patch / per_em:.3f}")


def binary_bitmap(path: str, size: float, char: str) -> np.ndarray:
    return load_glyph(path, size, char).coverage >= 0.5


def shape_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Share of the union of two binarised bitmaps' ink where only one of them has ink, after centring them."""
    rows, cols = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])

    def centre(m):
        out = np.zeros((rows, cols), bool)
        r0, c0 = (rows - m.shape[0]) // 2, (cols - m.shape[1]) // 2
        out[r0:r0 + m.shape[0], c0:c0 + m.shape[1]] = m
        return out

    A, B = centre(a), centre(b)
    union = (A | B).sum()
    return float((A ^ B).sum() / union) if union else 0.0


def zero_marked(zero: np.ndarray) -> bool:
    """Whether the zero carries a slash or dot: any ink in the middle 30% of its rows and columns."""
    h, w = zero.shape
    inner = zero[int(0.35 * h):max(int(0.35 * h) + 1, int(0.65 * h)), int(0.35 * w):max(int(0.35 * w) + 1, int(0.65 * w))]
    return bool(inner.any())


def screen(path: str, patch: int) -> dict[str, Any]:
    """The measures and verdicts of section 7.1 (a) to (d) for one face."""
    size, adv = size_for_advance(path, patch)
    per_em = load_glyph(path, 100.0, "x").advance_px / 100.0
    inks, overhangs, not_contained = [], [], []
    for c in CHARS:
        g = load_glyph(path, size, c)
        w = g.coverage.shape[1]
        inks.append(w)
        over = max(0, -g.left, g.left + w - patch)
        overhangs.append(over)
        if over > 0:
            not_contained.append(c)
    x_rows = load_glyph(path, size, "x").coverage.shape[0]
    cap_rows = load_glyph(path, size, "H").coverage.shape[0]
    o0 = shape_diff(binary_bitmap(path, size, "O"), binary_bitmap(path, size, "0"))
    marked = zero_marked(binary_bitmap(path, size, "0"))
    il1 = min(shape_diff(binary_bitmap(path, size, a), binary_bitmap(path, size, b)) for a, b in (("I", "l"), ("I", "1"), ("l", "1")))
    pass_a = not not_contained and min(inks) >= MIN_INK
    pass_b = X_ROWS[0] <= x_rows <= X_ROWS[1]
    pass_c = o0 >= SHAPE_DIFF and marked
    pass_d = il1 >= SHAPE_DIFF
    return {
        "font": Path(path).stem, "file": Path(path).name, "version": font_version(path), "advance_per_em": per_em,
        "size_px": size, "advance_px": adv, "x_rows": x_rows, "cap_rows": cap_rows, "min_ink": min(inks),
        "max_overhang": max(overhangs), "not_contained": "".join(not_contained), "o0_diff": o0, "zero_marked": marked,
        "il1_min_diff": il1, "pass_a": pass_a, "pass_b": pass_b, "pass_c": pass_c, "pass_d": pass_d,
        "passes": pass_a and pass_b and pass_c and pass_d,
    }


def confusions(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """(O/0 confusions, I/l/1 confusions, letter errors S + D + I, reference letters) on space-stripped strings."""
    import jiwer

    r, h = normalise_text(ref).replace(" ", ""), normalise_text(hyp).replace(" ", "")
    out = jiwer.process_characters(r, h)
    o0 = il1 = 0
    for ch in out.alignments[0]:
        if ch.type == "substitute":
            for a, b in zip(r[ch.ref_start_idx:ch.ref_end_idx], h[ch.hyp_start_idx:ch.hyp_end_idx]):
                if {a, b} == {"O", "0"}:
                    o0 += 1
                elif a in "Il1" and b in "Il1" and a != b:
                    il1 += 1
    return o0, il1, out.substitutions + out.deletions + out.insertions, out.hits + out.substitutions + out.deletions


def pretest(fonts: Sequence[Mapping[str, Any]], reader_name: str, pages: int, seed: int, out: Path,
            readers_cfg: Mapping[str, Any], defaults: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Render the same pages in every passing face's slot layout, read them, count the confusions."""
    from mrr import readers

    reader_cfg = next(c for c in readers_cfg["readers"] if c["name"] == reader_name)
    allow = {"qwen": (105, 26), "smolvlm": (40, 40), "smolvlm500m": (60, 60)}[reader_name]
    cfgs = {}
    for f in fonts:
        cfgs[f["font"]] = grid.resolve_render_cfg(
            defaults, reader_cfg, font_file=f["path"], font_size_px=float(f["size_px"]),
            line_pitch_px=render.unaligned_pitch_px(float(f["size_px"])), render_mode="patch_aware",
            phase_x_px=0.0, phase_y_px=0.0, max_phase_x_px=allow[0], max_phase_y_px=allow[1])
    tightest = max(cfgs.values(), key=lambda c: c["line_pitch_px"])  # the largest pitch holds the fewest lines
    texts = text_source.paginate(text_source.random_text(100_000, seed), lambda t: render.fits_page(t, tightest), pages)
    for name, cfg in cfgs.items():
        assert all(render.fits_page(t, cfg) for t in texts), name
    hashes = [hashlib.sha1(t.encode("utf-8")).hexdigest()[:10] for t in texts]
    model = readers.load_reader(reader_cfg)
    prompt = reader_cfg["prompt"]
    rows = []
    try:
        for f in fonts:
            cfg = cfgs[f["font"]]
            d = out / "pretest" / f["font"]
            d.mkdir(parents=True, exist_ok=True)
            o0 = il1 = errors = letters = 0
            for i, text in enumerate(texts):
                page = render.render_page(text, cfg)
                limit = math.ceil(1.5 * readers.count_text_tokens(model, text))
                tr = readers.transcribe_full(model, page, prompt, max_new_tokens=limit)
                (d / f"{i:02d}.txt").write_text(tr.text, encoding="utf-8")
                a, b, e, n = confusions(text, tr.text)
                o0 += a; il1 += b; errors += e; letters += n
            rows.append({"font": f["font"], "pages": len(texts), "letters": letters, "o0_confusions": o0,
                         "il1_confusions": il1, "letter_errors": errors, "letters_only_cer": errors / letters,
                         "page_hashes": " ".join(hashes)})
            print(f"{f['font']:24s} O/0 {o0:3d}  I/l/1 {il1:3d}  letter errors {errors:4d} of {letters}  "
                  f"({errors / letters:.4f})", flush=True)
    finally:
        readers.unload_reader(model)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items() if k in columns})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fonts", nargs="+", required=True, help="candidate .ttf files")
    p.add_argument("--patch", type=int, default=14)
    p.add_argument("--out", default="reports/tables/study2_prerun")
    p.add_argument("--pretest", action="store_true", help="also run the reader pre-test on the faces that pass")
    p.add_argument("--reader", default="qwen")
    p.add_argument("--pages", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--readers-config", default="configs/readers.yaml")
    p.add_argument("--render-config", default="configs/render_defaults.yaml")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    screened = []
    for path in args.fonts:
        row = screen(path, args.patch)
        row["path"] = path
        screened.append(row)
        flags = "".join("abcd"[i] if row[f"pass_{k}"] else "-" for i, k in enumerate("abcd"))
        print(f"{row['font']:24s} size {row['size_px']:.3f} adv {row['advance_px']:.3f} x {row['x_rows']} cap {row['cap_rows']} "
              f"ink>= {row['min_ink']} overhang {row['max_overhang']} [{row['not_contained']}] O/0 {row['o0_diff']:.2f} "
              f"marked {row['zero_marked']} Il1 {row['il1_min_diff']:.2f}  {flags} {'PASS' if row['passes'] else 'fail'}", flush=True)
    run_cfg = {"fonts": [{"file": r["file"], "path": r["path"], "version": r["version"]} for r in screened],
               "patch_px": args.patch, "criteria": {"min_ink_px": MIN_INK, "x_rows": list(X_ROWS), "shape_diff": SHAPE_DIFF},
               "pretest": None, "status": "pre-run step 7.1"}
    rows = []
    if args.pretest:
        passing = [r for r in screened if r["passes"]]
        if not passing:
            print("no face passes the screening; no pre-test", flush=True)
        else:
            readers_cfg = yaml.safe_load(Path(args.readers_config).read_text(encoding="utf-8"))
            defaults = yaml.safe_load(Path(args.render_config).read_text(encoding="utf-8"))
            run_cfg["pretest"] = {"reader": args.reader, "pages": args.pages, "seed": args.seed}
            rows = pretest(passing, args.reader, args.pages, args.seed, out, readers_cfg, defaults)
    write_manifest(out, run_cfg, run_id=new_run_id("study2", "font_pretest"))
    write_csv(out / "font_screen.csv", screened, SCREEN_COLUMNS)
    if rows:
        write_csv(out / "font_pretest.csv", rows, PRETEST_COLUMNS)
        best = min(rows, key=lambda r: (r["o0_confusions"] + r["il1_confusions"], r["letter_errors"]))
        print(f"chosen by the rule of section 7.1: {best['font']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
