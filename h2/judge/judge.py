# PURPOSE: The H2 judge: builds the judge pages, runs one kernel attempt on them, and scores it.
# WHAT GOES HERE: Kernel inputs from a layout, the reference for any input, the static check, the runner, scoring, the command line.
# FILLED IN AT STEP: W4.4.
# DO NOT: Change the pages, the tolerance or the scoring after attempt 1. Do not modify kernels to make them pass.
"""Step W4.4: build the judge pages, run one kernel attempt on them, and score it."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from mrr import render

sys.path.insert(0, str(Path(__file__).resolve().parent))     # compare.py sits beside this file
from compare import compare, page_passes  # noqa: E402


def to_bytes(coverage: np.ndarray) -> np.ndarray:
    """Coverage back to the FreeType bytes it came from (load_glyph divides them by 255)."""
    k = np.rint(coverage * 255.0).astype(np.uint8)
    assert np.array_equal(k / 255.0, coverage), "coverage is not bytes / 255"
    return k


def pack_bitmaps(blocks: list[np.ndarray], rows: list[int], cols: list[int]) -> dict:
    """The six glyph inputs of the kernel interface."""
    sizes = np.array([b.size for b in blocks], dtype=np.int64)
    return {"bitmaps": np.concatenate([b.ravel() for b in blocks] + [np.zeros(0, np.uint8)]),
            "offsets": np.cumsum(sizes) - sizes,
            "heights": np.array([b.shape[0] for b in blocks], dtype=np.int32),
            "widths": np.array([b.shape[1] for b in blocks], dtype=np.int32),
            "rows": np.array(rows, dtype=np.int32),
            "cols": np.array(cols, dtype=np.int32)}


def pack_page(placed, render_cfg) -> dict:
    """Kernel inputs for a laid-out page: the glyphs that render_coverage loads."""
    blocks, rows, cols = [], [], []
    for p in placed:
        g = render.load_glyph(render_cfg["font_file"], render_cfg["font_size_px"], p.char,
                              p.x_64 & 63, p.baseline_64 & 63)
        blocks.append(to_bytes(g.coverage))
        rows.append((p.baseline_64 >> 6) - g.top)
        cols.append((p.x_64 >> 6) + g.left)
    return pack_bitmaps(blocks, rows, cols)


def reference_from_packed(packed: dict, height: int, width: int, text_rgb, background_rgb,
                          aa_mode: str) -> np.ndarray:
    """The reference output for any kernel input, by the three steps of render_page."""
    cov = np.zeros((height, width))
    keys = ("offsets", "heights", "widths", "rows", "cols")
    for o, h, w, r, c in zip(*(packed[k].tolist() for k in keys)):
        glyph = packed["bitmaps"][o:o + h * w].reshape(h, w) / 255.0
        r0, c0, r1, c1 = max(r, 0), max(c, 0), min(r + h, height), min(c + w, width)
        if r1 > r0 and c1 > c0:
            cov[r0:r1, c0:c1] = np.minimum(1.0, cov[r0:r1, c0:c1] + glyph[r0 - r:r1 - r, c0 - c:c1 - c])
    page = np.empty((height, width, 3))
    page[:] = np.asarray(background_rgb, dtype=np.float64) / 255.0
    render.composite_coverage(page, cov, 0, 0, text_rgb, aa_mode)
    return np.floor(255.0 * np.clip(page, 0.0, 1.0) + 0.5).astype(np.uint8)


ALLOWED_IMPORTS = {"__future__", "math", "typing", "functools", "itertools", "dataclasses",
                   "numpy", "torch", "triton"}
FLAGGED_CALLS = {"open", "exec", "eval", "compile", "__import__", "getattr", "globals", "breakpoint"}
FLAGGED_ATTRS = {"save", "savez", "tofile", "fromfile", "hub", "cpp_extension", "load_inline",
                 "load_library", "system", "popen", "urlopen"}


def _is_jit(d: ast.expr) -> bool:
    d = d.func if isinstance(d, ast.Call) else d       # @triton.jit(...) as well as @triton.jit
    return (isinstance(d, ast.Attribute) and d.attr == "jit") or (isinstance(d, ast.Name) and d.id == "jit")


def static_check(source: str) -> dict:
    """What the kernel file defines, and the lines both authors must look at before it runs."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"syntax_error": f"line {e.lineno}: {e.msg}", "has_render": False, "jit": [], "flags": []}
    flags = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            flags += [f"line {n.lineno}: import {a.name}" for a in n.names
                      if a.name.split(".")[0] not in ALLOWED_IMPORTS]
        elif isinstance(n, ast.ImportFrom):
            if n.level or (n.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                flags.append(f"line {n.lineno}: from {'.' * n.level}{n.module or ''} import ...")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FLAGGED_CALLS:
            flags.append(f"line {n.lineno}: {n.func.id}()")
        elif isinstance(n, ast.Attribute):
            is_tl_load = n.attr == "load" and isinstance(n.value, ast.Name) and n.value.id == "tl"
            if n.attr in FLAGGED_ATTRS or (n.attr == "load" and not is_tl_load):
                flags.append(f"line {n.lineno}: .{n.attr}")
    return {"syntax_error": None,
            "has_render": any(isinstance(n, ast.FunctionDef) and n.name == "render" for n in tree.body),
            "jit": [n.name for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and any(_is_jit(d) for d in n.decorator_list)],
            "flags": flags}


RUN_KERNEL = Path(__file__).with_name("run_kernel.py")
SECRET = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)


def _fail(out_dir: Path, page_id: str, error: str) -> None:
    (out_dir / f"{page_id}.result.json").write_text(json.dumps({"ok": False, "error": error}), encoding="utf-8")


def run_attempt(kernel: Path, in_dir: Path, out_dir: Path, page_ids: list[str],
                runner: Path = RUN_KERNEL, timeout_s: int = 120, max_hangs: int = 3) -> None:
    """Run the kernel on each page in a fresh process. A crash or a hang fails that page only."""
    out_dir.mkdir(parents=True)                # a new folder for every attempt: FileExistsError otherwise
    env = {k: v for k, v in os.environ.items() if not SECRET.search(k)}    # the kernel never sees the API key
    hangs = 0
    for pid in page_ids:
        if hangs == max_hangs:                 # judge.yaml stop rule: changes the page count, never the verdict
            _fail(out_dir, pid, f"not run: {max_hangs} earlier pages hung")
            continue
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cwd:
            try:
                p = subprocess.run([sys.executable, "-I", "-X", "utf8", str(runner), str(kernel),
                                    str(in_dir), str(out_dir), pid],
                                   cwd=cwd, env=env, capture_output=True, encoding="utf-8",
                                   errors="replace", timeout=timeout_s)
                code, tail = p.returncode, p.stderr[-2000:]
            except subprocess.TimeoutExpired:
                code, tail = None, f"no result within {timeout_s} s"
        if not (out_dir / f"{pid}.result.json").exists():     # it died or hung before the first call returned
            hangs += code is None
            _fail(out_dir, pid, f"exit code {code}: {tail}")


def score_page(out_dir: Path, ref_dir: Path, page_id: str, aa_mode: str) -> dict:
    """One row of the judge table. Any error fails the page; so does a result outside the tolerance."""
    try:
        result = json.loads((out_dir / f"{page_id}.result.json").read_text(encoding="utf-8"))
    except ValueError:                         # killed while writing its result
        return {"passed": False, "error": "result file unreadable"}
    if not result["ok"]:
        return {"passed": False, "error": (result["error"].strip().splitlines() or ["no message"])[-1]}
    if not result["device"].startswith("cuda"):
        return {"passed": False, "error": f"output on {result['device']}, not on the GPU"}
    try:      # np.load refuses pickled objects by default, so a kernel's .npy cannot run code here
        r = compare(np.load(out_dir / f"{page_id}.npy"), np.load(ref_dir / f"{page_id}.npy"))
    except ValueError as e:
        return {"passed": False, "error": str(e)}
    return {**r, "passed": page_passes(r, aa_mode), "error": "", "kernel_ms": result["kernel_ms"],
            "repeat_identical": result["repeat_identical"]}


# ---- The command line: build, run and summary (build guide 4.4 gives them as steps, not as code; written at W4.4) ----

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import statistics  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import yaml  # noqa: E402

from mrr import grid, text_source  # noqa: E402
from mrr.manifest import new_run_id, write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
JUDGE_YAML = Path(__file__).with_name("judge.yaml")
PAGES_CSV = Path(__file__).with_name("judge_pages.csv")
REVIEWS_CSV = Path(__file__).with_name("reviews.csv")
BUILD_DIR = ROOT / "data" / "interim" / "h2" / "judge_v01"
RUNS_ROOT = ROOT / "data" / "interim" / "h2" / "runs"
TABLES_DIR = ROOT / "reports" / "tables" / "h2"
KERNELS_DIR = ROOT / "h2" / "kernels"
KEYS = ("bitmaps", "offsets", "heights", "widths", "rows", "cols")      # as run_kernel.py reads them
PAGE_COLUMNS = ("page_id", "reader", "layout", "aa_mode", "text_type", "colour_case", "edge_case",
                "phase_x_px", "phase_y_px", "height", "width", "n_glyphs", "inputs_sha256", "reference_sha256")
RESULT_COLUMNS = ("max_abs_error", "frac_pixels_differing", "passed", "kernel_ms", "repeat_identical", "error")
SUMMARY_COLUMNS = ("attempt", "kernel_file", "kernel_sha256", "static_flags", "ran", "pixel_work_in_triton",
                   "pages_passed", "passed_all", "worst_max_abs_error", "median_kernel_ms")
REVIEW_COLUMNS = ("attempt", "reviewer", "safe_to_run", "pixel_work_in_triton", "note", "date")
REVIEWERS = ("A", "B", "owner")     # D23: A Claude Opus 5.5, B a fresh Claude Sonnet 5.5 agent, owner the final go
BLACK, WHITE = (0, 0, 0), (255, 255, 255)


def load_judge_cfg(path: Path = JUDGE_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def full_run_settings(final_runs: Path, names: Sequence[str]) -> dict[str, tuple[str, float, dict]]:
    """(font file, font size, reader block) of each reader's full run, from the frozen manifests (section 2.8).

    A full run's manifest names its sweep "full"; the size is the sweep's font_size_px for that reader (block Z's
    sizes are on pages of their own), and the font file is the sweep's, relative to the repository root.

    Raises:
        FileNotFoundError: a reader has no full-run manifest under final_runs (the freeze, W3.5, comes first).
        ValueError: a size is not a number, or two full runs of one reader disagree.
    """
    found: dict[str, tuple[str, float, dict]] = {}
    for manifest in sorted(Path(final_runs).glob("*/manifest.json")):
        cfg = json.loads(manifest.read_text(encoding="utf-8")).get("config") or {}
        if cfg.get("sweep_name") != "full":
            continue
        name, sweep = cfg["reader"]["name"], cfg["sweep"]
        size = sweep["font_size_px"][name] if isinstance(sweep["font_size_px"], Mapping) else sweep["font_size_px"]
        if not isinstance(size, (int, float)) or isinstance(size, bool):
            raise ValueError(f"{manifest}: font_size_px of {name} is {size!r}, not a number")
        setting = (sweep["font_file"], size, cfg["reader"])
        if name in found and found[name][:2] != setting[:2]:
            raise ValueError(f"two full runs of {name} disagree: {found[name][:2]} and {setting[:2]}")
        found[name] = setting
    missing = [n for n in names if n not in found]
    if missing:
        raise FileNotFoundError(f"no full-run manifest under {final_runs} for {missing}: freeze first (W3.5), "
                                "or build the stand-in with --render-source defaults")
    return {n: found[n] for n in names}


def render_cfgs(judge_cfg: Mapping[str, Any], root: Path = ROOT, source: str = "final",
                final_runs: Path | None = None) -> dict[str, dict]:
    """Each reader's render config for the judge pages, before the layout, mode, colours and phase are set.

    source "final": the font and size of the reader's full run (full_run_settings); "defaults": the stand-in,
    configs/render_defaults.yaml (16 px). Either way the line pitch is the unaligned pitch of that size, and the
    phase allowances are the reader's largest, its sweep period less 1 (111, 41 and 63 px), so that the phase "max"
    of judge.yaml is a valid phase on both axes.
    """
    defaults = yaml.safe_load((root / "configs" / "render_defaults.yaml").read_text(encoding="utf-8"))
    readers = {r["name"]: r for r in
               yaml.safe_load((root / "configs" / "readers.yaml").read_text(encoding="utf-8"))["readers"]}
    names = list(judge_cfg["readers"])
    if source == "final":
        chosen = full_run_settings(final_runs or root / "results" / "final" / "runs", names)
    elif source == "defaults":
        chosen = {n: (defaults["font_file"], defaults["font_size_px"], readers[n]) for n in names}
    else:
        raise ValueError(f"source must be 'final' or 'defaults', got {source!r}")
    return {n: grid.resolve_render_cfg(defaults, reader_cfg, font_file=str(root / font), font_size_px=size,
                                       line_pitch_px=render.unaligned_pitch_px(size))
            for n, (font, size, reader_cfg) in chosen.items()}


def judge_texts(judge_cfg: Mapping[str, Any], root: Path = ROOT) -> dict[str, str]:
    """The natural text (the book from its marker, normalised) and the random text of judge.yaml."""
    natural, rnd = judge_cfg["texts"]["natural"], judge_cfg["texts"]["random"]
    book = text_source.BOOKS[natural["book"]]
    body = text_source.strip_gutenberg_boilerplate(text_source.load_book(root / text_source.TEXTS_DIR / book.file))
    return {"natural": text_source.normalise_text(
                text_source.skip_front_matter(body, natural["marker"], natural["occurrence"])),
            "random": text_source.random_text(rnd["n_chars"], rnd["seed"])}


def _phases(judge_cfg: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [(cfg["max_phase_x_px"], cfg["max_phase_y_px"]) if p == "max" else (p[0], p[1])
            for p in judge_cfg["phases_px"]]


def _row(page_id: str, aa_mode: str, height: int, width: int, n_glyphs: int, reader: str = "", layout: str = "",
         text_type: str = "", colour_case: str = "", edge_case: str = "", phase: tuple[float, float] = (0, 0)) -> dict:
    return {"page_id": page_id, "reader": reader, "layout": layout, "aa_mode": aa_mode, "text_type": text_type,
            "colour_case": colour_case, "edge_case": edge_case, "phase_x_px": f"{phase[0]:g}",
            "phase_y_px": f"{phase[1]:g}", "height": height, "width": width, "n_glyphs": n_glyphs}


def laid_out_page(text: str, base: Mapping[str, Any], layout: str, aa_mode: str, phase: tuple[float, float],
                  text_rgb: Sequence[int], background_rgb: Sequence[int]) -> tuple[dict, dict, np.ndarray]:
    """(render config, kernel inputs, reference) of one normal or colour page, checked as step 4 of build says.

    Raises:
        AssertionError: the reference from the packed inputs differs from render_page, or a glyph box leaves the page.
    """
    cfg = {**base, "render_mode": layout, "aa_mode": aa_mode, "phase_x_px": phase[0], "phase_y_px": phase[1],
           "text_color": list(text_rgb), "background_color": list(background_rgb)}
    packed = pack_page(render.layout_page(text, cfg), cfg)
    height, width = cfg["page_height_px"], cfg["page_width_px"]
    reference = reference_from_packed(packed, height, width, tuple(text_rgb), tuple(background_rgb), aa_mode)
    if not np.array_equal(reference, render.render_page(text, cfg)):
        raise AssertionError(f"{layout} {aa_mode} at {phase}: the packed reference differs from render_page")
    rows, cols = packed["rows"].astype(np.int64), packed["cols"].astype(np.int64)
    if not np.all((rows >= 0) & (cols >= 0) & (rows + packed["heights"] <= height) & (cols + packed["widths"] <= width)):
        raise AssertionError(f"{layout} {aa_mode} at {phase}: a glyph box leaves the page")
    return cfg, packed, reference


def make_pages(judge_cfg: Mapping[str, Any], cfgs: Mapping[str, Mapping[str, Any]],
               texts: Mapping[str, str]) -> list[dict]:
    """Every judge page in order (normal, colour, edge): its row, kernel inputs, meta and reference.

    Normal and colour pages also carry their text and render config, so that a test can render them again.
    """
    pages: list[dict] = []

    def add(row: dict, packed: dict, reference: np.ndarray, text_rgb, background_rgb, text=None, cfg=None):
        meta = {"height": row["height"], "width": row["width"], "text_rgb": list(text_rgb),
                "background_rgb": list(background_rgb), "aa_mode": row["aa_mode"]}
        pages.append({"row": row, "packed": packed, "meta": meta, "reference": reference, "text": text, "cfg": cfg})

    first = {}                       # the first page of each text that fits both layouts, per reader
    for reader in judge_cfg["readers"]:
        layout_cfgs = [{**cfgs[reader], "render_mode": layout} for layout in judge_cfg["layouts"]]
        for text_type, text in texts.items():
            first[reader, text_type] = text_source.paginate(
                text, lambda c: all(render.fits_page(c, cfg) for cfg in layout_cfgs), n_pages=1)[0]
    for reader in judge_cfg["readers"]:
        phases = _phases(judge_cfg, cfgs[reader])
        for layout in judge_cfg["layouts"]:
            for mode in judge_cfg["aa_modes"]:
                for k, phase in enumerate(phases):
                    for text_type in texts:
                        text = first[reader, text_type]
                        cfg, packed, ref = laid_out_page(text, cfgs[reader], layout, mode, phase, BLACK, WHITE)
                        row = _row(f"n_{reader}_{layout}_{mode}_ph{k}_{text_type}", mode, *ref.shape[:2],
                                   len(packed["offsets"]), reader, layout, text_type, phase=phase)
                        add(row, packed, ref, BLACK, WHITE, text, cfg)
    colour = judge_cfg["colour_pages"]
    for reader in judge_cfg["readers"]:
        for mode in judge_cfg["aa_modes"]:
            for case, c in judge_cfg["colour_cases"].items():
                text, phase = first[reader, colour["text"]], tuple(colour["phase_px"])
                cfg, packed, ref = laid_out_page(text, cfgs[reader], colour["layout"], mode, phase,
                                                 c["text_rgb"], c["background_rgb"])
                row = _row(f"c_{reader}_{mode}_{case}", mode, *ref.shape[:2], len(packed["offsets"]), reader,
                           colour["layout"], colour["text"], colour_case=case, phase=phase)
                add(row, packed, ref, c["text_rgb"], c["background_rgb"], text, cfg)
    fonts = {cfg["font_file"] for cfg in cfgs.values()}
    if len(fonts) != 1:
        raise ValueError(f"the readers use different fonts {sorted(fonts)}; the edge pages need one")
    font, size = fonts.pop(), judge_cfg["edge_size_px"]
    for case, spec in judge_cfg["edge_cases"].items():
        height, width = spec["page"]
        colours = (judge_cfg["colour_cases"][spec["colours"]] if "colours" in spec
                   else {"text_rgb": BLACK, "background_rgb": WHITE})
        if "glyph_on_first_grey" in spec:      # the first byte strictly between 0 and 255 lands on pixel (0, 0)
            char = spec["glyph_on_first_grey"]
            k = to_bytes(render.load_glyph(font, size, char).coverage)
            a, b = np.argwhere((k > 0) & (k < 255))[0]
            glyphs = [(char, -int(a), -int(b))]
        else:
            glyphs = [(g[0], int(g[1]), int(g[2])) for g in spec["glyphs"]]
        packed = pack_bitmaps([to_bytes(render.load_glyph(font, size, ch).coverage) for ch, _, _ in glyphs],
                              [r for _, r, _ in glyphs], [c for _, _, c in glyphs])
        for mode in judge_cfg["aa_modes"]:
            ref = reference_from_packed(packed, height, width, tuple(colours["text_rgb"]),
                                        tuple(colours["background_rgb"]), mode)
            row = _row(f"e_{case}_{mode}", mode, height, width, len(glyphs), colour_case=spec.get("colours", ""),
                       edge_case=case)
            add(row, packed, ref, colours["text_rgb"], colours["background_rgb"])
    return pages


def page_hashes(build_dir: Path, page_id: str) -> dict[str, str]:
    """inputs_sha256 (the .npz bytes followed by the .json bytes) and reference_sha256 (the .npy bytes)."""
    inputs = Path(build_dir) / "inputs"
    digest = hashlib.sha256((inputs / f"{page_id}.npz").read_bytes())
    digest.update((inputs / f"{page_id}.json").read_bytes())
    reference = (Path(build_dir) / "reference" / f"{page_id}.npy").read_bytes()
    return {"inputs_sha256": digest.hexdigest(), "reference_sha256": hashlib.sha256(reference).hexdigest()}


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows([{c: "" if row.get(c) is None else row.get(c) for c in columns} for row in rows])


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_build(out_dir: Path, pages: Sequence[Mapping[str, Any]], pages_csv: Path) -> list[dict]:
    """Steps 5 and 6 of build: inputs/<id>.npz and .json, reference/<id>.npy, then judge_pages.csv with the hashes.

    Raises:
        FileExistsError: out_dir exists; the pages are built once, before attempt 1.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; the judge pages are built once")
    (out_dir / "inputs").mkdir(parents=True)
    (out_dir / "reference").mkdir()
    rows = []
    for page in pages:
        pid = page["row"]["page_id"]
        np.savez(out_dir / "inputs" / f"{pid}.npz", **{k: page["packed"][k] for k in KEYS})
        (out_dir / "inputs" / f"{pid}.json").write_bytes(json.dumps(page["meta"]).encode("utf-8"))
        np.save(out_dir / "reference" / f"{pid}.npy", page["reference"])
        rows.append({**page["row"], **page_hashes(out_dir, pid)})
    write_csv(pages_csv, PAGE_COLUMNS, rows)
    return rows


def verify_build(build_dir: Path, rows: Sequence[Mapping[str, str]]) -> list[str]:
    """The page ids whose files are missing or whose hashes differ from judge_pages.csv."""
    bad = []
    for row in rows:
        try:
            hashes = page_hashes(build_dir, row["page_id"])
        except FileNotFoundError:
            bad.append(row["page_id"])
            continue
        if any(hashes[k] != row[k] for k in ("inputs_sha256", "reference_sha256")):
            bad.append(row["page_id"])
    return bad


def read_reviews(reviews_csv: Path, attempt: int) -> dict[str, dict[str, str]]:
    """The review rows of one attempt by reviewer: A, B and the owner's go (D23), verdicts lower-cased.

    Raises:
        ValueError: the file is missing, a reviewer has no row or two rows, or a verdict is not yes or no.
    """
    if not Path(reviews_csv).exists():
        raise ValueError(f"{reviews_csv} does not exist; the reviews come before the judge runs (D23)")
    verdicts: dict[str, dict[str, str]] = {}
    for r in read_csv(reviews_csv):
        if r["attempt"].strip() != str(attempt):
            continue
        who = r["reviewer"].strip()
        if who in verdicts:
            raise ValueError(f"attempt {attempt}: two rows from reviewer {who}")
        row = dict(r)
        for field in ("safe_to_run", "pixel_work_in_triton"):
            row[field] = (r[field] or "").strip().lower()
            if row[field] not in ("yes", "no"):
                raise ValueError(f"attempt {attempt}, reviewer {who}: {field} is {r[field]!r}, not yes or no")
        verdicts[who] = row
    missing = [who for who in REVIEWERS if who not in verdicts]
    if missing:
        raise ValueError(f"attempt {attempt}: no review row from {missing} in {reviews_csv}")
    return verdicts


def build_cmd(out_dir: Path, pages_csv: Path, source: str, root: Path = ROOT,
              judge_cfg: Mapping[str, Any] | None = None) -> int:
    """build: steps 1 to 6, then build.json beside the pages, saying where the render configs came from."""
    judge_cfg = judge_cfg or load_judge_cfg()
    out_dir, pages_csv = Path(out_dir), Path(pages_csv)
    if out_dir.exists():
        print(f"{out_dir} exists; the judge pages are built once (delete it only before attempt 1)", file=sys.stderr)
        return 1
    cfgs = render_cfgs(judge_cfg, root, source)
    for name, cfg in cfgs.items():
        print(f"{name}: {Path(cfg['font_file']).name} at {cfg['font_size_px']} px, pitch {cfg['line_pitch_px']} px, "
              f"page {cfg['page_height_px']} x {cfg['page_width_px']}, largest phase "
              f"({cfg['max_phase_x_px']}, {cfg['max_phase_y_px']}) px; render config from {source}")
    rows = write_build(out_dir, make_pages(judge_cfg, cfgs, judge_texts(judge_cfg, root)), pages_csv)
    info = {"render_source": source, "judge_yaml_sha256": hashlib.sha256(JUDGE_YAML.read_bytes()).hexdigest(),
            "readers": {n: {"font_file": Path(c["font_file"]).name, "font_size_px": c["font_size_px"],
                            "line_pitch_px": c["line_pitch_px"],
                            "max_phase_px": [c["max_phase_x_px"], c["max_phase_y_px"]]} for n, c in cfgs.items()},
            "pages": len(rows)}
    (out_dir / "build.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    counts = {kind: sum(r["page_id"].startswith(kind + "_") for r in rows) for kind in ("n", "c", "e")}
    print(f"{len(rows)} pages ({counts['n']} normal, {counts['c']} colour, {counts['e']} edge) in {out_dir}; "
          f"hashes in {pages_csv.as_posix()}")
    return 0


def run_cmd(kernel: Path, attempt: int, *, judge_cfg: Mapping[str, Any] | None = None, build_dir: Path = BUILD_DIR,
            pages_csv: Path = PAGES_CSV, reviews_csv: Path = REVIEWS_CSV, runs_root: Path = RUNS_ROOT,
            tables_dir: Path = TABLES_DIR, runner: Path = RUN_KERNEL, stand_in: bool = False) -> int:
    """run: steps 1 to 6. Paths are made absolute, because run_attempt starts each page in an empty temporary folder.

    Raises:
        ValueError: the reviews of the attempt are missing or malformed (read_reviews).
    """
    judge_cfg = judge_cfg or load_judge_cfg()
    kernel, build_dir = Path(kernel).resolve(), Path(build_dir).resolve()
    version = judge_cfg["version"]
    table = Path(tables_dir) / f"judge_{version}_{attempt}.csv"
    if table.exists():
        print(f"{table} exists; an attempt is judged once", file=sys.stderr)
        return 1
    rows = read_csv(pages_csv)
    bad = verify_build(build_dir, rows)
    if bad:
        print(f"{len(bad)} pages differ from {pages_csv} (first {bad[0]}); never rebuild after attempt 1",
              file=sys.stderr)
        return 1
    info = json.loads((build_dir / "build.json").read_text(encoding="utf-8"))
    if info["render_source"] != "final" and not stand_in:
        print(f"{build_dir} was built from {info['render_source']}, not from the full runs; rebuild it after the "
              "freeze and before attempt 1", file=sys.stderr)
        return 1
    reason = None
    if not kernel.exists():
        if not kernel.with_suffix(".md").exists():
            print(f"{kernel} does not exist, and there is no answer {kernel.with_suffix('.md').name} beside it",
                  file=sys.stderr)
            return 1
        reason = "no code block"
    else:
        check = static_check(kernel.read_text(encoding="utf-8"))
        print(json.dumps(check, indent=2))
        verdicts = read_reviews(reviews_csv, attempt)
        if any(v["safe_to_run"] != "yes" for v in verdicts.values()):
            reason = "safety"
        elif check["syntax_error"]:
            reason = "syntax error"
        elif not check["has_render"]:
            reason = "no top-level render"
    page_ids = [r["page_id"] for r in rows]
    if reason is None:
        out_dir = (Path(runs_root) / f"{version}_{attempt}").resolve()
        run_attempt(kernel, build_dir / "inputs", out_dir, page_ids, runner=Path(runner).resolve(),
                    timeout_s=judge_cfg["page_timeout_s"], max_hangs=judge_cfg["max_hangs"])
        results = {r["page_id"]: score_page(out_dir, build_dir / "reference", r["page_id"], r["aa_mode"])
                   for r in rows}
    else:
        results = {pid: {"passed": False, "error": f"not run: {reason}"} for pid in page_ids}
    out_rows = [{**{c: r[c] for c in PAGE_COLUMNS[:-2]}, **{c: results[r["page_id"]].get(c) for c in RESULT_COLUMNS}}
                for r in rows]
    write_csv(table, PAGE_COLUMNS[:-2] + RESULT_COLUMNS, out_rows)
    passed = sum(bool(results[pid]["passed"]) for pid in page_ids)
    print(f"attempt {attempt}: {passed} of {len(page_ids)} pages passed"
          + (f" (not run: {reason})" if reason else "") + f"; table {table.as_posix()}")
    return 0


def summary_cmd(*, judge_cfg: Mapping[str, Any] | None = None, pages_csv: Path = PAGES_CSV,
                reviews_csv: Path = REVIEWS_CSV, tables_dir: Path = TABLES_DIR, kernels_dir: Path = KERNELS_DIR) -> int:
    """summary: one row per judged attempt in summary.csv, after the folder's manifest."""
    judge_cfg = judge_cfg or load_judge_cfg()
    version, n_pages = judge_cfg["version"], len(read_csv(pages_csv))
    rows = []
    for attempt in range(1, judge_cfg["attempts"] + 1):
        table = Path(tables_dir) / f"judge_{version}_{attempt}.csv"
        if not table.exists():
            print(f"attempt {attempt}: no judge table {table.name}")
            continue
        results = read_csv(table)
        kernel = Path(kernels_dir) / f"kernel_{version}_{attempt}.py"
        source = kernel.read_bytes() if kernel.exists() else None
        try:
            verdicts = read_reviews(reviews_csv, attempt)
        except ValueError:
            verdicts = {}
        passed = sum(r["passed"] == "True" for r in results)
        errors = [int(r["max_abs_error"]) for r in results if r["max_abs_error"] != ""]
        times = [float(r["kernel_ms"]) for r in results if r["kernel_ms"] != ""]
        try:
            kernel_file = kernel.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            kernel_file = kernel.as_posix()
        rows.append({
            "attempt": attempt, "kernel_file": kernel_file,
            "kernel_sha256": hashlib.sha256(source).hexdigest() if source is not None else "",
            "static_flags": "; ".join(static_check(source.decode("utf-8"))["flags"]) if source is not None else "",
            "ran": "yes" if any(not r["error"].startswith("not run:") for r in results) else "no",
            "pixel_work_in_triton": "yes" if verdicts and all(v["pixel_work_in_triton"] == "yes"
                                                              for v in verdicts.values()) else "no",
            "pages_passed": passed, "passed_all": "yes" if passed == n_pages == len(results) else "no",
            "worst_max_abs_error": max(errors) if errors else None,
            "median_kernel_ms": statistics.median(times) if times else None})
    if not rows:
        print("no judge table yet", file=sys.stderr)
        return 1
    cfg = {"judge": dict(judge_cfg), "judge_pages_sha256": hashlib.sha256(Path(pages_csv).read_bytes()).hexdigest(),
           "attempts": rows}
    write_manifest(Path(tables_dir), cfg, run_id=new_run_id(f"h2_{version}", "judge"))   # first, for git_dirty
    write_csv(Path(tables_dir) / "summary.csv", SUMMARY_COLUMNS, rows)
    for r in rows:
        print(f"attempt {r['attempt']}: {r['pages_passed']} of {n_pages} pages, passed_all {r['passed_all']}, "
              f"pixel_work_in_triton {r['pixel_work_in_triton']}")
    return 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="The H2 judge (build guide 4.4): build the pages, run an attempt, sum up.")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build the judge pages and write judge_pages.csv")
    b.add_argument("--out", type=Path, default=BUILD_DIR)
    b.add_argument("--pages-csv", type=Path, default=PAGES_CSV)
    b.add_argument("--render-source", choices=("final", "defaults"), default="final",
                   help="final: each reader's full run under results/final/runs; defaults: the 16 px stand-in")
    r = sub.add_parser("run", help="run one attempt's kernel on every page and write its judge table")
    r.add_argument("--kernel", type=Path, required=True)
    r.add_argument("--attempt", type=int, required=True)
    r.add_argument("--build", type=Path, default=BUILD_DIR)
    r.add_argument("--pages-csv", type=Path, default=PAGES_CSV)
    r.add_argument("--reviews", type=Path, default=REVIEWS_CSV)
    r.add_argument("--stand-in", action="store_true", help="accept pages built from render_defaults (not for attempts)")
    s = sub.add_parser("summary", help="one row per attempt in reports/tables/h2/summary.csv")
    s.add_argument("--pages-csv", type=Path, default=PAGES_CSV)
    s.add_argument("--reviews", type=Path, default=REVIEWS_CSV)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_args(argv)
    if args.command == "build":
        try:
            return build_cmd(args.out, args.pages_csv, args.render_source)
        except (FileNotFoundError, ValueError) as e:
            print(str(e), file=sys.stderr)
            return 1
    if args.command == "run":
        try:
            return run_cmd(args.kernel, args.attempt, build_dir=args.build, pages_csv=args.pages_csv,
                           reviews_csv=args.reviews, stand_in=args.stand_in)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
    return summary_cmd(pages_csv=args.pages_csv, reviews_csv=args.reviews)


if __name__ == "__main__":
    sys.exit(main())
