# PURPOSE: Tests of the H2 judge: compare, the tolerance, the 186 pages and their references, the static check, the runner, the command line.
# WHAT GOES HERE: The cases of build guide 4.4 (F4 A4) on the CPU; the good and faulty kernels through run_kernel.py as model-marked tests (GPU, Triton).
# FILLED IN AT STEP: W4.4.
# DO NOT: Write a faulty kernel into the repository (they are made in tmp_path), loosen a tolerance, or read the pages from results/.
"""Tests for h2/judge/ (build guide 4.4). The pages are built once per module from configs/render_defaults.yaml and
the reader table (the 16 px stand-in), not from results/; the counts quoted from the guide are the stand-in's too."""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import PERSUASION, REPO_ROOT

JUDGE_DIR = REPO_ROOT / "h2" / "judge"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "h2"
FAKE_RUNNER = FIXTURES / "fake_runner.py"
DEFAULTS = yaml.safe_load((REPO_ROOT / "configs" / "render_defaults.yaml").read_text(encoding="utf-8"))
HAVE_DATA = (REPO_ROOT / DEFAULTS["font_file"]).exists() and PERSUASION.exists()
FAKE_KEY = "fake-zai-key-for-the-judge-tests"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"h2_judge_{name}", JUDGE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves postponed annotations through sys.modules
    spec.loader.exec_module(module)
    return module


judge = _load("judge")
compare_mod = _load("compare")
JUDGE_CFG = judge.load_judge_cfg()


@pytest.fixture(scope="module")
def pages():
    """The 186 judge pages in memory, from the stand-in render configs (about 20 s)."""
    if not HAVE_DATA:
        pytest.skip("font or book not present (W1.4)")
    cfgs = judge.render_cfgs(JUDGE_CFG, REPO_ROOT, "defaults")
    return judge.make_pages(JUDGE_CFG, cfgs, judge.judge_texts(JUDGE_CFG, REPO_ROOT))


def _by_id(pages):
    return {p["row"]["page_id"]: p for p in pages}


# ---- compare and the tolerance -----------------------------------------------------------------------------------

def test_compare_equal_and_one_channel_off():
    a = np.full((384, 384, 3), 200, np.uint8)
    assert judge.compare(a, a.copy()) == {"max_abs_error": 0, "frac_pixels_differing": 0.0}
    b = a.copy()
    b[100, 200, 1] = 201
    assert judge.compare(b, a) == {"max_abs_error": 1, "frac_pixels_differing": 1 / 147456}


def test_compare_never_wraps_uint8():
    three, five = np.full((2, 2, 3), 3, np.uint8), np.full((2, 2, 3), 5, np.uint8)
    assert judge.compare(three, five)["max_abs_error"] == 2
    assert judge.compare(np.zeros((2, 2, 3), np.uint8), np.full((2, 2, 3), 255, np.uint8)) == \
        {"max_abs_error": 255, "frac_pixels_differing": 1.0}


@pytest.mark.parametrize("kernel, reference", [
    (np.zeros((4, 4, 3), np.uint8), np.zeros((4, 5, 3), np.uint8)),
    (np.zeros((4, 4, 3), np.float32), np.zeros((4, 4, 3), np.uint8)),
    (np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8)),
    (np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.int16)),
])
def test_compare_refuses_shapes_and_dtypes(kernel, reference):
    with pytest.raises(ValueError):
        judge.compare(kernel, reference)


def test_page_passes_at_the_boundaries():
    passes = compare_mod.page_passes
    assert passes({"max_abs_error": 1, "frac_pixels_differing": 1e-3}, "coverage_linear")
    assert not passes({"max_abs_error": 1, "frac_pixels_differing": 1.1e-3}, "coverage_linear")
    assert not passes({"max_abs_error": 2, "frac_pixels_differing": 1e-6}, "coverage_linear")
    assert not passes({"max_abs_error": 1, "frac_pixels_differing": 1e-6}, "coverage_srgb")
    assert passes({"max_abs_error": 0, "frac_pixels_differing": 0.0}, "binary")
    assert not passes({"max_abs_error": 1, "frac_pixels_differing": 1e-6}, "binary")


def test_tolerance_equals_judge_yaml():
    tolerance = {mode: (t["max_abs_error"], t["frac_pixels_differing"]) for mode, t in JUDGE_CFG["tolerance"].items()}
    assert tolerance == compare_mod.TOLERANCE


# ---- the reference for any input: the worked examples of section 4.2 ------------------------------------------------

WORKED = {  # aa_mode: (black on white, white on black), each entry the value of all three channels
    "binary": ([[255, 255], [0, 0]], [[0, 0], [255, 255]]),
    "coverage_srgb": ([[255, 191], [127, 0]], [[0, 64], [128, 255]]),
    "coverage_linear": ([[255, 224], [187, 0]], [[0, 137], [188, 255]]),
}
RED, CREAM = (200, 30, 60), (250, 245, 230)


@pytest.mark.parametrize("mode", list(WORKED))
def test_reference_from_packed_worked_examples(mode):
    packed = judge.pack_bitmaps([np.array([[0, 64], [128, 255]], np.uint8)], [0], [0])
    for (text, background), expected in zip((((0, 0, 0), (255, 255, 255)), ((255, 255, 255), (0, 0, 0))), WORKED[mode]):
        page = judge.reference_from_packed(packed, 2, 2, text, background, mode)
        assert page.dtype == np.uint8 and page.shape == (2, 2, 3)
        assert page.tolist() == [[[v] * 3 for v in row] for row in expected]


def test_reference_from_packed_colour_and_sums():
    one = judge.pack_bitmaps([np.array([[128]], np.uint8)], [0], [0])
    expected = {"binary": [200, 30, 60], "coverage_srgb": [225, 137, 145], "coverage_linear": [227, 181, 173]}
    for mode, triple in expected.items():
        assert judge.reference_from_packed(one, 1, 1, RED, CREAM, mode)[0, 0].tolist() == triple
    for a, b, binary in ((100, 27, list(CREAM)), (100, 28, list(RED))):
        two = judge.pack_bitmaps([np.array([[a]], np.uint8), np.array([[b]], np.uint8)], [0, 0], [0, 0])
        assert judge.reference_from_packed(two, 1, 1, RED, CREAM, "binary")[0, 0].tolist() == binary
    clipped = judge.pack_bitmaps([np.array([[200]], np.uint8), np.array([[100]], np.uint8)], [0, 0], [0, 0])
    for mode in WORKED:
        assert judge.reference_from_packed(clipped, 1, 1, RED, CREAM, mode)[0, 0].tolist() == list(RED)


# ---- the 186 pages -----------------------------------------------------------------------------------------------

def test_page_ids_and_groups(pages):
    ids = [p["row"]["page_id"] for p in pages]
    assert len(ids) == len(set(ids)) == 186
    assert [sum(i.startswith(k) for i in ids) for k in ("n_", "c_", "e_")] == [144, 18, 24]
    assert "n_smolvlm500m_patch_aware_coverage_linear_ph3_random" in ids and "c_qwen_binary_red" in ids
    assert "e_tiny_coverage_srgb" in ids
    for p in pages:
        assert set(p["packed"]) == set(judge.KEYS) and p["packed"]["bitmaps"].dtype == np.uint8
        assert p["packed"]["offsets"].dtype == np.int64
        assert all(p["packed"][k].dtype == np.int32 for k in ("heights", "widths", "rows", "cols"))
        assert p["reference"].shape == (p["meta"]["height"], p["meta"]["width"], 3)


def test_committed_judge_pages_csv_lists_the_same_pages(pages):
    with open(judge.PAGES_CSV, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0]) == judge.PAGE_COLUMNS
    assert [r["page_id"] for r in rows] == [p["row"]["page_id"] for p in pages]
    for r in rows:
        assert len(r["inputs_sha256"]) == 64 and len(r["reference_sha256"]) == 64


def test_phases_and_colours_of_the_pages(pages):
    by_id = _by_id(pages)
    largest = {"qwen": "111", "smolvlm": "41", "smolvlm500m": "63"}
    for reader, phase in largest.items():
        row = by_id[f"n_{reader}_conventional_binary_ph3_natural"]["row"]
        assert (row["phase_x_px"], row["phase_y_px"]) == (phase, phase)
    assert (by_id["n_qwen_patch_aware_binary_ph2_random"]["row"]["phase_x_px"],
            by_id["n_qwen_patch_aware_binary_ph2_random"]["row"]["phase_y_px"]) == ("13.75", "7.5")
    assert by_id["c_smolvlm_coverage_linear_red"]["meta"]["text_rgb"] == list(RED)
    assert by_id["c_smolvlm_coverage_linear_wob"]["meta"]["background_rgb"] == [0, 0, 0]
    assert by_id["e_overlap_binary"]["meta"]["background_rgb"] == list(CREAM)


def test_first_page_fits_both_layouts(pages):
    """The guide's stand-in counts: 298, 298 and 471 characters of Persuasion, 269, 269 and 479 random."""
    by_id = _by_id(pages)
    counts = {(r, t): len(by_id[f"n_{r}_conventional_binary_ph0_{t}"]["text"])
              for r in ("qwen", "smolvlm", "smolvlm500m") for t in ("natural", "random")}
    assert counts == {("qwen", "natural"): 298, ("smolvlm", "natural"): 298, ("smolvlm500m", "natural"): 471,
                      ("qwen", "random"): 269, ("smolvlm", "random"): 269, ("smolvlm500m", "random"): 479}


def test_packed_reference_equals_render_page(pages, monkeypatch):
    """make_pages compares every normal and colour page with render_page and stops on a difference (build step 4),
    so 162 pages built means 162 of 162 identical; the check itself must be live."""
    laid_out = [p for p in pages if p["text"] is not None]
    assert len(laid_out) == 162
    p = laid_out[5]
    judge.laid_out_page(p["text"], p["cfg"], p["row"]["layout"], p["row"]["aa_mode"],
                        (p["cfg"]["phase_x_px"], p["cfg"]["phase_y_px"]), p["meta"]["text_rgb"],
                        p["meta"]["background_rgb"])
    monkeypatch.setattr(judge.render, "render_page", lambda text, cfg: p["reference"] ^ 1)
    with pytest.raises(AssertionError, match="differs from render_page"):
        judge.laid_out_page(p["text"], p["cfg"], p["row"]["layout"], p["row"]["aa_mode"],
                            (p["cfg"]["phase_x_px"], p["cfg"]["phase_y_px"]), p["meta"]["text_rgb"],
                            p["meta"]["background_rgb"])


def _integer_sum(packed, height, width, clip=True, swap=False):
    """Equation (1): the glyph bytes added as integers, clipped at 255 unless clip is False."""
    s = np.zeros((height, width), np.int64)
    keys = ("offsets", "heights", "widths", "rows", "cols")
    for o, h, w, r, c in zip(*(packed[k].tolist() for k in keys)):
        if swap:
            r, c = c, r
        g = packed["bitmaps"][o:o + h * w].reshape(h, w).astype(np.int64)
        r0, c0, r1, c1 = max(r, 0), max(c, 0), min(r + h, height), min(c + w, width)
        if r1 > r0 and c1 > c0:
            s[r0:r1, c0:c1] += g[r0 - r:r1 - r, c0 - c:c1 - c]
    return np.minimum(s, 255) if clip else s


@pytest.fixture(scope="module")
def sums(pages):
    """The unclipped integer sum of every page, by page id."""
    return {p["row"]["page_id"]: _integer_sum(p["packed"], p["meta"]["height"], p["meta"]["width"], clip=False)
            for p in pages}


def test_integer_sum_equals_reference_on_every_page(pages, sums):
    """Equation (1) composited by the reference compositor and rounded once. The output of a pixel depends on its S
    alone, so the 256 values of S are composited once per page, as a 1 x 256 page, and looked up."""
    from mrr.render import composite_coverage

    same = 0
    for p in pages:
        m = p["meta"]
        row = np.empty((1, 256, 3))
        row[:] = np.asarray(m["background_rgb"], dtype=np.float64) / 255.0
        composite_coverage(row, (np.arange(256) / 255.0)[None, :], 0, 0, m["text_rgb"], m["aa_mode"])
        lut = np.floor(255.0 * np.clip(row[0], 0.0, 1.0) + 0.5).astype(np.uint8)
        same += np.array_equal(lut[np.minimum(sums[p["row"]["page_id"]], 255)], p["reference"])
    assert same == 186


def test_edge_pages(pages, sums):
    by_id = _by_id(pages)
    assert by_id["e_tiny_coverage_srgb"]["reference"].tolist() == [[[153, 153, 153]]]
    zero = by_id["e_zero_binary"]["packed"]
    assert zero["offsets"][-1] == len(zero["bitmaps"]) and zero["heights"][-1] == 0
    assert len(by_id["e_empty_binary"]["packed"]["offsets"]) == 0
    assert (by_id["e_empty_coverage_linear"]["reference"] == 255).all()
    assert (by_id["e_outside_coverage_srgb"]["reference"] == 255).all()
    assert by_id["e_oblong_binary"]["reference"].shape == (37, 100, 3)
    assert sums["e_overlap_coverage_srgb"].max() > 255
    for case, glyphs in JUDGE_CFG["edge_cases"].items():
        if "glyphs" in glyphs:
            assert by_id[f"e_{case}_binary"]["row"]["n_glyphs"] == len(glyphs["glyphs"])


# ---- NumPy emulations of five kernels (on the stand-in: 26, 2, 60 and 176 pages fail, as the guide measured) ---------

def _decode(v, power=None):
    if power:
        return v ** power
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _encode(lin, power=None):
    with np.errstate(invalid="ignore", divide="ignore"):
        if power:
            return np.clip(lin, 0, None) ** (1 / power)
        return np.where(lin <= 0.04045 / 12.92, 12.92 * lin, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055)


def _mix(cov, meta, power=None):
    """R05 and R06 in float64 on coverage that may exceed 1 (no clip); cov of shape (n,) gives a table (n, 3)."""
    t = np.asarray(meta["text_rgb"], np.float64) / 255.0
    b = np.asarray(meta["background_rgb"], np.float64) / 255.0
    c = cov[..., None]
    if meta["aa_mode"] == "binary":
        a = (c >= 0.5).astype(np.float64)
        o = a * t + (1 - a) * b
    elif meta["aa_mode"] == "coverage_srgb":
        o = c * t + (1 - c) * b
    else:
        o = _encode(c * _decode(t, power) + (1 - c) * _decode(b, power), power)
    return np.floor(255.0 * np.clip(o, 0.0, 1.0) + 0.5).astype(np.uint8)


def _per_glyph(p):
    """Each glyph composited onto the page on its own, then one rounding: overlaps are mixed twice."""
    m, packed = p["meta"], p["packed"]
    height, width = m["height"], m["width"]
    t = np.asarray(m["text_rgb"], np.float64) / 255.0
    page = np.empty((height, width, 3))
    page[:] = np.asarray(m["background_rgb"], np.float64) / 255.0
    keys = ("offsets", "heights", "widths", "rows", "cols")
    for o, h, w, r, c in zip(*(packed[k].tolist() for k in keys)):
        r0, c0, r1, c1 = max(r, 0), max(c, 0), min(r + h, height), min(c + w, width)
        if r1 <= r0 or c1 <= c0:
            continue
        cov = packed["bitmaps"][o:o + h * w].reshape(h, w)[r0 - r:r1 - r, c0 - c:c1 - c, None] / 255.0
        region = page[r0:r1, c0:c1]
        if m["aa_mode"] == "binary":
            a = (cov >= 0.5).astype(np.float64)
            region[...] = a * t + (1 - a) * region
        elif m["aa_mode"] == "coverage_srgb":
            region[...] = cov * t + (1 - cov) * region
        else:
            region[...] = _encode(cov * _decode(t) + (1 - cov) * _decode(region))
    return np.floor(255.0 * np.clip(page, 0.0, 1.0) + 0.5).astype(np.uint8)


def _float32_table(meta):
    """good_kernel.py's arithmetic in single precision for S = 0 to 255: binary on integers, coverage_srgb as
    (S T + (255 - S) B) / 255, coverage_linear with exp2 and log2."""
    f = np.float32
    s = np.arange(256, dtype=np.int32)[:, None]
    t8, b8 = np.asarray(meta["text_rgb"], np.int32), np.asarray(meta["background_rgb"], np.int32)
    if meta["aa_mode"] == "binary":
        return np.where(s >= 128, t8, b8).astype(np.uint8)
    if meta["aa_mode"] == "coverage_srgb":
        return np.floor((s * t8 + (255 - s) * b8).astype(f) / f(255) + f(0.5)).astype(np.uint8)

    def dec(v):
        return np.where(v <= f(0.04045), v / f(12.92), np.exp2(f(2.4) * np.log2((v + f(0.055)) / f(1.055))))

    def enc(lin):
        with np.errstate(divide="ignore"):
            return np.where(lin <= f(0.04045 / 12.92), f(12.92) * lin,
                            f(1.055) * np.exp2(np.log2(lin) / f(2.4)) - f(0.055))

    c = s.astype(f) / f(255)
    o = np.clip(enc(c * dec(t8.astype(f) / f(255)) + (f(1) - c) * dec(b8.astype(f) / f(255))), f(0), f(1))
    assert o.dtype == np.float32
    return np.floor(f(255) * o + f(0.5)).astype(np.uint8)


def _swapped(p):
    m = p["meta"]
    s = _integer_sum(p["packed"], m["height"], m["width"], swap=True)
    return _mix(np.arange(256) / 255.0, m)[s]


EMULATIONS = {   # each maps (page, its unclipped integer sum) to the kernel's output
    "per_glyph": lambda p, s: _per_glyph(p),
    "no_clip": lambda p, s: _mix(np.arange(s.max() + 1) / 255.0, p["meta"])[s],
    "power_2_2": lambda p, s: _mix(np.arange(256) / 255.0, p["meta"], power=2.2)[np.minimum(s, 255)],
    "rows_cols_swapped": lambda p, s: _swapped(p),
    "float32": lambda p, s: _float32_table(p["meta"])[np.minimum(s, 255)],
}


def _failures(pages, sums, emulate):
    failed = {}
    for p in pages:
        r = judge.compare(emulate(p, sums[p["row"]["page_id"]]), p["reference"])
        if not judge.page_passes(r, p["meta"]["aa_mode"]):
            failed[p["row"]["page_id"]] = r["max_abs_error"]
    return failed


@pytest.mark.parametrize("name, n_failed, worst", [
    ("per_glyph", 26, 215), ("no_clip", 2, 200), ("power_2_2", 60, 8), ("rows_cols_swapped", 176, 255)])
def test_faulty_emulations_fail_some_page(pages, sums, name, n_failed, worst):
    """Each fails at least one page; on the 16 px stand-in, exactly the guide's counts [Measured 11 Oct]."""
    failed = _failures(pages, sums, EMULATIONS[name])
    assert len(failed) >= 1
    assert (len(failed), max(failed.values())) == (n_failed, worst)
    if name == "no_clip":
        assert {"e_overlap_coverage_srgb", "e_overlap_coverage_linear"} <= set(failed)
    if name == "power_2_2":
        assert all("coverage_linear" in pid for pid in failed)


def test_float32_emulation_passes_every_page(pages, sums):
    assert _failures(pages, sums, EMULATIONS["float32"]) == {}


# ---- static_check ------------------------------------------------------------------------------------------------

PLAUSIBLE = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _add(x_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + i, tl.load(x_ptr + i, mask=i < n) + 1, mask=i < n)


@triton.jit(do_not_specialize=["n"])
def _scale(x_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + i, tl.load(x_ptr + i, mask=i < n) * 2, mask=i < n)


def render(bitmaps, offsets, heights, widths, rows, cols, height, width, text_rgb, background_rgb, aa_mode):
    out = torch.zeros((height, width, 3), dtype=torch.uint8, device=bitmaps.device)
    _add[(triton.cdiv(out.numel(), 1024),)](out, out.numel(), BLOCK=1024)
    return out
'''


def test_static_check_plausible_triton_file():
    check = judge.static_check(PLAUSIBLE)
    assert check == {"syntax_error": None, "has_render": True, "jit": ["_add", "_scale"], "flags": []}


def test_static_check_flags_imports():
    check = judge.static_check("import os\nfrom subprocess import run\nfrom . import helper\nimport numpy as np\n")
    assert check["flags"] == ["line 1: import os", "line 2: from subprocess import ...", "line 3: from . import ..."]


def test_static_check_flags_calls_and_attributes():
    source = ("import numpy as np\nimport torch\nimport triton.language as tl\n\n"
              "def render(x):\n    f = open('x')\n    torch.save(x, 'y')\n    a = np.load('z')\n"
              "    g = getattr(x, 'y')\n    return tl.load(x)\n")
    flags = judge.static_check(source)["flags"]
    assert sorted(f.split(": ", 1)[1] for f in flags) == sorted(["open()", ".save", ".load", "getattr()"])
    assert "line 10: .load" not in flags


def test_static_check_render_jit_and_syntax():
    assert judge.static_check("class K:\n    def render(self):\n        pass\n")["has_render"] is False
    check = judge.static_check("from triton import jit\n\n@jit\ndef k(x):\n    pass\n\ndef render():\n    pass\n")
    assert check["jit"] == ["k"] and check["has_render"] and check["flags"] == []
    broken = judge.static_check("import torch\n\ndef render(:\n    pass\n")
    assert broken["syntax_error"].startswith("line 3:") and broken["has_render"] is False


def test_static_check_of_the_good_kernels():
    good_kernel = judge.static_check((FIXTURES / "good_kernel.py").read_text(encoding="utf-8"))
    assert good_kernel["flags"] == [] and good_kernel["has_render"] and good_kernel["syntax_error"] is None
    assert set(good_kernel["jit"]) >= {"_accumulate", "_composite"}
    good_torch = judge.static_check((FIXTURES / "good_torch.py").read_text(encoding="utf-8"))
    assert good_torch == {"syntax_error": None, "has_render": True, "jit": [], "flags": []}


# ---- run_attempt and score_page with the fake runner ----------------------------------------------------------

def _fake_build(tmp_path: Path, ids: list[str]) -> Path:
    build = tmp_path / "build"
    (build / "inputs").mkdir(parents=True)
    (build / "reference").mkdir()
    for pid in ids:
        np.save(build / "reference" / f"{pid}.npy", np.full((4, 5, 3), 7, np.uint8))
    return build


def test_run_attempt_with_the_fake_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", FAKE_KEY)
    ids = ["ok", "crash", "badjson", "env", "cpu", "hang_1", "hang_2", "ok_late"]
    build = _fake_build(tmp_path, ids)
    out = tmp_path / "runs with space" / "v01_1"
    judge.run_attempt(tmp_path / "kernel.py", build / "inputs", out, ids, runner=FAKE_RUNNER, timeout_s=5,
                      max_hangs=2)
    scored = {pid: judge.score_page(out, build / "reference", pid, "binary") for pid in ids}
    assert scored["ok"]["passed"] is True and scored["ok"]["error"] == "" and scored["ok"]["max_abs_error"] == 0
    assert scored["crash"] == {"passed": False, "error": "exit code 3:"}
    assert scored["badjson"] == {"passed": False, "error": "result file unreadable"}
    assert scored["cpu"] == {"passed": False, "error": "output on cpu, not on the GPU"}
    for pid in ("hang_1", "hang_2"):
        assert scored[pid] == {"passed": False, "error": "exit code None: no result within 5 s"}
    assert scored["ok_late"] == {"passed": False, "error": "not run: 2 earlier pages hung"}
    assert json.loads((out / "env.result.json").read_text(encoding="utf-8"))["key_visible"] is False
    assert scored["env"]["passed"] is True
    with pytest.raises(FileExistsError):
        judge.run_attempt(tmp_path / "kernel.py", build / "inputs", out, ["ok"], runner=FAKE_RUNNER, timeout_s=5)


# ---- write_build, verify_build and the command line -----------------------------------------------------------

SMALL = ["n_qwen_conventional_binary_ph1_natural", "c_smolvlm500m_coverage_linear_red", "e_tiny_coverage_srgb"]


def _small_build(pages, tmp_path: Path, source: str = "defaults") -> tuple[Path, Path]:
    by_id = _by_id(pages)
    build, pages_csv = tmp_path / "judge v01", tmp_path / "judge_pages.csv"
    judge.write_build(build, [by_id[pid] for pid in SMALL], pages_csv)
    (build / "build.json").write_text(json.dumps({"render_source": source}), encoding="utf-8")
    return build, pages_csv


def test_write_build_files_hashes_and_refusal(pages, tmp_path):
    build, pages_csv = _small_build(pages, tmp_path)
    rows = judge.read_csv(pages_csv)
    assert [r["page_id"] for r in rows] == SMALL and tuple(rows[0]) == judge.PAGE_COLUMNS
    assert judge.verify_build(build, rows) == []
    with np.load(build / "inputs" / f"{SMALL[0]}.npz") as z:
        assert list(z.files) == list(judge.KEYS)
        assert np.array_equal(z["bitmaps"], _by_id(pages)[SMALL[0]]["packed"]["bitmaps"])
    meta = json.loads((build / "inputs" / f"{SMALL[2]}.json").read_text(encoding="utf-8"))
    assert meta == {"height": 1, "width": 1, "text_rgb": [0, 0, 0], "background_rgb": [255, 255, 255],
                    "aa_mode": "coverage_srgb"}
    again, again_csv = tmp_path / "again", tmp_path / "again.csv"
    judge.write_build(again, [_by_id(pages)[pid] for pid in SMALL], again_csv)
    assert pages_csv.read_bytes() == again_csv.read_bytes()        # np.savez writes the same bytes again
    with pytest.raises(FileExistsError):
        judge.write_build(build, [], pages_csv)
    np.save(build / "reference" / f"{SMALL[1]}.npy", np.zeros((2, 2, 3), np.uint8))
    (build / "inputs" / f"{SMALL[2]}.json").unlink()
    assert judge.verify_build(build, rows) == SMALL[1:]
    assert judge.build_cmd(build, tmp_path / "other.csv", "defaults") == 1     # step 1: refuse an existing folder


def _review_rows(attempt: int, **verdicts) -> list[dict]:
    rows = []
    for who in ("A", "B", "owner"):
        safe, pixel = verdicts.get(who, ("yes", "yes"))
        rows.append({"attempt": attempt, "reviewer": who, "safe_to_run": safe, "pixel_work_in_triton": pixel,
                     "note": "", "date": "2026-10-23"})
    return rows


@pytest.fixture
def cli(pages, tmp_path):
    build, pages_csv = _small_build(pages, tmp_path)
    kernels = tmp_path / "kernels"
    kernels.mkdir()
    reviews = tmp_path / "reviews.csv"
    judge.write_csv(reviews, judge.REVIEW_COLUMNS, _review_rows(1) + _review_rows(2, B=("yes", "no"))
                    + _review_rows(3, owner=("no", "yes")) + _review_rows(4) + _review_rows(5))
    kw = dict(build_dir=build, pages_csv=pages_csv, reviews_csv=reviews, runs_root=tmp_path / "runs",
              tables_dir=tmp_path / "tables", runner=FAKE_RUNNER, stand_in=True)
    return {"build": build, "kernels": kernels, "kw": kw, "tables": tmp_path / "tables", "reviews": reviews,
            "pages_csv": pages_csv}


def _table(cli, attempt):
    return judge.read_csv(cli["tables"] / f"judge_v01_{attempt}.csv")


def test_run_and_summary_end_to_end(cli, capsys):
    kernels, kw = cli["kernels"], cli["kw"]
    good = kernels / "kernel_v01_1.py"
    shutil.copyfile(FIXTURES / "good_torch.py", good)
    assert judge.run_cmd(good, 1, **kw) == 0
    rows = _table(cli, 1)
    assert tuple(rows[0]) == judge.PAGE_COLUMNS[:-2] + judge.RESULT_COLUMNS
    assert [r["passed"] for r in rows] == ["True"] * 3 and {r["kernel_ms"] for r in rows} == {"0.25"}
    assert "3 of 3 pages passed" in capsys.readouterr().out
    assert judge.run_cmd(good, 1, **kw) == 1                                  # an attempt is judged once
    shutil.copyfile(FIXTURES / "good_torch.py", kernels / "kernel_v01_2.py")
    assert judge.run_cmd(kernels / "kernel_v01_2.py", 2, **kw) == 0          # B says no to pixel work: it still runs
    (kernels / "kernel_v01_3.py").write_text("import os\n\ndef render(*a):\n    os.system('x')\n", encoding="utf-8")
    assert judge.run_cmd(kernels / "kernel_v01_3.py", 3, **kw) == 0          # the owner says not safe
    assert {r["error"] for r in _table(cli, 3)} == {"not run: safety"}
    assert not (kw["runs_root"] / "v01_3").exists()
    assert judge.summary_cmd(judge_cfg=JUDGE_CFG, pages_csv=cli["pages_csv"], reviews_csv=cli["reviews"],
                             tables_dir=cli["tables"], kernels_dir=kernels) == 0
    summary = judge.read_csv(cli["tables"] / "summary.csv")
    first, second, third = summary
    assert (first["ran"], first["pixel_work_in_triton"], first["pages_passed"], first["passed_all"]) == \
        ("yes", "yes", "3", "yes")
    assert first["kernel_sha256"] == __import__("hashlib").sha256(good.read_bytes()).hexdigest()
    assert first["worst_max_abs_error"] == "0" and first["median_kernel_ms"] == "0.25"
    assert second["pixel_work_in_triton"] == "no"
    assert (third["ran"], third["pages_passed"], third["passed_all"]) == ("no", "0", "no")
    assert third["static_flags"] == "line 1: import os; line 4: .system"
    manifest = json.loads((cli["tables"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"].split("_")[1:4] == ["h2", "v01", "judge"]


@pytest.mark.parametrize("source, reason", [
    ("import torch\n\ndef render(:\n    pass\n", "not run: syntax error"),
    ("import torch\n\ndef draw(*a):\n    pass\n", "not run: no top-level render"),
])
def test_run_not_run_reasons(cli, source, reason):
    kernel = cli["kernels"] / "kernel_v01_4.py"
    kernel.write_text(source, encoding="utf-8")
    assert judge.run_cmd(kernel, 4, **cli["kw"]) == 0
    assert {r["error"] for r in _table(cli, 4)} == {reason} and {r["passed"] for r in _table(cli, 4)} == {"False"}


def test_run_without_code_block(cli):
    kernel = cli["kernels"] / "kernel_v01_5.py"
    assert judge.run_cmd(kernel, 5, **cli["kw"]) == 1                        # neither .py nor .md: a wrong path
    kernel.with_suffix(".md").write_text("I would rather not.\n", encoding="utf-8")
    assert judge.run_cmd(kernel, 5, **cli["kw"]) == 0                        # no reviews needed: nothing to read
    assert {r["error"] for r in _table(cli, 5)} == {"not run: no code block"}


def test_run_refusals(cli):
    kernel = cli["kernels"] / "kernel_v01_6.py"
    shutil.copyfile(FIXTURES / "good_torch.py", kernel)
    with pytest.raises(ValueError, match="no review row"):
        judge.run_cmd(kernel, 6, **cli["kw"])
    assert judge.run_cmd(kernel, 1, **{**cli["kw"], "stand_in": False}) == 1      # pages from the stand-in
    np.save(cli["build"] / "reference" / f"{SMALL[0]}.npy", np.zeros((1, 1, 3), np.uint8))
    assert judge.run_cmd(kernel, 1, **cli["kw"]) == 1                             # a page changed after build
    assert not (cli["tables"] / "judge_v01_1.csv").exists()


def test_read_reviews_rejects_bad_rows(tmp_path):
    reviews = tmp_path / "reviews.csv"
    rows = _review_rows(1)
    rows[0]["safe_to_run"] = "probably"
    judge.write_csv(reviews, judge.REVIEW_COLUMNS, rows)
    with pytest.raises(ValueError, match="not yes or no"):
        judge.read_reviews(reviews, 1)
    judge.write_csv(reviews, judge.REVIEW_COLUMNS, _review_rows(1) + _review_rows(1)[:1])
    with pytest.raises(ValueError, match="two rows"):
        judge.read_reviews(reviews, 1)
    with pytest.raises(ValueError, match="does not exist"):
        judge.read_reviews(tmp_path / "none.csv", 1)


# ---- render configs from the frozen full runs ----------------------------------------------------------------

def _manifest(folder: Path, sweep_name: str, reader: str, size, font="data/external/fonts/x.ttf"):
    folder.mkdir(parents=True)
    cfg = {"sweep_name": sweep_name, "reader": {"name": reader}, "sweep": {"font_file": font, "font_size_px": size}}
    (folder / "manifest.json").write_text(json.dumps({"config": cfg}), encoding="utf-8")


def test_full_run_settings(tmp_path):
    runs = tmp_path / "final" / "runs"
    _manifest(runs / "a_ladder_qwen", "ladder", "qwen", 99)
    _manifest(runs / "b_full_qwen", "full", "qwen", {"qwen": 14, "smolvlm": 16})
    _manifest(runs / "c_full_smolvlm", "full", "smolvlm", {"qwen": 14, "smolvlm": 16})
    settings = judge.full_run_settings(runs, ["qwen", "smolvlm"])
    assert {n: s[:2] for n, s in settings.items()} == {"qwen": ("data/external/fonts/x.ttf", 14),
                                                        "smolvlm": ("data/external/fonts/x.ttf", 16)}
    with pytest.raises(FileNotFoundError, match="smolvlm500m"):
        judge.full_run_settings(runs, ["qwen", "smolvlm500m"])
    _manifest(runs / "d_full_qwen", "full", "qwen", {"qwen": "from the pilot"})
    with pytest.raises(ValueError, match="not a number"):
        judge.full_run_settings(runs, ["qwen"])


def test_render_cfgs_use_the_largest_allowance_and_the_unaligned_pitch():
    cfgs = judge.render_cfgs(JUDGE_CFG, REPO_ROOT, "defaults")
    assert {n: (c["max_phase_x_px"], c["max_phase_y_px"]) for n, c in cfgs.items()} == \
        {"qwen": (111, 111), "smolvlm": (41, 41), "smolvlm500m": (63, 63)}
    assert {c["line_pitch_px"] for c in cfgs.values()} == {20} and {c["font_size_px"] for c in cfgs.values()} == {16}
    with pytest.raises(FileNotFoundError):
        judge.render_cfgs(JUDGE_CFG, REPO_ROOT, "final", final_runs=REPO_ROOT / "results" / "no such folder")


# ---- model: the good and faulty kernels through run_kernel.py on the GPU (make test-models, W4.4) -------------

@pytest.fixture(scope="module")
def built(pages, tmp_path_factory):
    """All 186 pages on disk (about 100 MB), for run_kernel.py."""
    build = tmp_path_factory.mktemp("judge") / "judge_v01"
    judge.write_build(build, pages, build.parent / "judge_pages.csv")
    return build


def _needs_triton():
    if importlib.util.find_spec("triton") is None:
        pytest.fail("triton-windows is not installed (W4.2)")


@pytest.mark.model
@pytest.mark.parametrize("name", ["good_torch.py", "good_kernel.py"])
def test_good_kernels_pass_every_page(built, pages, tmp_path, name):
    _needs_triton()
    out = tmp_path / "run"
    ids = [p["row"]["page_id"] for p in pages]
    judge.run_attempt(FIXTURES / name, built / "inputs", out, ids, timeout_s=120, max_hangs=3)
    scored = {p["row"]["page_id"]: judge.score_page(out, built / "reference", p["row"]["page_id"],
                                                    p["meta"]["aa_mode"]) for p in pages}
    failed = {pid: r for pid, r in scored.items() if not r["passed"]}
    assert failed == {} and len(scored) == 186
    assert all(r["repeat_identical"] for r in scored.values())


GOOD_TORCH = (FIXTURES / "good_torch.py").read_text(encoding="utf-8")
SIGNATURE = "def render(bitmaps, offsets, heights, widths, rows, cols, height, width, text_rgb, background_rgb, aa_mode):\n"
RETURN = "return torch.floor(255.0 * out.clamp(0, 1) + 0.5).to(torch.uint8)"
WRAP = ("\n\n_CALLS = [0]\n\n\ndef render(*args):\n    _CALLS[0] += 1\n    out = _render(*args)\n{body}    return out\n")
FAULTY = {   # name: (source, pages, what run_attempt and score_page record)
    "ndarray": (GOOD_TORCH.replace(RETURN, RETURN + ".cpu().numpy()"), 1,
                "TypeError: render returned ndarray, not a torch.Tensor"),
    "inplace": (GOOD_TORCH.replace(SIGNATURE, SIGNATURE + "    cols += 1\n"), 1,
                "RuntimeError: render changed one of its input tensors"),
    "exit": (GOOD_TORCH.replace(SIGNATURE, SIGNATURE + "    import os\n    os._exit(7)\n"), 1, "exit code 7:"),
    "float": (GOOD_TORCH.replace(RETURN, RETURN + ".float()"), 1, "dtypes float32 and uint8; both must be uint8"),
    "syntax": (GOOD_TORCH.replace("def render(", "def render(:"), 1, "SyntaxError"),
    "sleep": (GOOD_TORCH.replace(SIGNATURE, SIGNATURE + "    import time\n    time.sleep(600)\n"), 3, None),
    "sleep_second": (GOOD_TORCH.replace("def render(", "def _render(")
                     + WRAP.format(body="    if _CALLS[0] == 2:\n        import time\n        time.sleep(600)\n"), 1, None),
    "flip": (GOOD_TORCH.replace("def render(", "def _render(")
             + WRAP.format(body="    if _CALLS[0] > 1:\n        out[0, 0, 0] ^= 1\n"), 1, None),
}
FAULTY_PAGES = {"e_oblong_coverage_linear": "coverage_linear", "e_overlap_coverage_srgb": "coverage_srgb",
                "e_cut_tl_binary": "binary"}


@pytest.mark.model
@pytest.mark.parametrize("name", list(FAULTY))
def test_faulty_kernels(built, tmp_path, name):
    _needs_triton()
    source, n_pages, expected = FAULTY[name]
    assert source != GOOD_TORCH
    kernel = tmp_path / f"{name}.py"
    kernel.write_text(source, encoding="utf-8")
    out, ids = tmp_path / "run", list(FAULTY_PAGES)[:n_pages]
    judge.run_attempt(kernel, built / "inputs", out, ids, timeout_s=20, max_hangs=2)
    scored = [judge.score_page(out, built / "reference", pid, FAULTY_PAGES[pid]) for pid in ids]
    if name == "sleep":
        assert [r["error"] for r in scored] == ["exit code None: no result within 20 s"] * 2 + \
            ["not run: 2 earlier pages hung"]
    elif name == "sleep_second":
        assert scored[0]["passed"] is True and scored[0]["kernel_ms"] is None
    elif name == "flip":
        assert scored[0]["passed"] is True and scored[0]["repeat_identical"] is False
    elif name == "exit":
        assert json.loads((out / f"{ids[0]}.result.json").read_text(encoding="utf-8"))["error"].startswith(expected)
    else:
        assert scored[0]["passed"] is False and expected in scored[0]["error"]
