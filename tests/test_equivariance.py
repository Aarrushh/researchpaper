# PURPOSE: Tests for the shift-equivariance error and the vision towers behind paper Figure 0.
# WHAT GOES HERE: Equation (1) on synthetic arrays (CI); with the models, layer 0 against the notes' patch embedding, exact whole-patch moves, two forward passes, Qwen's two orders, SmolVLM's position ids in float32, and the layer-0 numbers of appendix C.3 (build guide section 2.1).
# FILLED IN AT STEP: W1.9.
# DO NOT: Load two towers at once, or a whole reader beside a tower: the laptop has 8 GiB. Do not mark the synthetic tests model: they run in CI.
"""Shift-equivariance error (build guide section 2.1)."""

from __future__ import annotations

import ast
import functools
import gc
import inspect
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from conftest import PERSUASION, READERS, REPO_ROOT
from mrr import equivariance, grid
from mrr.equivariance import e_curve, e_summary, e_value, shift_right

P = 14
NAMES = ["qwen", "smolvlm", "smolvlm500m"]
LAYERS = {"qwen": [0, 1, 8, 16, 32], "smolvlm": [0, 1, 9, 18, 27], "smolvlm500m": [0, 1, 4, 8, 12]}
SHAPES = {"qwen": (32, 32, 1280), "smolvlm": (27, 27, 1152), "smolvlm500m": (32, 32, 768)}
POSITION_IDS = {"smolvlm": 729, "smolvlm500m": 1024}
# Appendix C.3 test ranges on the E page: (E at 0.25 px, plateau), by reader and aa_mode.
APPC_RANGES = {
    ("qwen", "coverage_srgb"): ((0.32, 0.40), (1.22, 1.36)),
    ("qwen", "coverage_linear"): ((0.42, 0.51), (1.25, 1.38)),
    ("smolvlm", "coverage_srgb"): ((0.18, 0.25), (0.82, 0.96)),
    ("smolvlm", "coverage_linear"): ((0.26, 0.35), (0.89, 1.01)),
    ("smolvlm500m", "coverage_srgb"): ((0.29, 0.36), (1.19, 1.34)),
    ("smolvlm500m", "coverage_linear"): ((0.38, 0.47), (1.22, 1.36)),
}
DEFAULTS = yaml.safe_load((REPO_ROOT / "configs" / "render_defaults.yaml").read_text(encoding="utf-8"))


def _tokens(seed: int, shape: tuple[int, ...] = (4, 6, 5)) -> np.ndarray:
    """Random tokens (rows, columns, D)."""
    return np.random.default_rng(seed).normal(size=shape)


# Equation (1) on synthetic arrays: no torch, run in CI.


def test_the_page_against_itself_scores_zero():
    z0 = _tokens(1)
    assert e_value(z0, z0, _tokens(2), 0.0, P) == (0.0, 0)


@pytest.mark.parametrize("s", [2 * P, 2 * P - 0.25])
def test_a_move_of_two_patches_is_undone(s):
    """At 2P - 0.25 px the candidates are 1 and 2 patches, so the true move is among them."""
    z0 = _tokens(3)
    assert e_value(z0, shift_right(z0, 2), _tokens(4), s, P) == (0.0, 2)


def test_the_blank_page_scores_one():
    zw = _tokens(6)
    assert e_value(_tokens(5), zw, zw, 0.0, P) == (1.0, 0)


def test_no_move_compares_every_column():
    """With k = 0 the slices hold all n columns, so a change in the last column alone counts."""
    z0 = _tokens(7)
    zs = z0.copy()
    zs[:, -1] += 1.0
    e, k = e_value(z0, zs, _tokens(8), 0.0, P)
    assert k == 0
    assert e > 0


@pytest.mark.parametrize("move", [1, 2])
def test_the_move_with_the_smaller_error_is_kept(move):
    z0 = _tokens(9)
    assert e_value(z0, shift_right(z0, move), _tokens(10), 1.5 * P, P) == (0.0, move)


def test_a_tie_goes_to_the_smaller_move():
    """Columns that are all alike match after either move."""
    z0 = np.ones((4, 6, 5))
    assert e_value(z0, z0, np.zeros_like(z0), 1.5 * P, P) == (0.0, 1)


def test_shift_right_moves_columns_and_fills_the_left():
    z = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
    out = shift_right(z, 1, fill=-1.0)
    assert np.array_equal(out[:, 1:], z[:, :3])
    assert np.all(out[:, 0] == -1.0)
    assert np.array_equal(shift_right(z, 0), z)
    assert not np.any(shift_right(z, 9))


def test_shift_right_refuses_a_fractional_move():
    with pytest.raises(ValueError):
        shift_right(_tokens(1), 1.5)


@pytest.mark.parametrize(
    ("zs_shape", "s", "patch"),
    [((4, 5, 5), 0.0, P), ((4, 6, 5), -0.25, P), ((4, 6, 5), 6 * P, P), ((4, 6, 5), 0.0, 0)],
    ids=["shapes-differ", "negative-shift", "no-column-left", "no-patch-size"],
)
def test_e_value_refuses_bad_input(zs_shape, s, patch):
    with pytest.raises(ValueError):
        e_value(_tokens(1), _tokens(1, zs_shape), _tokens(2), s, patch)


def test_e_value_refuses_a_text_page_equal_to_the_blank_page():
    z = _tokens(1)
    with pytest.raises(ValueError):
        e_value(z, z, z, 0.0, P)


TOY_P = 8
TOY_POSITIONS = 0.1 * np.arange(8)[None, :, None]  # one learned position per column, as SigLIP adds them


def _toy_render(s: float) -> np.ndarray:
    """A 1 x 64 page: a bar of ink from 10 + s to 14 + s px, each pixel holding the share of it the bar covers."""
    x = np.arange(64, dtype=np.float64)
    return np.clip(np.minimum(x + 1, 14 + s) - np.maximum(x, 10 + s), 0, 1)[None]


def _toy_embed(page: np.ndarray) -> np.ndarray:
    """A patch embedding with kernel = stride = 8 on the 1 x 64 page: (1 row, 8 columns, D = 8)."""
    return page.reshape(1, 8, TOY_P) * np.arange(1, TOY_P + 1)


def _toy_embed_with_positions(page: np.ndarray) -> np.ndarray:
    return _toy_embed(page) + TOY_POSITIONS


def test_e_curve_is_zero_at_whole_patch_moves_only():
    shifts = [0.0, 0.5, 3.0, 8.0, 16.0]
    out = e_curve(_toy_render, _toy_embed, shifts, TOY_P, np.zeros((1, 64)))
    assert out.shape == (5, 1, 2)
    e, k = out[:, 0, 0], out[:, 0, 1]
    assert list(e[[0, 3, 4]]) == [0.0, 0.0, 0.0]
    assert list(k[[0, 3, 4]]) == [0.0, 1.0, 2.0]
    assert e[1] > 0 and e[2] > 0


def test_absolute_positions_spoil_whole_patch_moves():
    """Section 2.1's prediction for SmolVLM beyond layer 0: minima, not zeros, at multiples of P."""
    out = e_curve(_toy_render, _toy_embed_with_positions, [8.0, 16.0], TOY_P, np.zeros((1, 64)))
    assert np.all(out[:, 0, 0] > 0)


def test_e_curve_keeps_each_layer_in_its_place():
    """A mapping of layers gives one column per layer, in the mapping's order."""
    shifts = [0.0, 3.0, 8.0]
    blank = np.zeros((1, 64))

    def both(page):
        return {8: _toy_embed_with_positions(page), 0: _toy_embed(page)}

    out = e_curve(_toy_render, both, shifts, TOY_P, blank)
    assert out.shape == (3, 2, 2)
    assert np.array_equal(out[:, :1], e_curve(_toy_render, _toy_embed_with_positions, shifts, TOY_P, blank))
    assert np.array_equal(out[:, 1:], e_curve(_toy_render, _toy_embed, shifts, TOY_P, blank))


def test_e_curve_refuses_layers_that_change_between_pages():
    def embed(page):
        return {0 if page.any() else 1: _toy_embed(page)}

    with pytest.raises(ValueError):
        e_curve(_toy_render, embed, [0.0], TOY_P, np.zeros((1, 64)))


def test_e_summary_follows_appendix_c3():
    shifts = np.arange(0, 2 * P + 0.25, 0.25)
    e = np.where(np.isclose(shifts % P, 0), 0.0, 1.0)
    e[shifts == 0.25] = 0.3
    e[shifts == 3] = 2.0
    plateau_shifts = [s for s in range(1, 2 * P + 1) if s % P]  # 1 to 13 and 15 to 27
    want = {
        "at_multiples_of_p": 0.0,
        "at_quarter_px": 0.3,
        "plateau": (len(plateau_shifts) + 1.0) / len(plateau_shifts),
        "largest": 2.0,
    }
    assert e_summary(shifts, e, P) == pytest.approx(want)


def test_e_summary_needs_the_shifts_it_reports():
    with pytest.raises(ValueError):
        e_summary([0.0, 1.0, 14.0], [0.0, 1.0, 0.0], P)  # no 0.25 px


def test_equivariance_imports_only_numpy():
    """Section 2.1 keeps equivariance.py NumPy only, so the tests above run in CI without torch."""
    imported = set()
    for node in ast.walk(ast.parse(inspect.getsource(equivariance))):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"__future__", "math", "typing", "numpy"}


# Figure 0 from CSV-shaped rows: pandas and matplotlib, which the test extra installs; no torch.


def _fake_curves():
    """Rows as scripts/03_equivariance.py reads them back: every reader, both text types, two layers each."""
    import pandas as pd

    rows = []
    for name in NAMES:
        patch, s_max = READERS[name]["patch_size"], 2 * grid.sweep_period_px(READERS[name])
        for text_type in ("natural", "random62"):
            for layer in LAYERS[name][:2]:
                for s in np.arange(0, s_max + 0.25, 0.25):
                    rows.append((name, text_type, layer, float(s), float(s % patch > 0), int(s // patch)))
    return pd.DataFrame(rows, columns=["reader", "text_type", "layer", "shift_px", "e", "k_used"])


def test_plot_equivariance_draws_a_panel_per_reader_with_the_period_lines(tmp_path, monkeypatch):
    """Three panels in readers.yaml order; four curves each (two layers x two text types); a vertical line at
    every multiple of P up to s_max, plus one at every multiple of the attention window where the reader has one."""
    from matplotlib.figure import Figure

    from mrr import plotting

    saved = []
    real_savefig = Figure.savefig
    monkeypatch.setattr(
        Figure, "savefig", lambda fig, *args, **kwargs: (saved.append(fig), real_savefig(fig, *args, **kwargs))[1]
    )
    out = tmp_path / "fig0.pdf"
    plotting.plot_equivariance(_fake_curves(), [READERS[name] for name in NAMES], out)
    assert out.read_bytes()[:5] == b"%PDF-"
    (fig,) = saved
    assert [ax.get_title() for ax in fig.axes] == NAMES
    for ax, name in zip(fig.axes, NAMES):
        cfg = READERS[name]
        s_max = 2 * grid.sweep_period_px(cfg)
        curves = [line for line in ax.lines if len(line.get_xdata()) > 2]
        rules = [line for line in ax.lines if len(line.get_xdata()) == 2]
        assert len(curves) == 4
        assert {line.get_linestyle() for line in curves} == {"-", "--"}
        want = s_max // cfg["patch_size"] + (s_max // cfg["window_px"] if cfg.get("window_px") else 0)
        assert len(rules) == want
        assert ax.get_xlim() == (0.0, float(s_max))


@pytest.mark.parametrize("spoil", ["text_type", "reader"])
def test_plot_equivariance_refuses_an_unknown_text_type_or_a_missing_reader(tmp_path, spoil):
    from mrr import plotting

    df = _fake_curves()
    if spoil == "text_type":
        df.loc[df.index[0], "text_type"] = "prose"
    else:
        df = df[df["reader"] != "smolvlm"]
    with pytest.raises(ValueError):
        plotting.plot_equivariance(df, [READERS[name] for name in NAMES], tmp_path / "fig0.pdf")


# With the models: the float32 towers of section 2.1, one at a time.


@pytest.fixture(scope="module", autouse=True)
def _free_gpu_after_the_module():
    yield
    if "torch" in sys.modules:
        import torch

        gc.collect()  # the last tower, which pytest has released by now
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@pytest.fixture(scope="module", params=NAMES)
def tower(request):
    """Each reader's float32 tower, loaded once: pytest runs every test of one tower before the next."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from mrr import readers

    gc.collect()  # the previous tower, which pytest has released by now
    torch.cuda.empty_cache()
    return readers.load_vision_tower(READERS[request.param])


@functools.cache
def _e_page(name: str, text_type: str) -> tuple[dict, str]:
    """The E page of section 2.1 for one reader: (render config, text). text_type is natural or random62."""
    from mrr import render, text_source

    font = REPO_ROOT / DEFAULTS["font_file"]
    if not (PERSUASION.exists() and font.exists()):
        pytest.skip("book or font not present")
    s_max = 2 * grid.sweep_period_px(READERS[name])
    cfg = grid.resolve_render_cfg(
        DEFAULTS, READERS[name], font_file=str(font), max_phase_x_px=s_max, max_phase_y_px=0
    )  # font_file as a str for freetype-py
    if text_type == "natural":
        book = text_source.strip_gutenberg_boilerplate(text_source.load_book(PERSUASION))
        source = text_source.normalise_text(text_source.skip_front_matter(book, "CHAPTER I.", 2))
    else:
        source = text_source.random_text(200_000, 0)
    return cfg, text_source.paginate(source, lambda candidate: render.fits_page(candidate, cfg), n_pages=1)[0]


def _render(cfg: dict, text: str, s: float) -> np.ndarray:
    from mrr import render

    return render.render_page(text, {**cfg, "phase_x_px": s})


def _blank(page: np.ndarray) -> np.ndarray:
    """A page of the background colour, the size of page."""
    return np.full(page.shape, DEFAULTS["background_color"], dtype=np.uint8)


def _layer0(tower, page: np.ndarray) -> np.ndarray:
    from mrr import readers

    return readers.vision_hidden_states(tower, page, [0])[0]


def _notes_embed(img: np.ndarray, kernel: np.ndarray, bias: np.ndarray, mean, std) -> np.ndarray:
    """Layer-0 patch embedding of a grey uint8-valued image (H, W), copied from the notes' embed
    (scripts/make_figures.py:242). Returns (rows, cols, D) in float64.

    Valid padding, stride = kernel = P: rows and columns that do not fill a patch are dropped.
    """
    x = (img[None] / 255.0 - np.asarray(mean, dtype=np.float64)[:, None, None]) / np.asarray(
        std, dtype=np.float64
    )[:, None, None]
    d, c, p, _ = kernel.shape
    hn, wn = x.shape[1] // p, x.shape[2] // p
    t = x[:, : hn * p, : wn * p].reshape(c, hn, p, wn, p).transpose(1, 3, 0, 2, 4)
    t = t.reshape(hn * wn, c * p * p)
    return (t @ kernel.reshape(d, -1).T + bias).reshape(hn, wn, d)


@pytest.mark.model
def test_hidden_states_have_the_tower_shapes(tower, native_page):
    from mrr import readers

    _, page = native_page(tower.name)
    states = readers.vision_hidden_states(tower, page, LAYERS[tower.name])
    assert list(states) == LAYERS[tower.name]
    for z in states.values():
        assert z.shape == SHAPES[tower.name]
        assert z.dtype == np.float64


@pytest.mark.model
def test_layer0_matches_the_notes_patch_embedding(tower, native_page):
    """The notes' float64 embed of the same page, with the tower's own weights: the notes saved theirs from
    the pinned revisions (weights_record.json in the notes). Qwen's two temporal kernel slices are summed, as
    the notes' load_reader (:222) does, because the processor puts the page in both frames."""
    _, page = native_page(tower.name)
    assert (page == page[..., :1]).all()  # a grey page: the notes' embed takes one channel
    if tower.name == "qwen":
        kernel = tower.module.patch_embed.proj.weight.detach().cpu().double().numpy().sum(axis=2)
        bias = np.zeros(kernel.shape[0])
    else:
        conv = tower.module.embeddings.patch_embedding
        kernel = conv.weight.detach().cpu().double().numpy()
        bias = conv.bias.detach().cpu().double().numpy()
    image_processor = tower.processor.image_processor
    want = _notes_embed(page[..., 0], kernel, bias, image_processor.image_mean, image_processor.image_std)
    got = _layer0(tower, page)
    assert got.shape == want.shape
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-5


@pytest.mark.model
def test_layer0_is_exact_at_whole_patch_moves(tower):
    cfg, text = _e_page(tower.name, "natural")
    p = tower.patch_px
    page = _render(cfg, text, 0)
    z0, zw = _layer0(tower, page), _layer0(tower, _blank(page))
    for move in (1, 2):
        e, k = e_value(z0, _layer0(tower, _render(cfg, text, move * p)), zw, float(move * p), p)
        assert k == move
        assert e < 1e-6


@pytest.mark.model
def test_two_forward_passes_agree_at_every_layer(tower, native_page):
    from mrr import readers

    _, page = native_page(tower.name)
    layers = LAYERS[tower.name]
    first, second = (readers.vision_hidden_states(tower, page, layers) for _ in range(2))
    blank = readers.vision_hidden_states(tower, _blank(page), layers)
    for layer in layers:
        assert e_value(first[layer], second[layer], blank[layer], 0.0, tower.patch_px)[0] < 1e-6


@pytest.mark.model
def test_qwen_block0_input_is_the_patch_embedding(tower, native_page):
    """Block 0 gets the patch embedding in window order; qwen_to_raster must undo both orders exactly."""
    if tower.name != "qwen":
        pytest.skip("Qwen only: SmolVLM has no window order")
    import torch
    from mrr import readers

    _, page = native_page("qwen")
    seen = []
    hook = tower.module.blocks[0].register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().cpu()))
    try:
        layer0 = _layer0(tower, page)
    finally:
        hook.remove()
    (block0_input,) = seen
    grid_thw = torch.tensor([[1, page.shape[0] // tower.patch_px, page.shape[1] // tower.patch_px]])
    raster = readers.qwen_to_raster(block0_input, grid_thw, tower.module, window_order=True)
    assert np.array_equal(raster.numpy(), layer0)


@pytest.mark.model
def test_float32_smolvlm_tower_sees_every_position_id(tower, native_page):
    """D15: 729 and 1,024 distinct ids on the first page; load_vision_tower adds the fix of section 1B.5 only
    if the float32 tower needs it, and says so in position_ids_fixed."""
    if tower.name not in POSITION_IDS:
        pytest.skip("SmolVLM only: Qwen has no position table")
    torch = pytest.importorskip("torch")

    _, page = native_page(tower.name)
    seen = []
    table = tower.module.embeddings.position_embedding
    hook = table.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().cpu()))
    try:
        _layer0(tower, page)
    finally:
        hook.remove()
    (ids,) = seen
    assert ids.unique().numel() == POSITION_IDS[tower.name]
    assert torch.equal(ids[0], torch.arange(POSITION_IDS[tower.name]))


@pytest.mark.model
@pytest.mark.parametrize("text_type", ["natural", "random62"])
def test_layer0_e_at_a_quarter_pixel_and_on_the_plateau(tower, text_type):
    """Appendix C.3: on the E page, E(0.25) and the mean over whole-pixel shifts that are not multiples of P
    lie in the test ranges of the aa_mode in configs/render_defaults.yaml."""
    cfg, text = _e_page(tower.name, text_type)
    quarter, plateau = APPC_RANGES[(tower.name, cfg["aa_mode"])]
    shifts = [0.25] + [float(s) for s in range(1, cfg["max_phase_x_px"] + 1)]
    out = e_curve(
        lambda s: _render(cfg, text, s),
        lambda page: _layer0(tower, page),
        shifts,
        tower.patch_px,
        _blank(_render(cfg, text, 0)),
    )
    numbers = e_summary(shifts, out[:, 0, 0], tower.patch_px)
    assert numbers["at_multiples_of_p"] < 1e-6
    assert quarter[0] <= numbers["at_quarter_px"] <= quarter[1]
    assert plateau[0] <= numbers["plateau"] <= plateau[1]


@pytest.mark.model
def test_qwen_to_raster_puts_the_processor_patches_back_in_place(cpu_reader, native_page):
    """Without the GPU: Qwen's processor patches, put back in raster order, rebuild the normalised page."""
    pytest.importorskip("torch")
    from mrr import readers

    reader = cpu_reader("qwen")
    _, page = native_page("qwen")
    image_processor = reader.processor.image_processor
    p = image_processor.patch_size
    inputs = readers.processor_inputs(reader, page, reader.cfg["prompt"])
    patches = inputs["pixel_values"].reshape(-1, 3, image_processor.temporal_patch_size, p, p)[:, 0, 0]
    visual = SimpleNamespace(spatial_merge_size=image_processor.merge_size)  # all window_order=False reads
    raster = readers.qwen_to_raster(
        patches.reshape(len(patches), p * p), inputs["image_grid_thw"], visual, window_order=False
    )
    gh, gw, _ = raster.shape
    rebuilt = raster.reshape(gh, gw, p, p).permute(0, 2, 1, 3).reshape(gh * p, gw * p).numpy()
    want = (page[..., 0] / 255.0 - image_processor.image_mean[0]) / image_processor.image_std[0]
    np.testing.assert_allclose(rebuilt, want, atol=1e-5)
