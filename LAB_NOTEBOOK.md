<!--
PURPOSE: Append-only log of daily work by both authors.
WHAT GOES HERE: One dated entry per working session: what you did, found, broke and will do next.
FILLED IN AT STEP: Every step, starting at W1.1.
DO NOT: Edit or delete past entries. Do not paste secrets, tokens or personal data.
-->

# Lab notebook

**Rule: append only.** Add new entries at the bottom. Never edit or delete an old entry. If an old entry is wrong, add a new entry that corrects it and says which entry it corrects.

## Entry template (copy this, do not fill it in place)

```
## YYYY-MM-DD, <name>
Did:
Found:
Broke:
Next:
```

<!-- Entries start below this line. -->

## 2026-10-12, Claude (D22)
Did:
- Re-planned the calendar for one laptop (D20, D21). G1 Wed 14 Oct 12:00; `prereg-v1` Wed 14 Oct 18:00; G2 Fri 16 Oct 12:00; full run Fri 16 Oct to Wed 21 Oct 06:00; freeze and D16 Wed 21 Oct; G3 Sat 24 Oct; submission Wed 28 Oct; deadline Fri 30 Oct, 19:00 UTC. Slip rule: if G1 has not passed by Wed 14 Oct 12:00, every later date moves by one day, D16 included, and the cut order applies (stretch analyses, then H2, then SmolVLM-500M, then the anti-aliasing factor).
- Logged D7 to D19 as open and D20 to D23 as decided in `docs/05_decisions.md`. Appended steps W1.8 to W1.10 and W3.6 to W3.8 to `docs/03_pipeline.md`, moved the gate dates and added the freeze gate.
- Created the environment: uv-managed CPython 3.11.15 at `%APPDATA%\uv\python\cpython-3.11.15-windows-x86_64-none`, with a virtual environment `.venv` in the working tree (ignored by git). torch 2.14.1+cu130 and torchvision 0.29.1+cu130 come from `https://download.pytorch.org/whl/cu130`. Also installed: transformers 5.19.0, accelerate 1.15.0, numpy 2.4.6, scipy 1.17.1, jiwer 4.0.0, pillow 12.3.0, freetype-py 2.5.1, matplotlib 3.11.2, pandas 3.0.6, pytest 9.1.1, PyYAML 6.0.3, huggingface-hub 2.2.0, tokenizers 0.23.3.
- Fetched each reader's weights at its pinned revision into `%USERPROFILE%\.cache\huggingface\hub`: `Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3`, `HuggingFaceTB/SmolVLM-Instruct@81cd9a775a4d644f2faf4e7becff4559b46b14c7`, `HuggingFaceTB/SmolVLM-500M-Instruct@a7da5b986cb59b408707209984f360a5f4ad7e47`.
Found:
- Machine: Windows 11 Home; 31.7 GB RAM; NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB, driver 610.74; long paths enabled (`LongPathsEnabled` = 1).
- The torch check printed `2.14.1+cu130 13.0 True NVIDIA GeForce RTX 5060 Laptop GPU ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`, then `transformers 5.19.0 numpy 2.4.6 freetype (2, 13, 2)`.
- GNU make is not installed. Until it is, each `make` target is run as its recipe.
Broke:
- Nothing.
Next:
- W1.3: `requirements.txt`, `requirements.lock.txt`, the `model` marker (D17), the Makefile, CI and `scripts/00_check_env.py`.

## 2026-10-12, Claude (D22): W1.3
Did:
- `requirements.txt` gains `freetype-py` and loses `qwen-vl-utils`, which nothing imports. `pyproject.toml` gains `freetype-py` in `dependencies`, `scipy` and `jiwer` in the `test` extra, and the `model` marker (D17). CI and `make test` run `-m "not todo and not model"`; `make test-models` runs `-m model -v`.
- `scripts/00_check_env.py` now fails unless the GPU's architecture is in the torch wheel's list and freetype-py bundles FreeType 2.13.2. It prints the `nvidia-smi` line and only warns when `triton-windows` is missing.
- Installed `requirements.txt` (which added jupyter and its dependencies) and ran `pip install -e ".[test]"`. `pip check` found no broken requirements, and no pinned version moved.
- Wrote `requirements.lock.txt`: the cu130 index line first, then four header comment lines, then `pip freeze --exclude-editable` (138 packages).
Found:
- `make check-env`, run as `python scripts/00_check_env.py` because make is not installed, exits 0. It prints `torch CUDA build 13.0`, `Total VRAM 7.96 GiB`, `Architecture sm_120`, `Wheel arches sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`, `nvidia-smi NVIDIA GeForce RTX 5060 Laptop GPU, 610.74, 8151 MiB` and `FreeType (2, 13, 2)`.
- `pytest -m "not todo and not model"`: 23 passed, 11 deselected.
Broke:
- Nothing.
Next:
- W1.6: `configs/readers.yaml`. Then the CPU renderer track (W1.5, W1.4).

## 2026-10-12, Claude (D22): W1.5, colour
Did:
- `src/mrr/color.py`: the exact piecewise sRGB decode and encode, vectorised in float64. The encode switches pieces at 0.04045/12.92, computed in code (D24, which supersedes D12). The docstring cites W3C CSS Color 4 for the constants, and the DO NOT line now reads "Approximate the transfer curve with one power law (D24)."
- `tests/test_color.py`: removed the `todo` mark; added the decode of 128/255 (0.21586050011389926) and the encode of 0.5 (187.516, which rounds to 188).
- New `tests/test_encoding.py`: every `open`, `read_text` and `write_text` call in `src/`, `scripts/`, `h2/*.py` and `h2/judge/` must name its encoding, because Python 3.11 on Windows otherwise uses the locale's code page. The scaffold has no offenders. A planted file was flagged at the right lines. A `zipfile` handle's `.open(...)` is flagged too; that is dealt with when `text_source.py` reads EPUB files (W1.4).
- `references/citations.md`: the sRGB row, checked against the dated draft. `docs/05_decisions.md`: D12 superseded by D24.
Found:
- The build guide (section 1A.1, and its text for D12) says the W3C CSS Color 4 sample code uses the matched threshold. It does not: `gam_sRGB` in the Candidate Recommendation Draft of 9 October 2026, section 19, uses 0.0031308. CSS Color 4 is cited for the constants only, and the threshold is our choice (D24). The learning notes of 8 October had it right.
- Round trip over 100,001 points: largest error 1.665e-16 with 0.04045/12.92 and 2.959e-8 with 0.0031308, at v = 0.04045. The power piece starts 2.33e-9 above the end of the straight piece; the two thresholds differ by 4.954e-9.
- Python reads lines 1 and 2 of a source file for an encoding declaration (PEP 263), so a header line there that contains "coding=" or "coding:" followed by letters, digits, "-", "_" or "." breaks the file. The first draft of the WHAT GOES HERE line of `tests/test_encoding.py` ended in "encoding=.", and pytest stopped with "SyntaxError: unknown encoding: .". Header lines 1 and 2 of a Python file must not contain that pattern.
- `make test`, run as its recipe: 29 passed, 8 deselected.
Broke:
- Nothing.
Next:
- W1.5 continues: `src/mrr/grid.py`, then glyph loading in `render.py`.

## 2026-10-12, Claude (D22): W1.5 and W1.6, grid
Did:
- `src/mrr/grid.py`: `token_cell_px` (checks patch × merge against the config), `sweep_period_px` (the lcm of patch, cell and window: 112, 42 and 64 px), `token_count` (((h // P) // m) · ((w // P) // m): 256, 81 and 64 on the native pages), `blind_strip_px` (0, 6 and 0 px), `page_size_px` (replaces `native_size`, D8), `qwen_smart_resize` (a copy of `smart_resize` in transformers 5.19.0), `split_phase_64`, `phase_offsets` and `snap_to_grid`.
- New `resolve_render_cfg(render_defaults, reader_cfg, **overrides)`. It sets each reader's page size, the margin (one patch: 14, 14 and 16 px), the phase allowances (the sweep period − 1 where the default is null), the patch-aware slot (one patch where null) and the blind strip, then applies the overrides and rejects unknown keys. This closes two gaps in the guide: the allowance keys had no default without a reader, and Phase 1 never said how a reader's geometry reaches the render config.
- `snap_to_grid` now returns the smallest multiple of the cell at or above x, as section 1A.7 of the guide says; the scaffold's docstring said the largest multiple at or below. Nothing called it.
- New `tests/test_grid.py`: 41 tests, with the readers read from `configs/readers.yaml`.
Found:
- `qwen_smart_resize` agrees with transformers' `smart_resize` (with the snapshot's 3136 and 12845056) on all 185,556 sizes tried: heights 28 to 1999 px in steps of 3, widths in steps of 7.
- The scaffold's token formula gives 83.6 tokens for SmolVLM-2.2B's 384 px frame; the encoder's 27 × 27 patches, merged 3 × 3, give 81.
- `make test`, run as its recipe: 70 passed, 8 deselected.
Broke:
- Nothing. The comment at `scripts/01_check_readers.py:38` still names `grid.native_size`; the rewrite of that script at W1.6 replaces it.
Next:
- Glyph loading in `render.py` (W1.5).

## 2026-10-12, Claude (D22): W1.5, glyphs
Did:
- `src/mrr/render.py`: `Glyph` and `load_glyph`. Unhinted outlines (`FT_LOAD_RENDER | FT_LOAD_NO_HINTING | FT_LOAD_NO_BITMAP`); the size set with `set_char_size(0, floor(64 · size + 0.5), 72, 72)`; the phase, in 1/64 px, applied to the outline with `set_transform` and reset after every glyph, failure included; the advance from `linearHoriAdvance / 65536` (D11). Faces are cached per font and size, glyphs per argument tuple, and the cached coverage is read-only. A character the font lacks raises ValueError naming its code point, and so does a phase outside 0 to 63. `load_glyph_coverage` returns the phase-zero coverage.
- Fonts are opened through a Python file stream, not by path (see Found).
- `tests/test_render.py`: the module-level todo mark became one mark on each of the two `render_page` tests. Nine new glyph tests: the seven in the plan, a phase outside 0 to 63, and a font under a non-ASCII path.
Found:
- freetype-py 2.5.1: `face.size.ascender` and `face.size.descender` are in 26.6 fixed point: 1024 and −384 for Source Sans 3 at 16 px, that is 16 px and −6 px. `Face.get_char_index` returns 0 for a character the font lacks (U+4E00), 42 for `o` and 1 for the space.
- freetype-py hands a path to FreeType as UTF-8 bytes. A copy of the font in a folder named `fönts_ü` failed with "cannot open resource", and a `pathlib.Path` failed with AttributeError; a binary file stream opened both.
- Source Sans 3 at 16 px, as section 1A.2 of the guide measured: the coverage of `o` over phases k/8 sums to 25.388 to 25.420 (spread 0.124%); the linear advances of `m` and `i` are 13.264 and 3.936 px, against a hinted `advance.x` of 13.0 px for `m`; half a pixel of downward phase moves the centroid of `H` 0.522 px down; the advance of `m` scales by 1.031248 from 16 to 16.5 px.
- The guard search of `render.py` for `ImageDraw`, `ImageFont`, `truetype(`, `FT_LOAD_DEFAULT`, `FT_LOAD_TARGET`, `advance.x` and `set_pixel_sizes` finds nothing.
- `make test`, run as its recipe: 79 passed, 8 deselected.
Broke:
- Nothing.
Next:
- `composite_coverage` (W1.5).

## 2026-10-12, Claude (D22): W1.5, compositing
Did:
- `src/mrr/render.py`: `composite_coverage` mixes the text colour over the page once, with alpha from equation (4) of section 1A.3 and the mix of equation (5): stored sRGB values for `binary` and `coverage_srgb`, linear light for `coverage_linear`. x and y must be Python or NumPy integers, otherwise TypeError, because sub-pixel placement now belongs to `load_glyph`. The colour is 0 to 255 and is divided inside. The placement is clipped to the page, the page is written in place, and nothing is rounded. The page and the coverage must be float64 in [0, 1], so a uint8 page or a page still in 0 to 255 is refused. `AA_MODES` names the three modes.
- `tests/test_render.py`: the black-on-white table; for each mode, every page level at c = 0 and every text level at c = 1; coloured text at partial coverage; the coordinate, mode and contract checks; and three clipping cases (far corner, negative origin, wholly off the page).
Found:
- In `coverage_srgb`, all 16,777,216 combinations of 8-bit coverage k, page level B and text level T round, through our code and equation (6), to the exact round-half-up of n/255 with n = (255 − k)·B + k·T. None comes closer to a rounding boundary than 0.0019607843137, which is 1/510 up to float error. Two correct implementations therefore agree byte for byte in this mode, as the H2 judge in Phase 4 of the guide assumes.
- In `coverage_linear`, white decodes and encodes to 0.9999999999999999, not 1.0, because 1.055 − 0.055 is not exact in float64; it still stores as 255. The c = 0 and c = 1 tests compare stored bytes, not floats.
- `make test`, run as its recipe: 106 passed, 8 deselected.
Broke:
- Nothing.
Next:
- `src/mrr/text_source.py` (W1.4).

## 2026-10-13, Claude (D22): W1.4, text
Did:
- `src/mrr/text_source.py`: four functions from section 1A.8 of the guide. PR #1's `load_book` and `strip_gutenberg_boilerplate` are unchanged.
  - `skip_front_matter(text, marker, occurrence)` drops everything up to and including the given occurrence of a chapter heading. The marker's words match across any run of whitespace, case-sensitively and as whole words, so "Letter 1" cannot match "Letter 12"; the guide does not ask for whole words, but a partial match would skip to the wrong place without an error. Too few occurrences raise ValueError.
  - `normalise_text` (D13): NFC; ‘ and ’ to ', “ and ” to ", the em dash to -; whitespace collapsed and the ends stripped. Case and the other non-ASCII characters are kept.
  - `paginate(text, fits, n_pages, start=0)` replaces the stub's `paginate(text, chars_per_page)`. Each page takes as many words as `fits` allows, found by doubling and then halving, which assumes `fits` is monotone; greedy line breaking is. `start` is a word index, so a later run can begin after the last word used (section 2.4). A word that fits no page by itself raises ValueError.
  - `random_text(n_chars, seed, word_len=5)`: `np.random.default_rng(seed).integers(0, 62, size=n_chars)` over digits, upper case and lower case, joined into words of five and cut to exactly `n_chars`. It ends with a space when `n_chars` is a multiple of six.
- `data/raw/texts/README.md`: the table of front-matter markers, and the open choice on later chapter headings as TODO[W2.2] (proposed: keep them).
- `tests/test_text_source.py`: 22 new tests. The three markers on the committed books, a heading broken over two lines, whole-word matching and missing occurrences; the normalisation map and NFC; the two pinned random strings, length, alphabet, words of five, uniformity and determinism; pagination (pages join back, each fits, each is full, a second call resumes at a word, the search is not linear, a word that fits no page is refused); every character of the three normalised books loads in both fonts (Review Focus 2).
Found:
- In pg105 and pg84 the first occurrence of the marker is in the contents list. pg1661's contents list spells the title "I. A Scandal in Bohemia", so its first case-sensitive match is the heading, which breaks over two lines: "I.\nA SCANDAL IN BOHEMIA". A plain substring search for the marker finds it 0 times.
- The skip removes 390, 422 and 607 characters; pg1661's editorial note ("THERE IS AN ILLUSTRATED EDITION ...", at character 12 of the stripped text) goes with it.
- After the skip and normalisation the books have 463,186, 417,812 and 558,577 characters (83,279, 74,973 and 104,419 words), 4.48 characters per word, punctuation included. All three were NFC already. The map changes 10,136 of 1,439,575 characters (0.70%). Ten other non-ASCII characters remain, 95 times in all: £ 37, æ 27, é 14, ê 6, œ 4, ô 2, è 2, ½ 1, à 1, â 1. Both fonts draw every character.
- `paginate` called `fits` 1,399 times for 100 pages of 100 words; one word at a time would call it 10,099 times.
- In 62,000 characters of `random_text` every one of the 62 appears, 944 to 1,047 times. `random_text(200000, 20261022)`, the random text of the Phase 4 checks, has 33,334 words.
- `make test`, run as its recipe: 128 passed, 8 deselected.
Broke:
- Nothing. Nothing called the old `paginate`.
Next:
- Layout and `configs/render_defaults.yaml` (W1.5).

## 2026-10-13, Claude (D22): W1.5, layout
Did:
- `src/mrr/render.py`: the layout of section 1A.4 of the guide.
  - `layout_block(text, cfg)` lays a page out once, in block coordinates. `layout_page` adds floor(64·margin + 0.5) + floor(64·phase + 0.5) on each axis and refuses a phase outside [0, its allowance]. `fits_page` counts lines against floor((h − a − d)/p) + 1.
  - `layout_lines(text, font, width_px, line_pitch_px)` stays as the word-wrap helper. Wrapping is greedy. A word wider than the line starts a new line, is split where the line is full and is logged on the `mrr.render` logger; a single character wider than the line raises ValueError.
  - Added beyond the guide's interface: `LayoutFont` (the advances of one layout: linear, or slots of q with the centring offset), `block_size_px(cfg)` (page − 2·margin − allowance − blind strip; an unresolved config raises ValueError naming `grid.resolve_render_cfg`), `unaligned_pitch_px(size)` and `RENDER_MODES`.
  - `PlacedGlyph.x_64` is the glyph's origin, its pen position plus the patch-aware centring offset, because section 1A.5 draws each glyph at `x_64` and adds nothing.
- `configs/render_defaults.yaml`: lines 324 to 336 of the guide, plus a 448 × 448 page overwritten per reader (plan finding F4), `max_phase_x_px` and `max_phase_y_px` null (F5), and the DO NOT line of D8. Two departures from the guide's block:
  - the `aa_mode` comment names D11 and Gate 1 instead of "see the decision below", which points at a box in the guide, not in the file;
  - the scaffold's enum comments on `aa_mode` and `render_mode` stay.
  `aa_mode` changes from the scaffold's `coverage_linear` to `coverage_srgb`, as section 1A.9 proposes. D11 stays open until Gate 1.
- `tests/test_render.py`: 46 new tests:
  - the unaligned pitch;
  - the filled-in defaults;
  - an unresolved config and an unknown render mode are refused;
  - the block origin;
  - linear advances and the space;
  - patch-aware slots and centring;
  - the same (character, line) sequence at phases 0, 0.5 and 111;
  - a whole-pixel phase moves every glyph by 64·s on its own axis only;
  - margin plus phase on the 500M page;
  - phases outside the allowance are refused;
  - a 60-character word is split inside the block and logged;
  - a character wider than the line is refused;
  - the line limit in both directions;
  - the line counts, block sizes and page capacities of section 2.4;
  - no ink in SmolVLM-2.2B's blind strip at the largest phase, in both layouts.
- Moved: the two coherence tests of section 1A.4 (pooled R1 about 0.022 on a conventional 2,500-glyph Persuasion text; per-class 1.000 for `patch_aware`) need the R_k of section 2.2, so they move to Task P2.2 (plan finding F7).
Found:
- `fits_page` reproduces the page-capacity table of section 2.4 exactly: all 27 character counts, the 18 line counts and the 3 block sizes.
- All 86 characters of the three normalised books are narrower than one 14 px slot at 12, 14 and 16 px. The widest is "œ", 13.424 px at 16 px. Patch-aware layout is therefore one glyph per patch.
- The guide says a word wider than the block should never happen at the sizes used. It happens once.
  - In pg84 an em dash joins "industrious" and "painstaking,"; D13 maps it to a hyphen, which makes a 24-character word.
  - In 14 px slots that word is 336 px, over the 315 px block of Qwen and the 310 px block of SmolVLM-2.2B at every size, so patch-aware layout splits it and logs it.
  - In the 500M's 16 px slots it is 384 px and fits the 420 px block.
  - In conventional layout the widest word is 163.4 px at 16 px, and nothing splits.
  - Proposed for Gate 1: keep the guide's rule.
- Coherence by construction, on a Qwen page at phase 0:
  - conventional glyphs use all 64 fractions of a pixel, and 8 of 32 character classes hold one position mod 14;
  - patch-aware glyphs use one fraction, and all 30 classes hold one position;
  - a 20 px pitch puts the baselines at 7 phases mod 14 and 7 mod 28, while a 28 px pitch puts them all at one.
- At the largest phase with the default allowances (sweep period − 1), the last ink on a first page is:
  - Qwen: row 424 and column 431 of 448;
  - SmolVLM-2.2B: row 354 and column 361 of 384, short of the blind strip at 378;
  - SmolVLM-500M: row 478 and column 494 of 512.
- The space advance at 16 px is 3.20001220703125 px, which is 3.2 in 16.16 fixed point.
- Paginating 200 Qwen pages of Persuasion took 0.20 s: 2,854 calls to `fits_page`, 0.07 ms each with the glyphs cached.
- `make test`, run as its recipe: 174 passed, 8 deselected.
Broke:
- Nothing. The old `layout_lines` stub had no callers. The two `todo` tests in `tests/test_render.py` still read the unresolved defaults; Task P1.9 gives them a resolved Qwen config when `render_page` exists.
Next:
- `render_coverage`, `render_page` and `overlay_grid` (W1.5, sections 1A.5 and 1A.6).

## 2026-10-13, Claude (D22): W1.5, page
Did:
- `src/mrr/render.py`:
  - `render_coverage(text, cfg)` puts the coverage of every glyph from `layout_page` on one page-sized float64 canvas. Each glyph is rasterised at f = v & 63 and placed at n = v >> 6, with its first row at n_y − top and its first column at n_x + left. Coverage is added with `np.minimum(1, ...)` inside a slice that is clipped to the page before it is indexed.
  - `render_page(text, cfg)` checks `background_color`, composites once with `composite_coverage(page, coverage, 0, 0, text_color, aa_mode)` and applies equation (6) once: floor(255·clip(o, 0, 1) + 0.5). Step 5 of the plan writes the conversion without the clip, which equation (6) has.
  - `overlay_grid(img, cell_px, patch_px=None, window_px=None, blind_from_px=None)` returns a new uint8 image. Its lines are blended with the page, so the ink under them stays visible: patch lines faint, cell lines strong, window lines two pixels wide, and the blind strip shaded.
- `tests/test_render.py`:
  - The two `todo` tests now run on a resolved Qwen config (allowances 111 px), with `render_page` imported at the top. The shared setup (`READERS`, `DEFAULTS`, `_cfg` and the Persuasion fixture) moved to the top of the file.
  - 21 new tests:
    - exact translation by s ∈ {1, 7, 13, 14, 28, 111} in x and in y, with the vacated strip white;
    - the sub-pixel changes of "minimum";
    - identical bytes after the glyph cache is cleared;
    - no ink lost at the largest phase, in both layouts, compared with `math.fsum` (plan finding F18);
    - no coverage in SmolVLM-2.2B's blind strip at the largest phase;
    - left-edge clipping at margin 0 (Review Focus 5);
    - the configured colours, and a bad background refused;
    - the overlay: a copy, the line strengths, the window width, rows and columns, the blind strip and bad input;
    - a guard that no module under `src/mrr/` and no script except `02_render_check.py` uses `overlay_grid` (section 1A.6).
Found:
- "minimum" at 16 px moved by 0.25, 0.5 and 0.75 px changes 343, 388 and 408 pixels by at most 64, 128 and 192 levels, exactly the guide's values. Its total coverage changes by −0.041%, −0.051% and −0.032%.
- The coverage of a Qwen first page sums to the same value at phase 0 and at (111, 111), in both layouts: 12,163.6549 conventional and 5,602.4706 patch-aware. `np.sum` happened to agree as well; `math.fsum` is the check that does not depend on the order of addition.
- "j" starts one column left of its pen position (bitmap_left −1). At margin 0 its first column is cut off, and nothing reaches the right edge.
- One Qwen page of 651 characters renders in about 17 ms with the glyphs cached (16.9 and 17.1 ms in two runs of 20 pages).
- On that page `binary` uses 2 levels, `coverage_srgb` 256 and `coverage_linear` 183. The mean pixel is 239.7, 239.5 and 244.4: mixing in linear light draws lighter text, as note A4 predicts.
- `make test`, run as its recipe: 197 passed, 6 deselected. `pytest tests/test_render.py -q`, the guide's acceptance command: 105 passed.
Broke:
- Nothing. `render_page` and `overlay_grid` were stubs; `overlay_grid` gained four keyword arguments with defaults.
Next:
- `scripts/02_render_check.py` and `make render-check` (W1.5, section 1A.10).

## 2026-10-13, Claude (D22): W1.5, render check
Did:
- `scripts/02_render_check.py` (section 1A.10):
  - `--reader` takes the cell, the native page and the grids from `configs/readers.yaml`, through `grid.token_cell_px` and `grid.resolve_render_cfg`. Without `--reader`, the cell is `--cell-px` (default 28) on the render defaults' own 448 × 448 page, with allowances of one cell less a pixel and only cell lines drawn.
  - It renders the first page of Persuasion after its front matter at `grid.phase_offsets(cell, 4)` and at 0.5 px. Each phase moves the text right and down by the same amount, so one image checks both axes.
  - For each phase it saves `render_check_<reader>_<phase>.png`, the page under `overlay_grid` with the reader's patch, cell and window lines and blind strip. It also saves `render_check_<reader>_<phase>_crop.png`, the overlay around the first word, "Sir", enlarged 10 times by repeating pixels. The crop box is the same at every phase: the whole cells that hold the word's ink at all five phases, plus the next cell line. That is rows and columns 0–56 for Qwen, 0–84 for SmolVLM-2.2B and 0–128 for SmolVLM-500M.
  - It prints the SHA-1 of each raw page before the overlay, the commit, and whether the tree had uncommitted changes.
  - The scaffold's run folder and manifest under `results/runs/` are gone: a render check is not a run, and section 1A.10 keeps its record in this notebook.
- `Makefile`: `render-check` runs the script for qwen, smolvlm and smolvlm500m and stops at the first failure. It calls `$(PYTHON)` unquoted, like every other recipe; the plan had it in quotes.
- This laptop has no `make`, so the recipe's loop ran by hand with `.venv/Scripts/python.exe`: 4.3 s for the three readers.
- The 30 images (1.5 MB) are committed. Appendix B.2 of the guide lists `reports/figures/render_check/` among the files programs write and marks the outputs that git ignores; this folder is not marked.
- `docs/03_pipeline.md`: W1.5 ticked. Both test files pass without todo marks, and the render check writes its overlays.
Found:
- The SHA-1 of each raw page (uint8, H × W × 3). The pages hold 651 characters on Qwen and SmolVLM-2.2B and 1,207 on SmolVLM-500M, in Source Sans 3 at 16 px, `coverage_srgb`, conventional layout, pitch 20 px and the default allowances. The renderer is that of commit e113777.

| Reader | Page (px) | Phase (px) | SHA-1 of the raw page |
|---|---|---|---|
| qwen | 448 × 448 | 0 | c5870615c5911407077fe39e7c403b18a772020a |
| qwen | 448 × 448 | 7 | 23fef6d11c538a82bf4937308319a9739a6ec5fe |
| qwen | 448 × 448 | 14 | 8d72f65c6e0dc826620e168cfee82564c96f2436 |
| qwen | 448 × 448 | 21 | 7cbe1d4d375a1005dbaa90d8b19e013fc0cb1fed |
| qwen | 448 × 448 | 0.5 | bd3fe51d8f33c8e9222412ba6a792a7243d1d6d6 |
| smolvlm | 384 × 384 | 0 | 9f5a0fa3de1062c9ce2716cc8d28833a80237907 |
| smolvlm | 384 × 384 | 10.5 | 8aa1c6c888e9c82544403e9f6fe333aa731336f0 |
| smolvlm | 384 × 384 | 21 | b876632151f2bc54050ecf3d2a641bdae2ce66c3 |
| smolvlm | 384 × 384 | 31.5 | d5f0b06b082bdac7cd6fc92ea8657b178436f312 |
| smolvlm | 384 × 384 | 0.5 | b314195c4e09e4f1290d2fe02324beec0b8e3344 |
| smolvlm500m | 512 × 512 | 0 | 05c9d5a73ba29806723b54752787abc8b4181ab6 |
| smolvlm500m | 512 × 512 | 16 | 49653040226dd950132020dbee75d360d13656f2 |
| smolvlm500m | 512 × 512 | 32 | 0e6e3918e6f9c176aa057be52107a21f5399e470 |
| smolvlm500m | 512 × 512 | 48 | 50bc130864774935d97374292b2dfbeab0e98fbf |
| smolvlm500m | 512 × 512 | 0.5 | 4370419e99f0f3de3627bb4e26a889aa4430dfb5 |

- On the same pages, every whole-pixel phase equals phase 0 moved right and down by the phase, with white where the text moved away: Qwen 7, 14 and 21, SmolVLM-2.2B 21, and SmolVLM-500M 16, 32 and 48.
- SmolVLM-2.2B's cell is 42 px, so `phase_offsets(42, 4)` gives 0, 10.5, 21 and 31.5: two of its four cell phases are half-pixel phases. The 10.5 and 31.5 pages equal the 0.5 page moved by 10 and 31 px. The fraction sets the coverage and the whole part only moves it, so on the crops the 10.5 and 31.5 letters match the 0.5 letters, not the 0 letters.
- The 0.5 px phase changes 28,060 pixels, by up to 192 levels, on the Qwen and SmolVLM-2.2B pages (the same 651 characters in the same 309 px block), and 52,180 pixels on the SmolVLM-500M page.
- I looked at seven of the 30 images: the Qwen crops at 0, 7 and 0.5; the SmolVLM-2.2B crops at 0 and 21 and its page at 31.5; and the SmolVLM-500M crop at 48. Whole-pixel phases keep the letters' shapes and move them against the grid. The 0.5 crop has different edge greys; for example, the stem of "i" spreads over two grey columns. No ink reaches the SmolVLM-2.2B blind strip. The authors' own inspection at 1,000% zoom (section 1A.11, check 2) is still to come.
- Versions: Source Sans 3 "Version 3.052" and Source Serif 4 Display "Version 4.005", read from name ID 5 of the committed files; FreeType 2.13.2, bundled with freetype-py 2.5.1.
- Section 1A.11, check by check:
  1. `pytest -m "not todo" -q`: 197 passed, 6 deselected. The 6 are the todo tests of later steps; no `model` tests exist yet.
  2. Five overlays and five crops per reader. My look is above; the authors' is to come.
  3. `grep -rnE "ImageDraw|ImageFont|truetype\(" src/` prints nothing when run as `/usr/bin/grep`. Plain `grep` in Claude's shell here stops with "Unmatched ( or \(" on this pattern, though the same pattern works on piped text, so something between the command and grep drops the backslash.
  4. `grep -rnE "FT_LOAD_DEFAULT|FT_LOAD_TARGET|advance\.x" src/` prints nothing.
  5. This entry records the versions and the hashes.
- Plan findings F5, F6 and F7 at the end of Phase 1A:
  - F5 (allowance keys): closed. `render_defaults.yaml` sets them null, and `resolve_render_cfg` fills in the sweep period − 1 for each reader (111, 41 and 63 px). The narrower allowances of section 2.4 will come in as overrides.
  - F6 (reader geometry): closed by `resolve_render_cfg` for the page size, margin, slot, blind strip and allowances. Still open: `line_pitch_mode` exists only in the sweep configs (`TODO[W2.2]`, `TODO[W3.2]`) and in the results columns. The sweep runner (Phase 2) must turn it into `line_pitch_px`: `aligned_pitch_px` from `readers.yaml` (28, 42 and 64 px), or floor(1.25 · size + 0.5).
  - F7 (coherence test): moved to Task P2.2, as the layout entry says.
Broke:
- Nothing. The script's interface changed: `--config` now has a default, `--reader` and `--readers-config` are new, `--out-dir` defaults to `reports/figures/render_check`, and `--output-root` is gone. Only the `render-check` recipe called the script.
Next:
- `src/mrr/readers.py` and the position-id fix (W1.6, sections 1B.4 and 1B.5).

## 2026-10-13, Claude (D22): W1.6, readers
Did:
- `src/mrr/readers.py` (sections 1B.4 and 1B.5). Every torch and transformers import is inside a function.
  - `Transcription` (frozen): the text, the number of new tokens, whether the limit cut it off, and the latency. `Reader`: name, model, processor, config and image token id.
  - `load_reader`:
    - It refuses a revision that is not a 40-character commit hash.
    - It loads `AutoProcessor`, then the model class the guide names for that processor: `Qwen2_5_VLForConditionalGeneration` for `Qwen2_5_VLProcessor`, `Idefics3ForConditionalGeneration` for `Idefics3Processor`.
    - The model gets the config's dtype and `device_map`, `attn_implementation="sdpa"`, and `max_memory` only when the config has it (rung 2 of section 1B.8). Then `.eval()`.
    - When the config says `position_id_fix: pre_hook`, the fix goes on before any other hook.
    - The image token's id must equal the id the processor expands, `processor.image_token_id`. `readers.yaml` has the ids only in comments, and Task P1.12 pins 151655, 49153 and 49190.
  - `install_position_id_fix` is copied from P1:540–553, with `import torch` moved inside the function.
  - `transcribe_full(reader, image, prompt, max_new_tokens=None)`. None takes the config value. Anything other than a positive integer is refused, so the `TODO[W1.7]` in `readers.yaml` stops a run with a clear message.
    - The page goes in as `Image.fromarray(image)`, which must give mode RGB, with the chat messages of P1:520 and the processor called with keywords and `do_resize=False`.
    - Under `torch.inference_mode()`: synchronise, time `generate` with the four greedy arguments, synchronise. Qwen gets the whole processor output, `mm_token_type_ids` included.
    - Only the new tokens are decoded, with `skip_special_tokens=True, clean_up_tokenization_spaces=False`.
  - Checks on every call, before the inputs move to the GPU:
    - Qwen's `image_grid_thw` is [[1, H/14, W/14]].
    - SmolVLM's `pixel_values` is (1, 1, 3, H, W) with an all-true `pixel_attention_mask`.
    - `input_ids` holds `grid.token_count` image tokens. S:138 asks for the count on every call too.
  - `count_vision_tokens` builds the same inputs with the reader's own prompt and counts the image tokens. `transcribe` returns `transcribe_full(...).text`.
- `tests/test_smoke.py`: `test_readers_import_without_torch` imports `mrr.readers` in a fresh interpreter and checks that neither torch nor transformers is in `sys.modules`. The test can fail: an interpreter that imports torch first does show it there.
- Three choices the plan's text does not make:
  - `hit_limit` is true when the reader produced `max_new_tokens` tokens and the last one is not an end token (`generation_config.eos_token_id`). Section 1B.4, step 6, says "equals max_new_tokens", and its dataclass comment says "stopped at max_new_tokens, not at an end token". The two differ only when the end token is the last one allowed, and then the text is complete.
  - `new_tokens` counts the end token when there is one.
  - The text is the raw decode. It starts with a space after "Assistant:", which `normalise_text` trims before scoring (D13).
Found:
- The three pinned processors on CPU, one native page each, with `do_resize=False` (transformers 5.19.0):

| Reader | Processor | Vision input | Image tokens | `<\|im_start\|>` in input_ids |
|---|---|---|---|---|
| qwen | Qwen2_5_VLProcessor | `pixel_values` 1,024 × 1,176; `image_grid_thw` [[1, 32, 32]] | 256 | 3 (system, user, assistant) |
| smolvlm | Idefics3Processor | `pixel_values` (1, 1, 3, 384, 384); mask all true | 81 | 1 |
| smolvlm500m | Idefics3Processor | `pixel_values` (1, 1, 3, 512, 512); mask all true | 64 | 1 |

- Qwen's processor output also holds `mm_token_type_ids`.
- For both SmolVLM readers, undoing the normalisation, x = (p + 1) · 127.5 rounded, gives back the page exactly, so the no-resize test of Task P1.12 should pass for them. With `do_resize=False` the native page still passes through `resize_for_vision_encoder` at its own size, and `split_images` keeps it whole, so `do_image_splitting` need not be set.
- The chat template is part of the input:
  - Qwen's template adds its default system line, "You are a helpful assistant.", when the messages have none.
  - SmolVLM's prompt is `<|im_start|>User:<image>…<end_of_utterance>\nAssistant:`, and its tokenizer adds no second `<|im_start|>`.
- Plan finding F19: Pillow 12.3.0 gives no deprecation warning for `Image.fromarray(page, mode="RGB")`. The code infers the mode and checks it anyway.
- Loading SmolVLM prints "[transformers] Model config: pad_token_id must be `None` or an integer within the vocabulary (between 0 and 31999), got 128002." Nothing is padded at batch size 1.
- GPU smoke run of the new code on the first page of Persuasion (the render-check page), SmolVLM readers only. Qwen waits for its placement measurement (Task P1.14).

| Reader | Image tokens | Peak memory | New tokens at a limit of 400 | Ids seen by a hook registered after the fix |
|---|---|---|---|---|
| smolvlm500m | 64 | 1.00 GiB | 54; `hit_limit` false | 1,024 distinct, equal to 0 to 1,023 |
| smolvlm | 81 | 4.32 GiB | 400; `hit_limit` true | 729 distinct, equal to 0 to 728 |

- At a limit of 10, both readers give 10 tokens with `hit_limit` true.
- The config's `TODO[W1.7]` limit is refused with the message above.
- The latencies (8.5 s and 30.4 s) include the first call's warm-up. W1.7 measures them properly.
- D15, measured on CPU with the pinned `Idefics3VisionEmbeddings`, random weights (the ids do not depend on them):
  - In bfloat16, one 384 frame gets 441 distinct ids of 729 without the fix (21 distinct rows and 21 columns), and 729 with it. The guide's 441 is reproduced.
  - In float32 it gets 729 without the fix, so the guide's other option works too.
  - Rows and columns 15, 17, 19, 21, 23 and 25 lose their own id. For example, 15/27 = 0.555556 becomes 0.554688 in bfloat16, below its bucket boundary, so row 15 takes row 14's id.
  - SmolVLM-500M gets all 1,024 ids in bfloat16 even without the fix, because k/32 is exact in bfloat16. For it the fix changes nothing.
- D15, one page on the GPU:
  - SmolVLM-2.2B without the fix stops by itself after 139 tokens, with no loop. It gets "Somersetshire" and "Baronetage" right, but drops "his own" and writes "an idylle", "a frisses", "contemplatig" and "the earliest earls".
  - With the fix it gets "for his own amusement" and "occupation for an idle hour; and consolation in a distressed one" right, but writes "Somerset" and "Baronette". From character 210 on, it repeats ";" up to the limit.
  - One page decides nothing. The question is which ids SmolVLM saw in training: if the bfloat16 cast entered transformers after SmolVLM was trained, the fix restores its training conditions. This is to check before D15 is decided at W1.8.
- Greedy decoding with `repetition_penalty=1.0`, which the guide requires on every call, can loop, as on the page above. D13 keeps such a transcription and flags it. Its insertions count against the CER, and the page costs the full limit in time, which W1.7's t has to include.
Broke:
- Nothing. `load_reader` now returns a `Reader`, which its callers only pass back to `count_vision_tokens` and `transcribe`.
- The `make test` recipe, run by hand: 198 passed, 6 deselected.
Next:
- Task P1.12 on the GPU: the no-resize and reader-facts tests (W1.8). D15 gets more pages, with and without the fix, and the history of the bfloat16 cast, for your decision at Gate 1.

## 2026-10-13, Claude (D22): W1.8, no-resize test and reader facts
Did:
- `tests/test_no_resize.py` (section 1B.6, B2), marked `model`. It needs the processors only, no weights.
  - Each reader's native page goes through `readers.processor_inputs`, which builds the inputs of every reader call. The processor's rescale and normalisation are then undone, x = (p · std + mean) / rescale_factor rounded half up, and the result must equal the rendered bytes everywhere.
  - Two pages per reader: the first page of Persuasion, and uniform random noise. The text page is grey (R = G = B), so it cannot catch swapped channels; the noise page has every level in every channel.
  - `unpatchify_qwen(pixel_values, grid_thw)` inverts `patchify` (lines 177–210 of the pinned `image_processing_qwen2_vl.py`). It is tested first on an 84 × 140 px coordinate image, not square, cut by the pinned `patchify` itself. Qwen's two time copies must be equal.
  - Beyond the plan: the model casts `pixel_values` to bfloat16 before the patch embedding. A test checks that the 256 grey levels still give 256 distinct value triples after that cast.
  - Negative controls: with their defaults, Qwen's processor makes a 450 px page 448 px, and SmolVLM's makes 17 frames and 1,377 image tokens of a 384 px page. `processor_inputs` refuses a 450 px Qwen page and SmolVLM pages of 378, 400, 500 and 520 px.
- `tests/test_reader_facts.py` (section 1B.6), marked `model`.
  - Processors: 256, 81 and 64 image tokens per native page, each equal to `grid.token_count`; image-token ids 151655, 49153 and 49190; transformers' `smart_resize` maps 448, 1000 and 70 px to 448, 1008 and 56; `grid.qwen_smart_resize` agrees with it on all 25 pairs of 448, 384, 512, 1000 and 70 px, and its default bounds and the copy in `readers.yaml` are the processor's (3,136 and 12,845,056 pixels); one `<|im_start|>` in each SmolVLM prompt.
  - GPU, one model at a time: SmolVLM's position ids without and with the fix, recorded by a hook registered after the fix (P1:584); columns 378–383 and rows 378–383 of a SmolVLM-2.2B page, with column 377 as the control; the same page read twice, 256 tokens, for each reader.
- `tests/conftest.py`: the reader configs and two session fixtures, `native_page` (the first page of Persuasion at a reader's native size) and `cpu_reader` (the pinned processor in a `Reader` without a model).
- `src/mrr/readers.py`: `_inputs` is now public as `processor_inputs`. The tests need the exact inputs, and so will the equivariance runs (W1.9). Its behaviour is unchanged.
- Every model test starts with `pytest.importorskip("torch")`, binding the name `torch` only where the test uses it (section 1B.2, rule 4). With torch and transformers missing, all 62 skip.
- Mutation check on a temporary copy: `unpatchify_qwen` with rows and columns swapped fails the coordinate test and both Qwen pages; SmolVLM's channels reversed fail the noise pages only.
Found:
- `make test-models`, run by hand with `HF_HUB_OFFLINE=1` on the RTX 5060 Laptop GPU (8 GiB): 62 passed, 204 deselected, in 190 s.
- Undoing the processor gives back the page exactly for every reader, on text and on noise. Before rounding, the largest error is 2.5e-5 of a level for Qwen and 3.8e-6 for SmolVLM.
- D15 on the GPU with the real weights: SmolVLM-2.2B gets 441 distinct position ids without the fix and 729 with it, equal to 0 to 728; SmolVLM-500M gets 1,024 either way.
- The blind strip (D8): noise in columns 378–383 or in rows 378–383 leaves both the vision encoder's output and the connector's output bit-identical. Noise in column 377 changes both.
- In bfloat16 the 256 grey levels stay distinct for all three readers, but one channel loses a level. Qwen's normalisation puts blue levels 253 and 254 at 2.117457 and 2.131677, and both become 2.125: CLIP's blue standard deviation spaces the levels 0.01422 apart, and bfloat16 values between 2 and 4 are 0.015625 apart. Red and green still separate them, so grey text loses nothing; coloured text near white could.
- `do_resize=False` does not stop SmolVLM's processor from resizing. With image splitting on, `resize_for_vision_encoder` takes each side up to a multiple of the frame: 378 px becomes 384, and 400 px becomes 768, then 2 × 2 tiles and a global image. The shape check in `processor_inputs` refuses every such page; only the native page passes unchanged.
- Qwen's determinism test loaded Qwen with `device_map: cuda` and passed in 77 s, load included. Whether the weights spilled into shared system memory is measured in Task P1.14.
Broke:
- Nothing. The `make test` recipe, run by hand: 198 passed, 68 deselected.
Next:
- Task P1.13: `scripts/01_check_readers.py` and `make readers` (W1.6).
- D15 for Gate 1: when the bfloat16 cast entered transformers, and CER with and without the fix on more pages.

## 2026-10-13, Claude (D22): W1.6, check readers
Did:
- `scripts/01_check_readers.py` (section 1B.7), committed as e4ec23b and run as the `make readers` recipe by hand, with `HF_HUB_OFFLINE=1`. Before and after the run, `git status --porcelain -- src scripts configs pyproject.toml requirements.lock.txt` printed nothing.
  - One native page per reader, the first page of Persuasion, in place of the scaffold's three sizes and `token_check.csv`: under D8 each reader has one page size, and the shape check in `readers.processor_inputs` refuses any other.
  - Checks before the transcription: the model is in the configured dtype; with `device_map: cuda`, every parameter is on the GPU; the processor takes the page unchanged; `readers.count_vision_tokens` equals `grid.token_count`. A failed check stops the script, and a reader's manifest is written only after its checks pass (section 1B.7, item 2).
  - Until W1.7 sets `max_new_tokens`, the limit is ⌈1.5 × the page's reference tokens⌉, the rule of section 1B.9 applied to this page.
- D7 and D8 decided; D1 superseded by D7. `docs/01_research_plan.md:27` names the three readers with their patches and cells, `docs/02_learning_guide.md` links the SmolVLM-2.2B card, read at the pinned revision, and `docs/04_data_dictionary.md` gives D8's page sizes.
Found:

| Reader | Page | Encoder input | Vision tokens (`grid.token_count`) | New tokens, ended by | First call | Peak allocated |
|---|---|---|---|---|---|---|
| qwen | 448 × 448 | `image_grid_thw [[1, 32, 32]]`: 32 × 32 patches of 14 px, 16 × 16 tokens | 256 (256) | 150 of 197, end token | 47.6 s | 7.07 GiB |
| smolvlm | 384 × 384 | `pixel_values (1, 1, 3, 384, 384)`: 27 × 27 patches of 14 px, 9 × 9 tokens | 81 (81) | 200 of 200, the limit | 57.7 s | 4.29 GiB |
| smolvlm500m | 512 × 512 | `pixel_values (1, 1, 3, 512, 512)`: 32 × 32 patches of 16 px, 8 × 8 tokens | 64 (64) | 54 of 462, end token | 15.4 s | 1.00 GiB |

- All three load in bfloat16, with every parameter on the GPU and no `hf_device_map`. Manifests: `results/runs/20261013-1123_checkreaders_qwen_e4ec23b/`, `20261013-1124_checkreaders_smolvlm_e4ec23b/` and `20261013-1125_checkreaders_smolvlm500m_e4ec23b/`.
- Reading, with whitespace collapsed (`score.py` comes at W2.4):
  - Qwen: the transcription equals the page's 651 characters, CER 0.
  - SmolVLM-2.2B: CER 0.677. It reads the first 209 characters, then writes 151 semicolons until the limit.
  - SmolVLM-500M: CER 0.825. It paraphrases the opening ("Walter Elliot, Kellylynn Hall, was a man who found his own amusement, …") and stops after 234 of the page's 1,207 characters.
  - All three pages use 16 px text. The size ladder (W2.x) chooses the sizes the sweep reads at.
- Provenance: the Qwen manifest has `git_dirty: false`; the two SmolVLM manifests have `git_dirty: true`. That is only because the Qwen run folder was untracked when they were written: `git_dirty` reads `git status --porcelain`, which lists untracked files. Section 2.5 of the guide keeps `git_dirty` as it is and adds `code_dirty` for the runner (Task P2.5).
- Power: the GPU's enforced power limit was 30 W of a possible 100 W, and while generating it drew 27 to 30 W (nvidia-smi every 5 s). At this morning's 20 W limit the same three calls took 172.3, 157.6 and 19.8 s. These latencies are first calls, with warm-up, not W1.7's t.
- Memory: during Qwen's call nvidia-smi showed at most 7,424 MiB used of 8,151 MiB, so Qwen fits in the card's own memory with about 0.7 GiB to spare. The sysmem fallback is on (a probe allocated 9.00 GiB on the 7.96 GiB card), so a shortage of memory would show as slowness, not as an error. Task P1.14 settles the placement (D18).
Broke:
- Nothing. This morning's first run used the uncommitted script, so its manifests had `git_dirty: true`. They were moved out of the repository and the run was repeated after the commit.
Next:
- Task P1.14 (D18): Qwen's placement, from the peak memory above.
- Task P1.15 (W1.7): t per reader at full power, and `max_new_tokens`.

## 2026-10-13, Claude (D22): Phase 1 review, readers and environment
Did:
- A fresh reviewer read the readers and environment code of Phase 1 against the checklists of sections 1A.11 and 1B.10. Changes from its findings:
  - `src/mrr/readers.py`:
    - `load_reader` refuses a SmolVLM block without a `position_id_fix` key, which must say `pre_hook`, or null to read without the fix. A block that lost the key used to load without the fix, unnoticed (D15).
    - The position-id fix replaces the ids of a full frame only: n ids, of which exactly one is 0 in each image. Padded patches get id 0, and a frame that is not square puts several patches in bucket 0, so such frames can also have n ids; they now keep theirs.
    - `processor_inputs` refuses a page that is not uint8 (H, W, 3) before the processor sees it, and a Qwen page that is not whole 28 px token cells, which used to fail inside `patchify` with a bare reshape error.
  - `scripts/00_check_env.py` checks the setting of section 1B.2: it allocates and fills 256 MiB blocks up to 1.1 times the card's memory, and fails if the driver allows it, which means the sysmem fallback is on.
  - `scripts/01_check_readers.py`: the manifest records every generate argument, `readers.GREEDY` and the limit (section 1B.4), and the realised placement: the parameters' devices, `hf_device_map` and the peak allocated memory (D18). The script's own token-count check went, because `processor_inputs`, which builds the inputs of every reader call, makes the same check.
  - `pyproject.toml` lists Pillow, which `readers.py` imports.
  - `configs/readers.yaml`: the comments on `min_pixels` and `max_pixels` say what reads them, `grid.page_size_px`, and that the processor never gets them.
  - Tests:
    - `tests/test_readers.py` (new): the page checks, in CI; `transcribe_full` on a fake model, marked `model` because it imports torch: `hit_limit`, the greedy arguments and the limit passed to `generate`, and the refusal of a limit that is not a positive integer.
    - `tests/test_reader_facts.py`: `load_reader` refuses a SmolVLM block without the key. On the pinned embeddings in bfloat16, without weights, the fix changes the ids of a full frame only: the full 27 × 27 frame goes from 441 distinct ids to 0 to 728, while a padded frame (14 of 27 columns valid) and a 9 × 81 frame keep theirs. On the GPU, the fix as `load_reader` installs it gives a full frame the ids 0 to n − 1. The determinism test also requires a non-empty text.
    - `tests/test_no_resize.py`: the 450 px Qwen page is now refused by the cell check, with its own message.
Found:
- On the GPU, with `HF_HUB_OFFLINE=1`: `pytest -m "not todo" tests/test_readers.py tests/test_no_resize.py tests/test_reader_facts.py`, 88 passed in 209 s.
- `scripts/00_check_env.py` exits 1: "the driver let this script allocate 1.1 x the card's memory, so the CUDA sysmem fallback is on". Everything else passes, as at W1.3. It will exit 0 once the NVIDIA Control Panel's CUDA - Sysmem Fallback Policy is set to Prefer No Sysmem Fallback (owner action 2).
Broke:
- Nothing. The `make test` recipe, run by hand on the working tree, which also holds the render review's tests (next entry): 222 passed, 85 deselected.
- `scripts/01_check_readers.py` was not rerun for these changes. Its next run is `make readers` at Task P1.16, with the limits of W1.7.
Next:
- The render review's tests (next entry), then W1.7.

## 2026-10-13, Claude (D22): Phase 1 review, render
Did:
- A second fresh reviewer read the rendering code, `render.py`, `grid.py` and `text_source.py`, and its tests against section 1A.11. It found no wrong result, but behaviour that no test pinned, so that a later change could break it unnoticed. New and stronger tests:
  - `test_a_sub_pixel_phase_moves_the_ink_by_the_phase`: a phase of 0.25, 0.5 or 0.75 px in x or in y moves the centroid of the ink by that amount along that axis (within 0.02 px) and not along the other (within 0.01 px). The table of section 1A.5 shifts x only, so the size of a y phase had no test.
  - `test_half_pixel_down_moves_H_down` requires a move of half a pixel (0.522 measured, within 0.05), not merely a move down.
  - `test_overlapping_glyphs_add_their_coverage_clipped_at_one`: two glyphs whose boxes overlap, at the same point, a third of a pixel apart and one pixel apart. The page's coverage is their sum clipped at 1 (section 1A.5, step 2), and the clip is reached.
  - `test_a_line_may_end_exactly_at_the_block_edge`: a word fits if it ends at the edge (section 1A.4). In patch-aware layout every character, the space too, takes one 14 px slot, so a 308 px block holds exactly 22, and such ties are common (D10).
  - `test_transform_reset_between_glyphs` now reads the cached face itself: after a load at a phase of 32/64 px, the next load from that face is unshifted, whoever makes it.
  - `test_src_draws_no_text_with_pillow_or_hinted_freetype`: checks 3 and 4 of section 1A.11, the greps over `src/`, now run in CI.
  - `test_paginate_puts_a_text_that_fits_on_one_page`: texts of 1, 2, 3, 4, 5 and 8 words that fit come back as one page. The doubling search must try the whole rest of the text also when its length is a power of two.
- Docstrings: `render_page` says that it does not check the text against the page; `grid.page_size_px` says what it checks and what it leaves to the no-resize test and `processor_inputs`; `_epub_to_text` says that spine items that are not XHTML are skipped.
- Mutation check on a copy of the tree, outside the repository: ten one-line changes to `render.py` and `text_source.py`, each run against the CPU tests of `test_render.py` and `test_text_source.py`. The unchanged copy passed, 152 tests. Every change failed the tests written for it:

| Change | Tests that failed |
|---|---|
| The y phase ignored, halved or doubled | the sub-pixel test at all three phases; for halved and doubled, also the half-pixel test |
| Coverage added without the clip, by maximum instead of sum, or overwritten | the overlap test at all three distances |
| `<` for `<=` in the line-wrap or in the word-fit comparison | the block-edge test |
| The face's transform not reset after a glyph | the transform-reset test |
| `<` for `<=` in paginate's doubling loop | the one-page test at 2, 4 and 8 words |

Found:
- The `make test` recipe, run by hand: 222 passed, 85 deselected, in 14.6 s.
- For W1.10 (Task P2.2), from the reviewer's recomputation of the coherence table of D10 on its own cut of the text: in the conventional rows the straddle shares range from 0.384 to 0.425, against the guide's 0.384 to 0.430; R2(14) is 0.018 against 0.019, and R1(28) is 0.014 against 0.013. Both flip with a change of about 10 glyphs in where the text is cut. The patch-aware row matches exactly. Nothing to fix now; Task P2.2 measures them on its own pages.
Broke:
- Nothing.
Next:
- W1.7: timing, memory and `max_new_tokens`.

## 2026-10-13, Claude (D22): W1.7, timing, memory and `max_new_tokens`
Did:
- Section 1B.9, with a throwaway script outside the repository, run with `HF_HUB_OFFLINE=1` while nvidia-smi logged every 5 s. For each reader, one at a time: the first page after the front matter of Persuasion, Frankenstein and The Adventures of Sherlock Holmes, and three random pages (`random_text`, seeds 9101 to 9103), at 16 px on the native page with the render defaults. After one unrecorded 8-token warm-up call, each page was transcribed three times with the limit below.
- `max_new_tokens` in `configs/readers.yaml` is ⌈1.5 × the longest reference⌉, counted in the reader's own tokens over both text types (section 1B.9, item 3). The random pages set it: 575 to 581 characters of random text are 388 to 419 tokens, while 646 to 651 characters of prose are 131 to 145 (SmolVLM-500M's larger page: 1,067 to 1,073 and 1,187 to 1,207 characters, 761 to 771 and 258 to 308 tokens).
- `scripts/01_check_readers.py` takes the limit from `readers.yaml`; the per-page rule it used until now is gone.
- D4's open fields are tagged `TODO[G1]` instead of `TODO[W1.7]`: Gate 1 records the G1 outcome as D4 (section 2.10).
Found:

| Reader | Reference tokens, prose; random | `max_new_tokens` | t, median of 18 calls | Prose: median latency, calls cut by the limit | Random: median latency, calls cut by the limit | Peak allocated (reserved) |
|---|---|---|---|---|---|---|
| qwen | 131, 140, 142; 388, 389, 394 | 591 | 70.8 s | 42.5 s, 0 of 9 | 82.2 s, 0 of 9 | 7.07 (7.12) GiB |
| smolvlm | 133, 141, 145; 413, 415, 419 | 629 | 104.8 s | 76.5 s, 6 of 9 | 289.5 s, 9 of 9 | 4.36 (4.45) GiB |
| smolvlm500m | 258, 271, 308; 761, 763, 771 | 1,157 | 131.0 s | 66.8 s, 3 of 9 | 178.5 s, 9 of 9 | 1.03 (1.09) GiB |

- Placement: every parameter of every reader on the GPU, no `hf_device_map`. During Qwen's calls nvidia-smi showed at most 7,423 MiB used of 8,151 MiB.
- New tokens: Qwen wrote 141 to 158 on prose and 397 to 405 on random pages, always stopping by itself. SmolVLM-2.2B wrote its full 629 on two of the three prose pages and on every random page; SmolVLM-500M wrote its full 1,157 on one prose page and on every random page.
- The three repeats gave identical text on every page, for every reader.
- Reading, as CER with whitespace collapsed (jiwer; `score.py` comes at W2.4): Qwen 0.000 on all nine prose calls, and a median of 0.107 on random pages (0.099 to 0.113); SmolVLM-2.2B medians of 0.896 on prose and 1.024 on random pages; SmolVLM-500M 0.825 and 0.949. At 16 px both SmolVLM readers fall into loops on most pages: with greedy decoding and no repetition penalty, once a token or a short pattern repeats, repeating it stays the likeliest continuation until the limit, so each such page costs the whole limit.
- The conditions were not the pilot's, so these t are provisional:
  - The laptop slept three times. The nvidia-smi log stops for 351 s (13:53 to 13:59), 1,911 s (14:04 to 14:36) and 6,496 s (16:07 to 17:55), and the longest calls, 494 s and 1,997 s for Qwen and 6,624 s for SmolVLM-500M, span those gaps. The medians barely move; the means (195.5, 182.9 and 522.8 s) mean nothing.
  - The laptop was on battery whenever it was checked (17:58, 18:19 and 18:29), on the Balanced plan, and the GPU's enforced power limit moved between 20 and 96 W during the run.
  - Decoding was limited by the CPU, not the GPU. At 17:58, during SmolVLM-500M's calls, the Python main thread kept one core 83% busy at about 2.4 GHz, while the GPU was 6 to 29% busy and drew 16 to 24 W. The same work ran at very different rates: Qwen took 28.8 to 93.1 s for the same 150 to 158 prose tokens, and SmolVLM-2.2B's random pages slowed from 5.4 to 1.1 new tokens per second over its stage. SmolVLM-500M's first prose page took 59 to 68 s for 54 tokens in all three repeats, against about 6 tokens per second on its other prose pages; that is not explained.
  - Review agents and test runs shared the CPU during the Qwen and SmolVLM-2.2B stages.
- Pilot budget (D20: one laptop, the readers in series): 1,000 × 70.8 + 1,240 × 104.8 + 1,080 × 131.0 = 342,260 s, or 95.1 h, against 43,200 s (12 h). It does not fit. At the fastest rate each reader reached (11.9, 8.5 and 10.2 new tokens per second), the same calls would still need 236,770 s, or 65.8 h, because the SmolVLM readers write their whole limit on most pages. Section 2.12 takes t from the ladder's rows at the chosen size (section 2.9), under the pilot's conditions (plugged in; sleep, hibernation and the lid action off), and if a laptop still does not fit, cuts both text types to the largest equal page count that fits. At these t, one page of each type would fit.
- D18: Qwen's peak of 7.07 GiB allocated is below the 7.5 GiB of the first rung, but section 1B.8 measures it with the sysmem fallback off, and the fallback is still on. Task P1.14 waits for that.
Broke:
- Nothing. The `make test` recipe, run by hand: 222 passed, 85 deselected. No test reads the limit in `readers.yaml`; the model tests set their own.
Next:
- For your decision, with Gate 1: how to make the pilot fit (power and sleep settings, the decode's CPU cost, the cost of loops, the page count).
- Task P1.14 once the sysmem fallback is off; Task P1.16, including `make readers` with these limits.

## 2026-10-14, Claude (D22): Phase 1 GPU checks at b6d0dc5 (Tasks P1.12 and P1.16)
Did:
- `make test-models` by hand at b6d0dc5, the review-fixes commit: `pytest -m model -v`, 79 passed, 257 deselected, 3 min 16 s (`tests/test_no_resize.py`, `tests/test_reader_facts.py`, `tests/test_readers.py`), no skips.
- `make readers` by hand at b6d0dc5, for manifests after the review (finding 8): `results/runs/20261014-0755_checkreaders_qwen_b6d0dc5/`, `20261014-0756_checkreaders_smolvlm_b6d0dc5/` and `20261014-0757_checkreaders_smolvlm500m_b6d0dc5/`.
Found:
- Every check passed: bfloat16, every parameter on the GPU, the page itself at the encoder, 256, 81 and 64 image tokens. On AC power the first calls took 24.3, 68.4 and 7.8 s (47.6, 57.7 and 15.4 s on 13 October, on battery).
- Qwen and SmolVLM-500M read exactly as on 13 October: 150 new tokens ending in an end token, and 54. SmolVLM-2.2B now runs to the new limit of 629 tokens (200 of 200 then), and its peak allocation grows from 4.29 to 4.36 GiB with the longer cache; Qwen 7.07 GiB and SmolVLM-500M 1.00 GiB as before.
- `git_dirty` is True in the two SmolVLM manifests and False in Qwen's, on a clean tree: `git status --porcelain` lists the untracked run folder the previous reader had just written. The build guide (section 2.8) knows this: keep `git_dirty` as it is and add `code_dirty` and `code_changed_since` at the runner (Task P2.8). Until then a manifest's `git_dirty` is read with that in mind.
Broke:
- Nothing.
Next:
- Task P1.14 (D18, Qwen's placement) still waits for the sysmem fallback to be off; `make check-env` exits 1.
- Push and CI (owner item: the token's `workflow` scope), then PR #2 ready for review.
## 2026-10-14, Claude (D22): W1.9, shift-equivariance error per layer and paper Figure 0
Did:
- Section 2.1 of the build guide. `src/mrr/equivariance.py` (NumPy only: `shift_right`, `e_value`, `e_curve`, `e_summary`); in `src/mrr/readers.py` `load_vision_tower`, `vision_hidden_states`, `qwen_to_raster`, `count_text_tokens`, `unload_reader` and `_ieee_float32`; `scripts/03_equivariance.py`; `plot_equivariance` in `src/mrr/plotting.py`; `tests/test_equivariance.py`; the `equivariance` recipe in the Makefile.
- Each vision tower runs alone on the GPU in float32: the whole model is loaded on the CPU in bfloat16, the tower is cast and moved, the rest deleted. Forward hooks on the patch embedding and on the chosen blocks copy each output to the CPU as float64; the forward runs under `inference_mode` with TensorFloat-32 off (`_ieee_float32`: in torch 2.14.1 cuDNN's switch is on by default and matmul's off; both are set False and restored afterwards). Qwen's block outputs come back to raster order through `argsort(window_index)` and the processor's merge-unit grouping; SmolVLM's vision model is called directly with a full-frame `patch_attention_mask`.
- E pages: the first page of Persuasion after the front matter (natural: 510, 616 and 1,151 characters on qwen, smolvlm and smolvlm500m) and the first page of `random_text(200_000, 0)` (random62: 479, 575 and 1,031), conventional layout, Source Sans 3 at 16 px, line pitch 20 px (not a multiple of P), margin P, `coverage_srgb`, `max_phase_y_px` 0 and `max_phase_x_px` = s_max = twice the sweep period: 224, 84 and 128 px, so the text block is 196 × 420, 266 × 350 (the 6 px blind strip taken off) and 352 × 480 px. Shifts from 0 to s_max in 0.25 px steps: 897, 337 and 513 renders per page.
- Layers: qwen 0, 1, 8, 16, 32 (8, 16 and 32 are the outputs of the full-attention blocks 7, 15 and 31); smolvlm 0, 1, 9, 18, 27; smolvlm500m 0, 1, 4, 8, 12. Layer 0 is the patch embedding.
- `pytest -m model tests/test_equivariance.py`: 22 passed, 3 skipped in 7 min 06 s. The skips are the SmolVLM position-id test on qwen and the Qwen-only block-0 test on the two SmolVLMs. The file's 25 CI tests include two for Figure 0 on made-up rows; pandas and matplotlib join the `test` extra of `pyproject.toml` so CI can draw it.
- `python scripts/03_equivariance.py --config configs/readers.yaml`, 09:03 to 09:33: towers loaded in 35, 14 and 3 s; pages measured in 600 and 586 s (qwen, 0.67 s per shift), 134 and 126 s (smolvlm), 147 and 144 s (smolvlm500m). Outputs: six `results/equivariance/<reader>_<text_type>.csv`, `results/equivariance/manifest.json` and `reports/figures/fig0_equivariance.pdf`, drawn from the CSV files alone, each panel's legend below it. The manifest says `git_dirty: true` because the run preceded this commit; the code it ran is this commit's.
Found:
- Layer 0, natural / random62, against appendix C.3 (`coverage_srgb`):

| Reader | E at multiples of P | E(0.25) | Plateau | Largest E | Appendix C.3 test range: E(0.25), plateau |
|---|---|---|---|---|---|
| qwen | 0 / 0 | 0.363 / 0.359 | 1.270 / 1.307 | 1.416 / 1.371 | 0.32 to 0.40, 1.22 to 1.36 |
| smolvlm | 0 / 0 | 0.218 / 0.213 | 0.872 / 0.901 | 0.983 / 0.964 | 0.18 to 0.25, 0.82 to 0.96 |
| smolvlm500m | 0 / 0 | 0.327 / 0.322 | 1.242 / 1.284 | 1.351 / 1.343 | 0.29 to 0.36, 1.19 to 1.34 |

- Layer 0 behaves as notes A7 predict on all six pages: E is 0 to the last bit at every multiple of the patch, and E(0.25), the plateau and the largest E fall inside the ranges appendix C.3 measured on 11 October (largest E 1.371 to 1.416, 0.964 to 0.996 and 1.343 to 1.354, which the tests do not pin).
- Beyond layer 0, natural / random62. The plateau is the mean E over whole-pixel shifts that are not multiples of P. The dip is the mean over k of E(kP) divided by the mean of E(kP − 1) and E(kP + 1): 0 is exact, 1 is no dip at all.

| qwen, layer | E(0.25) | Plateau | Mean E at multiples of 14 px | E at 112 px, 224 px | Dip |
|---|---|---|---|---|---|
| 1 | 0.293 / 0.293 | 1.094 / 1.140 | 0.036 / 0.039 | 1.5e-7, 1.4e-7 / 1.4e-7, 1.4e-7 | 0.044 / 0.046 |
| 8 | 0.164 / 0.160 | 0.728 / 0.738 | 0.26 / 0.25 | 0.046, 0.050 / 0.046, 0.049 | 0.51 / 0.49 |
| 16 | 0.135 / 0.149 | 0.716 / 0.744 | 0.58 / 0.63 | 0.113, 0.116 / 0.125, 0.183 | 0.89 / 0.93 |
| 32 | 0.269 / 0.193 | 0.630 / 0.630 | 0.61 / 0.61 | 0.308, 0.286 / 0.255, 0.235 | 0.99 / 0.98 |

| smolvlm, layer | E(0.25) | Plateau | Mean E at multiples of 14 px | Dip |
|---|---|---|---|---|
| 1 | 0.151 / 0.154 | 0.993 / 0.995 | 0.85 / 0.82 | 0.90 / 0.88 |
| 9 | 0.104 / 0.104 | 0.771 / 0.762 | 0.68 / 0.66 | 0.93 / 0.93 |
| 18 | 0.091 / 0.097 | 0.535 / 0.577 | 0.47 / 0.48 | 0.95 / 0.94 |
| 27 | 0.123 / 0.097 | 0.616 / 0.620 | 0.62 / 0.55 | 1.00 / 0.94 |

| smolvlm500m, layer | E(0.25) | Plateau | Mean E at multiples of 16 px | Dip |
|---|---|---|---|---|
| 1 | 0.209 / 0.205 | 0.914 / 0.937 | 0.30 / 0.29 | 0.45 / 0.44 |
| 4 | 0.117 / 0.116 | 0.668 / 0.669 | 0.44 / 0.44 | 0.82 / 0.81 |
| 8 | 0.034 / 0.039 | 0.416 / 0.424 | 0.42 / 0.42 | 1.02 / 1.02 |
| 12 | 0.076 / 0.062 | 0.402 / 0.375 | 0.35 / 0.36 | 0.95 / 0.99 |

- Qwen after block 0, which attends inside 112 px windows: only moves of 112 and 224 px are exact (1.5e-7, float32 noise); the other multiples of 14 px cost 0.04, since a 14 px move slides patches across window edges. From block 7, the first full-attention block, 112 px moves cost something too: 0.05 at layer 8, 0.3 at the output. [Inference] This is the "page edges aside" clause of section 2.1: every text token attends to the blank tokens as well, and the move changes where the blank area lies around the text. With depth the plateau falls from 1.27 to 0.63, but the 14 px dip is gone by layer 32 (0.99) while the 112 px dip survives to the output (0.3 against 0.63). Beyond the embedding the window is the period that matters: D9's 112 px sweep period stands.
- SmolVLM-2.2B: minima, not zeros, at multiples of 14 px beyond layer 0, as section 2.1 predicts for learned absolute positions, and shallow ones from layer 1 on (0.85 against 0.99). [Inference] The learned position vector is a large part of what a token carries, so a 14 px move costs nearly as much as any other. The plateau falls to 0.54 to 0.77 in the deeper layers, with dips of 0.93 to 1.00.
- SmolVLM-500M: a deep dip at multiples of 16 px at layer 1 (0.30 against 0.91), weaker at layer 4 (0.82), gone from layer 8 (1.02); the plateau falls to 0.38 to 0.40 by layer 12.
- In all three towers E(0.25) shrinks with depth: qwen 0.36 at layer 0, 0.29 at layer 1, 0.13 to 0.27 deeper; smolvlm 0.22, 0.15, then 0.09 to 0.12; smolvlm500m 0.33, 0.21, then 0.03 to 0.12. The towers smooth the rasteriser's sub-pixel changes far more than whole-pixel misalignment.
- Natural and random62 pages agree within 0.05 at layer 0 and within 0.08 at every layer; the largest gaps are qwen's E(0.25) at layer 32 (0.269 against 0.193) and its E at 224 px at layer 16 (0.116 against 0.183).
- E(0) = 0 at every layer of every page: with TensorFloat-32 off, two forward passes of one page agree bit for bit.
- D15: `load_vision_tower` counts the distinct ids that reach the position table on a blank native page and installs the fix only if the count is short. In float32 both SmolVLM towers saw every id (729 and 1,024), so the run log says `position-id fix added: False` for both and `test_float32_smolvlm_tower_sees_every_position_id` passed. The bfloat16 readers keep the pre-hook of W1.8.
- Addendum, 14 October 10:49 UTC: `scripts/03_equivariance.py` rerun on a clean checkout of ef4aa8a (the W1.9 commit after the rebase onto build/p1), outputs written outside the repository. All six CSVs byte-identical to the committed ones and the figure identical apart from the PDF dates; only `results/equivariance/manifest.json` was replaced, so it now records `git_dirty` false, the commit that holds the code, and the freetype-py version.
Broke:
- Nothing. `make test` by hand in this worktree (af34524 plus this work): 247 passed, 110 deselected.
- Cost: Qwen's E page takes 10 min at 0.25 px steps. A rerun that only checks the figure could step by 1 px.
Next:
- Gate 1 batch: D9 and D15, recorded as decided on this evidence, for your confirmation.
- Rerun `scripts/03_equivariance.py` on the committed code, after the Phase 1 GPU jobs, so the manifest is clean (about 25 min), and commit the rerun.
- Task P2.2 of the build plan.

## 2026-10-14, Claude (D22): W1.10, coherence of the layouts (D10)
Did:
- `src/mrr/render.py`: `InkBox` and `glyph_ink_boxes`, which read each glyph's bitmap extent off the layout at the glyph's own fractional phase, as `render_coverage` places it, without clipping to the page. `src/mrr/grid.py`: `coherence` (R_k(T) = |mean exp(2πi k c / T)| over the ink centres c), `class_coherence` (the count-weighted mean over the characters with at least 20 glyphs) and `straddle_share` (the share of glyphs that a patch edge cuts after a whole-pixel shift). Tests first, in `tests/test_grid.py` and `tests/test_render.py`, including the two carried over from W1.5.
- `scripts/05_coherence.py` (`make coherence`): the first 2,500 glyphs of Persuasion after the front-matter skip and `normalise_text`, in Qwen's 308 px block (`max_phase_x_px` 112), Source Sans 3 at 16 px, pitch 20 px, phase 0, both render modes, every value checked against the table of build guide section 2.2.
Found:
- The table reproduces. Every value of the patch_aware row matches the guide to three decimals and its per-class R₁(14) is 1 to 1e-12; the conventional row is within 0.006 of the guide (it allows about 0.01, because the repository loads each glyph at its own fractional phase while the table used phase-0 bitmaps).

| Layout | Glyphs per line | R₁(14) | R₂(14) | R₁(28) | R₁(112) | Per-class R₁(14) | Straddle share over shifts 0 to 13 |
|---|---|---|---|---|---|---|---|
| conventional | 35.7 (70 lines) | 0.022 | 0.018 | 0.014 | 0.104 | 0.081 | 0.384 at shift 2 to 0.425 at shift 12 |
| patch_aware | 16.4 (152 lines) | 0.986 | 0.947 | 0.022 | 0.088 | 1.000 | 0.000 at shift 0 to 1.000 at shift 7 |

- Straddle share by shift. Conventional: 0.397 0.386 0.384 0.394 0.403 0.394 0.397 0.409 0.416 0.418 0.421 0.424 0.425 0.424. Patch_aware: 0.000 0.004 0.042 0.078 0.610 0.853 0.924 1.000 0.884 0.761 0.438 0.048 0.042 0.000.
- Reading the rows. Conventional glyphs sit at every offset against the 14 px grid, so R₁(14) is at the 1/√N floor (0.020 for N = 2,500) and a block shift leaves the page's set of offsets as it was: the page has no phase. The patch_aware layout puts every glyph of a character at one offset in its slot (per-class 1.000); the pooled 0.986 is the spread of glyph widths between characters, and R₂(14), the same offsets read at a 7 px period, is 0.947 because that spread costs more at half the period. R₁(28) is at the floor in both rows: consecutive 14 px slots alternate between the two halves of a 28 px period and cancel. [Inference] R₁(112) of about 0.1 in both rows is the block's shape, not the glyphs': 308 px is 2.75 periods of 112 px, so the columns the text occupies do not cover the three periods evenly. The straddle share says the same in counts: 38 to 43 percent of conventional glyphs are cut by a patch edge at every shift, while the patch_aware layout goes from none cut at shift 0 to all cut at shift 7.
- [Inference] On these numbers a block of patch_aware text can show H1 at 14 px and a conventional block cannot, whatever the per-glyph effect; that is the case for D10's blocks H and C. D10 stays open until Gate 1.
Broke:
- Nothing. `make test` by hand in this worktree: 299 passed, 110 deselected.
Next:
- The size ladder (`configs/sweeps/ladder.yaml`, GPU), the second half of W1.10, follows the runner (Task P2.9).
- Tasks P2.3 to P2.8.

## 2026-10-14, Claude (D22): W2.2, blocks, phase levels and condition expansion (D10); the ladder and pilot configs (W1.10)
Did:
- `configs/sweeps/ladder.yaml` (new) and `configs/sweeps/pilot.yaml`, copied verbatim from build guide section 2.7: a sweep is a list of blocks, each one layout swept along one axis, not the Cartesian product the scaffold described. The pilot's font sizes are "from the ladder" placeholders. `configs/sweeps/full.yaml` converted to the same format with W3.2 placeholders for the seed, the page counts and the sizes. `Makefile`: a `ladder` target, `READERS` and `DEADLINE` variables, quoted config paths. `docs/01_research_plan.md`: the three TODO[W2.2] rows filled; the font-file row records the one-typeface proposal for Gate 1. Commit 4db65eb, pulled ahead of Task P2.3 because its tests read every file in `configs/sweeps/`.
- `grid.phase_levels(spec, reader_cfg, block=None)`: the range is `patch` (P), `cell` or `sweep`; the step a number or a map by reader; an explicit `phases_px` list is checked to lie in [0, P). A step that does not divide both the range and P is refused with the block, the reader and the apparent period lcm(σ, T) in the message.
- `src/mrr/sweep.py` (new): `Condition`, the ten keys of the section 2.5 hash plus `paginate`, with `hash_keys()` casting every number to float; `expand_conditions(sweep, reader_cfg)`: blocks in file order, the top-level `font_file`, `font_size_px` (per reader) and `aa_mode` inherited and overridden by the block, phase levels on the block's axis with the other axis 0, list-valued keys crossed in the order `render_mode`, `line_pitch_mode`, `font_size_px`, `aa_mode` with the phases innermost; `line_pitch_px` is floor(1.25·size + 0.5) unaligned or the reader's `aligned_pitch_px`; `patch_aware_slot_px` is P for `patch_aware` and None otherwise. An unknown block key, a value outside its enumeration, or a string where a number belongs raises `ValueError` naming the key.
- Tests first: `tests/test_grid.py` (the four calls of the guide's table, the two aliasing refusals, placeholders, malformed blocks) and the new `tests/test_configs.py` on every file in `configs/sweeps/`: every block passes `phase_levels` for every reader; the pilot gives 50, 62 and 54 conditions per page per text type; the run count in each header equals the computed count (3320 for the pilot, 216 for the ladder; `full.yaml` is skipped while its count is a TODO); (H, 0) and (S, 0) differ only in their block label; a block C condition at phase 7 gives exactly the test-vector dictionary of section 2.5; block Z gives 12 per-condition conditions per reader. `mrr.sweep` added to the smoke MODULES.
Found:
- Conditions per page per text type: Qwen V 14 + H 14 + C 16 + S 4 + A 2 = 50; SmolVLM-2.2B 21 + 14 + 21 + 4 + 2 = 62; SmolVLM-500M 16 + 16 + 16 + 4 + 2 = 54, as the table of section 2.3 says.
- The aliasing refusals read as the guide predicts: Qwen's block C at σ 4 "repeats every 28 px", SmolVLM-2.2B's at σ 3 "repeats every 42 px".
- `expand_conditions` on the pilot as it stands raises `ValueError` naming `font_size_px`, so the pilot cannot start until the ladder has filled the sizes (Task P2.9).
Broke:
- Nothing. Full CPU suite by hand in this worktree: 349 passed, 1 skipped, 110 deselected.
Next:
- Task P2.4: `Page` and `plan_pages` with the phase allowances; the header-count test then takes the page count from the plan rather than from the config's `per_text_type`.
- `docs/03_pipeline.md` W2.2 stays unticked until the ladder's sizes are in `configs/sweeps/pilot.yaml`.

## 2026-10-14, Claude (D22): W2.2, the pagination plan: shared pages across layouts with phase allowances (section 2.4), block Z pages (section 3.5)
Did:
- `src/mrr/text_source.py`: `Book`, `BOOKS` (pg105, pg84 and pg1661 with the front-matter skips of `data/raw/texts/README.md`), `TEXTS_DIR` and `book_text(book_id, root)`, the body of a book as the pages see it, read once per file and cached.
- `src/mrr/sweep.py`: `Page` (page_id, book_id, text_type, start_word, text); `phase_allowances`, the ceiling of the largest φx and of the largest φy over the shared conditions; `pagination_key`, `shared` or `z:<render_mode>:<size>` for a block Z layout; and `plan_pages(sweep, reader_cfg, render_cfg, *, root, readers)`. A shared page fits when `render.fits_page` holds under every distinct shared layout at those allowances (conventional unaligned, conventional aligned and patch_aware unaligned in the pilot). Natural pages come from the books in rotation from their skip points, random pages from one `random_text(200_000, seed)` stream, interleaved natural then random. Block Z: each (size, render mode) layout is paginated alone at phase 0 with zero allowances; page i of every Z layout starts at the same word; passages of one source are spaced by the fullest page among the layouts; passages begin where the sweep's shared pages ended. `pages.natural.start_after` replans the sweeps it lists for every reader they list and starts each book after the last word any of them used; the seed must be new. A string where a page count or the seed belongs raises `ValueError` naming the key.
- Tests first: `tests/test_sweep.py` (21 tests) and five registry tests in `tests/test_text_source.py`. `tests/test_configs.py` now takes each header's run count from the plan (conditions × the pages each one reads × repeats, as the runner will count them) and checks that `full.yaml` refuses to plan while its placeholders stand. `docs/04_data_dictionary.md`: the `page_id` and `book_id` rows now give the `pg<id>`, `rnd<seed>` and block Z formats.
Found:
- The shared page at 16 px holds exactly the characters of the table of section 2.4: 379 (Qwen), 298 (SmolVLM-2.2B), 419 (SmolVLM-500M). The shared page is the shortest of the three layouts' pages, because each layout's fit is monotone in the word count.
- The pilot's allowances come out as the guide says: 105/26, 40/40 and 60/60 px.
- In the ladder, Qwen's 10 px conventional page is the fullest of the twelve and sets the passage spacing; its 16 px conventional Z page, on a 420 × 420 block, holds more than the 840 characters of the same layout under the 105/26 allowances.
- Planning is cheap: the pilot's 20 shared pages take 1.7 s per reader on the CPU, the ladder's 72 pages 1.3 s.
- Page numbering: NNNN is the page's index among the pages of its text type in the plan (pg105_p0000, pg84_p0001, pg1661_p0002, pg105_p0003), the same rule for shared and Z pages. Ids are unique within one run and reader; `start_word` and `references/<page_id>.txt` (W2.3) identify the text.
- [Rule, strict reading] `start_after` starts each book after the last word any reader of the earlier sweeps used. Section 2.4, step 6 does not say whether "used" means by this reader or by any; the strict reading can never be called reuse and costs nothing, the books being long. Say so if you want it per reader.
Broke:
- Nothing. Full CPU suite by hand in this worktree: 375 passed, 1 skipped, 110 deselected.
Next:
- Task P2.5: `condition_hash`, the 33-column schema (D19), `code_dirty`; then P2.6 (`score.py`), P2.8 (the runner).
- `docs/03_pipeline.md` W2.2 stays unticked until the ladder's sizes are in `configs/sweeps/pilot.yaml`.

## 2026-10-14, Claude (D22): the slip rule of the calendar fired at 12:00 (D21, D25)
Did:
- G1 was set for Wed 14 Oct 12:00 CEST (D21). At 12:00 the ladder (P2.9) had not run, because the runner (P2.8) is not built and the GPU still has the sysmem fallback on (D18), and the Gate 1 decisions had not been put to the owner. G1 therefore did not pass. The slip rule of the plan, approved with it on 12 October, moves every later date by one day, D16 included, and applies the cut order from the top. Recorded as D25; D4 stays open with a note; D16's date moved.
Found:
- New dates (CEST): G1 Thu 15 12:00; `prereg-v1` Thu 15 18:00; pilot Thu 15 night; Phase 3 Fri 16; G2 Sat 17 12:00; full run Sat 17 to Thu 22 06:00; Z.ai key Tue 20; `data-freeze-v1` and D16 Thu 22; H2 paid calls Fri 23 to Sat 24; G3 and `spec-v1` Sun 25; paper Sun 25 to Thu 29; buffer Fri 30. The deadline, Fri 30 Oct 19:00 UTC, does not move.
- Cut 1 of the cut order is the stretch tasks of Phase 3 except P3S.1. The plan's only stretch task is P3S.1, which stays because it runs on the CPU, so cut 1 removes nothing. Cut 2 would be H2, which D16 already governs by date.
- G1 still waits on the owner for: the NVIDIA sysmem-fallback setting (D18), the look at `reports/figures/render_check/`, and the Gate 1 batch (D9, D10, D11, D15, the typeface, the chapter headings, the target CER, the pilot budget).
Broke:
- Nothing.
Next:
- P2.5 to P2.8 today; the ladder as soon as the GPU is free of the sysmem fallback.

## 2026-10-14, Claude (D22): W2.3, the results schema (D19), the condition hash and the code-cleanliness checks (section 2.5)
Did:
- `src/mrr/manifest.py`: `RESULTS_COLUMNS` is now the 33 names of section 2.5 in order; `CODE_PATHS` (`src`, `scripts`, `configs`, `pyproject.toml`, `requirements.lock.txt`); `code_dirty(paths, cwd)` from `git status --porcelain -- <paths>` and `code_changed_since(commit, paths, cwd)` from `git diff --name-only <commit> HEAD -- <paths>`, both through the existing `_git` argument-list runner, both None when git cannot answer. `git_dirty` kept as it was.
- `src/mrr/sweep.py`: `condition_hash(cond)`, the ten `HASH_KEYS` only, numbers cast to float, `font_file` reduced to its name, `json.dumps(sort_keys=True, separators=(",", ":"))`, SHA-1, the first 12 hexadecimal characters; a missing key raises `ValueError`, extra keys are ignored.
- `docs/04_data_dictionary.md`: the 33-row table (D19), the phase rows over the reader's sweep period (D9), the hash rule with its test vector, and `coherence.csv`, `determinism.csv`, `progress.log` and `references/<page_id>.txt` in the run-folder layout.
- Tests first: `tests/test_manifest.py` (new: the 33 names, `CODE_PATHS`, the two checks on a throwaway repository in `tmp_path / "repo with space"`) and five hash tests in `tests/test_sweep.py`. `tests/test_smoke.py` is unchanged: `mrr.sweep` was already in MODULES and its column regex already forbids digits.
Found:
- The guide's test vector recomputed: `9c811d1777d4`; the same dictionary with the integer 7 for `phase_x_px` gives `bc2ca5dcb62f`, so the float cast is load-bearing.
- Every pilot condition of every reader has a distinct hash, and (H, 0) and (S, 0) differ by their block label only, which is what `determinism.csv` will compare.
- `TRACKED_PACKAGES` has tracked freetype-py since W1.3 under the key `freetype-py`; the guide's `"freetype": "freetype-py"` would duplicate it, so the key stays as the equivariance manifest already records it.
- [Flag, runner] `transcriptions/<page_id>__<condition_hash>.txt` has no repeat in its name while the resume key has one. Every sweep config has `repeats: 1`, so nothing collides; if a repeat count above 1 is ever set, the file name needs a repeat suffix and a dictionary change.
Broke:
- Nothing. Full CPU suite by hand in this worktree: 386 passed, 1 skipped, 110 deselected.
Next:
- Task P2.6: `score.char_counts`, `pooled_cer`, the `todo` mark off `tests/test_score.py`; then P2.8, the runner.

## 2026-10-14, Claude (D22): W2.4, score.py: the four character counts and the pooled CER (section 2.6, D13)
Did:
- `src/mrr/score.py`: `CharCounts` (frozen: hits, substitutions, deletions, insertions), `char_counts` from `jiwer.process_characters`, `pooled_cer` = (S + D + I) / (H + S + D) over every row, `cer` as the pooled CER of one row, `wer` from `jiwer.process_words(...).wer`. Both strings pass through `text_source.normalise_text` first; an empty normalised reference raises `ValueError`, because `jiwer.cer("", "abc")` returns 3 instead of failing. The docstring now says Learn B5.
- `tests/test_score.py`: the `todo` mark is off; the two scaffold tests are unchanged; eleven more from the table of section 2.6, every value reproduced with jiwer 4.0.0 before the tests were written.
- `pyproject.toml`: jiwer moved from the test extras to the core dependencies, because `mrr.score` imports it at module level and the runner scores every row with it. `requirements.lock.txt` already pins 4.0.0; CI installs `.[test]` and is unchanged.
Found:
- jiwer 4.0.0 reproduces every value of the guide's table: the transposition is D = 1, I = 1 (CER 2/7); the raw strings with a line break give 1/11 and 0 after normalisation; an empty transcription is three deletions and CER 1.0 with no exception; `CharacterOutput` has no `wer`.
- `pooled_cer` on the pages `ab`/`xb` and `abcd`/`abcd` is 1/6; the mean of the two page CERs would be 1/4, the number D13 forbids.
Broke:
- Nothing. Full CPU suite by hand in this worktree: 403 passed, 1 skipped, 106 deselected (the `todo` tests of `tests/test_stats.py` are among those that remain).
Next:
- Task P2.8: the runner (`sweep.run_sweep`, `scripts/10_run_sweep.py`), then the ladder on the GPU (P2.9).
- `docs/03_pipeline.md` W2.4 stays unticked until `stats.py` lands with the Phase 3 analysis.

## 2026-10-14, Claude (D22): W2.3, the runner (section 2.8): run_sweep, scripts/10_run_sweep.py, resume, shuffle, deadline
Did:
- `src/mrr/sweep.py`: `run_sweep(sweep_path, readers_cfg, render_defaults, backend, only_readers, resume, save_images, output_root, require_clean, deadline, *, root)`. One reader at a time, in the sweep's order. Every reader is planned (`expand_conditions`, `plan_pages`) before any folder is made, so a placeholder stops the run with nothing on disk. Per reader: a run folder `output_root/new_run_id(<sweep stem>, <reader>)`, refused if it already exists; `manifest.json` with the resolved config (the sweep file and its SHA-1, the reader block, the render config at the phase allowances, the greedy generation settings with the `max_new_tokens` rule, placement under D18, rows planned, pages, save_images, deadline); `references/<page_id>.txt` for every page; then the loop: the deadline check at each page boundary, `load_reader`, the vision-token check against `grid.token_count` on the first page (RuntimeError naming both counts, D8), `coherence.csv` rows per page per block layout at phase 0 (periods 14, 28, 112 for Qwen, 14 and 42 for SmolVLM-2.2B, 16 and 64 for SmolVLM-500M; harmonics 1 to 3), the page's `max_new_tokens = ceil(1.5 x count_text_tokens(reference))`, the (condition, repeat) pairs shuffled with `np.random.default_rng([seed, page_index]).permutation(n)`, and per row: render at the condition's phases, `transcribe_full`, `score.char_counts`, the transcription file written before the row, the row appended with flush and fsync, the image kept under `save_images`. After each page: `determinism.csv` rows for every pair of conditions that differ only in their block label, once both are done; one `progress.log` line with rows done of planned and a projected finish from the mean row latency of the last ten pages. `unload_reader` in a `finally`, so a failed page still frees the card.
- Resume: the folder's manifest must name the reader and carry the same sweep SHA-1; `code_dirty` refuses a start and `code_changed_since(<manifest commit>)` refuses a resume when `require_clean` (the default; the tests pass False); the resume key (reader, page_id, condition_hash, repeat) is read from `results.csv`; an incomplete last line, one without a line break, is cut off, logged at WARNING on `mrr.sweep` and its row redone; coherence, determinism and progress entries already on disk are not repeated.
- `_layout_cfg` gained a `phase` argument, (0, 0) by default, so the runner renders at the condition's phases through the same function `plan_pages` paginates with. `_now()` returns aware UTC and is the runner's only clock (deadline check, row timestamps, progress), so the tests can replace it. `load_results` is a `NotImplementedError("TODO[W3.2] ...")` stub.
- `scripts/10_run_sweep.py`: `--config`, `--readers-config`, `--render-config`, `--readers a,b`, `--resume <run folder>`, `--save-images {all,first,none}`, `--deadline <ISO 8601>`; passes the `mrr.readers` module as the backend; `require_clean=True`. Makefile: `RESUME=...` on `ladder`, `pilot` and `sweep`.
- `docs/04_data_dictionary.md`: `timestamp_utc` ends in `+00:00` (plan Review Focus 4; `manifest.json` keeps `Z`), and the run-folder lines for `coherence.csv`, `determinism.csv`, `progress.log`, the `__r<repeat>` suffix and `images/`.
- Tests first: 23 runner tests in `tests/test_sweep.py` on a fake backend (the five functions of readers.py without a model, one fixed transcription): every row and column of a two-block run, the per-page `max_new_tokens`, images and references, coherence rows, determinism rows, the manifest, progress, interrupt and resume, the cut-off line, the shuffle, four placeholders, unknown readers and an unknown `save_images`, `only_readers` with the whole readers file, a past deadline, a deadline during the first page (naive and aware), dirty and changed code, a token mismatch, three resume refusals, `save_images` all and none, a vertical block, a mismatching determinism pair, the `load_results` stub.
Found:
- [Flag, judgement] Pages run in plan order, natural then random; when a sweep has block Z paginations as well as shared pages, the paginations are interleaved page index by page index, so an interrupted ladder holds every layout on the same passages. The guide does not say in which order a run visits several paginations.
- [Flag, judgement] A repeat above 0 adds `__r<repeat>` to the transcription and image file names; repeat 0 keeps the guide's name. Every sweep has `repeats: 1`, so nothing changes until a repeat count is set (closes the W2.3 schema flag).
- [Flag, judgement] Row timestamps end in `+00:00`, the plan's Review Focus 4, not the dictionary's old `Z`; `manifest.json` keeps `Z` from `build_manifest`.
- [Flag, judgement] `run_sweep` takes a keyword-only `root` beyond the guide's signature, so the tests can run from any directory; the scripts leave it at the current directory, the repository root.
- [Flag, judgement] A run folder that already exists is refused rather than reused: `new_run_id` has minute resolution, so two starts of one reader within one minute collide; start again a minute later, or resume.
- The determinism check is general: any two conditions equal in the nine other hash keys and different in block; `condition_hash_a` is the earlier block in file order. In the pilot that is (H, 0) and (S, 0) only; the vertical-block test confirms no file is written when no such pair exists.
- The shuffle is per page and seeded by (seed, page index): two output roots give identical orders, and neither page of the test sweep follows file order.
- The deadline test freezes `_now()`: the first call (the check before page 1) is 10:00, every later call 11:00, the deadline 10:30; exactly the first page's five rows are written and the run stops at the boundary, for a naive and for an aware deadline alike.
Broke:
- Nothing. Full CPU suite by hand in this worktree: 430 passed, 1 skipped, 106 deselected.
Next:
- Task P2.9: the size ladder on the GPU (waits for the sysmem-fallback owner action), then the Gate 1 batch, the preregistration and the pilot.

## 2026-10-14, Claude (D22): W2.1, the preregistration draft (section 2.11), for the owner's approval
Did:
- `PREREGISTRATION.md`: every row of the guide's table (2.11) drafted from the repository's own values. Section 1: H1 as clauses (a), (b), (c) plus the block Z description, the prediction for the control block C (flat; any amplitude at harmonic k of period T scaled by R_k(T), with W1.10's R₁(14) = 0.022 conventional and 0.986 patch-aware as the reference values) and H2 in full: `glm-5.3`, one user message holding `prompt_v01.md`, `reasoning_effort` max, temperature 1.0, `max_tokens` 131072 (128000 if refused), three attempts, the 186 judge pages (144 + 18 + 24), the tolerance table of guide 4.2, `pixel_work_in_triton` by D23, the criterion at least one of three, D16 as D25 moved it (Thursday 22 October). Section 2: the three Hub ids and revisions, grids and token counts of `configs/readers.yaml`, bf16 on the GPU (D18), `pre_hook`, the four greedy arguments, the `max_new_tokens` rule, the prompt verbatim. Section 3: the block table of `configs/sweeps/pilot.yaml` with the levels per reader (50, 62 and 54 conditions per page per text type), the geometry and allowances, the line pitches, render modes, anti-aliasing levels, block Z of `ladder.yaml` and `full.yaml`. Section 4: `normalise_text`, pooled CER from the four counts (D13), WER, the reported quantities. Section 5: the books and skip points, the random62 seeds (20261012 ladder, 20261013 pilot, 20261016 full run), the ladder as a design measurement at target 0.15, the pilot at 10 + 10 pages, the page-count rule with N_min = 10, s_target = 0.01 and the guide's λ_q table, the budget windows after D25, `paired_mde`. Sections 6 and 7: α = 0.05, the model and F-test of D14, the groups table of guide 2.3, Holm, the three families, the G2 criterion. Section 8: the exclusion rules. Section 9: the analysis plan step by step with 2,000 page-bootstrap resamples at seed 20261014 and s* with δ = 0.01 and the plateau as the pooled CER at 20 and 24 px (design §9). Sections 10 and 11: the failure criteria and the tag rule (`git rev-list -n 1 prereg-v1`; runner commit 495f1f3).
- `docs/01_research_plan.md:56` now points at sections 6 and 7 of the preregistration; `h2/README.md:24` says three attempts. No `TODO[W2.1]` remains anywhere in the repository.
- Not tagged. The ACCEPT of 2.11 also needs the ladder's three font sizes (TODO[W1.10], section 3), the approval date and the ladder commit (section 11). The tag `prereg-v1` waits for the ladder (P2.9, GPU) and the owner's approval (owner action 7), due Thursday 15 October 18:00 under D25.
Found:
- [Flag, judgement] Sidedness: the F-tests are one-sided by construction; the paired Wilcoxon tests are two-sided, and clause (c) counts as supported only when its Holm-adjusted p is below α and the median paired difference is negative. The guide fixes α and names the tests but not the sidedness.
- [Flag, judgement] WER is descriptive only: `score.py` stores a row WER but no word counts, so WER cannot be pooled the way CER is and is summarised as the mean of row WERs; no test uses it. The guide calls WER secondary without saying how it is summarised.
- [Flag, judgement] Pilot budget: if 1,000 t_Q + 1,240 t₂ + 1,080 t₅ does not fit the 57,600 s window at the ladder's median latencies, both text types are cut to the largest equal page count that fits, before the tag. The guide says the budget is checked, not how a cut is shared between text types.
- [Flag, judgement] The bootstrap seed 20261014 of the design document is kept although D25 moved the analysis to Friday 16 October; the seed is a label, not a date.
- [Flag, judgement] The λ_q values written in section 5 are the guide's 1,000-degrees-of-freedom values for q = 3, 6, 7, 12, 13, 14 and 15 at α/3, the only q of the primary family; `stats.power_noncentrality` recomputes them with the real degrees of freedom at analysis, as the guide says.
- [Flag, judgement] An H2 attempt is a reply with HTTP status 200: a transport failure is repeated and not counted; a reply cut off by `max_tokens` counts, and its first Python code block is judged verbatim. The guide fixes three attempts and the criterion, not what a failed request is.
- [Flag, judgement] The prediction for block C is written as preregistered, and a significant group in C with none in V or H is written as a contradiction of D10 in section 10, as guide 2.11 asks.
- D13 and D14 stay open until the tag, as 2.11 says; the preregistration's section 11 records that they are marked decided when the tag goes on.
Broke:
- Nothing. Full CPU suite by hand in this worktree: 430 passed, 1 skipped, 106 deselected.
Next:
- Owner: read the draft, answer the flags above, approve (owner action 7). Then P2.9 (the ladder) fills the three sizes, section 11 gets the date and the commit, `pilot.yaml` gets its sizes, and the tag goes on.
- Meanwhile, without the GPU: the Gate 1 evidence (P2.10) and `stats.py` (P3.1) on `build/p3`.

## 2026-10-14, Claude (D22): Gate 1 evidence assembled (section 2.10, Task P2.10); items 2, 5 and 6 open, so G1 waits
Did:
- Ran the checklist of section 2.10 against the notebook and the repository at 05a3de9. Nothing new was measured; this entry gathers the evidence for your decision (D4), item by item, and the batch of decisions that goes with it.
Found:
- Item 1, the no-resize test and the reader facts: passes. `make test-models` at b6d0dc5: 79 passed, no skips (`tests/test_no_resize.py`, `tests/test_reader_facts.py`, `tests/test_readers.py`). Undoing the processor gives back every page byte for byte for all three readers, on text and on noise (largest error before rounding 2.5e-5 of a level); 256, 81 and 64 image tokens, equal to `grid.token_count`; `make readers` at b6d0dc5 passed every check (bfloat16, every parameter on the GPU, the page itself at the encoder). One laptop under D20, so "both laptops" is one.
- Item 2, the overlays inspected and their hashes recorded: half. The 15 SHA-1 hashes of the raw pages are in the W1.5 render-check entry (renderer of e113777), and I looked at seven of the 30 images, named there. Your look at `reports/figures/render_check/` (owner action 4, "overlays OK") has not arrived. Open.
- Item 3, Figure 0: passes. `reports/figures/fig0_equivariance.pdf` exists (W1.9; rerun on a clean checkout of ef4aa8a, CSVs byte-identical). At layer 0, E at every multiple of P is 0 to the last bit on all six pages, so below 1e-6; E(0.25) is 0.363, 0.218 and 0.327 and the plateau 1.270, 0.872 and 1.242 on the natural pages (random62 within 0.04), inside appendix C.3's ranges (0.32 to 0.40 and 1.22 to 1.36; 0.18 to 0.25 and 0.82 to 0.96; 0.29 to 0.36 and 1.19 to 1.34). The predicted periods show: the patch at layer 0 for all three readers; for Qwen only 112 and 224 px stay exact after block 0 (1.5e-7) and the 112 px dip survives to the output (0.3 against a plateau of 0.63) while the 14 px dip is gone by layer 32 (0.99); both SmolVLM towers show minima, not zeros, beyond layer 0.
- Item 4, `scripts/05_coherence.py` reproduces the table of section 2.2: passes (W1.10). Conventional R₁(14) 0.022 against the 1/√N floor of 0.020, straddle share 0.384 to 0.425 at every shift; patch_aware R₁(14) 0.986, R₂(14) 0.947, per-class 1.000, straddle share 0 at shift 0 and 1 at shift 7. The patch_aware row matches the guide to three decimals, the conventional row within 0.006.
- Item 5, the ladder has run and the three sizes are chosen: not run. `configs/sweeps/ladder.yaml` and the runner are ready (W2.3, 495f1f3). The run waits for `make check-env` to exit 0 (the CUDA sysmem fallback is still on, owner action 1), for the power plan and sleep settings (owner action 2) and for the Epic and EOS processes to be closed (owner action 3). At W1.7's provisional medians (70.8, 104.8 and 131.0 s a row) its 72 rows per reader take about 6.1 h; less on AC at best performance, more where the SmolVLM readers run to their token limit. Open.
- Item 6, `tests/test_sweep.py` passes and the runner has resumed after a real interruption: half. The runner tests pass (the suite is 430 passed, 1 skipped at 05a3de9), including interrupt-and-resume and the cut-off last line on the fake backend. No real run has been interrupted yet. The ladder is the first real run, and I will stop it once after its first page (Ctrl-C), resume it with `--resume` and record both here. Open until then.
- Decisions the guide requires decided before G1: D7, D8, D15 and D17 are decided, and D12 is superseded by D24, decided. D11, the renderer contract, is still open although every rule in it is built, tested (`tests/test_render.py`, `tests/test_color.py`, W1.5) and stated in section 3 of the preregistration; what it lacks is your yes to `coverage_srgb` as the anti-aliasing baseline. D18, Qwen's placement, is still open: the first rung fits on the measurements so far (7.07 GiB allocated, at most 7,423 MiB of 8,151 used on nvidia-smi; W1.7 and the GPU checks at b6d0dc5), but section 1B.8 measures it with the sysmem fallback off, which it is not; Task P1.14 reruns it once owner action 1 is done. Gate 1 cannot pass with either open.
- Promised for G1 in the W1.8 entry and not done: the history of the bfloat16 cast in transformers, and CER with and without the position-id fix on more pages. W1.9 showed instead that in float32 both SmolVLM towers see every position id (729 and 1,024), and W1.8 that in bfloat16 SmolVLM-2.2B sees 441 without the fix and 729 with it. The CER comparison costs GPU time; say if you want it before the pilot.
- Gate 1 is not passed today. Under D25 the decision is due Thursday 15 October 12:00. The path there: owner actions 1 to 4, then the ladder with its deliberate interruption (item 5, item 6), then P1.14 (D18), then your yes to the batch below.
- Batch 2, for one yes or itemised answers:
  - (a) D4, the G1 outcome: to be written as go, with the date, once items 2, 5 and 6 close; the evidence lines above are its Evidence field.
  - (b) D10: confirm on the coherence table of W1.10. Blocks V, H and C as `pilot.yaml` holds them; H1 restated in section 1 of the preregistration with the prediction for C beside it.
  - (c) D9: confirm. Decided at W1.9 on Figure 0 and recorded on 14 October; the 112, 42 and 64 px sweep periods stand.
  - (d) D11: the renderer contract as written, with `coverage_srgb` as the baseline.
  - (e) D15: the position-id fix stays as a `pre_hook` on the bfloat16 readers, on the counts above.
  - (f) D18: the first rung, the whole model on the GPU in bfloat16, pending the fallback-off measurement of P1.14.
  - (g) Departures from the guide not yet in `docs/05_decisions.md`, each recorded in its notebook entry:
    - W1.5, layout: one word wider than the block exists, where the guide says it should never happen: in pg84 an em dash joins "industrious" and "painstaking,", which D13's hyphen mapping makes a 24-character word, 336 px in 14 px slots against the 315 and 310 px blocks; the patch-aware layout splits it and logs it.
    - W1.5 to W2.3, interfaces beyond the guide's: `LayoutFont`, `block_size_px`, `InkBox` and `glyph_ink_boxes`, `processor_inputs` made public, `run_sweep`'s keyword-only `root`.
    - W2.2: `start_after` under the strict reading (each book starts after the last word any reader of the earlier sweeps used); page ids numbered by the page's index among the pages of its text type in the plan, the same rule for shared and block Z pages.
    - W2.3: `TRACKED_PACKAGES` keeps the key `freetype-py` (the guide's `"freetype"` would duplicate it); the five runner flags (several paginations interleaved page index by page index; `__r<repeat>` on file names above repeat 0; row timestamps ending `+00:00` while `manifest.json` keeps `Z`; `root`; an existing run folder refused rather than reused).
    - W2.4: jiwer moved from the test extra to the core dependencies, since `mrr.score` imports it at module level.
    - W2.1: the seven preregistration flags of the previous entry.
Broke:
- Nothing. No code changed.
Next:
- Owner actions 1 to 4; then P2.9, the ladder, with one deliberate interruption and resume; P1.14 for D18; then your G1 decision (D4) and batch 2.
- Meanwhile, without the GPU: `src/mrr/stats.py` (Task P3.1, W2.4) on `build/p3`.

## 2026-10-14, Claude (D22): W2.4, `stats.py` (section 3.1): the harmonic model, Holm, the page bootstrap, Wilcoxon, power and the Fano floor
Did:
- `src/mrr/stats.py`: the sixteen functions of section 3.1, by name. `bin_groups`, `harmonic_design` and `group_f_test` are the guide's snippet (`group_f_test` raises `ValueError("F-test undefined: ...")` when a group adds no column, when no residual degree of freedom is left or when the residual sum of squares is zero). `holm`: sort, multiply the i-th smallest by m − i + 1, running maximum, clip at 1, back in input order. `group_rms`: the group's columns rebuilt at the block's sampled levels, times the group's coefficients, RMS over the levels (equation (1)). `harmonic_fit`: the design from every bin of the groups in increasing order, one `group_f_test` per group, one full fit with page dummies, σ = √(RSS₁/df2), each group's RMS and its share of the summed squared RMS (equation (2)); returns the groups (bins, columns, q, F, df1, df2, p, rms, share), `sigma`, `n_rows`, `n_pages`, `names` and `beta_harm`. `harmonic_component`: the fitted sum on any grid. `page_bootstrap_rms`: pages numbered by `np.unique(..., return_inverse=True)`, `default_rng(seed)`, `rng.integers(0, n_pages, size=n_pages)` per resample, rows concatenated in draw order, the i-th drawn page labelled i, refit, `group_rms`, `np.quantile` at α/2 and 1 − α/2. `pooled_cer_by_level`: errors and reference characters summed per page and level with `np.add.at`, pooled CER per level, one page draw shared by every level for the band; returns `levels`, `cer`, `low` and `high`. `paired_wilcoxon`: d = a − b, SciPy 1.17.1's `method="auto"` rule resolved in the module (more than 50 pairs, asymptotic; no zero and no tie, exact; at most 13 pairs, `PermutationMethod()`; otherwise asymptotic), `sps.wilcoxon(..., zero_method="wilcox", correction=False, method=<resolved>)`, the method reported as `"exact"`, `"permutation"` or `"asymptotic"`. `bootstrap_ci` and `median_paired_difference` as the stubs said. `fano_floor_bits`, `power_noncentrality` (`sps.f.isf`, then `optimize.brentq` on `sps.ncf.sf` between 1e-6 and 400), `pages_needed` and `paired_mde` as equations (4), (5) and (8) of sections 2.11 and 3.1. Module header and `__all__` as the guide's file conventions ask.
- `tests/test_stats.py`: the `todo` mark removed, the two original tests kept, 44 tests in all. Every row of the guide's table, with the generator and the notes' two curves at seed 0 (F to three decimals, RMS values and intervals to five, p-values to a relative 1e-3); then Holm on one value, the clip and the refusals; the sizes of the full-sweep groups (7, 7 and 42 bins); `group_rms` on a pure cosine; `harmonic_component` equal to the design times the coefficients, and periodic; the one-page refusal of `group_f_test`; the bootstrap's indifference to the seed's type and to string page ids; `pooled_cer_by_level` on three pages and four levels, with a zero-width band when the pages are identical; `alternative="greater"` as half the two-sided p; the length and NaN refusals; `bootstrap_ci` reproducible and bracketing the median; Fano at P_e = 61/62 and at 1.
Found:
- [Flag, judgement] The shares of the noiseless curves (one page, all 112 levels) are tested from the coefficients of a direct least-squares fit (`harmonic_design`, `np.linalg.lstsq`, `group_rms`), not through `harmonic_fit`: with one page and 112 levels the saturated model leaves no residual degree of freedom, so the guide's `group_f_test` raises and `harmonic_fit` cannot run there. The shares are the quantity of equation (2) all the same: 1.000 at 14 px for the patch curve; 0.4082 at 14 px and 0.5918 at 28 px for the merge curve.
- [Flag, judgement] `fano_floor_bits(1.0)` is log₂(62/61) = 0.0235, not 0. Fano's bound is zero at P_e = 61/62, uniform guessing, and rises again towards P_e = 1; the clip at 0 covers rounding at 61/62, as the notes' `fano_min_bits` does. The test pins both values; my first draft of the test expected 0 and was wrong.
- [Flag, judgement] Refusals beyond the guide's text, all `ValueError`: `holm` on an empty family, a NaN or a value outside [0, 1]; `bin_groups` on a period that does not divide the sweep; `page_bootstrap_rms` on a period that is not one of the groups; `pooled_cer_by_level` on a level with no reference characters; `paired_wilcoxon` on an unknown alternative. The guide names the Wilcoxon length and NaN refusals and the Fano range; the rest are mine.
- [Flag, judgement] Each group in `harmonic_fit`'s result also carries `columns`, the names of its design columns, beyond the guide's list (bins, q, F, df1, df2, p, rms, share); the figures can use it to draw one group's curve.
- At Qwen block C's 16 levels the 14 px group is bin 8 alone, the Nyquist bin: `cos8` is +1 or −1 at every level, so the group's RMS at the sampled levels equals the magnitude of its coefficient, where over the 112 whole-pixel levels it would be that magnitude over √2. Equation (1) takes the RMS at the block's own levels, so the 14 px amplitude of block C and the 14 px amplitude of block V (bins 2, 4 and 6 of 14 levels) are not on one footing. A note for the analysis, not a change.
- The method rule is read from the pinned SciPy, 1.17.1 (`scipy/stats/_wilcoxon.py`): the thresholds 50 and 13 count every pair, zeros included, and ties are sought among the non-zero absolute differences after `"wilcox"` drops the zeros. The module does the same.
- No `TODO[W2.4]` remains in the repository.
Broke:
- Nothing. Full CPU suite by hand in this worktree: 474 passed, 1 skipped, 104 deselected in 83 s. `tests/test_stats.py` alone takes about 40 s: the page bootstrap refits the 1,120-row model 500 times per interval.
Next:
- Task P3.2: `load_results` and `scripts/20_analyze.py` (section 3.2); then P3.3, the figures. Both without the GPU. The ladder (P2.9) and P1.14 wait for owner actions 1 to 3.

## 2026-10-14, Claude (D22): W2.4, `load_results` and `scripts/20_analyze.py` (section 3.2): the preregistration's tables from the run folders
Did:
- `src/mrr/sweep.py`: `load_results(sweep_path, runs_root)` in place of the stub, with `sweep_run_dirs` to find the folders: the SHA-1 of the sweep file, then the sorted folders of `runs_root` whose `manifest.json` carries the same `sweep_name` (the file's stem) and the same SHA-1. A folder of the same name with another SHA-1 raises `ValueError` ("a frozen config was edited"); folders without a manifest and folders of other sweeps are ignored; a missing `runs_root` gives no folders. Each `results.csv` is read with `RESULTS_DTYPES` (every id and hash a string, `repeat` and the counts int64, the measurements float64, `hit_limit` bool, `keep_default_na=False` so no cell becomes a missing value); a header other than `RESULTS_COLUMNS` raises; a `run_dir` column is added; a repeated `RESULTS_KEY` (reader, page_id, condition_hash, repeat) raises `ValueError` naming the key and the folders. Sorted by reader, page_id, block, condition_hash. `_reader_periods` became the public `reader_periods`.
- `scripts/20_analyze.py`: `--config`, `--runs-root` (default `results/runs`) and `--out-dir` (default `reports/tables`); the local `sweep` renamed `sweep_cfg`. A `RunFolder` per run folder rebuilds the plan from the manifest's `config.sweep` and `config.reader` (`expand_conditions`, the condition hashes per `pagination_key`, the repeats), reads `coherence.csv`, `determinism.csv` and `references/`, and tells a block Z page from a shared page by its id (`_z<size>_<conv|pa>_p<nnnn>`). Section 8 of the preregistration: a page leaves every table but `checks.csv` when any planned (condition, repeat) of its pagination is missing; nothing else is dropped. Six tables into `reports/tables/<sweep_name>/`, each with `status` and `family`, then `write_manifest(out_dir, cfg, run_id=new_run_id(sweep_name, "analysis"))` recording the sweep's SHA-1, the run folders, the readers planned and analysed, the rows loaded and analysed, the excluded pages per run, the resamples (2,000), the seed (20261014), α, δ and the plateau sizes.
  - `checks.csv`, one row per run folder: rows done and planned, repeated keys, missing transcription files, `hit_limit` count and share, determinism pairs and mismatches, pages seen and excluded (with their ids), median latency.
  - `harmonics.csv`, per reader, text type and block V, H, C, one row per period group of `stats.bin_groups(L, periods, n)`, L the block's range (cell, patch or sweep) and the periods the reader's that divide L; `stats.harmonic_fit` at the block's levels and `stats.page_bootstrap_rms` per group; period, bins, q, F, df1, df2, p, Holm p, RMS and its interval, share, σ, pages, levels, rows. Families: `a_<text_type>` for the patch group of H and the patch and cell groups of V (Holm over the three tests of one reader), `control_C` for every group of C (Holm within reader and text type).
  - `paired.csv`: `H_phi0_vs_C_pooled` (a = the page's CER pooled over every level of C, b = the page's (H, φ = 0) CER; family `c_<text_type>`, Holm across the readers) and `A_<mode>_vs_C_phi0` (a = (C, φ = 0), b = the mode's row; family `A`, Holm over the modes of one reader and text type); n, n_zero, statistic, p, method, Holm p, the median of b − a and its bootstrap interval.
  - `subpixel.csv`: block S, one F-test per reader and text type (`stats.harmonic_design(phase, 1, [1, 2])`, cos1, sin1, cos2, q = 3) and the pooled CER at each phase with its page-bootstrap band.
  - `sizes.csv`: block Z per reader, text type, render mode and size: pooled CER with its band, vision tokens, characters per token, symbols per token (counted in `references/`), the plateau CER (pooled over the 20 and 24 px sizes) and s* (the smallest size within δ = 0.01 of the plateau; missing when none is).
  - `coherence_summary.csv`: per reader, text type, block, render mode, line-pitch mode, axis, period and harmonic: mean, min and max of R over pages, beside the RMS of the matching `harmonics.csv` row.
- `tests/test_analyze.py`: 14 tests. One run of a six-block Qwen sweep on a fake backend whose transcription drops 2 to 32 percent of the reference's characters by the SHA-1 of the rendered page (CER varies with condition and page, identical pixels give identical text, about one row in ten claims `hit_limit`), analysed through `subprocess` as `make analyze` would. The guide's five cases and more: every expected row of every table (bins, q, levels, df1, df2, families, Holm's first step, V's shares summing to 1, the band around each CER, s* against the plateau, the matching RMS beside R); one H row of a random62 page removed, so that page leaves every table but `checks.csv` and that text type's counts drop by one page; a copied run folder raising on the repeated key from the function and from the script; a folder of another sweep, a folder without a manifest and a stray file ignored, an edited sweep refused; the hashes `1e5123456789` and `000000000042` read back as strings; `page_key`, `write_table`, and exit code 1 without a run folder.
- `tests/test_sweep.py`: the `load_results` not-implemented test replaced by one on the real run folder of the runner tests (every id a string, `hit_limit` a bool, 256 tokens a row, sorted keys) and one on an empty runs root.
Found:
- [Flag, judgement] `coherence_summary.csv` carries `text_type`, `render_mode`, `line_pitch_mode` and `axis` beyond the guide's "reader, block, period and harmonic": the RMS of `harmonics.csv` is per text type, so R is summarised per text type to sit beside it; a coherence row's text type comes from its page id through `results.csv`.
- [Flag, judgement] `sizes.csv` is labelled `description`, as `coherence_summary.csv` is: block Z describes the trade-off and tests nothing, and the guide's family table names no family for it.
- [Flag, judgement] Columns beyond the guide's lists: `comparison`, `axis`, `range_px` and `n_rows` in the test tables; `cer_low_at_<phase>` and `cer_high_at_<phase>` beside `cer_at_<phase>` in `subpixel.csv`; `cer_plateau` in `sizes.csv`; `determinism_pairs`, `pages_seen` and `excluded_page_ids` in `checks.csv`; `run_dir` from `load_results`. `repeated_keys` in `checks.csv` is always 0, since `load_results` raises on a repeated key before any table is written.
- [Flag, judgement] New public names in `sweep.py`: `sweep_run_dirs`, `RESULTS_DTYPES`, `RESULTS_KEY` and `reader_periods`; the figures (P3.3) will use them.
- [Flag, judgement] The plan is rebuilt from each run folder's manifest, not from the sweep file: a page's planned conditions are what that run planned. The sweep file only finds the folders (stem and SHA-1).
- [Flag, judgement] A V or H period group that is neither the patch nor the cell would be written `exploratory`, family `other_<block>`; none arises for the three readers' sweeps, since every period of H divides the patch and every period of V the cell. `sizes.csv` is written with its header alone when the sweep has no block Z, so every sweep leaves the six files. The script exits 1 with "no run folder of sweep ..." when `runs_root` holds no run of the sweep, instead of writing empty tables.
- Pooled CER is a ratio of sums everywhere (`pooled_cer` over `CharCounts`, D13); every band resamples pages, never rows (`stats.pooled_cer_by_level`, `stats.page_bootstrap_rms`); every Holm family is one `stats.holm` over the family's defined p-values. A Wilcoxon pair whose differences are all zero (SciPy raises) is written with `method = "undefined: ..."` and no p, so it cannot enter a family unseen.
- `make analyze` has yet to write a table on a real run, the done-when of W2.4 at `docs/03_pipeline.md:66`; the box stays open until then.
Broke:
- Nothing. `tests/test_analyze.py` and `tests/test_sweep.py`: 68 passed in 69 s. Full CPU suite by hand in this worktree: 489 passed, 1 skipped, 104 deselected in 140 s.
Next:
- Task P3.3, the figures (section 3.3), without the GPU. The ladder (P2.9) and P1.14 wait for owner actions 1 to 3; `make analyze` on the first real run once it exists.

## 2026-10-14, Claude (D22): W2.5, `plotting.py` and `scripts/30_make_figures.py` (section 3.3): Figures 1, 2 and 3 from the run folders
Did:
- `src/mrr/plotting.py`: `plot_phase_curve(df, out_path, n_resamples=2000, seed=0, *, reader_cfgs=None)` in place of the stub. Readers in rows, blocks V, H and C in columns. Points: the pooled CER at each level (D13) with the page-bootstrap band of `stats.pooled_cer_by_level` as error bars, natural in blue, random62 in red. Line: `stats.harmonic_component` of `stats.harmonic_fit` (page fixed effects; the groups of `stats.bin_groups` over the reader's periods that divide the block's range) on a 241-point grid, added to the block's pooled mean CER, drawn only where the model is defined (two or more levels, a tested period dividing the range, a residual degree of freedom). Light vertical lines at the multiples of the patch, darker at the multiples of the token cell; the x axis from 0 to the block's range, read from the levels (n × step when they are 0, step, ..., (n − 1) step); the swept axis is whichever phase column varies. Rows of S, A and Z are ignored; a panel without rows says so. Reader configs from the keyword or, when it is None, from each run folder's `manifest.json` (`config.reader`) through the frame's `run_dir` column. `plot_tradeoff(df, out_path, pe_assumed=(0.0, 0.03, 0.1))`: one panel per text type of block Z, the pooled CER per size against the pages' mean reference characters per vision token, colour per reader, solid for conventional and dashed for patch-aware, each point labelled with its size in pixels; a last panel with `stats.fano_floor_bits` (M = 62) times the symbols per token at each assumed P_e, every line labelled "(assumed)"; no measured CER enters it. `plot_mtf(curves, out_path, px_per_degree=None)`: gain against cycles per pixel, one line per name, the Nyquist line at 0.5, a top axis in cycles per degree when a scale is given. Every figure is a `Figure` without pyplot, `pdf.fonttype` 42, `bbox_inches="tight"`, and the PDF's `CreationDate` is dropped, so the same rows and seed give the same bytes; `plot_equivariance` saves through the same helper.
- `scripts/30_make_figures.py`: `--config`, `--runs-root` (default `results/runs`) and `--out-dir` (default `reports/figures`); `sweep_run_dirs` (exit 1 with "no run folder of sweep ..." when none), a `RunFolder` per folder for the reader configs, `load_results`, then the section 8 exclusion through `complete_rows`, the call `scripts/20_analyze.py` makes, so the figures draw the rows the tables used. `fig1_phase_<sweep_name>.pdf` always; `fig2_tradeoff_<sweep_name>.pdf` when the rows hold block Z; 2,000 page resamples and seed 20261014 as section 9 says; nothing read from `reports/tables/`.
- `src/mrr/sweep.py`: `RunFolder`, `excluded_pages` and the block Z page-id pattern moved here from `scripts/20_analyze.py`, unchanged in behaviour, plus `complete_rows(df, folders)` returning the kept rows and the excluded pages per run folder; the analysis script imports them and its `main` calls `complete_rows`.
- `tests/test_plotting.py`: 12 tests on synthetic frames (a cosine of the patch period plus a page effect and noise, with hits, substitutions, deletions and insertions summing to the reference). Figure 1 for three readers with and without S, A and Z rows, for one reader, with an empty panel and a one-page block, with reader configs from a fake manifest and the two errors without them, the same bytes for the same seed and other bytes for another seed, and the error without V, H or C rows; Figure 2 with two Fano settings and the error without block Z; Figure 3 with and without the degree axis and its three errors; Figure 0 from a synthetic frame; the script through `subprocess` on a fake-backend run of the V, H, C and Z blocks of the analysis sweep (both PDFs, exit 0, no page excluded) and exit 1 without a run folder. The synthetic figures were drawn and looked at (PNG through the same code): nine panels, error bars, fitted lines, the patch and cell lines, one legend; the Z lines labelled by size beside the Fano panel; the MTF curve with its degree axis.
Found:
- [Flag, judgement] Figure 1 draws the band as error bars, not as a shaded strip: the levels are discrete (4, 2 and 8 per block), and a strip between them would interpolate the points the guide says never to smooth. The fitted model is the only curve.
- [Flag, judgement] The line of Figure 1 is drawn only where the harmonic model has a residual degree of freedom and a tested period divides the block's range; a block with one page or one level keeps its points and no line, instead of a fit that could not be checked.
- [Flag, judgement] `plot_phase_curve` takes a `reader_cfgs` keyword beyond the guide's `(df, out_path, n_resamples, seed)`; without it the configs come from the run folders' manifests through `run_dir`. The script passes them from `RunFolder`, so a frame from `load_results` draws either way.
- [Flag, judgement] `plot_mtf` (Figure 3, section 3S.4, cut with the P3S tasks on 14 October) is implemented generically, since it costs nothing and the section may return; no script calls it.
- [Flag, judgement] `scripts/03_equivariance.py` keeps its `draw_figure` calling `plotting.plot_equivariance(df, reader_cfgs, figure)`: the guide's "move the drawing into `plot_equivariance(csv_dir, out_path)`" applies when W1.9 drew in the script, and W1.9 drew through `plotting.py` from the start.
- [Flag, judgement] The PDFs carry no `CreationDate`, so `make figures` on the same rows gives byte-identical files (a test checks it). `plot_equivariance` changes the same way, so `reports/figures/fig0_equivariance.pdf` regenerated at this commit would differ from the committed file in that one field.
- [Flag, judgement] Plan P3.3 says "importing matplotlib inside the function with Agg". The functions build a `Figure` without pyplot, which needs no backend and sets none, as `plot_equivariance` did at W1.9; `matplotlib.use("Agg")` is not called, so the module changes no global state of the process that imports it.
- [Flag, judgement] The plan's commit `W2.5: Figure 1 from the pilot` needs a pilot run, which waits for the GPU; committed as `W2.5: Figures 1, 2 and 3 from the run folders`, with the figure on a real run still to come.
- Colours are fixed: natural blue, random62 red in Figures 1 and 2; Figure 0 keeps its line styles for the text types and viridis for the layers.
- `make figures` has yet to write Figure 1 on a real run, the done-when of W2.5 at `docs/03_pipeline.md`; the box stays open until then, as the W2.4 box does.
Broke:
- Nothing. `tests/test_plotting.py`, `tests/test_analyze.py` and `tests/test_sweep.py`: 80 passed in 114 s. Full CPU suite by hand in this worktree: 501 passed, 1 skipped, 104 deselected in 146 s.
Next:
- Gate 2 (P3.4) needs the ladder and the pilot, which wait for owner actions 1 to 3 and the GPU; `make analyze` and `make figures` on the first real run once it exists. The full sweep config (P3.5) once the ladder gives the sizes.

## 2026-10-14, Claude (D22): W3.5, `scripts/40_freeze.py` (section 3.8): the freeze and its check, ahead of the run
Did:
- `scripts/40_freeze.py`: `freeze(results, final, replace=False, root=None)` copies `runs/`, `equivariance/`, `mtf/` and `probe/` (those present, in that order) from `results/` into `results/final/` with `shutil.copytree`, skipping `images/` and `__pycache__`; `FileExistsError` when any of those folders already exists under `final`, unless `--replace`, which removes them and the README first. `write_readme(final, copied, root)`: `results/final/README.md` with the UTC time, the commit the freeze was made at (`manifest.git_commit`) and whether the tree was dirty, one line per folder copied saying what it holds, a table of the run folders from their manifests (run id, sweep, reader, commit with `*` when the manifest records a dirty tree, rows in `results.csv`, rows planned, start time), then a fenced block "SHA-256 of every file" in `sha256sum` format (hash, two spaces, POSIX path relative to `final`), sorted, every file but the README itself. `check_hashes(final)`: recomputes every hash and names the missing, changed and unlisted files. `check_tables(final, sweeps, tables_root)`: for each sweep runs `scripts/20_analyze.py --config <sweep> --runs-root <final>/runs --out-dir <tmp>` and compares every CSV of `<tmp>/<sweep_name>/` with `<tables_root>/<sweep_name>/` byte for byte; `manifest.json` is not compared; a missing table folder says to run `make analyze` first. Command: `--results`, `--final`, `--replace`; `--check` with a repeatable `--sweep` and `--tables`; exit 1 on any problem, on the refusal and when `results/` holds none of the four folders.
- `Makefile`: `freeze` and `freeze-check` (`SWEEP=...`). `results/README.md`: the layout now shows `references/`, the run logs, `equivariance/` and what `final/` holds; the rule names the script and the check. `docs/03_pipeline.md`: the W3.5 files line names the script and its tests; the box stays open until the real freeze and the tag.
- `tests/test_freeze.py`: 6 tests on a fake-backend run of the V and H blocks of the analysis sweep, beside a manifest-only folder like those of `make readers`, an equivariance CSV and an `images/` file: the copy without images and the README's folder lines, run table and hash list (as many hashes as files, the README not listed, one hash recomputed); the refusal and `--replace` restoring a damaged file; nothing to freeze; `check_hashes` naming a changed, a missing and an unlisted file, and a missing README; `check_tables` clean on the fresh tables, naming an appended `paired.csv` and a deleted `sizes.csv`, and a missing table folder; the command end to end (freeze, refusal, `--check --sweep` passing, exit 1 after an edit).
Found:
- [Flag, judgement] `images/` is not copied into `final/`: the rendered pages are git-ignored (`.gitignore`, `results/runs/**/images/`), large, and regenerable from a row's page id, condition hash and seed with the code at the frozen commit (`determinism.csv` checks that the renderer repeats itself); the guide's freeze is of the CSVs. The transcriptions and references are copied: they are the text the CER was computed from and cannot be regenerated without the model.
- [Flag, judgement] The hashes live inside `README.md` in `sha256sum` format, as guide 3.8 words it, rather than in a separate `SHA256SUMS` file; `make freeze-check` does what `sha256sum -c` would, and names the file that changed.
- [Flag, judgement] Whole run folders are copied, not the CSVs alone: `manifest.json`, `progress.log`, `transcriptions/` and `references/` go with `results.csv`, so every table can be rebuilt from `final/` and the transcription behind any row can be read. `results/README.md` said "frozen CSVs only"; it now says what is copied.
- [Flag, judgement] `--check` compares the CSVs only: `reports/tables/<name>/manifest.json` carries the analysis run id and timestamp, which change on every run (guide 3.8, step 3: every CSV byte-identical, only `manifest.json` may differ).
- [Flag, judgement] `results/equivariance/` (W1.9, build/p2) is frozen with the runs, so the Figure 0 numbers trace to `final/` as the README rule asks of every number in the paper.
- [Flag, judgement] A freeze over a freeze is refused; `--replace` removes the previous copy whole, so redoing a signed-off freeze is a visible act for the notebook and a new tag, not an accident of `make freeze` run twice.
- Not done here: the freeze itself (needs the full run and `make analyze` on it), the owner's sign-off and the tag `data-freeze-v1` (step 4, which also decides D16), the fresh review (step 5).
Broke:
- Nothing. `tests/test_freeze.py`: 6 passed in 24 s. Full CPU suite by hand in this worktree: 507 passed, 1 skipped, 104 deselected in 167 s.
Next:
- P3S.1 (W3.6), the layer-0 MTF from the patch-embedding weights, CPU only; the full sweep config (P3.5) once the ladder gives the sizes. Gate 2 and every run wait for owner actions 1 to 3 and the GPU.

## 2026-10-14, Claude (D22): W3.6, `src/mrr/mtf.py` and `scripts/04_mtf.py` (section 3S.1): the layer-0 MTF from the patch-embedding weights, CPU only
Did:
- `src/mrr/mtf.py` (NumPy only): `grey_kernels(weight, image_std)`, equation (1), each colour slice divided by the processor's `image_std` and the slices summed, a five-dimensional Qwen weight summed over its two frames first; `kernel_response(K, f, theta)`, equation (2), the direct Fourier sum with u the column and v the row of `K[d, v, u]`, theta = 0 a grating whose bars are vertical; `m0(K, f, theta)`, equation (3); `radial_profile(K, nfft=128, n_rings=32)` and `horizontal_profile(K, nfft=128)` as in notes A8 (zero-padded FFT, the power averaged over the filters, square root, divided by the largest value); `csf_mannos_sakrison(f)`.
- `scripts/04_mtf.py` (`make mtf`): for each reader of `configs/readers.yaml`, the shard from `model.safetensors.index.json` (or `model.safetensors` when there is none), the safetensors header by two range requests, dtype, shape, shard and byte count checked against the table of guide 3S.1, exactly the tensor's bytes by one range request (a reply other than 206 is refused, so a server that ignores the range never sends the checkpoint), SHA-256 checked, then `preprocessor_config.json` at the same revision for sigma_c. Saves `<reader>__patch_embedding.npy` (float32, lossless from BF16), `<reader>__preprocessor_config.json` and `<reader>__record.json` under `data/external/weights/` (git-ignored; a README is committed). Writes `results/mtf/layer0_<reader>.csv` (`profile`, `f_cyc_per_px`, `value`: 32 ring rows, 64 horizontal rows, 32 `m0_0deg` and 32 `m0_45deg` rows at the ring centres) and `manifest.json` (`write_manifest`, run id `..._mtf_layer0_<sha>`). `--offline` reads the saved files, checks the hash again and fetches nothing. 5.5 MB fetched in all; no checkpoint, no model, no GPU.
- `plot_mtf(curves, out_path, px_per_degree=None, *, reference=None)`: a reference curve drawn as a plain black line; the top axis label names rho as assumed; legend at the lower right, where no curve goes. `scripts/30_make_figures.py`: `--mtf-dir` (default `results/mtf`), `--config` now optional; Figure 3a (`reports/figures/fig3a_layer0_mtf.pdf`) from the ring rows of every `layer0_*.csv`, with the Mannos-Sakrison CSF placed at rho = 40 px per degree by f_px = f_deg / rho; exit 1 when there is neither a sweep nor MTF files. `docs/04_data_dictionary.md`: a section for `results/mtf/`. `results/README.md`; `docs/03_pipeline.md` W3.6 ticked; `Makefile` `mtf`.
- Ran it: all three SHA-256 values equal the table; Qwen 1280 filters of 14 px, SmolVLM-2.2B 1152 of 14 px, SmolVLM-500M 768 of 16 px. The ring table of notes A8 reproduced: Qwen peak ring 0.258, lowest ring 0.198, ring at 0.070 0.481, last ring 0.832, no ring after the peak below half of it; SmolVLM-2.2B peak in the lowest ring (1.000), 0.555 at 0.070, last ring 0.389, below half of the peak from 0.289 on; SmolVLM-500M peak 0.164, lowest 0.239, 0.787 at 0.070, last 0.936, none below half. The committed results come from a detached worktree at the code commit (0c52814) with the output folder under the ignored `data/external/`, so the manifest records a clean tree; the CSVs are byte-identical to the first run in this worktree, whose manifest said dirty because the files it was writing were untracked.
- `tests/test_mtf.py`: 13 CPU tests (`grey_kernels` with differing frames and the error cases; `kernel_response` against the FFT on its grid and `m0` at zero frequency; equation (4) on a random bank of 8 kernels with float gratings on 6 x 6 patches at four phases, five frequencies and four directions, ratio within 1e-12, at two contrasts; the delta kernel giving 1 everywhere; a Gaussian falling with frequency; the CSF peak at 7.89 cycles per degree with 0.981; the script against a local HTTP server with Range support: the fetch, the saved files, the 160 rows and the manifest; a hash mismatch stopping the second reader with no manifest; a shard in another place than the table says; a server that ignores Range; `--offline` reusing the files byte for byte and stopping on a changed tensor or a missing folder; an unknown reader and one without a table row) and 3 model-marked tests that reproduce the A8 ring table from the real tensors within 0.002. `tests/test_plotting.py`: the reference curve, Figure 3a from fake CSVs without a sweep (same bytes on a second run), nothing to draw.
Found:
- [Flag, judgement] "First ring below half the peak" in the A8 table is counted from the peak onwards (Qwen's lowest ring, 0.198, is below half of its peak but comes before it); the script's log and the model-marked test read it that way and say "gain, not a resolution limit" beside it (A8, "Gain is not loss").
- [Flag, judgement] The tensors are kept under `data/external/weights/` with a record of the fetch, and `--offline` reuses them: the model-marked test, and section 3S.2 should it return, read the same bytes without the network; nothing of it is committed, the manifest carries the hashes, shapes and revisions.
- [Flag, judgement] Figure 3a puts the CSF on the same y axis as the ring profiles. The profiles are each divided by their own peak, so 1 means "the frequency this layer passes best"; the CSF is Mannos and Sakrison's fitted sensitivity, which peaks at 0.981. The shared axis is labelled "gain", the legend calls the CSF what it is, and rho = 40 is written in the top axis label as assumed (guide 3S.4: never a match).
- [Flag, judgement] `scripts/30_make_figures.py` draws Figure 3a whenever `results/mtf/layer0_*.csv` exist, before the sweep figures and also in a `--config` run, so `make figures` redraws it each time; the bytes are deterministic (the test checks a second run gives the same file).
- [Flag, judgement] The fontTools font-subsetting log, which the PDF backend emits at INFO level for every glyph it embeds, is silenced in the figure script; nothing else of the logging changes.
- The results manifest: `git_dirty` counts untracked files (`git status --porcelain`), so a script that writes its CSVs before its manifest can never record a clean tree from inside the repository; the runner writes its manifest first and has no such problem. Noted for any future one-off script: run it from a detached worktree with the outputs under an ignored path, as here and as for Figure 0.
Broke:
- Nothing. `tests/test_mtf.py`: 16 passed in 13 s (the 3 model-marked ones against the real tensors). Full CPU suite by hand in this worktree: 522 passed, 1 skipped, 107 deselected in 181 s.
Next:
- P3.5, the full sweep config, once the ladder gives the sizes; P4.1, the spec. Every run, Gate 2, the freeze and the tag wait for owner actions 1 to 3 and the GPU.

## 2026-10-14, Claude (D22): W3.2, `configs/sweeps/full.yaml` as the template of section 3.5, numbers at Gate 2
Did:
- `configs/sweeps/full.yaml` rewritten to the template of build guide section 3.5: the pilot's blocks V, H, C, S and A unchanged, block Z as the ladder's design (10, 12, 14, 16, 20 and 24 px, both render modes, `paginate: per_condition`), seed 20261016 (the ladder's is 20261012, the pilot's 20261013), `pages.natural.start_after: [configs/sweeps/ladder.yaml, configs/sweeps/pilot.yaml]`, `z_per_text_type: 5`. The page counts per reader, the sizes, the run count and the hours are `TODO[W3.2]` strings, which `plan_pages` and `expand_conditions` refuse, so the file cannot start a run before Gate 2 fills it (D5).
- `tests/test_configs.py`: the full sweep names both earlier sweeps, a new seed and five Z pages per text type; for every reader its blocks expand to the pilot's conditions plus the ladder's twelve Z layouts (62, 74 and 66 conditions); no page shares text with the ladder or the pilot, every natural page starts after the last word any earlier reader used and the random stream is new; every Z layout holds 2 x N_Z pages, each after this sweep's shared pages; page i of the twelve layouts starts with the same word. While the placeholders remain, the three plan tests run on a stand-in with the file's structure and small counts (16 px, 2 shared and 1 Z page per text type, the earlier sweeps replanned from concrete copies with one page each); once the numbers are in they run on the file itself. `test_header_run_count_equals_the_computed_count` now passes `readers=` to `plan_pages`, which `start_after` needs.
Found:
- The header test and the Gate 2 fill: when Gate 2 sets the counts, `total runs` and the hours in the header must be written in the same edit, or that test fails on the file.
- `start_after` replans the pilot, so `pilot.yaml` must hold its sizes (not "from the ladder") before the full sweep can be planned at all; the ladder run writes them.
- If the owner takes the Sunday plan (no separate pilot), `pilot.yaml` leaves `start_after`; nothing else in the file changes.
Broke:
- Nothing. `tests/test_configs.py`: 31 passed, 1 skipped (the full sweep's run count, until Gate 2) in 8 s.
Next:
- Gate 2: the counts from `pages_needed` under the 108 h budget, the header, the pilot's sizes; commit as `W3.2: full sweep config and budget (D5)`. Then P4.1 on `build/p4`.

## 2026-10-15, Claude (D22): W4.1, `spec/RENDERING_SPEC.md` R01 to R11, the contract checks and the reader facts table
Did:
- `spec/RENDERING_SPEC.md` rewritten to build guide section 4.1: Scope, Definitions, the rule template, R01 to R11 each in the six fields RULE, WHY, EVIDENCE, PARAMETERS, APPLIES WHEN and TEST, the Kernel interface with the signature and the argument table, and the Conformance table. Every EVIDENCE names files under `reports/` and nothing else: R01 to R06 and the Conformance table cite `reports/tables/contract/contract_checks.csv`, R07 and R08 cite `reports/tables/readers/reader_facts.csv`, R09 to R11 cite the full run's tables and figures and carry `TODO[W4.1]` for their numbers until the freeze. Line 10 names `h2/README.md`, as the guide's text does (F2); kept.
- `scripts/08_contract_checks.py` (`make contract`): 70 checks for R01 to R06 and the Conformance table, 70 passed, written as `reports/tables/contract/contract_checks.csv` (columns `rule, check, expected, observed, passed`) after a manifest; exit 1 on any failure. Each expected cell is the spec's number; each observed cell is recomputed through the renderer (`load_glyph`, `layout_page`, `composite_coverage`, `render_page`) or, for the three Conformance facts, from the spec's own equations over all 16,777,216 (S, T, B) byte triples with the script's own single- and double-precision sRGB functions. No `round()`, `np.round` or `torch.round` anywhere; `tests/test_contract_checks.py` greps for them.
- `scripts/01_check_readers.py --out reports/tables/readers/reader_facts.csv` (`make reader-facts`, GPU): one row per reader, 25 columns (`FACTS_COLUMNS`), with the blind-strip noise test of `tests/test_reader_facts.py` built in (columns 378 to 383, rows 378 to 383, column 377 as the control, noise from `default_rng(2)`); the manifest first, the table once every reader has passed. Not run yet: the GPU half waits for owner actions 1 to 3. The schema is in `docs/04_data_dictionary.md`; the guide names the file and not its columns, so the column list is my proposal (F4).
- `tests/test_spec.py` (structure, forbidden content, evidence paths under `reports/`, the tolerance table, halves up), `tests/test_contract_checks.py` (the run, the worked examples through the renderer, the exhaustive Conformance rows, the manifest, a forced wrong expectation returns 1, no library rounding), `tests/test_check_readers.py` (the column list against the dictionary, `encoder_input_px`, `facts_row` for a native Qwen page, for SmolVLM with the strip facts and for a tiled or resized frame, `write_facts`). Only the two W4.1 "done when" tests are `todo`-marked (every cited file exists; no `TODO[W4.1]` remains), so the rest run in CI now (F10).
- `Makefile`: `contract` and `reader-facts`. `docs/03_pipeline.md`: the W4.1 files and done-when. `docs/04_data_dictionary.md`: the two tables.
Found:
- Two numbers of the first draft were wrong, and the script caught them. A power of 2.2 in place of the sRGB curve changes 11 of the 27 `coverage_linear` bytes of the R05 examples by at most 2 levels (the draft said "up to 8"), and by 9 levels at worst over all byte triples. Single precision moves 148 `coverage_linear` triples by one level for either way of writing the mix (the draft said "148 to 160, depending on how the arithmetic is written"; that range was not computed, and a scratch run through `mrr.color`, which upcasts to double inside, gave 163 and 162, which is not a single-precision count either). The spec now quotes the measured values and the Conformance paragraph names the rows. Rule for the file from here: a stated number changes only to what `make contract` measured, with an entry here saying why the earlier one was wrong.
- The smallest distance of 255·o from a half-integer in `coverage_srgb` is exactly 1/510 over every triple, as the parity argument of the Conformance section says, and single precision moves no `coverage_srgb` triple: the two zero tolerances are measured, not only argued.
- On the first Persuasion page at 16 px, 192 pixels lie inside two glyph boxes and none has an unclipped sum above 255, so the R04 clip never acts on that page; the row reports the count instead of asserting it, and the clip itself is tested by the (200, 100) pair.
- `tests/test_smoke.py::test_results_columns_match_data_dictionary` matched every `| \`name\` |` row of the whole dictionary and had been failing since the MTF table was documented at W3.6 (three extra columns); it now reads the `## results.csv` section only.
- Loading a script with `importlib.util.spec_from_file_location` and `exec_module` without putting the module in `sys.modules` breaks `@dataclasses.dataclass` under `from __future__ import annotations` (`'NoneType' object has no attribute '__dict__'`); the two new test loaders register the module first.
Broke:
- Nothing. `tests/test_spec.py`, `tests/test_contract_checks.py`, `tests/test_check_readers.py`, `tests/test_smoke.py`: 57 passed, 12 deselected (todo) in 13 s. Full CPU suite in this worktree: 571 passed, 1 skipped, 119 deselected in 202 s.
Next:
- `make contract` from a detached worktree at this commit with the outputs under an ignored path, the table and its clean manifest committed as `W4.1: contract checks table`. `make reader-facts` once the GPU is free (owner actions 1 to 3). R09 to R11 and the todo marks after the freeze. Re-run `make contract` at the `spec-v1` commit before Gate 3 (build guide 4.1).
