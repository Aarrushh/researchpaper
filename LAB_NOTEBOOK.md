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
