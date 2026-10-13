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
