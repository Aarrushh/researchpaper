<!--
PURPOSE: Defines every column of results.csv and the run folder layout.
WHAT GOES HERE: Column name, type, unit, allowed values, example. Run ID format.
FILLED IN AT STEP: Scaffolded at setup. Updated only if the schema changes (log in 05_decisions.md).
DO NOT: Change a column without also changing src/mrr/manifest.py RESULTS_COLUMNS and logging a decision.
-->

# Data dictionary

## results.csv

One row per page per condition per repeat. Columns appear in this order. The same list is in `src/mrr/manifest.py` as `RESULTS_COLUMNS`, and `tests/test_smoke.py` checks the two agree.

Examples are illustrative formats only, not results.

| Column | Type | Unit | Allowed values | Example |
|---|---|---|---|---|
| `run_id` | string | none | run ID format below | `20261005-1432_pilot_qwen_ab12cd3` |
| `reader` | string | none | a `name` in `configs/readers.yaml` | `qwen` |
| `reader_revision` | string | none | HF revision hash | `<40-char hash>` |
| `page_id` | string | none | `<book_id>_p<NNNN>` (shared pages) or `<book_id>_z<size>_<mode>_p<NNNN>` (block Z; `<mode>` is `conv` or `pa`); NNNN is the page's index among the pages of its text type in the plan | `pg105_p0007` |
| `book_id` | string | none | `pg<Gutenberg id>`, a key of `mrr.text_source.BOOKS` (`data/raw/texts/README.md`), or `rnd<seed>` for the sweep's random stream | `pg105` |
| `text_type` | string | none | `natural`, `random62` (from the page plan) | `random62` |
| `block` | string | none | `V`, `H`, `C`, `S`, `A`, `Z` (the block of the sweep config, D10) | `H` |
| `condition_hash` | string | none | 12 hexadecimal characters, `mrr.sweep.condition_hash` (rule below) | `9c811d1777d4` |
| `repeat` | int | none | 0 to repeats minus 1 | `0` |
| `font_file` | string | none | a file name under `data/external/fonts/`, the name only | `SourceSans3-Regular.ttf` |
| `font_size_px` | float | pixels | positive | `16.0` |
| `line_pitch_px` | float | pixels | positive: floor(1.25 × size + 0.5) when `unaligned`, the reader's token cell when `aligned` | `20.0` |
| `line_pitch_mode` | string | none | `aligned`, `unaligned` | `unaligned` |
| `aa_mode` | string | none | `binary`, `coverage_srgb`, `coverage_linear` | `coverage_srgb` |
| `render_mode` | string | none | `conventional`, `patch_aware` | `conventional` |
| `phase_x_px` | float | pixels | 0 to the reader's sweep period (exclusive), D9: 112, 42 or 64 | `7.0` |
| `phase_y_px` | float | pixels | 0 to the reader's sweep period (exclusive), D9 | `0.0` |
| `image_width_px` | int | pixels | the reader's native page size (D8): 448, 384 or 512 | `448` |
| `image_height_px` | int | pixels | the reader's native page size (D8): 448, 384 or 512 | `448` |
| `vision_tokens` | int | tokens | positive; equals `grid.token_count` for the page | `256` |
| `reference_chars` | int | characters | positive | `379` |
| `transcription_file` | string | none | path relative to the run folder | `transcriptions/pg105_p0007__9c811d1777d4.txt` |
| `output_tokens` | int | tokens | 0 or more (`Transcription.new_tokens`) | `412` |
| `hit_limit` | bool | none | `True`, `False` (`Transcription.hit_limit`: the output stopped at `max_new_tokens`, not at an end token) | `False` |
| `cer` | float | ratio | 0 or more (can exceed 1) | `0.031` |
| `wer` | float | ratio | 0 or more (can exceed 1) | `0.084` |
| `hits` | int | characters | 0 or more (jiwer, `score.char_counts`) | `371` |
| `substitutions` | int | characters | 0 or more (jiwer) | `6` |
| `deletions` | int | characters | 0 or more (jiwer) | `2` |
| `insertions` | int | characters | 0 or more (jiwer) | `1` |
| `latency_s` | float | seconds | positive | `4.2` |
| `timestamp_utc` | string | ISO 8601 | aware UTC from `datetime.isoformat(timespec="seconds")`, ends in `+00:00` (manifest.json keeps its older `Z` form) | `2026-10-14T14:32:10+00:00` |
| `git_commit` | string | none | short or full SHA | `ab12cd3` |

## Run folder layout

```
results/runs/<run_id>/
├── manifest.json                                  written by mrr.manifest.write_manifest
├── results.csv                                    columns above, one row per page, condition and repeat
├── coherence.csv                                  one set of rows per page at phase 0, per block layout: page_id, block, render_mode, line_pitch_mode, axis, period_px (the reader's patch, token cell and attention window), harmonic (1 to 3), R (section 2.2)
├── determinism.csv                                one row per page for each pair of conditions that differ only in their block label, once both are done: page_id, condition_hash_a (the earlier block in file order), condition_hash_b, identical (True or False) (section 2.8)
├── progress.log                                   one line per page: `<time> page <i>/<N> <page_id> rows <done>/<planned> page_latency_s <mean> projected_finish <time>`, times aware UTC (section 2.8)
├── references/<page_id>.txt                       the reference text of each page, so every CER can be recomputed from this folder alone
├── transcriptions/<page_id>__<condition_hash>.txt raw model output, one file per row; a repeat above 0 adds `__r<repeat>` before `.txt`
└── images/<page_id>__<condition_hash>.png         rendered pages, the first page's by default (`--save-images first`; `all`, `none`); the same repeat suffix; gitignored
```

`<condition_hash>` is `mrr.sweep.condition_hash(cond)` (build guide section 2.5): a dictionary of exactly ten keys, `aa_mode`, `block`, `font_file` (the file name only), `font_size_px`, `line_pitch_mode`, `line_pitch_px`, `patch_aware_slot_px` (`null` for `conventional`, the reader's patch in pixels for `patch_aware`), `phase_x_px`, `phase_y_px` and `render_mode`, every number cast to float, serialised with `json.dumps(cond, sort_keys=True, separators=(",", ":"))` and hashed with SHA-1; the hash is the first 12 hexadecimal characters. Test vector: `{"aa_mode":"coverage_srgb","block":"C","font_file":"SourceSans3-Regular.ttf","font_size_px":16.0,"line_pitch_mode":"unaligned","line_pitch_px":20.0,"patch_aware_slot_px":null,"phase_x_px":7.0,"phase_y_px":0.0,"render_mode":"conventional"}` hashes to `9c811d1777d4`; with the integer 7 in place of 7.0 it would hash to `bc2ca5dcb62f`, which is why every number is cast first. The resume key of a row is (`reader`, `page_id`, `condition_hash`, `repeat`). The hash never includes a path or a page size, never uses Python's `hash()`, and its key set does not change after the pilot without an amendment.

## results/mtf/ (W3.6, the layer-0 MTF from the weights)

```
results/mtf/
├── manifest.json            written by mrr.manifest.write_manifest; config.readers holds, per reader, the tensor fetched (hf_id, revision, tensor, shard, dtype, shape, bytes, sha256), image_std, filters D and patch_px P; config.nfft (128), n_rings (32), m0_angles_deg ([0, 45])
└── layer0_<reader>.csv      160 rows per reader, written by scripts/04_mtf.py: profile, f_cyc_per_px, value
```

| Column | Type | Unit | Allowed values | Example |
|---|---|---|---|---|
| `profile` | string | none | `ring` (32 rows: the response averaged over direction, notes A8 equation (6), at the ring centres (k + 0.5)/64); `horizontal` (64 rows: the response along the horizontal frequency axis, at k/128); `m0_0deg` and `m0_45deg` (32 rows each: M_0 of build guide 3S.1, equation (3), at the ring centres, for a grating whose wave runs at 0 or 45 degrees to the columns) | `ring` |
| `f_cyc_per_px` | float | cycles per pixel | 0 to 0.5 | `0.0078125` |
| `value` | float | none | `ring` and `horizontal`: the square root of the mean power over the filters divided by its largest value, so the peak is 1; `m0_*`: the root mean square over the filters of the kernels' Fourier magnitude, in normalised-pixel units, not rescaled | `0.198` |

Every row comes from the grey kernels K_d = sum over c of W_{d,c} / sigma_c (Qwen's two frame slices summed first), with sigma_c the processor's `image_std` at the pinned revision (`mrr.mtf.grey_kernels`). The tensors live under `data/external/weights/`, git-ignored; the manifest records their SHA-256. A value is a gain of one linear layer, never a resolution limit (notes A8, "Gain is not loss").

## reports/tables/contract/contract_checks.csv (W4.1, the evidence of R01 to R06 and the Conformance table)

One row per check, written by `scripts/08_contract_checks.py` (`make contract`) beside the folder's `manifest.json`, which is written first so that its `git_dirty` describes the tree the checks ran on. Rules R01 to R06 of `spec/RENDERING_SPEC.md` cite the rows whose `rule` names them; the Conformance table cites the `Conformance` rows. Every `expected` cell is a value the spec states or a bound it implies; every `observed` cell is recomputed, from the renderer where the rule describes the renderer and from the spec's equations where it states arithmetic (the three Conformance facts over all 16,777,216 byte triples). The script exits 1 if any row has `passed` False. A number the spec states may not be changed to match the script without a notebook entry saying why the earlier number was wrong.

| Column | Type | Unit | Allowed values | Example |
|---|---|---|---|---|
| `rule` | string | none | `R01` to `R06`, or `Conformance` | `R05` |
| `check` | string | none | what the row checks, in words | `coverage_linear: byte 128 of (200, 30, 60) on (250, 245, 230)` |
| `expected` | string | none | the spec's value, or a bound such as `< 1e-12` | `[227, 181, 173]` |
| `observed` | string | none | the computed value | `[227, 181, 173]` |
| `passed` | string | none | `True` or `False` | `True` |

## reports/tables/readers/reader_facts.csv (W4.1, the evidence of R07 and R08)

One row per reader of `configs/readers.yaml`, written on the GPU by `scripts/01_check_readers.py --out reports/tables/readers/reader_facts.csv` (`make reader-facts`) once every reader's checks have passed, beside the folder's `manifest.json` (written first). The column list is `FACTS_COLUMNS` in that script; `tests/test_check_readers.py` checks that it and this table agree. The blind-strip columns come from replacing a region of the native page with the bytes of `numpy.random.default_rng(2)` and comparing the encoder's and the connector's outputs with the page's, as `tests/test_reader_facts.py` does.

| Column | Type | Unit | Allowed values | Example |
|---|---|---|---|---|
| `reader` | string | none | `qwen`, `smolvlm`, `smolvlm500m` | `smolvlm` |
| `hf_id` | string | none | the Hub id | `HuggingFaceTB/SmolVLM-Instruct` |
| `revision` | string | none | the pinned 40-character commit | `81cd9a775a4d644f2faf4e7becff4559b46b14c7` |
| `dtype` | string | none | `bfloat16` | `bfloat16` |
| `placement` | string | none | the `hf_device_map` summary the script prints, or `no hf_device_map` | `hf_device_map {'': 0}` |
| `patch_px` | int | px | 14, 16 | `14` |
| `merge` | int | patches per token side | 2, 3, 4 | `3` |
| `token_cell_px` | int | px | 28, 42, 64 | `42` |
| `window_px` | int or empty | px | 112 for `qwen`; empty for the SmolVLM readers, which attend over the whole frame | `` |
| `page_height_px` | int | px | 448, 384, 512 | `384` |
| `page_width_px` | int | px | 448, 384, 512 | `384` |
| `encoder_input` | string | none | the processor output's shape and the patch and token grids, in words | `pixel_values (1, 1, 3, 384, 384): 27 x 27 patches of 14 px, 9 x 9 tokens of 3 x 3 patches` |
| `encoder_input_px` | string | frames x px x px | what reached the encoder; one frame of the page size for a native page | `1x384x384` |
| `vision_tokens` | int | tokens | the image tokens counted in the prompt | `81` |
| `grid_token_count` | int | tokens | the formula's count, `grid.token_count` | `81` |
| `native_frame` | bool | none | True when one frame of exactly the page size reached the encoder and the two counts agree (R07) | `True` |
| `blind_from_px` | int or empty | px | 378 for `smolvlm`; empty for the others | `378` |
| `blind_strip_px` | int | px | 6 for `smolvlm`, 0 for the others | `6` |
| `blind_strip_invisible` | bool or empty | none | True when noise in the columns and in the rows from `blind_from_px` onward leaves the encoder's output identical (R08); empty for a reader without a strip | `True` |
| `control_column_visible` | bool or empty | none | True when noise in the column before the strip changes the output; empty for a reader without a strip | `True` |
| `max_new_tokens` | int | tokens | from `configs/readers.yaml` | `629` |
| `transcription_tokens` | int | tokens | new tokens of the one transcription, the end token included | `412` |
| `hit_limit` | bool | none | the transcription stopped at `max_new_tokens`, not at an end token | `False` |
| `latency_s` | float | s | the first generate call after loading, warm-up included, to 3 decimals | `21.374` |
| `peak_gib` | float | GiB | `torch.cuda.max_memory_allocated`, to 3 decimals | `5.012` |

## Run ID format

`YYYYMMDD-HHMM_<sweep>_<reader>_<short-git-sha>`

- Date and time are UTC.
- `<sweep>` is the sweep config file stem, for example `pilot` or `full`.
- `<short-git-sha>` is the 7-character commit, or `nogit` outside a git repo.
- Generated by `mrr.manifest.new_run_id`.
