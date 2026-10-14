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

## Run ID format

`YYYYMMDD-HHMM_<sweep>_<reader>_<short-git-sha>`

- Date and time are UTC.
- `<sweep>` is the sweep config file stem, for example `pilot` or `full`.
- `<short-git-sha>` is the 7-character commit, or `nogit` outside a git repo.
- Generated by `mrr.manifest.new_run_id`.
