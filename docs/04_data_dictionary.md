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

## results.csv under `results_schema: study2` (Study 2, D27)

A sweep config with `results_schema: study2` (`configs/sweeps/study2.yaml`) writes the 33 columns above and then these five, 38 in all (`STUDY2_SCORE_COLUMNS` and `RESULTS_SCHEMAS` in `src/mrr/manifest.py`). Study 1's sweeps have no `results_schema` key and keep the 33. `load_results` and a resume both refuse a header that is not the sweep's own. All five are computed from the same normalised strings as `hits` to `insertions` (`score._normalised`).

| Column | Type | Unit | Allowed values | Example |
|---|---|---|---|---|
| `hits_ns` | int | characters | 0 or more: jiwer's hits with every space removed from both strings (`score.char_counts_letters_only`), the letters-only CER of PREREGISTRATION_2.md section 4 | `228` |
| `substitutions_ns` | int | characters | 0 or more (jiwer, spaces removed) | `3` |
| `deletions_ns` | int | characters | 0 or more (jiwer, spaces removed) | `1` |
| `insertions_ns` | int | characters | 0 or more (jiwer, spaces removed) | `0` |
| `spaces_inserted` | int | characters | 0 or more: the spaces inside jiwer's insert chunks of the full alignment (`score.spaces_inserted`); a space that replaces a character is a substitution and is not counted. Divided by `reference_chars` it is H3's inserted spaces per reference character | `2` |

A block of a Study 2 sweep may carry `text_types: [random62]` (or `[natural]`): its conditions then read only that type's shared pages. `text_types` is not one of the ten hash keys, so it never changes a `condition_hash`. One of `pages.natural.per_text_type` and `pages.random62.per_text_type` may be 0 for a reader (smolvlm500m reads prose only); both 0 is refused. A sweep's `readers_config` names its reader file: `configs/readers_study2.yaml` for Study 2, `configs/readers.yaml` by default. Its `phase_allowances_px` fixes each reader's phase allowances (Study 2: Study 1's 105 x 26, 40 x 40 and 60 x 60 px, and 105 x 26 for the validation readers), and a block value may be a map by reader (Study 2's mono size and the phase of M7); neither is recorded per row, the manifest's `config.sweep` holds both.

## reports/tables/study2_prerun/ (Study 2, PREREGISTRATION_2.md section 7, before the run)

`font_screen.csv` (`scripts/23_font_pretest.py`, section 7.1, criteria (a) to (d)): one row per candidate monospace face at the size where its advance is the patch of Qwen2.5-VL-3B (P = 14).

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `font` | string | none | the font file's stem |
| `file` | string | none | the font file under `data/external/fonts/` |
| `version` | string | none | the face's version string (name table) |
| `advance_per_em` | float | em | the advance width of `x` at 100 px divided by 100, which every glyph of a monospace face shares |
| `size_px` | float | pixels | the largest size, in steps of 1/64 px down from P / `advance_per_em`, at which the advance is at most P |
| `advance_px` | float | pixels | the advance at `size_px`, at most P |
| `x_rows`, `cap_rows` | int | pixel rows | the bitmap heights of `x` and `H` at `size_px`; criterion (b) wants `x_rows` from 11 to 13 |
| `min_ink` | int | pixel columns | the narrowest bitmap over the 62 symbols; criterion (a) wants at least 7 |
| `max_overhang` | int | pixels | the largest distance any symbol's ink reaches outside its P-pixel cell, either side |
| `not_contained` | string | none | the symbols whose ink leaves the cell, concatenated; empty to pass (a) |
| `o0_diff` | float | share of pixels | of the union of the binarised `O` and `0` (coverage at least 0.5, centred), the share where only one has ink; criterion (c) wants at least 0.15 |
| `zero_marked` | bool | none | whether `0` has ink in the middle 30% of its rows and columns (a dot or a slash); criterion (c) wants it |
| `il1_min_diff` | float | share of pixels | the smallest such share over the pairs `I`/`l`, `I`/`1` and `l`/`1`; criterion (d) wants at least 0.15 |
| `pass_a` to `pass_d`, `passes` | bool | none | criteria (a) to (d) of section 7.1, and all four |

`font_pretest.csv`: one row per face that passed the screen, read by Qwen2.5-VL-3B on ten random-text pages (seed 20261022, paginated to the face with the largest line pitch so that every face reads the same pages) in patch-aware layout at `size_px`. On space-stripped strings: `pages`, `letters` (the reference's letters, H + S + D), `o0_confusions` and `il1_confusions` (substitutions between `O` and `0`, and among `I`, `l` and `1`), `letter_errors` (S + D + I), `letters_only_cer` (their ratio, pooled over the pages) and `page_hashes` (the first ten hexadecimal characters of each reference's SHA-1, space-separated). `pretest/<font>/NN.txt` holds the transcriptions; `screen_eight/` the screen of all eight candidates.

`feasibility.csv` (`scripts/24_feasibility.py`, section 7.2): one row per validation reader and page (`random`, `prose`).

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `reader`, `hf_id`, `revision`, `quantization` | string | none | the block of `configs/readers_study2.yaml` (`quantization` is empty for bf16) |
| `parameter_devices` | string | none | the device types the parameters sit on, space-separated; `cuda` alone to pass |
| `linear4bit_layers` | int | layers | 4-bit linear layers in the whole model |
| `vision_linear4bit_layers` | int | layers | 4-bit linear layers whose module name contains `visual`, `vision_tower` or `vision_model` (the vision tower; Qwen's merger sits under `visual`); 0 to pass |
| `blank_tokens`, `grid_token_count` | int | tokens | the processor's image tokens for a blank 448 x 448 page, and `grid.token_count`; equal to pass |
| `page`, `chars` | string, int | none, characters | which M0 page (Fira Mono at 23.333 px) and its reference length |
| `max_new_tokens`, `new_tokens`, `hit_limit` | int, int, bool | tokens | the limit (ceil(1.5 x the reference's tokens)), the tokens written, and whether the limit stopped the output |
| `latency_s` | float | seconds | the transcription's wall time |
| `cer` | float | ratio | the page's CER |
| `peak_gib` | float | GiB | `torch.cuda.max_memory_allocated` after the reader's pages |
| `passes` | bool | none | every check above, and a transcription that is not empty |

`feasibility_run1_tower_in_4bit/` keeps the first run, which failed with the vision tower in 4-bit; `feasibility/manifest.json` and `stdout.txt` belong to the passing run.

## reports/tables/study2/ (Study 2, PREREGISTRATION_2.md section 9)

Written by `scripts/22_analyze_study2.py` from the complete rows of the run folders of `configs/sweeps/study2.yaml` (section 8), with `manifest.json` (seed 20261021, 2,000 page resamples, the bounds 0.005 and 0.01, the readers analysed and planned, the pages excluded). Every table carries `status` and `family`. The condition codes are those of section 3: `S0`, `S5`, `S10` (block S), `Q0` (block Q), `M0`, `M3` and `M7`, or `M8` at P = 16 (block M), and block D by its size (`S28`). The groups are `conventional` (S), `slot` (Q0), `contained` (M0 and M3), `cut` (M at P / 2) and `density` (block D); a page's value for a group pools the counts of the group's rows on that page (D13). `role` is `validation` for `qwen7b` and `internvl3_2b` and `development` for Study 1's readers. `checks.csv` has the columns of Study 1's (above).

`paired.csv`: one row per family, reader, text type and contrast that the reader has rows for (smolvlm500m reads no random page).

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `family` | string | none | `gap` (H3), `containment` (H4), `advantage` (H5), `prose` (H6), `secondary`, or `wer` (exploratory) |
| `hypothesis` | string | none | H3 to H6; empty for secondary and WER rows. H7 is read from the `validation` rows of `gap` and `containment` |
| `contrast` | string | none | `<treatment>_vs_<baseline>`; the treatment is `contained` except for `M3_vs_M0` |
| `baseline`, `treatment` | string | none | a group or a condition code |
| `measure` | string | none | `cer` (spaces counted), `letters` (letters-only CER), `spaces` (inserted spaces per reference character) or `wer` (the mean row WER) |
| `n` | int | pages | pages with a value on both sides |
| `n_zero` | int | pages | pages whose difference is zero, dropped by the test |
| `statistic`, `p`, `method` | float, float, string | none | `stats.paired_wilcoxon` (two-sided); `method` is `undefined: ...` when no page holds both sides or every difference is zero (SciPy would give p = 1 up to 13 pages and NaN above 50), and `p` is then empty |
| `p_holm` | float | none | Holm across the readers of the family, for `gap` (Q0 against contained only), `containment`, `advantage` and `prose`; an undefined test counts as p = 1 in its family and keeps an empty `p_holm`; empty for rows outside a family |
| `median_diff` | float | the measure | the median over pages of treatment minus baseline; negative means the treatment has the lower error |
| `ci_low`, `ci_high` | float | the measure | the 95% page-bootstrap interval of `median_diff` |
| `sd_diff`, `mde` | float | the measure | the standard deviation of the differences, and the smallest detectable median difference (`stats.paired_mde`, 80% power) |
| `predicted_sign` | int | none | -1 where section 1 predicts the treatment lower (H3's test, H4, H5); empty otherwise |
| `significant` | bool | none | `p_holm` below 0.05; empty without `p_holm` |
| `supported` | bool | none | significant and `median_diff` of the predicted sign; empty without a predicted sign |
| `bound`, `passes` | float, bool | the measure | the non-inferiority bound (0.005 for H3's parity with conventional, 0.01 for H6) and whether `ci_high` is below it |

`letters.csv`: one row per reader, text type and condition: `group`, `block`, `phase_x_px`, `font_file`, `font_size_px`, `render_mode`, `n_pages`, `n_rows`; `cer` (pooled, spaces counted) with `cer_low` and `cer_high`, and `letters_cer` (pooled letters-only CER) with `letters_low` and `letters_high`, the 95% page-bootstrap intervals (one draw of pages serves every condition of a reader and text type, as `stats.pooled_cer_by_level`); `wer_mean`, the mean row WER.

`spaces.csv`: one row per reader, text type and condition: `group`, `n_pages`, `n_rows`, `spaces_inserted` (the sum of the rows' `spaces_inserted`), `reference_chars` (the sum of H + S + D), `spaces_per_char` (their ratio) with `low` and `high` as above.

`confusions.csv`: per reader, text type and condition (with `group`, `font_file`, `render_mode`, `letters`, the reference letters over its rows, and `n_rows`), one row per `kind`: `O/0`, `I/l/1` and `case_pair` (`count` = the substitutions between O and 0, between two of I, l and 1, and between the two cases of one letter), then `substitution` with `ref_char` and `hyp_char`, and `deletion` with `ref_char`. The counts come from jiwer's character alignment of the normalised strings with every space removed (`study2.letter_confusions`), so a misread space is never a letter confusion and a swap of two neighbouring letters aligns as a deletion and an insertion, not as two substitutions.

`capacity.csv`: per reader, text type and shared layout (`layout`, `render_mode`, `font_file`, `font_size_px`, `line_pitch_px`): the run's own planned pages of that type joined in plan order and paginated under that layout alone, at phase 0 with the run's allowances, for up to 20 pages, a last page the text ran out on dropped (`n_pages`). `chars_per_page`, `glyphs_per_page` (placed characters, spaces not) and `lines_per_page` are means over those pages; `glyphs_per_line` is glyphs over lines; `vision_tokens` is `grid.token_count` for the page; `chars_per_token` is `chars_per_page` over it; `shared_chars_per_page` is the mean length of the shared pages the run read.

Figures (`scripts/30_make_figures.py --config configs/sweeps/study2.yaml`, the same complete rows, seed 20261021): `reports/figures/fig3_conditions_study2.pdf`, the pooled CER of every condition with its page-bootstrap interval, one panel per text type and reader, coloured by group; `reports/figures/fig4_confusions_study2.pdf`, the O/0 and I/l/1 confusions per 1,000 reference letters (`study2.pair_rates`), one bar per reader and group, labelled with the count.

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

## reports/tables/exploratory/levels.csv (W4.1, exploratory, the per-level numbers behind R09 to R11)

One row per reader, text type, block and level of a sweep, written by `scripts/21_levels.py` (`make levels`) from the frozen run folders (`results/final/runs`) beside the folder's `manifest.json` (written first). The rows are those `scripts/20_analyze.py` analyses. Every count is pooled over the level's rows (D13), with N = hits + substitutions + deletions; each row's counts are recomputed from its reference and transcription first, and a difference stops the script. Every row is `exploratory`: the table describes the frozen rows and tests nothing. The column list is `COLUMNS` in that script; `tests/test_levels.py` checks that it and this table agree.

| Column | Type | Unit | Allowed values | Example |
|---|---|---|---|---|
| `reader` | string | none | `qwen`, `smolvlm`, `smolvlm500m` | `qwen` |
| `text_type` | string | none | `natural`, `random62` | `random62` |
| `block` | string | none | `V`, `H`, `C`, `S`, `A`, `Z` | `H` |
| `render_mode` | string | none | `conventional`, `patch_aware` | `patch_aware` |
| `line_pitch_mode` | string | none | `aligned`, `unaligned` | `unaligned` |
| `aa_mode` | string | none | `binary`, `coverage_srgb`, `coverage_linear` | `coverage_srgb` |
| `font_size_px` | float | pixels | the sizes of the sweep | `24.0` |
| `phase_x_px` | float | pixels | the level's horizontal phase | `0.0` |
| `phase_y_px` | float | pixels | the level's vertical phase | `0.0` |
| `rows` | int | rows | 1 or more | `47` |
| `pages` | int | pages | 1 or more | `47` |
| `ref_chars_per_row` | float | characters | N divided by `rows`: how much text a page of the level holds | `228.4` |
| `cer` | float | ratio | (S + D + I) / N, 0 or more | `0.194` |
| `substitution_rate` | float | ratio | S / N | `0.013` |
| `deletion_rate` | float | ratio | D / N | `0.001` |
| `insertion_rate` | float | ratio | I / N; the three rates add up to `cer` | `0.181` |
| `added_spaces_per_char` | float | spaces per reference character | (spaces in the normalised transcriptions − spaces in the references) / N; negative when more spaces were dropped than added | `0.180` |
| `spaces_inserted_in_words_per_char` | float | spaces per reference character | the spaces jiwer's alignment inserts between two characters of one word of the reference (a word is a run between spaces), divided by N | `0.178` |
| `inserted_beside_wide_share` | float | ratio | the share of those spaces next to a wide character, one whose linear advance at the level's font size exceeds the reader's patch (it takes two slots in the patch-aware layout); empty when no space was inserted | `0.94` |
| `gaps_beside_wide_share` | float | ratio | the share of the gaps between two characters of one word that are next to a wide character: the share the inserted spaces would have if they fell anywhere in a word | `0.36` |
| `cer_without_spaces` | float | ratio | the pooled CER of the same strings with every space removed | `0.017` |
| `hit_limit_share` | float | ratio | the share of rows whose output stopped at `max_new_tokens` | `0.000` |
| `status` | string | none | `exploratory` | `exploratory` |
| `family` | string | none | `description` | `description` |

## h2/judge/judge_pages.csv (W4.4, the 186 pages every H2 attempt is judged on)

One row per judge page, written by `python h2/judge/judge.py build --out data/interim/h2/judge_v01` with the pages themselves, which git ignores: `inputs/<page_id>.npz` (the six glyph arrays of the kernel interface under the names `bitmaps`, `offsets`, `heights`, `widths`, `rows`, `cols`), `inputs/<page_id>.json` (`height`, `width`, `text_rgb`, `background_rgb`, `aa_mode`), `reference/<page_id>.npy` (the reference renderer's page, uint8 H x W x 3) and `build.json` (where the render configs came from). The pages are defined by `h2/judge/judge.yaml`; the hashes fix them before attempt 1, and `judge.py run` refuses to start when any page's files hash differently. Page ids: `n_<reader>_<layout>_<aa_mode>_ph<k>_<text_type>` (k from 0 to 3 in the order of `phases_px`), `c_<reader>_<aa_mode>_<colour_case>`, `e_<edge_case>_<aa_mode>`.

| Column | Type | Unit | Allowed values | Example |
|---|---|---|---|---|
| `page_id` | string | none | as above | `n_qwen_patch_aware_coverage_linear_ph2_random` |
| `reader` | string | none | `qwen`, `smolvlm`, `smolvlm500m`; empty for an edge page | `qwen` |
| `layout` | string | none | `conventional`, `patch_aware`; empty for an edge page | `patch_aware` |
| `aa_mode` | string | none | `binary`, `coverage_srgb`, `coverage_linear` | `coverage_linear` |
| `text_type` | string | none | `natural`, `random`; empty for an edge page | `random` |
| `colour_case` | string | none | `wob`, `red`; empty for black on white | `red` |
| `edge_case` | string | none | `empty`, `cut_tl`, `cut_br`, `outside`, `overlap`, `oblong`, `zero`, `tiny`; empty for a laid-out page | `` |
| `phase_x_px`, `phase_y_px` | float | px | 0, 0.5 and 0.25, 13.75 and 7.5, or the reader's largest allowance (111, 41, 63); 0 for an edge page | `13.75` |
| `height`, `width` | int | px | the reader's native page, or the edge case's page | `448` |
| `n_glyphs` | int | glyphs | G, the length of `offsets`; spaces are not placed on a laid-out page, but count on `e_zero` | `225` |
| `inputs_sha256` | string | none | SHA-256 of the `.npz` bytes followed by the `.json` bytes | 64 hex digits |
| `reference_sha256` | string | none | SHA-256 of the `.npy` bytes | 64 hex digits |

## h2/judge/reviews.csv (W4.4 and P4.5, the reviews before the judge runs, D23)

Written by hand, before `judge.py run`, three rows per attempt: `A` (Claude Opus 5.5), `B` (a fresh Claude Sonnet 5.5 agent, blind to A) and `owner` (the final go). `judge.py run` refuses an attempt without all three rows, and records every page as `not run: safety` when any row says `safe_to_run` no; `summary` writes `pixel_work_in_triton` yes only when all three rows say yes.

| Column | Type | Unit | Allowed values | Example |
|---|---|---|---|---|
| `attempt` | int | none | 1, 2, 3 | `1` |
| `reviewer` | string | none | `A`, `B`, `owner` | `B` |
| `safe_to_run` | string | none | `yes`, `no` | `yes` |
| `pixel_work_in_triton` | string | none | `yes`, `no` | `yes` |
| `note` | string | none | free text | `flags explained: none` |
| `date` | string | none | ISO date | `2026-10-23` |

## reports/tables/h2/judge_v01_<n>.csv and summary.csv (W4.4, the judge's tables)

`judge_v01_<n>.csv`, one per attempt, written by `judge.py run --kernel h2/kernels/kernel_v01_<n>.py --attempt <n>`: the columns of `judge_pages.csv` from `page_id` to `n_glyphs`, then the six below, one row per page. `summary.csv`, written by `judge.py summary` after the folder's `manifest.json`, has one row per judged attempt, with the columns of the second table.

| Column | Type | Unit | Allowed values | Example |
|---|---|---|---|---|
| `max_abs_error` | int or empty | levels | 0 to 255, equation (3) of the spec; empty when the page failed before a comparison | `1` |
| `frac_pixels_differing` | float or empty | none | 0 to 1, equation (4) | `6.8e-05` |
| `passed` | bool | none | True when both are within the tolerance of the page's `aa_mode` | `True` |
| `kernel_ms` | float or empty | ms | `triton.testing.do_bench` after the first call; empty when timing failed, which never fails a page | `0.412` |
| `repeat_identical` | bool or empty | none | a later call gave the same bytes as the judged first call | `True` |
| `error` | string | none | empty on a page that ran and compared; otherwise the last line of the error, `exit code <c>: ...`, `result file unreadable`, `output on <device>, not on the GPU`, or `not run: <reason>` (safety, syntax error, no code block, no top-level render, `<k> earlier pages hung`) | `not run: safety` |

| Column (summary.csv) | Type | Unit | Allowed values | Example |
|---|---|---|---|---|
| `attempt` | int | none | 1, 2, 3 | `2` |
| `kernel_file` | string | none | the kernel's path from the repository root | `h2/kernels/kernel_v01_2.py` |
| `kernel_sha256` | string | none | SHA-256 of the kernel's bytes; empty when the answer had no code block | 64 hex digits |
| `static_flags` | string | none | `static_check`'s flags joined by `; `, empty when none | `line 3: import os` |
| `ran` | string | none | `yes` when any page ran, `no` when every page is `not run: ...` | `yes` |
| `pixel_work_in_triton` | string | none | `yes` only when all three review rows say yes | `yes` |
| `pages_passed` | int | pages | 0 to 186 | `186` |
| `passed_all` | string | none | `yes` when every page of `judge_pages.csv` passed | `yes` |
| `worst_max_abs_error` | int or empty | levels | the largest `max_abs_error` of the attempt | `1` |
| `median_kernel_ms` | float or empty | ms | the median `kernel_ms` of the attempt | `0.397` |

## Run ID format

`YYYYMMDD-HHMM_<sweep>_<reader>_<short-git-sha>`

- Date and time are UTC.
- `<sweep>` is the sweep config file stem, for example `pilot` or `full`.
- `<short-git-sha>` is the 7-character commit, or `nogit` outside a git repo.
- Generated by `mrr.manifest.new_run_id`.
