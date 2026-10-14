<!--
PURPOSE: Preregistration of the pilot and full experiment, frozen before data collection.
WHAT GOES HERE: Final hypotheses, readers, variables, sample size, tests, exclusion rules, failure criteria.
FILLED IN AT STEP: W2.1 (then tagged prereg-v1). Amendments only after that.
DO NOT: Edit anything above the Amendments section after the prereg-v1 tag. Do not add results.
-->

# Preregistration

> **WARNING:** Complete and tag `prereg-v1` BEFORE running the pilot. Changes after the tag go in a dated Amendments section at the bottom, never by editing above.

Drafted on 14 October 2026 by Claude (D22) from build guide section 2.11 and decisions D7 to D25 in `docs/05_decisions.md`; approved by the owner, whose approval stands for both authors (D22), on the date in section 11. The three values that the font-size ladder (W1.10) supplies are marked `TODO[W1.10]` and are filled in the commit that is tagged. Where this document and `docs/01_research_plan.md` differ, this document rules once tagged.

## 1. Hypotheses (final wording)

- **H1.** Restated under D10 as a claim about coherently placed text, in three clauses and a description. `docs/01_research_plan.md:22` holds the original wording.
  - **(a) Periodicity.** On layouts whose glyphs share a phase relative to the encoder's grid, a reader's character error rate (CER) varies periodically with that phase. Two layouts have this property and are tested: block V, conventional text whose line pitch equals the reader's token cell, swept vertically over one token cell (φy); and block H, patch-aware text with one glyph per patch-wide slot, swept horizontally over one patch (φx). Tested by the period groups of the harmonic model (section 6).
  - **(b) Period.** The significant groups identify the period: the patch (14, 14 and 16 px for Qwen2.5-VL-3B-Instruct, SmolVLM-2.2B and SmolVLM-500M) or the token cell (28, 42 and 64 px). This clause is read from the groups of clause (a); it is not a separate test. Prediction from paper Figure 0 (W1.9, `results/equivariance/`): at layer 0 every reader is exactly equivariant for shifts that are whole multiples of its patch and nowhere else, so the patch group is predicted in block H for all three readers; block V folds the patch and the cell into one range, and its cell group is predicted through the spatial merge (2 × 2, 3 × 3 and 4 × 4 patches per token).
  - **(c) Patch-aware rendering.** At the aligned phase, block H at φx = 0, patch-aware rendering has a lower CER than conventional rendering averaged over the phases of block C, on the same pages, read by the same reader, at the same vision-token count (256, 81 and 64 tokens per page under D8). Tested by the paired test of section 6.
  - **Description, not a test.** How the difference in (c) changes with font size, from block Z of the full run (section 3); the pilot has no block Z. `docs/01_research_plan.md:22` predicts the effect is largest at small sizes.
  - **Prediction for the control block C.** Block C is conventional text with unaligned line pitch, swept horizontally over the reader's sweep period (112, 42 and 64 px, D9). Its curve is predicted flat: a block shift moves every glyph by the same amount, but conventional glyphs already sit at every offset against the grid, so the page has no phase. Any amplitude at harmonic k of period T is predicted to scale with the page's coherence R_k(T) = |mean over glyphs j of exp(2πi k c_j / T)|, c_j being the ink centre of glyph j, which the runner records per page and layout in `coherence.csv`. Measured on 2,500 glyphs of Persuasion at 16 px (W1.10): conventional R₁(14) = 0.022, at the 1/√N floor of 0.020; patch-aware R₁(14) = 0.986 and R₂(14) = 0.947. A significant group in C with none in V or H would contradict this account (section 10).
- **H2.** As `docs/01_research_plan.md:23` states it: a language model (GLM-5.3) given only the written specification can write a Triton kernel whose output matches the reference renderer. Fixed here:
  - **Subject and settings.** `glm-5.3` through the Z.ai API, never GLM-5.3-Flash. One user message holding the frozen prompt `h2/prompts/prompt_v01.md`, which wraps `spec/RENDERING_SPEC.md` at tag `spec-v1` whole and nothing else: no system message, no examples, no code, no tests, no hints. Thinking enabled with `reasoning_effort` `max`; `temperature` 1.0; `max_tokens` 131072, or 128000 if the API refuses the larger value. The base URL and whether the reply streams are settled by the dummy calls and recorded as an amendment below before attempt 1. The specification is written by Claude (D22), so H2 tests GLM-5.3 implementing a Claude-written specification, and the paper discloses this.
  - **Attempts.** Three. An attempt is a reply with HTTP status 200; a transport failure is repeated and not counted. No second turn, no feedback, no second prompt version. What is judged is the first Python code block of the reply, verbatim; a reply cut off by `max_tokens` still counts as an attempt. `h2/README.md` records every attempt.
  - **Pages.** 186 pages, fixed before attempt 1 by `h2/judge/judge.yaml` and the hashes in `h2/judge/judge_pages.csv`: 144 normal pages (3 readers × 2 layouts × 3 anti-aliasing modes × 4 phases × 2 text types, black on white), 18 colour pages (3 readers × 3 modes × 2 colour pairs) and 24 edge cases (8 cases × 3 modes), as build guide section 4.4 lists them.
  - **Match criterion per page.** `h2/judge/compare.py` compares the kernel's page K with the reference renderer's page R, both H × W × 3 bytes: `max_abs_error`, the largest |K − R| over pixels and channels, and `frac_pixels_differing`, the share of pixels where any channel differs. A page passes when both are within the tolerance of its anti-aliasing mode: `binary` 0 and 0; `coverage_srgb` 0 and 0; `coverage_linear` 1 level and 0.001. Each page runs in its own process with a 120 s limit, and the attempt stops after three pages hang. A reply with no code block, a syntax error or no top-level `render` fails every page.
  - **`pixel_work_in_triton`.** Judged before the judge runs: Claude Opus 5.5 and a fresh Claude Sonnet 5.5 agent, each blind to the other's verdict, each with the flags of `static_check`; "no" if they disagree; the owner gives the final go (D23). A flag nobody can explain means the attempt is recorded as "not run: safety".
  - **Criterion.** H2 is supported if at least one of the three attempts passes all 186 pages and `pixel_work_in_triton` is yes. The paper reports k of 3 in every case.
  - **Condition.** D16 as D25 moved it: H2 runs only if `data-freeze-v1` is tagged by Thursday 22 October 2026. Otherwise the specification is still written and tagged `spec-v1`, and H2 is reported as not run.

## 2. Readers (exact model IDs and revision hashes)

From `configs/readers.yaml` (W1.6), read at these revisions with transformers 5.19.0 and torch 2.14.1+cu130 (`requirements.lock.txt`):

| Name | Hub id | Revision | Patch | Merge | Token cell | Native page | Vision tokens per page |
|---|---|---|---|---|---|---|---|
| `qwen` | `Qwen/Qwen2.5-VL-3B-Instruct` | `66285546d2b821cf421d4f5eb2576359d3770cd3` | 14 px | 2 × 2 | 28 px | 448 × 448 | 256 |
| `smolvlm` | `HuggingFaceTB/SmolVLM-Instruct` | `81cd9a775a4d644f2faf4e7becff4559b46b14c7` | 14 px | 3 × 3 | 42 px | 384 × 384 | 81 |
| `smolvlm500m` | `HuggingFaceTB/SmolVLM-500M-Instruct` | `a7da5b986cb59b408707209984f360a5f4ad7e47` | 16 px | 4 × 4 | 64 px | 512 × 512 | 64 |

- **Grids (D7, D9).** Qwen attends inside 112 px windows in 28 of its 32 vision blocks, so its sweep period is 112 px. SmolVLM-2.2B's 27 × 27 patches cover 378 of its 384 px, so columns and rows 378 to 383 never reach its encoder (W1.8) and the layout keeps text out of them. The sweep periods of the two SmolVLM readers are their token cells, 42 and 64 px.
- **No resize (D8).** Each page is rendered at the reader's native size and enters the encoder byte for byte (`tests/test_no_resize.py`, W1.8). On the first page of every run the runner checks that the processor's vision-token count equals the table above and stops otherwise.
- **dtype and placement.** bfloat16, every parameter on the GPU (`device_map: cuda`, the first rung of D18; measured peak 7.07 GiB allocated for Qwen on the 8 GiB RTX 5060 Laptop GPU, W1.7). Every run's `manifest.json` records the realised placement. Both SmolVLM readers get the position-id fix `pre_hook` (D15).
- **Generation.** Greedy: `do_sample=False`, `num_beams=1`, `repetition_penalty=1.0`, and `max_new_tokens` set per page by the runner to ⌈1.5 × the reference text's length in the reader's own tokens⌉ (build guide 1B.9). A transcription that stops at the limit is kept and flagged `hit_limit` (section 4). The limits in `configs/readers.yaml` (591, 629 and 1,157) are the longest 16 px references of W1.7 and serve `scripts/01_check_readers.py` only.
- **Prompt, verbatim**, as the text part of one user message that also holds the image, through the processor's chat template with the generation prompt added: `Transcribe all text in the image exactly. Output only the text.`

## 3. Independent variables and their levels

Matching `configs/sweeps/pilot.yaml` (W2.2); the full run (`configs/sweeps/full.yaml`, W3.2) repeats the same blocks on new pages and adds block Z.

- **Text type** (two levels, every block): `natural`, prose from the three books of section 5; `random62`, characters independent and uniform over the 62 alphanumerics `0-9A-Za-z` in five-character words (`text_source.random_text`), which the language decoder cannot repair (build guide 1A.8).
- **Font.** Source Sans 3 Regular (`data/external/fonts/source-sans-3/SourceSans3-Regular.ttf`): unhinted FreeType 2.13.2 outlines with linear advances; no kerning, ligatures, shaping, justification or hyphenation (D11). One typeface in every decoded block is a stated limitation; Source Serif 4 appears only in the equivariance and coherence measurements, which need no decoding.
- **Font size** (one per reader, every block except Z), chosen by the ladder (W1.10, section 5): `qwen` TODO[W1.10]: px; `smolvlm` TODO[W1.10]: px; `smolvlm500m` TODO[W1.10]: px.
- **Page geometry (D8).** Native page sizes as in section 2; margin one patch (14, 14 and 16 px); black text (0, 0, 0) on white (255, 255, 255). Shared pages are paginated so that the text fits under every shared layout with a phase allowance equal to the largest phase of the sweep, 105 × 26 px for Qwen, 40 × 40 for SmolVLM-2.2B and 60 × 60 for SmolVLM-500M, so that every phase of every block keeps all ink on the page and out of the blind strip: text blocks of 315 × 394, 310 × 310 and 420 × 420 px.
- **Line pitch** (two levels): `unaligned`, ⌊1.25 × size + 0.5⌋ px; `aligned`, the reader's token cell (28, 42 and 64 px), so that every line sits at the same vertical phase.
- **Render mode** (two levels): `conventional`, each glyph advances by its linear advance a; `patch_aware`, each glyph occupies a' = P × ⌈a / P⌉ pixels, P the patch, centred in its slots at a whole-pixel offset, and a space takes exactly one slot (`src/mrr/render.py`, D10). The density cost (glyphs per line) is reported with R_k.
- **Anti-aliasing** (three levels). `coverage_srgb`, the baseline (D11): text and background bytes blended by the glyph coverage c in the stored sRGB encoding, o = c × T + (1 − c) × B. `coverage_linear`: the same blend in linear light, encoded back to sRGB. `binary`: coverage thresholded at c ≥ 0.5. All three round once with ⌊255 × clip(o, 0, 1) + 0.5⌋.
- **Phase**, the swept factor, in pixels relative to the page's top-left corner and so to the grid, in steps of 1/64 px applied to the outline (D11). Each block sweeps one axis over the range in which its layout's glyphs share a phase, at a step σ that divides every tested period (the aliasing rule of build guide 2.3); the other axis is 0.

| Block | Layout | Axis and range | Qwen | SmolVLM-2.2B | SmolVLM-500M | Role |
|---|---|---|---|---|---|---|
| V | `conventional`, `aligned` pitch | φy over [0, cell) | σ 2, 14 levels | σ 2, 21 levels | σ 4, 16 levels | H1 (a), primary |
| H | `patch_aware`, `unaligned` pitch | φx over [0, P) | σ 1, 14 levels | σ 1, 14 levels | σ 1, 16 levels | H1 (a) and (c), primary |
| C | `conventional`, `unaligned` pitch | φx over [0, sweep period) | σ 7, 16 levels | σ 2, 21 levels | σ 4, 16 levels | Control, predicted flat |
| S | `patch_aware`, `unaligned` pitch | φx ∈ {0, 0.25, 0.5, 0.75} | 4 | 4 | 4 | Rasteriser sub-pixel phase, secondary |
| A | `conventional`, `unaligned` pitch, φ = 0 | `aa_mode` ∈ {`binary`, `coverage_linear`} | 2 | 2 | 2 | Anti-aliasing, secondary; the `coverage_srgb` baseline is C at φ = 0 |
| | **Conditions per page per text type** | | **50** | **62** | **54** | |

- (H, φ = 0) and (S, φ = 0) render identical pixels under different condition hashes; the runner compares their transcriptions on every page in `determinism.csv`.
- **Block Z** (full run only; the ladder reads the same design as a measurement, section 5): font size ∈ {10, 12, 14, 16, 20, 24} px × render mode ∈ {`conventional`, `patch_aware`}, `unaligned` pitch, φ = (0, 0), `coverage_srgb`, each condition on its own pages, page i of every Z condition starting at the same word.
- **Repeats.** One row per page per condition (`repeats: 1`).

## 4. Dependent variables

- **Normalisation (D13).** Reference and transcription both pass through `text_source.normalise_text` before scoring: Unicode NFC; curly quotes become straight ones and the em dash a hyphen; whitespace is collapsed to single spaces. No case folding and no punctuation removal.
- **Primary: CER.** Per row, `score.char_counts` stores the four character counts of jiwer 4.0.0's `process_characters`: hits H, substitutions S, deletions D and insertions I. The CER of any set of rows is pooled, (S + D + I) / (H + S + D) with the counts summed over the rows, never the mean of row CERs (`score.pooled_cer`). The response of the harmonic model is the row's own CER, one row per page and level; the paired tests use page CERs as section 9 defines them.
- **Secondary: WER.** `score.wer`, jiwer's `process_words` on the same normalised strings split on whitespace, stored per row. Word counts are not stored, so WER is summarised as the mean of row WERs over a cell and is descriptive; no test uses it.
- **Reported, not tested.** `output_tokens`; the count and share of rows with `hit_limit` true (the output stopped at `max_new_tokens`, not at an end token); `latency_s`; the `determinism.csv` mismatches; each page's coherence R_k per layout and period (`coherence.csv`); the characters and symbols per vision token of every page.

## 5. Page corpus and sample size, with justification

- **Books** (`data/raw/texts/README.md`, W1.4): Project Gutenberg EPUBs pinned by SHA-256: pg105 Persuasion (Austen), pg84 Frankenstein (Shelley) and pg1661 The Adventures of Sherlock Holmes (Doyle). The Gutenberg header and footer are stripped in code, the front matter is skipped at the second `CHAPTER I.`, the second `Letter 1` and the first `I. A SCANDAL IN BOHEMIA`, and the text is normalised (section 4). Later chapter headings stay in the text, as proposed for Gate 1.
- **Random text.** `random_text(200000, seed)`: 200,000 characters drawn with NumPy's default generator (NumPy 2.4.6) from the seed, in five-character words. Seeds: ladder 20261012, pilot 20261013, full run 20261016. No two sweeps share a seed.
- **Pages.** A page is the longest run of whole words, from where the previous page of the same source stopped, that fits the text block under every shared layout (section 3). Natural pages take the books in rotation (pg105, pg84, pg1661, …) from their skip points; the pilot and the full run interleave natural and random pages. Each page is read under every condition of every shared block by every reader of its sweep, so each page is its own control. Page texts are fixed by `references/<page_id>.txt` and `start_word` in each run folder.
- **No reuse.** The full run's natural pages start in each book after the last word the ladder or the pilot used (`pages.natural.start_after`), and its random pages come from a new seed. The ladder's and the pilot's rows are never pooled with the full run's.
- **The ladder, a design measurement.** `configs/sweeps/ladder.yaml`: 3 pages per text type, block Z's 12 conditions, 72 rows per reader. For each reader and size the random62 counts are pooled over both render modes; the size whose pooled CER is closest to the target 0.15, fixed before the run, is chosen, the larger size on a tie. The target leaves room for the curve to move both ways; near 0 the curve would be flat at the floor and near 1 at the ceiling. The result is recorded in `LAB_NOTEBOOK.md` and section 3.
- **Pilot.** 10 natural and 10 random pages per reader; 50, 62 and 54 conditions per page; 1,000, 1,240 and 1,080 rows. Its purposes are G2 (section 7) and the residual standard deviation σ for the page-count rule. With ten pages the primary F-test detects an RMS amplitude of about 0.02 at σ = 0.05 (Qwen's block H at α/3, 80% power), so the pilot can miss a real effect of the preregistered target size.
- **Full run: the page-count rule.** For each reader, σ is the residual standard deviation of the saturated harmonic model (section 9) on the pilot's random62 rows, taken separately for block V and block H. For each group g of the reader's primary family (section 6), with q columns and n levels,

  N_g = max(N_min, ⌈λ_q σ² / (n s_target²)⌉),

  where λ_q is the noncentrality for 80% power at α = 0.05/3 (Holm's smallest threshold in a family of three) from `stats.power_noncentrality(q, 1000, 0.05/3)`: 14.12 for q = 3, 17.40 for q = 6, 18.28 for q = 7, 21.94 for q = 12, 22.58 for q = 13, 23.19 for q = 14 and 23.77 for q = 15 (build guide 2.11, recomputed with the real degrees of freedom at analysis); N_min = 10 pages per text type; s_target = 0.01, an RMS amplitude of one CER percentage point, the smallest effect worth reporting. The reader's N is the largest N_g (`stats.pages_needed`). Worked example: Qwen's block H (n = 14, q = 13) with σ = 0.05 gives N = 41; s_target = 0.005 would give 162. N is set from σ only, never from the pilot's amplitudes.
- **Budget and cap (D20, one laptop, readers in series).** Pilot: 1,000 t_Q + 1,240 t₂ + 1,080 t₅ must fit the 57,600 s from Thursday 15 October 18:00 to Friday 16 10:00, t being each reader's median row latency at its chosen size in the ladder (`LAB_NOTEBOOK.md`, W1.10). If it does not fit, both text types are cut to the largest equal page count that fits, before the tag. Full run: the window is Saturday 17 October 12:00 to Thursday 22 06:00 (114 h, D25), of which 60% is planned, about 68 h, for the three readers in series plus block Z with N_Z = 5 pages per text type per Z condition, using the mean row latency of the pilot and of the ladder. If N does not fit, it is capped at the largest N that does, the smallest detectable amplitude s = σ √(λ_q / (N n)) is written into D5 and the paper, and any further cut follows the order of D21 (stretch analyses, H2, SmolVLM-500M, the anti-aliasing factor), each as a recorded decision at G2.
- **Clause (c).** The smallest detectable median paired difference at the chosen N is reported from `stats.paired_mde(σ_d, N)`, with σ_d the standard deviation of the pilot's paired differences: (z₀.₉₇₅ + z₀.₈) σ_d / √(3N/π).

## 6. Primary test and alpha

- **α = 0.05** for every family. F-tests are one-sided by construction. The paired Wilcoxon tests are two-sided; clause (c) counts as supported only when its Holm-adjusted p is below α and the median paired difference is negative (patch-aware lower).
- **The model (D14).** For one reader, one block and one text type, with p the page, j the level, φ_j the phase, L the block's range and G every identifiable bin of the block (1 ≤ k ≤ ⌊n/2⌋; the sine of the Nyquist bin k = n/2 is zero at every level and is dropped):

  CER_{p,j} = a_p + Σ_{k∈G} [b_k cos(2πkφ_j/L) + c_k sin(2πkφ_j/L)] + ε_{p,j}.

  With every identifiable bin included this equals treating phase as a factor with page fixed effects. Bins are grouped by period: bin k belongs to the shortest tested period T (the reader's patch, cell and window that divide L) for which k is a multiple of L/T (`stats.bin_groups`). Each group is tested with the F-test of the full model against the model without the group's q columns,

  F = [(RSS₀ − RSS₁) / q] / [RSS₁ / (n_rows − p₁)],

  with page dummies in both models (`stats.group_f_test`). The groups:

| Reader and block | L, σ, levels | Groups: bins (q columns) |
|---|---|---|
| Qwen V | 28, 2, 14 | 28 px: 1, 3, 5, 7 (q 7; bin 7 cosine only); 14 px: 2, 4, 6 (q 6) |
| Qwen H | 14, 1, 14 | 14 px: 1 to 7 (q 13) |
| Qwen C | 112, 7, 16 | 112 px: 1, 2, 3, 5, 6, 7 (q 12); 28 px: 4 (q 2); 14 px: 8 (q 1, cosine only) |
| SmolVLM-2.2B V and C | 42, 2, 21 | 42 px: 1, 2, 4, 5, 7, 8, 10 (q 14); 14 px: 3, 6, 9 (q 6) |
| SmolVLM-2.2B H | 14, 1, 14 | 14 px: 1 to 7 (q 13) |
| SmolVLM-500M V and C | 64, 4, 16 | 64 px: 1, 2, 3, 5, 6, 7 (q 12); 16 px: 4, 8 (q 3; bin 8 cosine only) |
| SmolVLM-500M H | 16, 1, 16 | 16 px: 1 to 8 (q 15) |

- **Holm's correction** within each family (`stats.holm`): order the m p-values, multiply the i-th smallest by m − i + 1, take the running maximum and clip at 1.
- **Families.** Random pages are primary because they measure the encoder; natural pages add the language model's repair (build guide 1A.8).
  1. **Primary, clause (a)** (`a_random62`): per reader, on random62 pages, three F-tests: the patch group of block H, the patch group and the cell group of block V. Holm over the three tests of one reader.
  2. **Primary, clause (c)** (`c_random62`): per reader, on random62 pages, the paired Wilcoxon signed-rank test over pages of the page's CER at (H, φ = 0) against the page's CER pooled over every level of block C. Holm across the three readers. Pooling C over its 16 or 21 levels estimates the page's conventional CER far more precisely than one render would, and C is predicted flat, so the pooled value is also what a conventional renderer, which cannot choose its phase, would give.
  3. **Secondary**: the same two families on natural pages (`a_natural`, `c_natural`); the control block C, every group, Holm within one reader and text type (`control_C`); block S, one F-test per reader and text type (`S`); block A, each mode against C at φ = 0, Holm over the two modes of one reader and text type (`A`).

## 7. Definition of "periodic signal" for G2

On the pilot's random62 pages, at least one group of the primary family `a_random62` has a Holm-adjusted p below 0.05 for at least one reader, read from `reports/tables/pilot/harmonics.csv` beside Figure 1. G2 decides the paper's framing only (`docs/03_pipeline.md`, G2; `docs/01_research_plan.md`, negative-result framing): the full run goes ahead either way, and its page count comes from σ, not from this outcome. Because the pilot's ten pages detect only amplitudes of about 0.02 at σ = 0.05, a no-go is weak evidence of no effect, and D5 says so.

## 8. Exclusion rules (what makes a run invalid)

- No row is excluded for its CER, its latency or the content of its transcription. There is no outlier rule.
- A condition that fails to run is re-run on resume (`--resume`), never dropped. The resume key is (reader, page_id, condition_hash, repeat), and a repeated key is an error in `load_results`.
- Rows that hit `max_new_tokens` are kept, scored as they are and counted (D13).
- A run does not start, and so has no rows, when the code under `src`, `scripts`, `configs`, `pyproject.toml` or `requirements.lock.txt` has uncommitted changes, when a config holds a placeholder, or when the processor's vision-token count differs from the table in section 2. A run does not resume when its sweep file, reader or code has changed since its manifest.
- A page leaves every analysis only if its reference is empty or fails `render.fits_page` under a shared layout, which the pagination already prevents, or if any of its planned conditions is missing when a run stops at its deadline (build guide 3.6). Such pages are counted in `checks.csv` and reported.
- Determinism mismatches in `determinism.csv` are reported, never excluded; if any occurs, the paper says that greedy decoding was not deterministic on this hardware.
- The ladder's rows are design data (section 5) and enter no test.

## 9. Analysis plan

Every table is written by `scripts/20_analyze.py` into `reports/tables/<sweep_name>/` with a manifest, and every figure by `scripts/30_make_figures.py` or `scripts/03_equivariance.py` into `reports/figures/`, from `results/` alone. Every table carries `status` (`preregistered` or `exploratory`) and `family`, with the Holm-adjusted value beside every preregistered p-value. Every resampling is by page, never by row, with 2,000 resamples and seed 20261014.

1. **Load.** `sweep.load_results` reads only the run folders whose manifest names the sweep file's SHA-1 and name, with explicit dtypes, and raises on a repeated key. Section 8 is applied and `checks.csv` is written: rows done and planned, repeated keys, missing transcription files, the `hit_limit` count and share, determinism mismatches, pages excluded, median latency.
2. **Harmonics** (`harmonics.csv`). For each reader, text type and block V, H and C: `stats.bin_groups(L, periods, n)`, `stats.harmonic_fit` at the block's levels, the F-test of each group, Holm within the family, each group's RMS amplitude A_T over the sampled levels (the RMS of the group's fitted component, in CER units) and its share of the fitted harmonic variance, and a 95% page-bootstrap interval for A_T (`stats.page_bootstrap_rms`: pages drawn with replacement, a page drawn twice entering as two pages). The share is the descriptive "dominant period".
3. **Paired** (`paired.csv`). For clause (c), per reader and text type and per page: a = the mean of the page's CER over the levels of C (every level reads the same reference, so this is the pooled CER), b = the page's CER at (H, φ = 0). `stats.paired_wilcoxon(a, b)` with `zero_method="wilcox"`, no continuity correction and the method resolved as SciPy 1.17.1's `method="auto"` would resolve it, reported as `exact`, `permutation` or `asymptotic`; `stats.median_paired_difference(a, b)` = median(b − a), negative when patch-aware lowers CER, with a 95% bootstrap interval (`stats.bootstrap_ci` on b − a). Block A the same way: a = the row at (C, φ = 0), b = the mode's row; Holm over the two modes.
4. **Sub-pixel** (`subpixel.csv`). Block S: `stats.harmonic_design(phase, 1, [1, 2])` gives cos1, sin1 and cos2 (sin2 is zero at quarter pixels), so one `group_f_test` of the three columns is the F-test of phase as a four-level factor with page fixed effects (q = 3), reported with the pooled CER at each of the four phases.
5. **Sizes** (`sizes.csv`, full run only). Block Z, per reader, text type, render mode and size: the pooled CER with its page-bootstrap interval; vision tokens (256, 81 and 64); characters per token (the pages' mean reference length over the tokens) and symbols per token (the same without spaces). The critical size s* = min{s : CER(s) ≤ CER∞ + δ} (learning notes A8), with δ = 0.01 absolute CER and the plateau CER∞ = the pooled CER of the rows at 20 and 24 px of the same reader, text type and mode; a missing s* is written as missing, never as the largest size. Descriptive, for Figure 2.
6. **Coherence** (`coherence_summary.csv`). Per reader, block, period and harmonic: the mean and range of R_k over pages beside the RMS amplitude of the matching group, the description that goes with the C prediction of section 1.
7. **Figures.** Figure 0, `reports/figures/fig0_equivariance.pdf`, the shift-equivariance error E(s) per layer (W1.9, already drawn). Figure 1, `fig1_phase_<sweep_name>.pdf`: readers in rows, blocks V, H and C in columns; the pooled CER at each level with its page-bootstrap band, the fitted harmonic component as the line, vertical lines at multiples of the patch and the token cell. Figure 2, `fig2_tradeoff_<sweep_name>.pdf` (full run): block Z's pooled CER against characters per token, one line per reader and render mode; beside it the Fano floor of bits per token against symbols per token at assumed symbol-error probabilities 0, 0.03 and 0.1, labelled as assumed; no measured CER enters the Fano bound.
8. **Exploratory.** Anything not named above, including comparisons with the layer-wise curves of Figure 0, the gap between natural and random62 CER, and WER versions of the tests, is labelled `exploratory` and reported as such.

## 10. What would count as H1 failing

- **Clause (a)** fails for a reader when no group of its primary family (`a_random62`) is significant after Holm on the full run, and fails overall when it fails for all three readers.
- **Clause (b)** is unsupported for a reader when its significant groups are neither the patch nor the cell group.
- **Clause (c)** fails for a reader when its Holm-adjusted paired test is not significant at α = 0.05 or its median paired difference is not negative.
- A significant group in the control block C with no significant group in V or H for the same reader contradicts the coherence account of D10, and the paper says so.
- G2's no-go changes the framing to the negative result of `docs/01_research_plan.md`; it does not by itself make H1 fail, which the full run decides.
- H2 fails when no attempt meets the criterion of section 1, and is reported as not run under D16.

## 11. Date and commit hash

- **Drafted:** 14 October 2026 (Claude, D22). **Approved by the owner, for both authors (D22):** TODO[W1.10]: the date, after the ladder has filled section 3.
- **Tag.** `prereg-v1` goes on the commit that adds this file's final text, which also holds `configs/sweeps/pilot.yaml` with the ladder's sizes and the runner (`src/mrr/sweep.py`, `scripts/10_run_sweep.py`); a commit cannot contain its own hash, so the tag names it: `git rev-list -n 1 prereg-v1`. The runner was committed as `495f1f3` (W2.3, 14 October 2026); the configs with the ladder's sizes in TODO[W1.10]: the hash of the W1.10 commit.
- D13 and D14 are marked decided when the tag goes on.

---

## Amendments

<!-- Add dated entries here after prereg-v1. Format: ### YYYY-MM-DD, what changed, why, who. -->
