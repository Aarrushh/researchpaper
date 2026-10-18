<!--
PURPOSE: Preregistration of Study 2, the cell-fitted monospace layout, frozen before its data collection.
WHAT GOES HERE: Hypotheses H3 to H7, readers (the three of Study 1 plus validation readers), conditions, sample size, tests, exclusion rules, failure criteria, and the pre-run steps that must be done before the tag.
FILLED IN AT STEP: W5.1 (draft 18 October 2026); tagged prereg-v2 once the owner approves and the pre-run steps of section 7 are recorded. Amendments only after that.
DO NOT: Edit anything above the Amendments section after the prereg-v2 tag. Do not add results. Do not pool Study 2 rows with Study 1 or with the pilots.
-->

# Preregistration, Study 2: one glyph per patch

> **STATUS: DRAFT, not approved.** Tag `prereg-v2` only after the owner approves and after the three pre-run steps of section 7 (font choice, validation-reader feasibility, pipeline changes) are done and recorded in section 11. Nothing below has been run.

Drafted on 18 October 2026 by Claude (D22) from the frozen results of Study 1 (`data-freeze-v1`, `spec-v1`), the mechanism analysis in `research-paper-notes/research/R6_mechanism.md`, and the exploratory pilots of 18 October (`exploratory_2026-10-18/explore7.py`, `explore9.py`, `explore10.py`), whose pages are excluded from this study. Where this document and the notes differ, this document rules once tagged. `PREREGISTRATION.md` (Study 1, `prereg-v1`) stands unchanged; its H2 is unaffected.

## 0. Why this study

Study 1 found that a reader's error depends on where glyphs sit against its patch grid (H1 (a), (b) supported), and that the patch-aware layout read worse than conventional text (clause (c) reversed, median paired difference +0.162, +0.114 and +0.046 on random characters). The post-freeze analysis explained the reversal as two mechanisms pulling apart:

- **Encoder.** With glyphs in patch-wide slots, a linear probe reads the character from Qwen's layer-0 patch vector at 82–88% against 11–13% for conventional text, and at the tower's output 98% against 86% (SmolVLM-2.2B 92–93% against 84%). When the same slot layout is shifted by half a patch so every glyph is cut by a patch edge, the probe falls back to 86–90%. Spaces removed, Qwen's character errors in the slot layout were 0.014 against 0.023 conventional. The condition is one whole glyph per patch at one offset (R6 §1).
- **Decoder.** The reader writes a space where the white between two glyphs exceeds a threshold it sets between the page's letter-gap and word-gap modes (Qwen: 50% at 8.5 px on slot pages; 1.84 million glyph pairs). The slot layout's two-slot characters leave 9–14 px of white, which it read as word gaps 81–91% of the time; the grid position of the gap moved the odds by a quarter of a pixel's worth (R6 §2).
- **Design.** Both are met when every glyph's ink is about 8 px to P − 2 px wide and sits in its own P px cell, with one empty cell per space: a monospace face at an advance of one patch. The pilots confirmed the decoder half causally (false spaces 0.170 → 0.0014 per character when the wide glyphs were condensed; at the conventional level with a real monospace face on both readers), found the contained-against-cut ordering on both readers at one to two standard errors, found the gain to depend on the reader (Qwen at parity, SmolVLM-2.2B about a quarter lower on random characters), and found a monospace layout to hurt SmolVLM-2.2B on prose (R6 §4, §4b).

Study 2 tests the design under preregistration, with a font chosen for a distinct O and 0 (the pilot's Source Code Pro made Qwen read O as 0 in 30 of 50 cases), and asks whether the two mechanisms hold on a reader none of this work has touched. The goal it serves is a rendering specification written in a model's patch size P and merge factor, whose rules transfer and whose coefficients are measured per reader.

## 1. Hypotheses (final wording)

All comparisons are between layouts of the same page read by the same reader, so each page is its own control. "Contained" means the mean over the two contained phases of the monospace layout (conditions M0 and M3 of section 3); "cut" means the monospace layout at phase P/2 (M7), where every glyph straddles a patch edge; "conventional" means the sans baseline pooled over its three phases (S0, S5, S10). "Letters-only CER" is the CER with every space removed from both strings (section 4).

- **H3, the gap mechanism (decoder).** On random characters, the inserted spaces per reference character of the contained monospace layout are lower than those of Study 1's proportional slot layout (condition Q0) on the same pages, and not higher than those of conventional text by more than 0.005 per character. Prediction: a fall by an order of magnitude, and parity with conventional text.
- **H4, containment (encoder).** On random characters, the letters-only CER of the contained monospace layout is lower than that of the cut monospace layout. Same glyphs, same font, same gaps, same page; only the glyphs' place against the grid differs. Prediction from the probe and the pilots: lower for every reader whose error is not at an alphabet floor.
- **H5, the reading advantage at equal glyph size.** On random characters, the CER (spaces counted) of the contained monospace layout differs from that of conventional sans text with the same x-height. Two-sided; the predicted sign is negative (monospace lower), and the paper reports the sign per reader. Study 1's finding that Qwen's remaining errors at 24 px are case-identical pairs (c/C, k/K, v/V, x/X, z/Z, j/J) predicts parity for Qwen and a gain for SmolVLM-2.2B.
- **H6, prose safety.** On prose, the contained monospace layout is not worse than conventional sans text by more than δ = 0.01 CER. A reader passes when the upper end of the 95% interval of the median paired difference is below δ. Prediction: Qwen passes; SmolVLM-2.2B fails (pilot median +0.028, 12 of 13 pages worse); SmolVLM-500M unknown. A reader that fails H6 gets the layout for unpredictable text only.
- **H7, generalisation.** H3 and H4 hold, with their predicted signs, on the validation reader(s) of section 2, which were chosen before any page of this study was rendered and have never been read in this project.
- **Descriptive, not tested.** The density-matched comparison (contained monospace against conventional sans shrunk to the monospace layout's glyphs per line, condition S18); the per-letter confusions, in particular O/0 and I/l/1; WER; the token-limit share; the capacity of each layout in characters per page and per vision token, which follows from geometry and is reported, not measured.

## 2. Readers

The three readers of Study 1 at their pinned revisions (`configs/readers.yaml`; `PREREGISTRATION.md` section 2): `qwen` (Qwen2.5-VL-3B-Instruct, P = 14, merge 2 × 2, 448 px, 256 tokens), `smolvlm` (SmolVLM-Instruct 2.2B, P = 14, merge 3 × 3, 384 px, 81 tokens, blind strip from column and row 378) and `smolvlm500m` (SmolVLM-500M-Instruct, P = 16, merge 4 × 4, 512 px, 64 tokens). `smolvlm500m` reads prose only: Study 1 put its random-character error at 0.94–1.00 at every size and phase, a ceiling at which no comparison can move.

**Validation readers**, fixed before the tag by the feasibility step of section 7. The owner named Qwen2.5-VL-7B-Instruct. Its weights are 16.6 GB in bfloat16 and the GPU holds 8 GB, so on this machine it can run only quantised:

| Role | Hub id | Revision | Patch | Merge | Page | Tokens | How it runs here |
|---|---|---|---|---|---|---|---|
| validation, named | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | 14 | 2 × 2 | 448 × 448 | 256 | 4-bit NF4 weights (bitsandbytes 0.50.2, to be added to the lock), vision tower in bfloat16; or unquantised on a cloud GPU if one is used |
| validation, different family | `OpenGVLab/InternVL2_5-2B` | `573169ee54df216786bb9a189e9a32a060a008cf` | 14 | pixel shuffle to 256 tokens | one 448 × 448 tile, dynamic tiling off | 256 | bfloat16, fits |
| alternate, same family | `Qwen/Qwen2-VL-2B-Instruct` | `895c3a49bc3fa70a340399125c650a463535e71c` | 14 | 2 × 2 | 448 × 448 | 256 | bfloat16, fits |

The study runs the named 7B if the feasibility page passes (section 7), and InternVL2.5-2B in any case, because a second architecture (InternViT-300M, a different connector, a different language model) is a stronger test of "general" than a larger sibling of a tested reader. A quantised reader is the object measured; the paper says so, and claims nothing about the unquantised 7B. If the 7B does not fit, Qwen2-VL-2B takes its row and the paper says so. The processor of every validation reader is pinned to one native page with no resize (D8), checked on the first page by the token count as in Study 1; InternVL's dynamic tiling is set to a single tile. Prompt, decoding and the per-page token limit as in Study 1 section 2.

## 3. Independent variables and conditions

Every condition is black on white, `coverage_srgb`, unaligned line pitch ⌊1.25 × size + 0.5⌋ px, margin one patch, phase_y = 0, on the shared pages of section 5 with the phase allowances of Study 1 (105 × 26, 40 × 40 and 60 × 60 px for the three readers; the Qwen allowances for the validation readers, whose P is 14). Phases are whole pixels.

- **Fonts.** Sans: Source Sans 3 Regular, as Study 1. Mono: one monospace face chosen by section 7, OFL-licensed, with advance 0.5–0.6 em so that an advance of one patch gives a glyph of about the sans baseline's size, and with a marked zero and an O that differs from it in shape. It enters `data/external/fonts/` with its licence file and SHA-256 before the tag.
- **Sizes.** Mono: the size at which the face's advance equals P (P / advance-per-em; 23.3 px for an 0.6 em face at P = 14, 26.7 px at P = 16), so that the slot layout gives exactly one cell per glyph. Sans baseline: the size whose x-height in pixel rows equals the mono's (24 px for P = 14 with Source Code Pro-like metrics; 27 px for P = 16); fixed in section 11 once the font is chosen. Sans density-matched: the size at which conventional sans text holds the same glyphs per line as the mono layout (18 px for P = 14; omitted for the prose-only reader).
- **Layouts.** `conventional` (linear advances) and `patch_aware` (slot of P px, glyph centred at a whole-pixel offset, a space one slot), both as implemented in Study 1 (`src/mrr/render.py`); with the mono face every glyph takes one slot by construction.

| Code | Font, size | Layout | Phase_x | Text | Role |
|---|---|---|---|---|---|
| S0, S5, S10 | sans, baseline size | conventional | 0, 5, 10 | both | conventional reference; the page's conventional CER is pooled over the three |
| Q0 | sans, baseline size | patch_aware | 0 | random | Study 1's slot layout, the gap-mechanism control (H3) |
| M0 | mono, advance P | patch_aware | 0 | both | contained, aligned |
| M3 | mono, advance P | patch_aware | 3 | both | contained, offset |
| M7 | mono, advance P | patch_aware | P/2 (7 or 8) | both | cut: every glyph straddles a patch edge |
| S18 | sans, density-matched size | conventional | 0 | random | descriptive density comparison |

Renders per page: 8 on random characters, 6 on prose (`smolvlm500m`: prose only, 6). The conventional phases 0, 5 and 10 are three renders of the same layout; Study 1 found no effect of the horizontal phase of conventional text (R09), so pooling them estimates the page's conventional CER with less noise, as Study 1 pooled block C.

## 4. Dependent variables

As Study 1 section 4 (normalisation D13; `score.char_counts`; pooled CER; WER descriptive), with two additions scored per row and stored in `results.csv`:

- **Letters-only counts.** `score.char_counts` on the reference and the transcription with every space removed from both, stored as `hits_ns`, `substitutions_ns`, `deletions_ns`, `insertions_ns`; the letters-only CER is pooled from them. This is the response of H4.
- **Inserted spaces per reference character.** The number of spaces in the transcription's insert chunks of the jiwer alignment, divided by the reference length (H3); stored as `spaces_inserted`.
- Per-letter confusion counts (substitutions by reference and hypothesis character, deletions by character) are computed by the analysis from the stored texts, not stored per row.

## 5. Page corpus and sample size

- **Random characters.** `text_source.random_text(400000, 20261020)`, five-character words; a seed no sweep or pilot has used (Study 1: 20261012, 20261013, 20261016; pilots: 20261018, 20261019).
- **Prose.** The three books of Study 1 in rotation, each starting after the last word the ladder and Study 1's full run used in it: `configs/sweeps/study2.yaml` sets `pages.natural.start_after: [configs/sweeps/ladder.yaml, configs/sweeps/full.yaml]`, the mechanism Study 1 used to start after the ladder. The pilot's prose pages came from Persuasion at character 200,000 onward of the skipped text; 60 pages of about 230 characters stay far below that offset, and the pagination asserts it.
- **Pages.** A page is the longest run of whole words that fits the text block under every shared layout of the reader, so the mono slot layout, the tightest, sets the length (about 230 characters for P = 14). Every page is read under every condition of its text type. Pages are planned in order, text types alternating, so that a run stopped early still holds the same pages under every condition.
- **Counts.** `qwen`: 60 random and 60 prose pages. `smolvlm`: 90 and 90 (its page-to-page spread is larger and it is cheap, 4 s a page). `smolvlm500m`: 60 prose. Each validation reader: 60 and 60. The counts are ceilings and the clock is the stopping rule, as in Study 1 (D26): each reader runs until its deadline in `configs/schedules/study2.yaml`, checked between pages.
- **Justification from the pilots** (R6 §4b, 16 pages each). Paired differences of page CER had standard deviation about 0.010 for Qwen on random characters (letters-only, contained against cut; observed difference 0.005–0.006) and about 0.10 for SmolVLM-2.2B (observed 0.03 contained against cut, 0.04 against conventional); on prose the SmolVLM-2.2B effect was 0.03–0.08. With the smallest detectable median difference of Study 1, (z₀.₉₇₅ + z₀.₈) σ_d / √(3N/π): Qwen at N = 60 detects 0.0037, below the observed effect; SmolVLM-2.2B at N = 90 detects 0.030, at the observed effect, which is why it gets 90 pages and why its primary tests are rank-based. The realised detectable differences are reported from the run's own σ_d, as in Study 1.
- **Budget.** Transcriptions: `qwen` 840, `smolvlm` 1,260, `smolvlm500m` 360, each validation reader 840; at the pilots' speeds (10, 4, about 20 and, for a 4-bit 7B, about 30 s a page) roughly 2.3, 1.4, 2, 7 and 1.4 hours: two nights on the laptop's GPU, one reader at a time for Qwen-sized models, the two SmolVLMs together as in Study 1.

## 6. Tests and alpha

- **α = 0.05.** Every test is the two-sided paired Wilcoxon signed-rank test over pages (`stats.paired_wilcoxon`, zero differences dropped, method as SciPy resolves it), reported with the median paired difference and its 95% page-bootstrap interval (2,000 resamples, seed 20261021). A hypothesis with a predicted sign counts as supported only when the Holm-adjusted p is below α and the median difference has that sign.
- **Per page.** a = the page's conventional CER (pooled over S0, S5, S10); b = the page's contained CER (pooled over M0 and M3); c = the page's cut CER (M7); q = the page's CER at Q0. For H3 the same with inserted spaces per character; for H4 with letters-only CER.
- **Families and Holm** (`stats.holm` across the readers within each family; validation readers included):
  1. `gap` (H3): per reader, Q0 against contained on inserted spaces per character; the non-inferiority part is read from the interval of contained minus conventional (upper end below 0.005 per character), not tested.
  2. `containment` (H4): per reader, cut against contained, letters-only CER.
  3. `advantage` (H5): per reader, conventional against contained, CER.
  4. `prose` (H6): per reader, the interval of contained minus conventional on prose; passes when its upper end is below 0.01. No p-value is needed for a pass; the paired test is reported.
  5. H7 is read from the validation readers' rows in families 1 and 2.
- **Secondary.** M0 against M3 (does the offset within the cell matter, as the pilots and Study 1's block H hint); S18 against contained (descriptive); the same contrasts on prose where they exist; per-letter confusion tables with the O/0 and I/l/1 counts; WER versions of the primary contrasts, exploratory.

## 7. Pre-run steps, recorded in section 11 before the tag

1. **Font choice.** Candidates: OFL monospace faces with a marked zero: Inconsolata, Roboto Mono, Anonymous Pro, IBM Plex Mono, JetBrains Mono, Fira Mono, DejaVu Sans Mono, and Source Code Pro (the pilot's). Measured criteria at the size where the advance equals 14 px, in this order: (a) every one of the 62 alphanumerics has its ink inside its own cell (bitmap left ≥ 0 and left + width ≤ 14) and at least 7 px wide; (b) x-height 11 to 13 rows; (c) the O and 0 bitmaps differ in at least 15% of their pixels after centring, and 0 carries a slash or dot; (d) I, l and 1 are pairwise distinct by the same measure. Among the faces that pass, a declared pre-test: 10 random-character pages (seed 20261022), the contained layout at phase 0, read by `qwen` only; the face with the fewest O/0 plus I/l/1 confusions is chosen, ties broken by the fewest letter errors. The 10 pages are excluded from the study, the counts are written in section 11, and the face goes into `data/external/fonts/` with its licence and SHA-256 under the rules of that folder's README.
2. **Validation-reader feasibility.** For each validation reader: install what it needs (bitsandbytes 0.50.2 for the 7B, pinned in `requirements.lock.txt`), load it as `configs/readers.yaml` will describe it, check the vision-token count on a blank native page, and transcribe one page of random characters and one of prose from the pilot's seed (excluded pages). A reader passes if it loads without CPU offload of the language model, the token count matches, and the two pages decode. The outcome, the realised placement and the peak memory are written in section 11; a reader that fails is replaced as section 2 says.
3. **Pipeline changes**, each with tests, committed before the tag: `configs/sweeps/study2.yaml` (per-block `font_file` and `font_size_px`, which `sweep.expand_conditions` already supports); the two reader blocks; `readers.load_reader` support for 4-bit loading when a reader block asks for it; `score.char_counts_letters_only` and `score.spaces_inserted`, written to `results.csv` by the runner; `scripts/22_analyze_study2.py` writing `reports/tables/study2/paired.csv` (family, reader, contrast, n, statistic, p, p_holm, median_diff, ci_low, ci_high, mde), `letters.csv`, `spaces.csv`, `confusions.csv` and `capacity.csv`, with `status` and `family` columns as Study 1's tables carry them; `scripts/30_make_figures.py` extended with Figure 3 (CER by condition and reader, with page-bootstrap bands) and Figure 4 (O/0 and I/l/1 confusions by font and layout). `make test` and `make contract` must pass on the tagged commit.

## 8. Exclusion rules

As Study 1 section 8, without change: no row excluded for its content; conditions that fail are re-run on resume; token-limit rows kept and counted; no run on a dirty tree or a wrong token count; a page leaves the analysis only if a planned condition is missing when the run stops; determinism mismatches reported. In addition: the font pre-test's 10 pages and the feasibility pages enter no table; Study 2 rows are never pooled with Study 1, the ladder or the pilots.

## 9. Analysis plan

`scripts/22_analyze_study2.py` reads the run folders through `sweep.load_results` and `complete_rows`, applies section 8, and writes to `reports/tables/study2/` with a manifest:

1. `checks.csv`: rows done and planned, repeated keys, missing files, token-limit count and share, pages excluded, median latency, per reader.
2. `paired.csv`: every contrast of section 6 with its family, n, statistic, p, Holm-adjusted p, median paired difference, 95% interval, and the smallest detectable median difference from the run's own σ_d.
3. `letters.csv` and `spaces.csv`: pooled letters-only CER and inserted spaces per character by reader, text type and condition, with page-bootstrap intervals.
4. `confusions.csv`: per reader, text type and condition, the substitution counts by (reference, hypothesis) character and the deletion counts by character; the O/0, I/l/1 and case-pair totals.
5. `capacity.csv`: per reader and layout, glyphs per line, lines per page, characters per page at capacity and per vision token, from the layout geometry on the study's own text.
6. Figures 3 and 4 as section 7 names them; Figure 3 marks the conventional, contained and cut conditions.
7. Freeze: `scripts/40_freeze.py` copies the run folders without images, hashes them, and `--check` must report that the tables regenerate byte for byte; tag `data-freeze-v2`. Everything not named above is `exploratory`.

## 10. What would count as failing

- **H3** fails for a reader when Q0 against contained is not significant with the contained rate lower, or when the upper end of the contained-minus-conventional interval is at or above 0.005 spaces per character. If H3 fails for the development readers, the gap mechanism of R6 is wrong or incomplete and the paper says so.
- **H4** fails for a reader when cut against contained is not significant with the contained letters-only CER lower. Failing for Qwen alone is consistent with the floor argument of section 1 and is reported as such; failing for every reader means the probe's accessibility gain does not reach the reading level, and the specification's one-glyph-per-patch rule rests on the decoder mechanism only.
- **H5** has no failure criterion; its sign and size are reported per reader.
- **H6** fails for a reader when the interval's upper end is at or above 0.01; that reader's rule is restricted to unpredictable text in the specification.
- **H7** fails when either validation reader does not show H3 and H4 with the predicted signs; the specification's rules are then stated for the readers tested, and the generalisation claim is withdrawn.
- A validation reader that cannot be made to run is reported as not run; it does not make the study fail.

## 11. Date, approvals and the pre-run record

- **Drafted:** 18 October 2026, Claude (D22). **Approved by the owner:** pending.
- **Font chosen (section 7.1):** pending; face, version, size at advance P, measured criteria, pre-test counts, page ids excluded.
- **Validation readers (section 7.2):** pending; realised placement, peak memory, token count, the two feasibility pages' ids.
- **Pipeline (section 7.3):** pending; commit hashes of the changes; `make test` and `make contract` results.
- **Tag.** `prereg-v2` goes on the commit that holds this file's final text, `configs/sweeps/study2.yaml`, `configs/schedules/study2.yaml`, the reader blocks and the font; `git rev-list -n 1 prereg-v2` names it.

---

## Amendments
