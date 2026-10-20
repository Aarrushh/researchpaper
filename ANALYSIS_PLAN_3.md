<!--
PURPOSE: The analysis plan of the paper's exploratory analyses: the scoring rules that split reading errors from behaviour, Study 3's stage 0 (the calibration method on five readers), and Study 2's exploratory re-scoring.
WHAT GOES HERE: Scope, the rules R1 to R5 with their fixed parameters, the stage-0 tables and figures, the Study 2 questions, the order of work, and the owner's decisions.
FILLED IN AT STEP: Drafted 20 October 2026 (research-paper-notes/research/R8_analysis_plan.md); the owner approved the four decisions of section 6 the same day; committed with src/mrr/behaviour.py and its tests before any stage-0 or Study 2 row was scored with them.
DO NOT: Change a rule or parameter of section 1 after a row has been scored with it; add an amendment at the bottom instead. Do not touch a preregistered verdict of Study 1 or Study 2 from here.
-->

# Analysis plan 3: reading errors and behaviour, the calibration method, and Study 2's exploratory analyses

> **STATUS: committed before scoring, 20 October 2026.** The owner's instruction of 20 October: "Scope the paper to what we measured: the mechanisms, Study 2's preregistered results and a calibration method shown on five readers. Treat data collection as done and plan the analysis: write down the scoring rules first, then the stage-0 and Study 2 analysis." The owner approved the four recommendations of section 6 the same day. Nothing in this document is confirmatory: every table it names carries `status: exploratory`. The rules were written after the diagnostics of 20 October, which read stage-0 transcriptions of the SmolVLM pair and of InternVL3 (LAB_NOTEBOOK.md, the two stage-0 entries of 20 October); the paper says so. Changes after this commit go in the Amendments section at the bottom.

Drafted by Claude (D22). Numbers quoted come from committed tables, named where they are used.

## 0. Scope and ground rules

The paper reports three things, all from data already collected and committed:

1. **The mechanisms.** Study 1 (`prereg-v1`, frozen as `data-freeze-v1`, specification `spec-v1`) and the exploratory analyses of R6: one glyph per patch makes a letter readable at the encoder and makes its gaps read as spaces at the decoder.
2. **Study 2's preregistered tests.** H3 to H7 (`prereg-v2`; tables and Figures 3 and 4 at d4817abf, 4,140 rows), frozen as `data-freeze-v2`: monospace at an advance of one patch.
3. **A calibration method, shown on five readers.** Study 3's stage 0 (exploratory; 896 rows in three sweeps: 784 rows committed in 31f72979, 112 in 46e15675): each reader's error as a function of font size, the size for a target error, and the characters a vision token carries there.

The paper claims no rule for a reader it has not measured. The rest of R7 (its no-calibration rule, tier A; its stages 1 and 2; the GPU cell renderer) is left to later work.

Ground rules:

- **No new reading.** Every number comes from the committed run folders: `results.csv` and the stored transcriptions.
- **Preregistered verdicts are final.** Study 1's and Study 2's confirmatory results stand as their analysis plans produced them, and nothing here can change one. Every new measure carries `status: exploratory` in its table and is called exploratory in the text.
- **The primary measure stays as frozen** (D13): `normalise_text` on both strings, case kept, every character counted, counts pooled over pages. Every table with a new measure reports the primary CER beside it.
- **Rules before results.** Section 1's rules are implemented in `src/mrr/behaviour.py` and tested on planted strings in `tests/test_behaviour.py`. They are committed with this document before any rule is applied to a stage-0 or Study 2 row.
- **The data are frozen first.** `data-freeze-v2` (local) hashes every run folder under `results/runs` into `results/final/`: Study 1's, Study 2's and the three stage-0 sweeps'. `scripts/40_freeze.py --check` then regenerated the tables of Study 1 (`full`) and Study 2 (`study2`) byte for byte. So the stage-0 run folders were hashed before section 1 touched them.

## 1. The scoring rules

**What they are for.** Above the size where a reader's cliff ends, most of its remaining error is not misreading. On 20 October all five readers showed some of these habits:

- skipped lines inside text they read correctly;
- a dropped unfinished sentence at the page's end;
- capitals written in lower case;
- once, a chat preamble.

These habits do not shrink with glyph size as misreading does, so they bend the curve the calibration fits. The rules split each row's errors into reading errors and behaviour. The paper can then report, side by side, what the reader resolves (its acuity) and what a user gets (the primary CER).

### 1.1 Per row, in this order (`mrr.behaviour.score_row`)

- **R1, format.** Before normalising, the transcription loses two kinds of line, and nothing else is removed:
  - every line that is only a code fence: three backticks, optionally a language name (`FENCE`);
  - then, if the first line that is not blank matches the pattern below ignoring case, that line (`PREAMBLE`):

    `^\s*(sure|certainly|of course|here is|here's|here’s|here are|below is|the text|the transcription|transcribed text)\b[^\n]*:\s*$`

  The characters removed, counted after normalisation, are the row's format characters. The prompt asks for the text only ("Transcribe all text in the image exactly. Output only the text."), so both are text it excludes. Every row R1 touches is listed and read by eye. That reading may only report on the pattern, never change it.
- **R2, alignment.** The rest is normalised and aligned to the reference by jiwer, as `score.char_counts` does. There is one alignment per row, and every later class is read from it.
- **R3, anchored runs.** A run is a delete or insert chunk of at least L = 15 characters (`MIN_RUN`).
  - **Anchoring.** A side of a run exists when the reference goes on past it. A run is anchored when at least one side exists and, on every side that exists, the 30 neighbouring reference characters (`ANCHOR_CHARS`, or as many as there are) hold at most 3 errors (`ANCHOR_ERRORS`). The errors counted are substituted and deleted characters, plus characters inserted strictly inside the window. Errors of a neighbouring run count like any other.
  - **Classes.** An anchored deletion is an **omission**: the reader skipped text inside a stretch it read correctly. It is labelled head (it starts at the first reference character), tail (it ends at the last) or interior. An anchored insertion is a **runaway insertion**, a loop or an invented continuation.
  - **Unanchored runs.** An unanchored run stays a reading error, so the long deletions of a page the reader cannot read are not called omissions. A run that covers the whole reference has no side, so an empty transcription is a reading error.
  - **Why these numbers.** In Study 1, 80% (SmolVLM-2.2B) and 60% (SmolVLM-500M) of deleted prose characters lay in runs of 15 or more, and Qwen's in none (LAB_NOTEBOOK.md, 18 October). A line at 42 px on a 448 px page holds about 18 characters (146 on eight lines, the plan of `s3_stage0_prose_internvl3.yaml`), so a skipped line there still counts.
  - **Lean.** The rules lean towards reading: two runs closer than 30 characters can fail each other's anchor and stay reading errors. Behaviour may be missed; misreading is not called behaviour.
- **R4, case.** A substitution of a letter by the same letter in the other case is a case error (Study 2's `case_pair`, `src/mrr/study2.py`). It is split by letter:
  - **Shape-alike letters** are those whose lower-case glyph is the capital at a smaller size. A shape-alike case error can be a misreading of size, so it is a reading error.
  - **Shape-distinct letters** (a/A, e/E, g/G and so on) are the rest. A case error on one cannot be misreading: the reader got the letter right and changed its case. It is behaviour, called case normalisation.
  - **The measure** (`shape_alike_letters`): at 48 px, each glyph is rasterised as the pages were (`mrr.render.load_glyph`, unhinted FreeType), and the lower-case glyph is scaled by cap height over x-height (the ink heights of H and x). Centred on its capital and moved by up to 2 px each way, it counts as shape-alike when its ink mask (grey ≥ 128) overlaps the capital's with IoU ≥ 0.6.
  - **Measured on 20 October with the study's renderer (Amendment 1):**
    - Source Sans 3: **c o s u v x z**;
    - Fira Mono: **c o q s v x z** (its u has a spur; q, at 0.603, lies just above the threshold).
  - **A threshold set after measuring.** The draft fixed IoU ≥ 0.8 before measuring. Measured first with Pillow's hinted rasteriser, that admitted no letter in either font (o/O reached 0.66 and 0.62), and the threshold was lowered to 0.6, the widest gap in Source Sans 3's ranking, before any row was scored. The study's renderer gives higher overlaps: 0.8 would admit only s (Source Sans 3) and z (Fira Mono), and 0.6 still lies in the widest gap of Source Sans 3's ranking (u .66, q .53). The owner kept 0.6 on 20 October. The overlaps:

    | Font | z | s | c | o | v | x | u | q | n | y | e | i | w | h | k | t | r | a | j | b | p | f | d | g | m | l |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
    | Source Sans 3 | .76 | .82 | .76 | .77 | .70 | .71 | .66 | .53 | .49 | .46 | .45 | .43 | .51 | .40 | .32 | .39 | .32 | .30 | .27 | .25 | .23 | .16 | .24 | .16 | .13 | .12 |
    | Fira Mono | .80 | .72 | .73 | .66 | .73 | .69 | .58 | .60 | .40 | .46 | .45 | .30 | .43 | .36 | .38 | .46 | .31 | .29 | .37 | .25 | .24 | .23 | .28 | .18 | .28 | .11 |

    w (0.51, 0.43), which the draft expected to be alike, is not.
  - **Sensitivity.** The case-folded reading CER of section 1.2 removes every case error, and the primary CER keeps every one, so the two bound any choice of this threshold.
- **R5, reading errors.** Everything else:
  - other substitutions, with O/0 and I/l/1 counted as Study 2 counts them;
  - deletions and insertions shorter than L, or unanchored;
  - spaces.

### 1.2 Measures

Per row (`RowScore`), then per cell by summing counts over pages (D13, never a mean of page rates):

- **CER**, the primary measure, as frozen.
- **Coverage** = 1 − omitted characters / reference characters.
- **Reading CER** = reading errors (R5 and shape-alike case errors) / reference characters not omitted. This is the error of what the reader did read, and the calibration fits it as acuity.
- **Behaviour**, per reference character: omitted (head, interior, tail), runaway inserted, format, and case-normalised. Also the share of pages with at least one event of each kind, and the stored token-limit share.
- **Case-folded reading CER**: reading CER with every case error removed. A sensitivity, for random text only.
- **A check that the parts add up.** On rows R1 does not touch, primary errors = reading errors + omitted + runaway + case-normalised, exactly (`RowScore.parts_add_up`). On rows it touches, the difference between the two alignments is reported per cell.

### 1.3 Fixed now, reported later

These are fixed: L = 15, the anchor of 30 characters with at most 3 errors, the R1 pattern, and the shape test with its two measured sets. Stage 0 also gets a sensitivity table at L = 10 and 30 and anchors of 20 and 50 characters. Nothing else is tuned.

### 1.4 Code and tests

`src/mrr/behaviour.py` is joined later by a script that writes the per-row counts; `results.csv` is never rewritten. `tests/test_behaviour.py` checks, on planted strings:

- a line skipped inside correct text is an interior omission;
- the same deletion inside misread text is a reading error;
- a dropped first or last line is a head or tail omission;
- the anchor allows 3 errors and not 4;
- a 14-character deletion is a reading error;
- an empty transcription is not an omission;
- a repeated block at the end is a runaway insertion;
- a preamble and its fences are format;
- case swaps split by the measured set, and O/0 stays reading;
- on 200 randomly edited rows, the classes add up to the primary count;
- the measured sets of the two study fonts.

## 2. Study 3, stage 0 (exploratory)

**Inputs.**

- `s3_stage0_random` (392 rows);
- `s3_stage0_prose` (392 rows);
- `s3_stage0_prose_internvl3` (112 rows);
- five readers in all.

The paper says the design followed Study 2 and the analysis followed the diagnostics of section 0.

### 2.1 Tables

All go in `reports/tables/s3_stage0/`, each regenerated byte for byte from the run folders.

- `pages.csv`: one row per results row, with section 1's counts.
- `cells.csv`, extended: per reader, text type and size, it holds:
  - the primary CER, reading CER, coverage and behaviour rates, each with a 95% passage-bootstrap interval;
  - the median and 90th-percentile page CER;
  - the token-limit share.
- `acuity.csv`, extended: the two-parameter curve fitted to the primary CER (as now) and to the reading CER, per reader and text type.
  - **Per fit:** a, β, s05, s01, κ05 and κ01 with intervals (2,000 passage resamples, seed 20261025), a/P and a/c, and in-range flags.
  - **Read off without a model:** s05 and s01, by log-linear interpolation between the bracketing sizes.
  - **Disagreements:** where the fitted and read-off s01 disagree beyond the fit's interval, the table says so. For example, InternVL3's prose s01 is 26.9 px [23.5, 32.6] fitted against about 22 read off (0.029 at 20 px, 0.003 at 24 px). The fit is pulled up mainly by the one page with a preamble at 42 px.
- `behaviour.csv`: behaviour rates by reader, text type and size.
- `tokens.csv`:
  - characters per text token from each reader's own tokenizer, on the stage-0 references, prose and random. R7 section 1.2 measured 4.54 and 1.48 for the Qwen tokenizer on Study 2's references.
  - the **token ratio** κ_ε / (characters per text token), with its interval. Above 1, the page image needs fewer tokens than the text.
- `sensitivity.csv`: section 1.3's L and anchor variants.
- `cost.csv`: GPU minutes (first to last page, model load apart) and pages per reader, from the run logs. This is the cost of the method.

### 2.2 Does one curve shape fit every reader?

This is the method's central claim.

- **Residuals:** per cell (observed against fitted, logit scale), and the deviance per reader.
- **Leave one size out:** refit without each size and predict it. Report the share of cells whose observed CER lies inside the prediction's 95% bootstrap band, per reader, for both CERs. Descriptive: no pass mark is set after seeing the data.

### 2.3 How cheap can calibration be?

Refit on subsets of the stage-0 design: 4 sizes instead of 7 (every other size), and 4 passages instead of 8 or 16. Report how far s01 moves from the full fit over 200 subsets. Descriptive.

### 2.4 What the paper states per reader

- **The 1% and 5% sizes on the primary CER, where reachable.** Reachable means within the ladder. Otherwise the paper says "not reached up to N px" and names the behaviour floor that stops it.
- **The reading-CER sizes beside them, with coverage at those sizes.**
- **κ and the token ratio at those sizes.**

### 2.5 Latency

Medians per reader, per page and per output token. Qwen's random-text passages 4 to 7 are left out (LAB_NOTEBOOK.md, 20 October).

### 2.6 Limits the paper states

- One font (Source Sans 3), the conventional layout, full pages at each reader's native frame, three English novels, greedy decoding.
- InternVL3's random-text s01 (37.0 px) lies one pixel past its ladder, and the SmolVLM pair's 1% sizes are extrapolated.
- a was measured on full pages, which read worse than the half-full pages of Studies 1 and 2 (R7 section 1.5).

### 2.7 Figures and freeze

- **Figure 5:** each reader's curves, primary and reading CER against size in px and in patches, with the cells, their intervals and s01 marked.
- **Figure 6:** behaviour rates against size.
- **Freeze:** the run folders are already hashed in `data-freeze-v2`. When the tables are done:
  - `scripts/40_freeze.py --check` gains the stage-0 fit, which takes `--configs` and `--name` rather than `--config`, so that it regenerates `reports/tables/s3_stage0/` from `results/final/runs`;
  - the tag `data-freeze-s3-stage0` (local) follows.

## 3. Study 2

### 3.1 Confirmatory, as preregistered (done)

From `reports/tables/study2/paired.csv` at d4817abf. The values are medians of paired differences, contained minus the comparison, so a negative value favours the contained monospace.

| Hypothesis | Qwen-3B | Qwen-7B (validation) | InternVL3-2B (validation) | SmolVLM-2.2B | SmolVLM-500M |
|---|---|---|---|---|---|
| H3, gap: spaces per character, vs Study 1's slot layout | −0.221, supported | −0.244, supported | −0.187, supported | −0.171, supported | not read on random text |
| H4, containment: letters-only CER, vs cut | 0.000, not supported | 0.000, not supported | −0.010, supported | −0.022, supported | — |
| H5, advantage: CER, vs conventional | 0.000, parity | +0.001, parity | −0.015, gain | −0.015, gain | — |
| H6, prose: upper end of the interval, δ = 0.01 | 0.000, passes | 0.000, passes | 0.000, passes | +0.002, passes | +0.056, fails |

- **H3.** The non-inferiority part holds for all four: the upper end of contained minus conventional is 0.000 spaces per character.
- **H6, SmolVLM-500M.** The median is +0.015 and the interval [−0.003, +0.056]. Non-inferiority is not shown, which is not the same as being shown worse. As PREREGISTRATION_2.md section 10 says, the layout is restricted to unpredictable text for this reader.
- **H7 fails** because Qwen-7B misses H4. The generalisation claim is withdrawn as worded, and the rules are stated for the readers tested.

### 3.2 Exploratory re-scoring with section 1

Every table here carries `status: exploratory`.

- **Measures:** per reader and condition, the reading CER, coverage and behaviour rates.
- **Tests:** families 2 to 4 are recomputed on reading CER with the same method: paired Wilcoxon, median difference, page-bootstrap interval, seed 20261021. They form a secondary table beside the confirmatory one, never in its place.

Three questions:

- **Q1. Is SmolVLM-500M's H6 result an omission habit or misreading?** Paired differences of coverage and of reading CER, contained minus conventional, on its 60 prose pages.
- **Q2. Is the Qwen pair's H4 null a letterform floor?** Compare letters-only reading CER without the O/0, I/l/1 and case classes, contained against cut. The preregistration anticipated the floor argument for Qwen-3B (section 10). This asks whether it also accounts for Qwen-7B.
- **Q3. How much of each layout's error is case?** Shape-alike and case-normalised errors by condition. The mono and the sans differ in letterforms, and their measured sets differ by u.

### 3.3 Linking Study 2 to stage 0 (exploratory, descriptive)

This compares each reader's Study 2 size with its stage-0 half-error size a (`reports/tables/s3_stage0/acuity.csv` at 6a1b10a9). The sans baseline was 25 px for P = 14 and 29 px for P = 16 (`configs/sweeps/study2.yaml`), and the mono had the same x-height.

| Reader | a, random (px) | 25 px / a | H4 median | H5 median | a, prose (px) | size / a | H6 upper end |
|---|---|---|---|---|---|---|---|
| Qwen-3B | 13.2 | 1.90 | 0.000 | 0.000 | 8.6 | 2.92 | 0.000 |
| Qwen-7B | 12.2 | 2.06 | 0.000 | +0.001 | 7.8 | 3.22 | 0.000 |
| InternVL3-2B | 19.6 | 1.28 | −0.010 | −0.015 | 14.9 | 1.68 | 0.000 |
| SmolVLM-2.2B | 25.6 | 0.98 | −0.022 | −0.015 | 17.6 | 1.42 | +0.002 |
| SmolVLM-500M | — | — | — | — | 25.1 | 1.15 (29 px) | +0.056 |

- **Pattern.** The layout effects appear where Study 2's size sat near the reader's cliff (size/a about 1.0 to 1.3), not where it sat at twice a or more. Likewise, the one prose failure is the reader read closest to its cliff.
- **This is R7's regime prediction** (F7, G3), seen after the fact on four readers.
- **Four readers cannot test it.** A two-sided exact Spearman test on four points cannot get below p = 2/24 ≈ 0.08, so the paper states it as an explanation to test, not a finding.
- **Caveats.** a comes from full pages of Source Sans 3. Study 2's pages were shorter, and its contained arm used Fira Mono at the same x-height.

## 4. Study 1 and the mechanisms

These are reported as frozen (`data-freeze-v1`, `spec-v1`), and R6's analyses as written, exploratory. Section 1's rules touch Study 1's transcriptions only to describe a habit the paper names, in an exploratory table. One example is SmolVLM-2.2B dropping the orphan clause at the top of a page.

## 5. Order of work

1. ~~The owner settles section 6.~~ Done, 20 October.
2. ~~Study 2's freeze: `data-freeze-v2` and its notebook entry.~~ Done, 20 October.
3. ~~Section 1 built: `src/mrr/behaviour.py`, its tests, the shape-alike sets measured, this document committed.~~ Done, 20 October, in the commit that adds this file.
4. **Stage 0:** ~~the per-row scoring script and the tables~~ (done 20 October, `scripts/25_fit_stage0.py`); ~~the freeze check, the tag~~ (done 20 October, `40_freeze.py --fit s3_stage0`, `data-freeze-s3-stage0`); Figures 5 and 6.
5. **Study 2:** the exploratory tables of section 3.
6. **Results text:** every claim names its table and its status.

All of it runs on the CPU, with no model loaded except the tokenizers for `tokens.csv`.

## 6. Decisions (approved by the owner, 20 October 2026)

The owner's words: "Approve all four section 6 recommendations."

1. **Case:** shape-distinct case errors count as behaviour, shape-alike ones as reading (R4). The shape threshold of 0.6 was set after this approval, for the reason given in R4, and before any row was scored.
2. **Omissions:** L = 15 characters, anchored by 30 characters with at most 3 errors (R3).
3. **The headline size per reader:** the 1% size on the primary CER where reachable, with the reading-CER size and coverage beside it.
4. **Where this plan lives:** in the repository, committed before any scoring.

## Amendments

1. **The shape test's rasteriser (20 October 2026, after stage 0 was scored and before any Study 2 row; confirmed by the owner the same day: "Yes: I confirm Amendment 1").**
   - **Why.** `shape_alike_letters` drew its glyphs with Pillow. `tests/test_render.py` forbids that anywhere under `src/` (Section 1A.11, checks 3 and 4: no text drawn with Pillow or hinted FreeType), and Pillow hints outlines that no page ever hinted. The full test suite found it; the commit that added R4 had run only the scoring tests.
   - **Change.** `_glyph` now rasterises with `mrr.render.load_glyph`, the rasteriser of every page, so the test measures the shapes the readers saw. The threshold (0.6), the size (48 px), the scaling and the shifts are unchanged.
   - **Effect.** Source Sans 3's set is unchanged (c o s u v x z), so no stage-0 row's score changes. Fira Mono's set gains q (0.603), and no Fira Mono row had been scored. R4's table above gives the new overlaps.
