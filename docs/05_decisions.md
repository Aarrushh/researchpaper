<!--
PURPOSE: Architecture-decision-record style log of every design and gate decision.
WHAT GOES HERE: One entry per decision with evidence and consequences.
FILLED IN AT STEP: G1, G2, G3, and any time a plan changes.
DO NOT: Delete or rewrite old entries. Supersede them with a new entry that references the old ID.
-->

# Decision log

## Template

```
### D<nn>. <title>
- ID: D<nn>
- Status: open | decided | superseded by D<nn>
- Date: YYYY-MM-DD
- Decision:
- Options considered:
- Evidence:
- Consequences:
```

---

### D1. Two readers with different grids
- ID: D1
- Status: superseded by D7
- Date: 2026-09-27 (recorded at scaffold time; decided before it)
- Decision: Use two VLM readers whose encoders have different token cell sizes (Qwen2.5-VL-3B-Instruct and SmolVLM-2.2B).
- Options considered: one reader; two readers with the same grid; two readers with different grids.
- Evidence: If error periodicity tracks each reader's own cell size, the effect can be attributed to the grid rather than to the font or text.
- Consequences: SmolVLM cell size must be confirmed at W1.6. Doubles compute for the full sweep.

### D2. Transcription CER as primary metric rather than QA
- ID: D2
- Status: decided
- Date: 2026-09-27 (recorded at scaffold time; decided before it)
- Decision: Primary metric is character error rate of a full transcription, computed with jiwer. WER is secondary.
- Options considered: question answering accuracy; transcription CER; transcription WER.
- Evidence: QA scores overstate text fidelity (ZeroSense, arXiv 2608.01848). TODO[W1.2]: verify this claim and log it in references/citations.md.
- Consequences: Outputs are long; `max_new_tokens` must cover a full page. Text normalisation must be fixed in the preregistration.

### D3. Render at processor-native sizes
- ID: D3
- Status: decided
- Date: 2026-09-27 (recorded at scaffold time; decided before it)
- Decision: Render every page at a size the processor will not resize, so the phase we set is the phase the encoder sees.
- Options considered: render at arbitrary size and let the processor resize; render at native size.
- Evidence: Qwen2.5-VL rounds image sides to multiples of 28, which would move glyphs relative to the grid.
- Consequences: Page sizes depend on the reader's cell size. `mrr.grid.native_size` must be verified at W1.6.

### D4. G1 outcome
- ID: D4
- Status: open
- Date: TODO[G1]: date of G1 decision
- Decision: TODO[G1]: go or no-go at G1
- Options considered: go; no-go.
- Evidence: TODO[G1]: token count check, overlay images, transcription test
- Consequences: TODO[G1]: what happens next

### D5. G2 outcome
- ID: D5
- Status: open
- Date: TODO[W2.5]: date of G2 decision
- Decision: TODO[W2.5]: periodic or not
- Options considered: go (periodic); no-go (negative-result framing).
- Evidence: TODO[W2.5]: Figure 1 and the preregistered periodicity test
- Consequences: TODO[W2.5]: framing for W3 onward

### D6. G3 outcome
- ID: D6
- Status: open
- Date: TODO[W4.4]: date of G3 decision
- Decision: TODO[W4.4]: kernel matches or not
- Options considered: go (matches); no-go (documented failure).
- Evidence: TODO[W4.4]: judge table from h2/judge/compare.py
- Consequences: TODO[W4.4]: how H2 is reported

### D7. Three readers: two share the 14 px patch, one has a 16 px patch
- ID: D7
- Status: decided
- Date: 2026-10-13 (decided at W1.6)
- Decision: Read with Qwen2.5-VL-3B-Instruct (patch 14 px, token cell 28 px, attention windows 112 px), SmolVLM-Instruct, the 2.2B model (patch 14 px, token cell 42 px), and SmolVLM-500M-Instruct (patch 16 px, token cell 64 px), each at the revision pinned in `configs/readers.yaml`.
- Options considered: the two readers of D1; those two plus SmolVLM-500M; SmolVLM-500M in place of SmolVLM-2.2B.
- Evidence: The `config.json` of each repository at its pinned revision. Qwen and SmolVLM-2.2B both cut 14 px patches, so with D1's two readers a 14 px period cannot separate the patch embedding from the merge. A third reader with a 16 px patch can: if the dominant period reads 14, 14 and 16 px, the patch embedding drives the effect; if it reads 28, 42 and 64 px, the merge does. `make readers` confirms the token counts at W1.6. It did on 13 October: 256, 81 and 64 vision tokens, each equal to `grid.token_count`, from 32 × 32 patches of 14 px, 27 × 27 patches of 14 px and 32 × 32 patches of 16 px (`LAB_NOTEBOOK.md`, W1.6).
- Consequences: A third entry in `configs/readers.yaml`. When this entry is decided, D1's status becomes "superseded by D7". SmolVLM-500M is the third item in the cut order.

### D8. Each reader's page is its native frame
- ID: D8
- Status: decided
- Date: 2026-10-13 (decided at W1.8)
- Decision: Render 448 × 448 pages for Qwen, 384 × 384 for SmolVLM-2.2B and 512 × 512 for SmolVLM-500M, and process them with `do_resize=False`.
- Options considered: the consequence of D3, page sides that are multiples of the reader's token cell; each reader's native frame.
- Evidence: 448 is a fixed point of Qwen's `smart_resize`, while 384 becomes 392 and 512 becomes 504. 384 is SmolVLM-2.2B's frame but is not a multiple of its 42 px cell, and its 27 × 27 patches cover only the first 378 px, so columns and rows 378 to 383 are never seen. `tests/test_no_resize.py` and `tests/test_reader_facts.py` check both at W1.8. Both passed on 13 October: undoing each processor gives back the rendered bytes, and noise in columns or rows 378 to 383 of a SmolVLM-2.2B page leaves the outputs of the vision encoder and the connector bit-identical (`LAB_NOTEBOOK.md`, W1.8).
- Consequences: D3's decision stands; its consequence about page sizes is superseded. The DO NOT line of `configs/render_defaults.yaml` changes. In `docs/04_data_dictionary.md`, `image_width_px` and `image_height_px` become "the reader's native page size". `grid.page_size_px` replaces `grid.native_size`.

### D9. Phases are recorded over each reader's longest period
- ID: D9
- Status: open
- Date: 2026-10-12 (opened; to be decided at W1.9)
- Decision: The sweep period is lcm(patch, cell, window): 112 px for Qwen, 42 px for SmolVLM-2.2B and 64 px for SmolVLM-500M. Every phase is recorded in [0, period), and each page leaves room for a shift of period − 1 px. Each block of the sweep samples the part of this range in which its layout's glyphs share a phase (D10).
- Options considered: phases over one token cell, as the scaffold has it; phases over the sweep period.
- Evidence: Qwen uses window attention in 28 of its 32 vision blocks, with full attention in blocks 7, 15, 23 and 31 and a window of 112 px (Hub config at the pinned revision). At layer 0 each reader's tokens are exactly equivariant for shifts that are whole multiples of the patch (learning notes A7). After Qwen's first block, exact equivariance is expected only at multiples of 112 px (inference in the notes, A7); paper Figure 0 tests this at W1.9.
- Consequences: In `docs/04_data_dictionary.md`, `phase_x_px` and `phase_y_px` become "0 to the reader's sweep period (exclusive)".

### D10. H1 is tested on layouts whose glyphs share a phase
- ID: D10
- Status: open
- Date: 2026-10-12 (opened; to be decided at Gate 1)
- Decision: Proposed: test H1 on two blocks whose glyphs share a phase. Block V is a vertical sweep of conventional text whose line pitch equals the token cell, so every line sits at the same vertical phase. Block H is a horizontal sweep of the patch-aware layout, one glyph per patch-wide slot (a' = P·ceil(a/P), centred at a whole-pixel offset). Block C, a horizontal sweep of conventional text, stays as the control that is predicted to be flat. H1 is restated in `PREREGISTRATION.md` as a claim about coherently placed text, with the prediction for C beside it. Every run records each page's coherence R_k, and the paper reports the density cost of every coherent layout.
- Options considered: (1) coherent blocks V and H with control C (proposed); (2) conventional sweeps only, as first planned, whose predicted result is a flat curve to be reported as phase averaging; (3) word starts on patch edges, giving partial coherence (R1 0.102) at a small density cost (30.5 glyphs per line instead of 35.7).
- Evidence: A block shift moves every glyph by the same amount, but in conventional text the glyphs' positions relative to the patch grid are already spread almost evenly. The page-level amplitude at harmonic k of period T is predicted to scale with the coherence R_k(T) = |mean over glyphs j of exp(2πi·k·c_j/T)|, where c_j is glyph j's ink centre. On 2,500 glyphs of Persuasion in Source Sans 3 at 16 px, on 308 px lines with P = 14 px: R1(14) is 0.022 for the conventional layout, close to the 0.020 of random phases, and R1 0.986 and R2 0.947 for the patch-aware layout. As a conventional block shifts, the share of glyphs that straddle a patch edge stays between 0.38 and 0.43; for the patch-aware layout it swings from none to all. `scripts/05_coherence.py` reproduces these values at W1.10.
- Consequences: The sweep configs hold blocks V, H and C, with S, A and Z beside them. H1 is restated in `PREREGISTRATION.md`. Every run folder holds `coherence.csv`.

### D11. The renderer contract
- ID: D11
- Status: open
- Date: 2026-10-12 (opened; to be decided at W1.5)
- Decision: Unhinted FreeType 2.13.2 outlines with linear advances. Phase in 1/64 px, applied to the outline. One page-sized coverage canvas, with coverage added and clipped at 1. Composite once, with the equations in `composite_coverage` (`src/mrr/render.py`). Round once, with floor(255·clip(o, 0, 1) + 0.5). The binary mode thresholds at c >= 0.5. Glyph bitmaps are placed at whole-pixel offsets. No kerning, ligatures, shaping, justification or hyphenation. The anti-aliasing baseline is `coverage_srgb` (proposed).
- Options considered: anti-aliasing baseline `coverage_srgb` (proposed) or the scaffold's `coverage_linear`.
- Evidence: Learning notes A4 to A7; `tests/test_color.py` and `tests/test_render.py` at W1.5.
- Consequences: Every rule becomes a rule in `spec/RENDERING_SPEC.md` (W4.1). The H2 kernel implements the accumulate and composite stages.

### D12. The sRGB encode threshold is 0.04045/12.92
- ID: D12
- Status: superseded by D24
- Date: 2026-10-12 (opened; to be decided at W1.5)
- Decision: Encode linear light with the threshold 0.04045/12.92 = 0.0031308049535603713, matched to the decode threshold 0.04045, and compute it rather than write it as a literal.
- Options considered: the rounded threshold 0.0031308 in the scaffold's docstring; the matched threshold 0.04045/12.92.
- Evidence: With the matched threshold, the round-trip test in `tests/test_color.py` (100,001 points at an absolute tolerance of 1e-12) passes with a largest error of 1.7e-16; with 0.0031308 the error reaches 3.0e-8 and the test fails. The sample code in W3C CSS Color 4 uses the matched pair (learning notes A4).
- Consequences: The docstring and the DO NOT line of `src/mrr/color.py` change. The specification cites the threshold.

### D13. How transcriptions are scored
- ID: D13
- Status: open
- Date: 2026-10-12 (opened; to be decided at W2.1)
- Decision: CER is primary and WER secondary, as in D2. Reference and transcription both pass through `normalise_text`: NFC; curly quotes become straight ones and the em dash a hyphen; whitespace is collapsed. There is no case folding and no punctuation removal. Each row stores the four jiwer counts. Any summary over pages pools the counts (total errors over total reference characters) instead of averaging page CERs. Random pages are scored the same way. A transcription that stops at `max_new_tokens` is kept and flagged.
- Options considered: the mean of page CERs or pooled counts; case-folded or exact comparison; dropping or keeping transcriptions that stop at the token limit.
- Evidence: The pooling, case and empty-reference behaviour of jiwer 4.0.0, measured for the build guide.
- Consequences: `src/mrr/score.py`; sections 4 and 8 of `PREREGISTRATION.md`. D2's TODO[W1.2] about ZeroSense stays with D2, and D2 is superseded only if its claim does not hold.

### D14. The statistical model
- ID: D14
- Status: open
- Date: 2026-10-12 (opened; to be decided at W2.1)
- Decision: For each reader and each block, an ordinary least-squares fit of row CER on page fixed effects plus cosine and sine terms for the harmonics of each period group, with the number of harmonics per group fixed before the pilot. Each group is tested with an F-test against the model without it, and the groups of one reader are corrected with Holm's method at the preregistered α. Harmonic columns that are identically zero at the sampled phases are dropped. Amplitudes get page-bootstrap intervals. The paired comparison of conventional and patch-aware pages is a Wilcoxon signed-rank test over pages: each page's CER under H at phase 0 against its CER pooled over the levels of C. Because every level reads the same reference, the pooled CER equals the mean of the page's C rows. The test reports the median paired difference with its bootstrap interval, and Holm's method runs across the three readers.
- Options considered: Wilcoxon tests on every pair of phases (the plan of 25 September); one harmonic regression per reader and block, with a period test per group.
- Evidence: Learning notes A7, equations (11) and (12); SciPy 1.17.1.
- Consequences: `src/mrr/stats.py`; sections 6, 7 and 9 of `PREREGISTRATION.md`.

### D15. SmolVLM position ids
- ID: D15
- Status: open
- Date: 2026-10-12 (opened; to be decided at W1.8)
- Decision: Replace SmolVLM's position ids with 0 to n − 1 through a forward pre-hook on the position table, or run the vision tower in float32. The position-id test in `tests/test_reader_facts.py` decides.
- Options considered: a forward pre-hook on the position table; a float32 vision tower; no change.
- Evidence: In bfloat16, one 384 px frame gets 441 distinct position ids instead of 729 (learning notes A2).
- Consequences: `src/mrr/readers.py` installs the fix in `load_reader`. The equivariance runs (W1.9) use the float32 tower anyway.

### D16. H2 runs only if the data are frozen by 21 October
- ID: D16
- Status: open
- Date: 2026-10-12 (opened; to be decided on 2026-10-21, or a day later under D21's slip rule)
- Decision: If `data-freeze-v1` is tagged by Wednesday 21 October, run H2 in full. Otherwise write the specification (W4.1), which states the paper's rendering rules, skip W4.2 to W4.4, and report H2 as not run.
- Options considered: run H2 whatever the date; run it only if the data are frozen by 21 October; drop it.
- Evidence: The guide of 6 October set this cut; H2 costs about three days.
- Consequences: Gate 3 tags `spec-v1` either way.

### D17. Tests that need models are marked and kept out of CI
- ID: D17
- Status: decided
- Date: 2026-10-12 (decided at W1.3)
- Decision: A pytest marker `model` for tests that need torch, transformers or downloaded weights. CI and `make test` run `-m "not todo and not model"`; `make test-models` runs the marked tests on the laptop.
- Options considered: run the model tests in CI; skip them silently when torch is missing; mark them and run them on the laptop.
- Evidence: The DO NOT line of `.github/workflows/tests.yml` forbids installing torch or downloading models.
- Consequences: The marker in `pyproject.toml`, the Makefile and the CI workflow. The instruction to run the no-resize test in CI is met by pasting the laptop results into `LAB_NOTEBOOK.md`.

### D18. Where Qwen's weights live
- ID: D18
- Status: open
- Date: 2026-10-12 (opened; to be decided at W1.7)
- Decision: One of: the whole model on the GPU in bfloat16; accelerate's offload with `max_memory={0: "6GiB", "cpu": "24GiB"}`; or the official AWQ build as a separate reader, `qwen_awq`. The first placement whose peak memory fits is chosen.
- Options considered: the three placements above.
- Evidence: The peak-memory measurement on one page, recorded in `LAB_NOTEBOOK.md`.
- Consequences: The placement goes in every run manifest. A 4-bit language model is a different reader and is named as one.

### D19. The results schema has 33 columns
- ID: D19
- Status: open
- Date: 2026-10-12 (opened; to be decided at W2.3)
- Decision: `RESULTS_COLUMNS` grows from the scaffold's 24 columns to 33, adding `text_type`, `block`, `condition_hash`, `output_tokens`, `hit_limit` and the four jiwer counts `hits`, `substitutions`, `deletions` and `insertions`.
- Options considered: keep 24 columns and recover the block, the text type and the counts at analysis time from page ids, configs and transcriptions; store them in every row.
- Evidence: D13 pools the jiwer counts over pages, so every row needs its counts. The analysis groups rows by block and text type. D13 keeps and reports the rows that stop at `max_new_tokens`, which needs `hit_limit`.
- Consequences: `RESULTS_COLUMNS` in `src/mrr/manifest.py` and the table in `docs/04_data_dictionary.md` change in one commit, as the DO NOT lines of both files require. The condition hash moves from `scripts/10_run_sweep.py` to `sweep.condition_hash`, and the data dictionary says so.

### D20. One laptop runs all three readers
- ID: D20
- Status: decided
- Date: 2026-10-12
- Decision: Every reader, test and run uses one laptop (an RTX 5060 Laptop GPU with 8 GB and 32 GB of RAM), serially. Wherever the plan says "on both laptops", the check runs once, on this laptop.
- Options considered: two laptops, one running the SmolVLM readers and the other Qwen; one laptop for everything.
- Evidence: The owner's choice on 12 October. The laptop reports 8,151 MiB of GPU memory and 31.7 GB of RAM.
- Consequences: Every budget is a serial sum over the readers. The equivariance runs (W1.9) come before any Qwen transcription, and one reader is loaded at a time.

### D21. The calendar is re-planned from Monday 12 October
- ID: D21
- Status: decided
- Date: 2026-10-12
- Decision: Gate 1 on Wednesday 14 October at 12:00; tag `prereg-v1` by 18:00 that day; the pilot that night. Gate 2 on Friday 16 October at 12:00; the full run from then until Wednesday 21 October at 06:00; the freeze and D16 on Wednesday 21 October. H2 on Thursday 22 and Friday 23 October; Gate 3 on Saturday 24 October; submission on Wednesday 28 October. Thursday 29 and Friday 30 October are buffer, and the deadline is Friday 30 October, 19:00 UTC. If Gate 1 has not passed by Wednesday 14 October at 12:00, every later date moves by one day, D16 included, and the cut order applies.
- Options considered: keep the calendar of 6 October; re-plan with a one-day slip rule; re-plan with no slack.
- Evidence: The build started on Monday 12 October, the date the earlier calendar set for Gate 1.
- Consequences: The gate dates in `docs/03_pipeline.md` change. The cut order is unchanged: the stretch analyses, then H2, then SmolVLM-500M, then the anti-aliasing factor. Never cut: the no-resize test, the equivariance measurement, the random-text condition and the harmonic model. Each cut is recorded as a decision.

### D22. Who writes what, and who signs
- ID: D22
- Status: decided
- Date: 2026-10-12
- Decision: Claude (Anthropic's Claude Opus 5.5, as a coding agent) writes the code, tests, configs, CI, scripts, the documents in `docs/`, the decision entries, `LAB_NOTEBOOK.md` entries, `spec/RENDERING_SPEC.md`, the draft of `PREREGISTRATION.md`, the LaTeX setup, the generated tables and figures, `references/references.bib`, `references/citations.md` and `paper/number_audit.csv`. The authors write every sentence in `paper/`, captions included. Judgement calls are put to the owner and wait for a yes. The owner's approval stands for both authors.
- Options considered: the authors write the code with an assistant's help; Claude writes everything except the paper's prose.
- Evidence: The owner's decision on 12 October. The authors write the paper themselves under the MLSys 2027 policy on LLM use, which is reread before submission (W5.4).
- Consequences: The paper discloses Claude's role. Because Claude writes the specification, H2 tests GLM-5.3 implementing a specification written by another language model; the paper states this. Claude explains each scientific choice to the owner as it is made.

### D23. Who judges `pixel_work_in_triton`
- ID: D23
- Status: decided
- Date: 2026-10-12
- Decision: For each H2 attempt, Claude Opus 5.5 and a fresh Claude Sonnet 5.5 agent each review the kernel, blind to each other's verdict, with the flags of `static_check`. The verdict is "no" if they disagree. The owner gives the final go before the judge runs. This replaces review by both authors independently.
- Options considered: both authors, independently; two language-model reviewers, blind to each other, with the owner's final go; the owner alone.
- Evidence: The owner's decision on 12 October.
- Consequences: `h2/judge/reviews.csv` records both verdicts and the owner's go. The paper discloses that language-model reviewers judged this field.

### D24. The sRGB encode threshold is 0.04045/12.92 (supersedes D12)
- ID: D24
- Status: decided
- Date: 2026-10-12
- Decision: As D12. Encode linear light on the straight piece when l <= 0.04045/12.92 = 0.0031308049535603713, where the decode's straight piece ends, and compute the threshold in code rather than write it as a literal. D24 replaces D12 only because D12's evidence named a source that does not use this threshold.
- Options considered: As D12: the rounded threshold 0.0031308; the matched threshold 0.04045/12.92.
- Evidence: Measured at W1.5. With the matched threshold, the round-trip test in `tests/test_color.py` (100,001 points, absolute tolerance 1e-12) passes with a largest error of 1.7e-16; with 0.0031308 the largest error is 3.0e-8, at v = 0.04045, and the test fails. Every value the decode's straight piece produces is at most 0.04045/12.92, and every value its power piece produces is above it (the power piece starts 2.3e-9 higher), so the encode applies the inverse of the piece that made each value. D12 also said that the sample code in W3C CSS Color 4 uses the matched pair. It does not: its `gam_sRGB` uses 0.0031308 (Candidate Recommendation Draft of 9 October 2026, section 19, read on 12 October 2026). That code is a source for the constants 0.04045, 12.92, 0.055, 1.055 and 2.4, not for this threshold.
- Consequences: As D12: the docstring and DO NOT line of `src/mrr/color.py` change, and the specification (W4.1) states the threshold as our choice, with the round-trip test as its reason. `references/citations.md` cites CSS Color 4 for the constants only. Section 1A.1 of the build guide makes D12's claim too; `LAB_NOTEBOOK.md` records the error.
