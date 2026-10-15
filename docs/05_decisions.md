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
- Note (2026-10-14): G1 had not passed by Wed 14 Oct 12:00 CEST; the slip rule applied (D25). G1 is now Thu 15 Oct 12:00.

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
- Status: decided
- Date: 2026-10-14 (decided at W1.9)
- Decision: The sweep period is lcm(patch, cell, window): 112 px for Qwen, 42 px for SmolVLM-2.2B and 64 px for SmolVLM-500M. Every phase is recorded in [0, period), and each page leaves room for a shift of period − 1 px. Each block of the sweep samples the part of this range in which its layout's glyphs share a phase (D10).
- Options considered: phases over one token cell, as the scaffold has it; phases over the sweep period.
- Evidence: Qwen uses window attention in 28 of its 32 vision blocks, with full attention in blocks 7, 15, 23 and 31 and a window of 112 px (Hub config at the pinned revision). At layer 0 each reader's tokens are exactly equivariant for shifts that are whole multiples of the patch (learning notes A7). After Qwen's first block, exact equivariance is expected only at multiples of 112 px (inference in the notes, A7); paper Figure 0 tests this at W1.9. Figure 0 did so on 14 October (W1.9; `results/equivariance/`, `LAB_NOTEBOOK.md`): at layer 0 every reader is exact at multiples of its patch and nowhere else, with E(0.25) and the plateau inside appendix C.3's ranges. After Qwen's block 0 only moves of 112 and 224 px are exact (1.5e-7); the other multiples of 14 px cost 0.04; and the 112 px dip survives to the output (0.3 against a plateau of 0.63) while the 14 px dip is gone by layer 32 (ratio 0.99 to its neighbours). Beyond the embedding the window is the period that matters, so the 112 px sweep period stands. Both SmolVLM towers show minima, not zeros, at multiples of P beyond layer 0 (ratios 0.90 and 0.45 at layer 1, fading with depth), as predicted for learned absolute positions. Their 42 and 64 px cell periods come from the pixel shuffle after the tower, which Figure 0 cannot see, so they rest on the structural argument above.
- Consequences: In `docs/04_data_dictionary.md`, `phase_x_px` and `phase_y_px` become "0 to the reader's sweep period (exclusive)".

### D10. H1 is tested on layouts whose glyphs share a phase
- ID: D10
- Status: decided
- Date: 2026-10-12 (opened); 2026-10-15 (decided by the owner with the Gate 1 batch: "Yes to all as proposed")
- Decision: Proposed: test H1 on two blocks whose glyphs share a phase. Block V is a vertical sweep of conventional text whose line pitch equals the token cell, so every line sits at the same vertical phase. Block H is a horizontal sweep of the patch-aware layout, one glyph per patch-wide slot (a' = P·ceil(a/P), centred at a whole-pixel offset). Block C, a horizontal sweep of conventional text, stays as the control that is predicted to be flat. H1 is restated in `PREREGISTRATION.md` as a claim about coherently placed text, with the prediction for C beside it. Every run records each page's coherence R_k, and the paper reports the density cost of every coherent layout.
- Options considered: (1) coherent blocks V and H with control C (proposed); (2) conventional sweeps only, as first planned, whose predicted result is a flat curve to be reported as phase averaging; (3) word starts on patch edges, giving partial coherence (R1 0.102) at a small density cost (30.5 glyphs per line instead of 35.7).
- Evidence: A block shift moves every glyph by the same amount, but in conventional text the glyphs' positions relative to the patch grid are already spread almost evenly. The page-level amplitude at harmonic k of period T is predicted to scale with the coherence R_k(T) = |mean over glyphs j of exp(2πi·k·c_j/T)|, where c_j is glyph j's ink centre. On 2,500 glyphs of Persuasion in Source Sans 3 at 16 px, on 308 px lines with P = 14 px: R1(14) is 0.022 for the conventional layout, close to the 0.020 of random phases, and R1 0.986 and R2 0.947 for the patch-aware layout. As a conventional block shifts, the share of glyphs that straddle a patch edge stays between 0.38 and 0.43; for the patch-aware layout it swings from none to all. `scripts/05_coherence.py` reproduces these values at W1.10. W1.10 (14 October) reproduced the table of build guide section 2.2 with the repository's layout code: conventional R₁(14) 0.022 (the 1/√N floor for 2,500 glyphs is 0.020), per-class 0.081, straddle share 0.38 to 0.43 at every shift; patch_aware R₁(14) 0.986, per-class 1.000, straddle share 0 at shift 0 and 1 at shift 7 (`LAB_NOTEBOOK.md`, W1.10).
- Consequences: The sweep configs hold blocks V, H and C, with S, A and Z beside them. H1 is restated in `PREREGISTRATION.md`. Every run folder holds `coherence.csv`.

### D11. The renderer contract
- ID: D11
- Status: decided
- Date: 2026-10-12 (opened); 2026-10-15 (decided by the owner with the Gate 1 batch, `coverage_srgb` as the baseline)
- Decision: Unhinted FreeType 2.13.2 outlines with linear advances. Phase in 1/64 px, applied to the outline. One page-sized coverage canvas, with coverage added and clipped at 1. Composite once, with the equations in `composite_coverage` (`src/mrr/render.py`). Round once, with floor(255·clip(o, 0, 1) + 0.5). The binary mode thresholds at c >= 0.5. Glyph bitmaps are placed at whole-pixel offsets. No kerning, ligatures, shaping, justification or hyphenation. The anti-aliasing baseline is `coverage_srgb`.
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
- Status: decided
- Date: 2026-10-14 (decided at W1.8 and W1.9)
- Decision: Both, by where the tower runs. The bfloat16 readers get the forward pre-hook of section 1B.5 on the position table in `load_reader`, which replaces the ids with 0 to n − 1 (W1.8). The equivariance measurement runs each vision tower alone in float32, where the table receives every id without a fix, so `load_vision_tower` installs nothing unless its own count falls short (W1.9).
- Options considered: a forward pre-hook on the position table; a float32 vision tower; no change.
- Evidence: In bfloat16, one 384 px frame gets 441 distinct position ids instead of 729 (learning notes A2). W1.8 (13 October): `tests/test_reader_facts.py` checks the 729 and 1,024 ids through the hook. W1.9 (14 October): `test_float32_smolvlm_tower_sees_every_position_id` sees 729 and 1,024 distinct ids in the float32 towers with no hook, and the run log records `position-id fix added: False` for both.
- Consequences: `src/mrr/readers.py` installs the fix in `load_reader` for the bfloat16 readers. `load_vision_tower` counts the distinct ids on a blank native page and installs the fix only if the count is short; `VisionTower.position_ids_fixed` records whether it did.

### D16. H2 runs only if the data are frozen by 21 October
- ID: D16
- Status: open
- Date: 2026-10-12 (opened; to be decided on 2026-10-21, or a day later under D21's slip rule)
- Decision: If `data-freeze-v1` is tagged by Wednesday 21 October, run H2 in full. Otherwise write the specification (W4.1), which states the paper's rendering rules, skip W4.2 to W4.4, and report H2 as not run.
- Options considered: run H2 whatever the date; run it only if the data are frozen by 21 October; drop it.
- Evidence: The guide of 6 October set this cut; H2 costs about three days.
- Consequences: Gate 3 tags `spec-v1` either way.
- Note (2026-10-14): the slip rule fired (D25), so the day later applies: the cut-off is Thursday 22 October.

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
- Status: decided
- Date: 2026-10-12 (opened); 2026-10-15 (decided: rung 1, with the owner's yes to the Gate 1 batch and the ladder's measurement)
- Decision: One of: the whole model on the GPU in bfloat16; accelerate's offload with `max_memory={0: "6GiB", "cpu": "24GiB"}`; or the official AWQ build as a separate reader, `qwen_awq`. The first placement whose peak memory fits is chosen.
- Options considered: the three placements above.
- Evidence: The peak-memory measurement on one page, recorded in `LAB_NOTEBOOK.md`. W1.7: 7.07 GiB allocated, 7,423 MiB on `nvidia-smi`. The ladder (15 October, 72 pages up to 1.5 times the longest reference in new tokens): 7,544 MiB (7.37 GiB) at most, under the 7.5 GiB of build guide 1B.8, with steady latencies. Rung 1, the whole model on the GPU in bfloat16, holds. The sysmem fallback was still on during both measurements; `make check-env` confirms the setting when the owner changes it (owner action 2).
- Consequences: The placement goes in every run manifest. A 4-bit language model is a different reader and is named as one.

### D19. The results schema has 33 columns
- ID: D19
- Status: decided
- Date: 2026-10-12 (opened); 2026-10-14 (decided at W2.3)
- Decision: `RESULTS_COLUMNS` grows from the scaffold's 24 columns to 33, adding `text_type`, `block`, `condition_hash`, `output_tokens`, `hit_limit` and the four jiwer counts `hits`, `substitutions`, `deletions` and `insertions`.
- Options considered: keep 24 columns and recover the block, the text type and the counts at analysis time from page ids, configs and transcriptions; store them in every row.
- Evidence: D13 pools the jiwer counts over pages, so every row needs its counts. The analysis groups rows by block and text type. D13 keeps and reports the rows that stop at `max_new_tokens`, which needs `hit_limit`. Landed on 14 October: `tests/test_manifest.py` pins the 33 names and `tests/test_smoke.py` checks the data dictionary against them.
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

### D25. The calendar slips one day (the slip rule of D21)
- ID: D25
- Status: decided
- Date: 2026-10-14
- Decision: G1 had not passed by Wed 14 Oct 12:00 CEST, so, by the slip rule the owner approved with the plan on 12 October, every later date of D21 moves by one day, D16 included: G1 Thu 15 12:00; `prereg-v1` Thu 15 18:00; the pilot Thu 15 night; Phase 3 Fri 16; G2 Sat 17 12:00; the full run Sat 17 to Thu 22 06:00; `data-freeze-v1` and D16's cut-off Thu 22; H2's paid calls Fri 23 to Sat 24; G3 and `spec-v1` Sun 25; the paper Sun 25 to Thu 29; buffer Fri 30. The deadline, Fri 30 Oct 19:00 UTC, does not move. The cut order applies from the top: cut 1 is the stretch tasks of Phase 3 except P3S.1, and the plan's only stretch task is P3S.1, so cut 1 removes nothing; cut 2 would be H2, which D16 governs by date.
- Options considered: Apply the rule as written; hold the dates and compress Phase 3 (rejected: the rule exists so that this choice is never made under pressure).
- Evidence: At 12:00 on 14 October the ladder (P2.9) had not run, because the runner (P2.8) was not built and the GPU still had the sysmem fallback on (D18), and the Gate 1 decisions had not been put to the owner. `LAB_NOTEBOOK.md`, 14 October.
- Consequences: D4 stays open until G1 on Thu 15. D16's date moves to Thu 22 October. One day of buffer remains before the deadline; a second slip would take it.

### D26. The experiment finishes by Saturday 17 October: no separate pilot, one full run stopped by the clock
- ID: D26
- Status: decided
- Date: 2026-10-15
- Decision: The owner set a new target on 14 and 15 October: the experiment run, the data frozen and every result interpreted by Saturday 17 October, 23:59 CEST. To meet it:
  1. **No separate pilot.** `configs/sweeps/pilot.yaml` is never run. Its blocks V, H, C, S and A, unchanged, are the shared blocks of `configs/sweeps/full.yaml`, beside block Z.
  2. **The ladder runs on Thursday 15 afternoon** as section 2.9 describes, its target CER 0.15 unchanged.
  3. **One full run, stopped by the clock.** `full.yaml` sets each reader's page count to a ceiling the run cannot reach, and every reader runs with `--deadline`. The runner checks the deadline only between pages, so every page it reports is complete. Pages run in plan order: natural and random alternate, and the twelve block Z layouts take turns with the shared pages. The pages done when the clock stops are therefore a prefix of the plan, the same for every condition and balanced between the text types. The page count is not chosen by looking at any result.
  4. **Gate 2 is not held.** Nothing measured during the run changes the run. The first finished pages are analysed only to check the pipeline (`checks.csv`, `determinism.csv`, `hit_limit`), never to test H1, and the notebook records each such look.
  5. **Power is reported, not planned.** The preregistration's page-count rule (N from σ and λ_q at 80% power) is replaced by the smallest detectable amplitude at the pages obtained, s = σ √(λ_q / (N n)), and `stats.paired_mde` for clause (c). These are written into the paper beside each test.
  6. **The two SmolVLM readers share the GPU.** They run as two processes at the same time, each at batch size 1 with greedy decoding as before; Qwen runs alone. This amends D20's "serially" for these two readers only.
  7. **Calendar.** Ladder Thursday 15 afternoon; G1 (D4), the preregistration's approval and `prereg-v1` Thursday 15 evening; the full run from then until Saturday 17 about 14:00; `make analyze`, `make figures`, the freeze, `data-freeze-v1` and the interpretations by Saturday 23:59. D16's condition (frozen by Thursday 22) is then met, so H2 runs: its code is built on the CPU while the GPU runs, and each paid call waits for the owner's go-ahead (Friday 16 and Saturday 17). The paper's prose (owner action 15) and the deadline (Friday 30 October, 19:00 UTC) do not move.
- Options considered: keep the pilot, cut to about 12 h, with G2 on Friday at noon and a full run from Friday 13:00 to Saturday 14:00 (the guide's gates kept, but the pilot's pages are never pooled, so the analysed data hold about half as many pages); no pilot and one clock-stopped full run (chosen by the owner, 15 October); dropping blocks S and A (not needed: they are 6 of Qwen's 50 conditions per page).
- Evidence: The owner's messages: on 14 October, "we should aim to be done by this sunday at latest, with the entire repo and all the interpretations of all results and the data all frozen and the experiment completed"; on 15 October, "we have until saturday night to finish the experiment completely", with the clock-stopped plan chosen from two options. At W1.7's median latencies (70.8, 104.8 and 131.0 s a row, measured on battery with three sleeps) the pilot alone needed 95 h, and the pilot plus the planned full run about 200 h, against about 59 h from Thursday 13:00 to Saturday 24:00. The ladder's first rows on 15 October, on mains power with the Epic processes closed, wrote 178 tokens in 6.0 s on SmolVLM-2.2B (about 30 tokens per second), while the two SmolVLM processes kept the GPU 91% busy; in W1.7 one process kept it 6 to 29% busy.
- Consequences: `PREREGISTRATION.md` sections 3, 5, 7, 9 and 11 are rewritten before the tag (no pilot; the clock as the stopping rule; no G2; reported power). `full.yaml` gets the ladder's sizes, the page ceilings and `start_after: [configs/sweeps/ladder.yaml]` only. `pilot.yaml`'s header says it was not run. D5 (G2) is closed as not held. `docs/03_pipeline.md` marks W2.3's pilot step "not run (D26)". The tables and figures are written under `reports/tables/full/` and `reports/figures/`.

