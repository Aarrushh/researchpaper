<!--
PURPOSE: A short state summary to paste into a new AI chat session or hand to your co-author.
WHAT GOES HERE: Thesis, current step, frozen tags, what is broken, next step, top open decisions.
FILLED IN AT STEP: Updated at the end of every working session, from W1.1 onward.
DO NOT: Exceed 400 words. Do not paste results that are not in results/final/.
-->

# Handoff

> Paste this at the start of any new AI chat session. Update at the end of every working session.
> Last updated: Wednesday 21 October 2026, 09:30 CEST, by Claude (D22).

## Thesis (one sentence)

A vision-language model's character error rate on text whose glyphs share a phase against the encoder's patch grid varies periodically with that phase, at the patch or the token-cell period, and rendering glyphs on the grid lowers it (H1, `PREREGISTRATION.md` section 1). Frozen result: clauses (a) and (b) supported for all three readers, mostly at the patch period; clause (c) reversed, for Qwen and SmolVLM-2.2B mostly through spaces inserted beside two-slot characters (exploratory). The paper reports the mechanisms, Study 2's preregistered tests and a calibration method shown on five readers (`ANALYSIS_PLAN_3.md` section 0).

## Current week and step

Data collection and analysis are finished and frozen (`analysis-freeze-v2`, which adds Amendment 2's load check to `analysis-freeze-v1`). The owner writes the paper from 21 October: `ANALYSIS_PLAN_3.md` section 5, item 6, every claim naming its table and its status.

## What is frozen (tags)

- `prereg-v1`: 3223ba91, approved 15 October 15:20
- `data-freeze-v1`: d25bf9b1, Saturday 17 October, signed off by the owner
- `spec-v1`: 1d633f69, signed off by the owner
- `prereg-v2`: badfe672, Study 2, 19 October
- `data-freeze-v2`: ad903d94, 20 October: every run folder (Study 1, Study 2, stage 0), 44,989 files hashed in `results/final/`
- `data-freeze-s3-stage0`: ad209d97, 20 October: stage 0's tables
- `analysis-freeze-v1`: 4987c291, 21 October: every table and figure that needs no GPU regenerates byte for byte from `results/final/` (`make freeze-check-all`; `LAB_NOTEBOOK.md`, 21 October)
- `analysis-freeze-v2`: the commit that adds this line, 21 October: the same, with the load check of `ANALYSIS_PLAN_3.md` Amendment 2 (`reports/tables/load/`, exploratory, post hoc)
- `submission`: not yet
- Every tag is local only: nothing is pushed until the owner says so.

## What is broken

Nothing known. The CUDA sysmem fallback is still on (owner action 2).

## Next step

The paper. Its numbers come from `reports/tables/` and `reports/figures/` only; `ANALYSIS_PLAN_3.md` says which table answers what, and whether it is confirmatory or exploratory. A changed table needs `make freeze-check-all` and a new tag.

## Top three open decisions

1. How the paper words clause (c): reversed under the preregistered CER, with the inserted spaces as an exploratory explanation; SmolVLM-500M's random-text tests reported with their detectable amplitudes.
2. How the paper reports H2: preregistered in `PREREGISTRATION.md` and never run (no attempt in `h2/kernels/`).
3. When to push the local tags to GitHub.
