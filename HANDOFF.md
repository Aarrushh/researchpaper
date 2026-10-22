<!--
PURPOSE: A short state summary to paste into a new AI chat session or hand to your co-author.
WHAT GOES HERE: Thesis, current step, frozen tags, what is broken, next step, top open decisions.
FILLED IN AT STEP: Updated at the end of every working session, from W1.1 onward.
DO NOT: Exceed 400 words. Do not paste results that are not in results/final/.
-->

# Handoff

> Paste this at the start of any new AI chat session. Update at the end of every working session.
> Last updated: Thursday 22 October 2026, 12:45 CEST, by Claude (D22).

## Thesis (one sentence)

A vision-language model's character error rate on text whose glyphs share a phase against the encoder's patch grid varies periodically with that phase, at the patch or the token-cell period, and rendering glyphs on the grid lowers it (H1, `PREREGISTRATION.md` section 1). Frozen result: clauses (a) and (b) supported for all three readers, mostly at the patch period; clause (c) reversed, for Qwen and SmolVLM-2.2B mostly through spaces inserted beside two-slot characters (exploratory). The paper reports the mechanisms, Study 2's preregistered tests and a calibration method shown on five readers (`ANALYSIS_PLAN_3.md` section 0).

## Current week and step

Data and analysis are frozen (`analysis-freeze-v2`). The authors write every sentence (D22); every claim names its table and status (`ANALYSIS_PLAN_3.md` section 5, item 6). The outline was settled on 22 October: headline C, "an honest map" (`paper/main.tex` header). Each `paper/sections/` file holds its paragraph jobs, evidence and evening as comments. Deadline: Friday 30 October.

## What is frozen (tags)

- `prereg-v1`: 3223ba91, approved 15 October 15:20
- `data-freeze-v1`: d25bf9b1, 17 October, signed off by the owner
- `spec-v1`: 1d633f69, signed off by the owner
- `prereg-v2`: badfe672, Study 2, 19 October
- `data-freeze-v2`: ad903d94, 20 October: all run folders, 44,989 files hashed in `results/final/`
- `data-freeze-s3-stage0`: ad209d97, 20 October: stage 0's tables
- `analysis-freeze-v1`: 4987c291, 21 October: every table and figure needing no GPU regenerates byte for byte (`make freeze-check-all`)
- `analysis-freeze-v2`: 265489ea, 21 October: the same, plus Amendment 2's load check (`reports/tables/load/`, exploratory, post hoc)
- `submission`: not yet
- All tags are local: nothing is pushed until the owner says so.

## What is broken

Nothing known. The CUDA sysmem fallback is still on (owner action 2).

## Next step

The introduction, evening 2 (`paper/sections/01_introduction.tex`). Numbers come only from `reports/tables/` and `reports/figures/`; a changed table needs `make freeze-check-all` and a new tag.

## Top three open decisions

1. D29's wording: H2 was not run although D16 said go; section 8 reports it as a deviation.
2. Which figures stay: Figures 0 and 2 may be cut, Figures 4 and 6 moved to an appendix.
3. When to push the tags, and how reviewers check them anonymously (`04_method.tex` P6).
