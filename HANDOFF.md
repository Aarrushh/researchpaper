<!--
PURPOSE: A short state summary to paste into a new AI chat session or hand to your co-author.
WHAT GOES HERE: Thesis, current step, frozen tags, what is broken, next step, top open decisions.
FILLED IN AT STEP: Updated at the end of every working session, from W1.1 onward.
DO NOT: Exceed 400 words. Do not paste results that are not in results/final/.
-->

# Handoff

> Paste this at the start of any new AI chat session. Update at the end of every working session.
> Last updated: Monday 19 October 2026, 12:15 CEST, by Claude (D22).

## Thesis (one sentence)

A vision-language model's character error rate on text whose glyphs share a phase against the encoder's patch grid varies periodically with that phase, at the patch or the token-cell period, and rendering glyphs on the grid lowers it (H1, `PREREGISTRATION.md` section 1). Frozen result: clauses (a) and (b) supported for all three readers, mostly at the patch period; clause (c) reversed, for Qwen and SmolVLM-2.2B mostly through spaces inserted beside two-slot characters (exploratory).

## Current week and step

W5.1: Study 2 (`PREREGISTRATION_2.md`): Fira Mono at an advance of one patch against conventional sans, Study 1's slot layout and the mono cut by patch edges; Study 1's readers plus Qwen2.5-VL-7B (NF4) and InternVL3-2B-hf. Goal: a specification written in patch size P and merge factor. Run: 19 October 12:30 to 00:30 CEST, in `.claude/worktrees/run2` at `prereg-v2`.

## What is frozen (tags)

- `prereg-v1`: 3223ba9, approved 15 October 15:20
- `data-freeze-v1`: d25bf9b1, Saturday 17 October, signed off by the owner
- `spec-v1`: 1d633f69, signed off by the owner
- `prereg-v2`: Study 2, 19 October, with the owner's choice of S28 (28 px) and the go for the run
- `submission`: not yet
- Every tag is local only: nothing is pushed until the owner says so.

## What is broken

Nothing known. The CUDA sysmem fallback is still on (owner action 2).

## Next step

Watch `run2/logs/schedule.log`; before any relaunch, list `10_run_sweep.py` and `11_run_schedule.py` processes. After `schedule done`: commit the run folders here without images, `scripts/22_analyze_study2.py`, Figures 3 and 4, `data-freeze-v2`. Then the rules in terms of P, and H2 with the owner's go-ahead (D23).

## Top three open decisions

1. The go-ahead for H2's dummy calls and each paid attempt (D23 reviews before each judge run).
2. How the paper words clause (c): reversed under the preregistered CER, with the inserted spaces as an exploratory explanation; SmolVLM-500M's random-text tests reported with their detectable amplitudes.
3. When to push the local tags to GitHub.
