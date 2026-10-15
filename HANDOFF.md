<!--
PURPOSE: A short state summary to paste into a new AI chat session or hand to your co-author.
WHAT GOES HERE: Thesis, current step, frozen tags, what is broken, next step, top open decisions.
FILLED IN AT STEP: Updated at the end of every working session, from W1.1 onward.
DO NOT: Exceed 400 words. Do not paste results that are not in results/final/.
-->

# Handoff

> Paste this at the start of any new AI chat session. Update at the end of every working session.
> Last updated: Thursday 15 October 2026, 15:45 CEST, by Claude (D22).

## Thesis (one sentence)

A vision-language model's character error rate on text whose glyphs share a phase against the encoder's patch grid varies periodically with that phase, at the patch or the token-cell period, and rendering glyphs on the grid lowers it (H1, `PREREGISTRATION.md` section 1).

## Current week and step

W3.3: the full run (D26), started 15:23 Thursday from the run worktree `.claude/worktrees/run` at `prereg-v1`, by `scripts/11_run_schedule.py` outside the chat session (`logs/launch_driver.cmd`). Phases: Qwen to Thu 23:00, SmolVLM pair to Fri 09:00, Qwen to Fri 14:00, pair to Sat 00:00, Qwen to Sat 04:00, pair to Sat 13:00 (a page may overrun by up to an hour).

## What is frozen (tags)

- `prereg-v1`: 3223ba9, approved 15 October 15:20
- `data-freeze-v1`: Saturday 17 October, after `make analyze`
- `spec-v1`: after H2 (R09 to R11 wait for the freeze)
- `submission`: not yet

## What is broken

Nothing known. The CUDA sysmem fallback is still on (owner action 2); Qwen fits at 7,544 MiB of 8,151, and `logs/nvsmi.csv` would show a spill.

## Next step

Saturday about 14:00: `scripts/20_analyze.py --config configs/sweeps/full.yaml`, `scripts/30_make_figures.py`, `scripts/50_summary.py`, `scripts/40_freeze.py` and its check, the tag `data-freeze-v1`, then the interpretations (`research-paper-notes/build_plan/INTERPRETATION_KEY.md`). H2 only after that: the specification's R09 to R11 from the frozen results, `spec-v1`, then the Z.ai key, dummy calls and three attempts, each with the owner's go-ahead (D26 note).

## Top three open decisions

1. The freeze sign-off on Saturday, before the interpretations are written.
2. After the spec: the go-ahead for H2's dummy calls and each paid attempt (D23 reviews before each judge run).
3. Whether the paper keeps SmolVLM-500M's random-text tests as primary given its ceiling CER (kept by the rule; report its minimum detectable amplitudes).
