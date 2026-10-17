<!--
PURPOSE: A short state summary to paste into a new AI chat session or hand to your co-author.
WHAT GOES HERE: Thesis, current step, frozen tags, what is broken, next step, top open decisions.
FILLED IN AT STEP: Updated at the end of every working session, from W1.1 onward.
DO NOT: Exceed 400 words. Do not paste results that are not in results/final/.
-->

# Handoff

> Paste this at the start of any new AI chat session. Update at the end of every working session.
> Last updated: Saturday 17 October 2026, 20:40 CEST, by Claude (D22).

## Thesis (one sentence)

A vision-language model's character error rate on text whose glyphs share a phase against the encoder's patch grid varies periodically with that phase, at the patch or the token-cell period, and rendering glyphs on the grid lowers it (H1, `PREREGISTRATION.md` section 1). Frozen result: clauses (a) and (b) supported for all three readers, mostly at the patch period; clause (c) reversed, for Qwen and SmolVLM-2.2B mostly through spaces inserted beside two-slot characters (exploratory).

## Current week and step

W4.1: the rendering specification. R09 to R11 of `spec/RENDERING_SPEC.md` were revised after an independent review: the conventional layout at the unaligned pitch with φ_y = 0 (R09), critical sizes measured on full pages, where size and text amount are confounded (R10), anti-aliasing per reader and kind of text (R11). The exploratory per-level table behind them is `reports/tables/exploratory/levels.csv`. Waiting for the owner's sign-off.

## What is frozen (tags)

- `prereg-v1`: 3223ba9, approved 15 October 15:20
- `data-freeze-v1`: d25bf9b1, Saturday 17 October, signed off by the owner
- `spec-v1`: after the owner signs off the revised specification, before H2
- `submission`: not yet

## What is broken

Nothing known. The CUDA sysmem fallback is still on (owner action 2).

## Next step

After the sign-off: `make contract` on the committed tree, then tag `spec-v1` and push it. Then H2 (`h2/README.md`): the owner types the Z.ai key into `.env` (never into a chat), `h2/build_prompt.py` freezes the prompt, then the judge pages, the dummy calls and each paid attempt with the owner's go-ahead (D23).

## Top three open decisions

1. The `spec-v1` sign-off of R09 to R11 as revised.
2. The go-ahead for H2's dummy calls and each paid attempt (D23 reviews before each judge run).
3. How the paper words clause (c): reversed under the preregistered CER, with the inserted spaces as an exploratory explanation; SmolVLM-500M's random-text tests reported with their detectable amplitudes.
