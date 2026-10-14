<!--
PURPOSE: The research plan: question, design, factors, statistics, gates, threats, positioning.
WHAT GOES HERE: Planning decisions. Factor levels are fixed at W2.2 and W3.2.
FILLED IN AT STEP: Read at W1.1. Levels filled at W2.2 and W3.2.
DO NOT: Put results or paper prose here. Change plans only via docs/05_decisions.md.
-->

# Research plan

## Question

Text rendering was designed for the human visual system. In visual text compression (VTC), long text is drawn onto images and read by a vision-language model (VLM) whose encoder cuts the page into fixed patches. We measure how that encoder reads rendered text as a function of where glyphs land on its patch grid, derive a rendering specification from the measurements, and test whether a language model can implement that specification as a GPU kernel.

## Why it matters

- VTC compresses text into images, so one image token stands in for several text tokens.
- Legibility at the encoder's resolution is the limit on how far that compression can go.
- Rendering choices (hinting, anti-aliasing, gamma, subpixel positioning) were inherited from design for the human eye, not for a patch grid.

## Hypotheses

- **H1:** Reading error (character error rate) varies periodically with glyph phase relative to the encoder's token grid, with period equal to that encoder's cell size. Patch-aware rendering lowers reading error at equal token count, most at small font sizes.
- **H2:** A language model (GLM-5.3) given only the written specification can write a Triton kernel whose output matches the reference renderer.

## Design

- **Three readers with different grids** (D7): Qwen2.5-VL-3B-Instruct (patch 14 px, token cell 28 px, attention windows 112 px), SmolVLM-2.2B (patch 14 px, token cell 42 px; its 27 × 27 patches see only the first 378 px of its 384 px page) and SmolVLM-500M (patch 16 px, token cell 64 px). If the error period tracks each reader's own grid, the effect is traced to the grid; the third reader's 16 px patch separates the patch embedding from the merge.
- **Factors** varied in blocks, each one layout swept along one axis (D10; table below). Blocks V and H test H1; C is the control; S, A and Z are the rasteriser, anti-aliasing and font-size factors.
- **Paired comparison:** the same pages are rendered under conventional and patch-aware modes and read by the same reader. Each page is its own control.

## Factors

| Factor | Levels | Why |
|---|---|---|
| Glyph phase within one token cell (x, y) | V: φy over one token cell, step 2 px (Qwen 14 levels, SmolVLM-2.2B 21) or 4 px (SmolVLM-500M 16). H: φx over one patch, step 1 px (14, 14, 16 levels). C: φx over the sweep period (112, 42, 64 px), step 7, 2, 4 px (16, 21, 16 levels). S: φx at 0, 0.25, 0.5, 0.75 px. A and Z: φ = 0. The step divides every tested period, since a curve of period T sampled every σ px repeats every lcm(σ, T) px (`configs/sweeps/pilot.yaml`) | Tests H1 periodicity directly |
| Font size in pixels | Block Z, the ladder: 10, 12, 14, 16, 20 and 24 px in both render modes (`configs/sweeps/ladder.yaml`). Every other block: one size per reader, read off the ladder at its target CER (W1.10); until the ladder has run, `configs/sweeps/pilot.yaml` holds placeholders the runner rejects | H1 predicts the effect is largest at small sizes |
| Line pitch | aligned to grid; not aligned to grid | Separates vertical grid alignment from glyph phase |
| Anti-aliasing mode | binary; coverage in sRGB; coverage in linear light | Tests whether the encoder prefers hard edges or physically correct coverage |
| Render mode | conventional; patch-aware | The paired comparison for H1's second clause |
| Font file | Source Sans 3 Regular (W1.4) in every decoded block. Proposed for Gate 1: Source Serif 4 only in the equivariance measurement and the coherence table, which need no decoding, with the one-typeface limit stated in the paper | Checks the effect is not specific to one typeface |
| Reader | TODO[W3.2]: both readers in full sweep | Traces period to cell size |

## Metrics

- **Primary:** character error rate (CER) of a transcription task, computed with `jiwer`.
- **Secondary:** word error rate (WER).
- **Why transcription, not QA:** QA scores overstate text fidelity (ZeroSense, arXiv 2608.01848; claim to be verified at W1.2 and logged in references/citations.md).
- **Bookkeeping:** vision token count, latency.

## Statistics

- Paired design: same pages, two conditions.
- Wilcoxon signed-rank test (`scipy.stats.wilcoxon`).
- Bootstrap confidence intervals.
- Effect size: median paired CER difference with its bootstrap interval.
- Alpha, sidedness and the periodicity criterion: TODO[W2.1]: fix in PREREGISTRATION.md.

## Gates and what each decision means

- **G1 (2 Oct):** Both readers transcribe a page, token counts match the formula, phase-shifted renders are correct by overlay. Go: start the pilot. No-go: fix the pipeline before anything else; the pilot slips.
- **G2 (9 Oct):** Is CER periodic in phase? Go: run the full experiment as planned. No-go: switch to the negative-result framing below; the full sweep still runs, the spec in W4 becomes "what does not matter".
- **G3 (23 Oct):** Does the H2 kernel match the reference? Go: report H2 as supported. No-go: report H2 as a documented failure with the judge's numbers; no retries beyond the preregistered attempts.

## Negative-result framing

If G2 fails, the paper's claim becomes: this encoder does not care where glyphs land, measured rigorously. That is still useful: it tells VTC builders they can ignore glyph phase and spend effort elsewhere.

## Threats to validity

- **Smart resize can move the grid.** Processors resize images to multiples of the cell size. Always render at native sizes so no resize happens.
- **Tokenizer or processor version changes** can change resize behaviour. Pin revisions and record them in every manifest.
- **Font licence** limits what rendered pages can be released.
- **Two models only.** The result may not generalise to other encoders.
- **One GPU class** (RTX 5060, 8 GB). Numerical behaviour in bf16 may differ on other hardware.

## Positioning against prior art

- Grid2Matrix (2604.09687) and Steal the Patch Size (2607.00174) show boundary effects for grid cells.
- CodeOCR (2602.01785) used patch-divisible resolution.
- The gap: glyph-level measurement of phase against the token grid, and a written, testable rendering specification.
- All of the above to be verified at W1.2 and logged in references/citations.md before citing.
