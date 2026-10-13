<!--
PURPOSE: Structured self-study for both authors before and during implementation.
WHAT GOES HERE: One explanation, one exercise and the listed resource per topic.
FILLED IN AT STEP: A1 to A6 at W1.2; A7 at W1.2; A8 at W4.2; B topics when the matching step starts.
DO NOT: Add guessed URLs. Replace a dead link only with a source you have read.
-->

# Learning guide

All links were verified to exist on 25 Sep 2026. If one is dead, replace it with a source you have read; never with a guessed URL.

Write exercise answers in `LAB_NOTEBOOK.md`.

---

## Part A: Theory

### A1. Vision Transformer patches

A Vision Transformer cuts an image into a grid of non-overlapping square patches, flattens each patch into a vector and projects it into a token. The model never sees pixels directly, only these patch tokens. A glyph that straddles a patch border is split across two tokens, while the same glyph centred in a patch lands in one. That is the mechanism H1 tests.

**Exercise:** Take a 56 by 56 pixel image and a patch size of 14. How many patches are there? If a vertical stroke 2 pixels wide sits at x = 13 to 14, which patches contain it? Repeat at x = 20 to 21.

**Resources:**
- Video: https://www.youtube.com/watch?v=TrdevFK_am4
- Paper: https://arxiv.org/abs/2010.11929 (sections 1 and 3.1)

### A2. Qwen resize and tiling

Qwen2.5-VL uses 14-pixel patches and merges each 2 by 2 block of patches into one token, so each token covers 28 by 28 pixels. The processor resizes images so both sides are multiples of 28. If your page is not already such a multiple, the resize moves every glyph relative to the grid and destroys the phase you set.

**Exercise:** For a 1000 by 700 image, compute the resized size by rounding each side to the nearest multiple of 28. How many vision tokens result? Now choose a page size that needs no resize.

**Resources:**
- https://huggingface.co/docs/transformers/model_doc/qwen2_vl
- https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ

### A3. Sampling and anti-aliasing

A pixel is a sample, not a little square. Drawing a continuous glyph outline onto a pixel grid is sampling, and sampling below the signal's frequency causes aliasing. Anti-aliasing replaces hard in/out decisions with coverage values. Binary rendering, coverage rendering and where the grid sits all change what the encoder receives.

**Exercise:** Draw a 1D edge at x = 3.3 on a row of 8 pixels. Write the pixel values for binary rendering (threshold 0.5) and for area coverage. Shift the edge by 0.5 px and repeat.

**Resources:**
- https://cs184.eecs.berkeley.edu/public/sp20/lectures/lec-3-sampling-aliasing-antialiasing/lec-3-sampling-aliasing-antialiasing.pdf
- https://alvyray.com/Memos/CG/Microsoft/6_pixel.pdf

### A4. Colour and light

Stored 8-bit values are sRGB-encoded, not proportional to light. Averaging or blending encoded values gives a different result from blending in linear light. Coverage-based anti-aliasing is physically correct only when done in linear light. This matters for the `coverage_srgb` versus `coverage_linear` factor.

**Exercise:** Blend pure black and pure white 50/50 twice: once on encoded values, once in linear light then re-encoded. Compare the two 8-bit results. (One of them is checked in `tests/test_color.py`.)

**Resources:**
- Video: https://www.youtube.com/watch?v=LKnqECcg6Gw
- Article: https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-24-importance-being-linear

### A5. Alpha compositing

Compositing puts a partially covered glyph pixel over the background. The "over" operator combines colour and coverage. Whether colours are premultiplied by alpha, and whether the maths happens in linear or encoded space, decides the final pixel.

**Exercise:** Composite black text at coverage 0.25 over white using "over", once in sRGB-encoded values and once in linear light. Write both 8-bit results.

**Resources:**
- https://ciechanow.ski/alpha-compositing/
- https://keithp.com/~keithp/porterduff/p253-porter.pdf

### A6. Font rasterisation

A font stores outlines. A rasteriser turns them into coverage at a given size, with choices about hinting (snapping outlines to the pixel grid) and subpixel positioning. Hinting was designed for human eyes on screens. Our question is whether a different grid, the encoder's, deserves its own rules.

**Exercise:** In your own words, list what hinting changes about a glyph at 10 px and why that might help or hurt a patch encoder.

**Resource:**
- https://freetype.org/freetype2/docs/hinting/text-rendering-general.html

### A7. Prior art

Titles and claims below must be verified at W1.2 by opening each paper. Fill the two right-hand columns only after reading.

| Paper | arXiv link | What it proves | What it does not do |
|---|---|---|---|
| Glyph | https://arxiv.org/abs/2510.17800 | TODO[W1.2]: read and summarise | TODO[W1.2]: read and summarise |
| DeepSeek-OCR | https://arxiv.org/abs/2510.18234 | TODO[W1.2]: read and summarise | TODO[W1.2]: read and summarise |
| LensVLM | https://arxiv.org/abs/2605.07019 | TODO[W1.2]: read and summarise | TODO[W1.2]: read and summarise |
| ZeroSense | https://arxiv.org/abs/2608.01848 | TODO[W1.2]: read and summarise | TODO[W1.2]: read and summarise |
| Grid2Matrix | https://arxiv.org/abs/2604.09687 | TODO[W1.2]: read and summarise | TODO[W1.2]: read and summarise |
| Steal the Patch Size | https://arxiv.org/abs/2607.00174 | TODO[W1.2]: read and summarise | TODO[W1.2]: read and summarise |
| CodeOCR (title unconfirmed) | https://arxiv.org/abs/2602.01785 | TODO[W1.2]: read, confirm title, summarise | TODO[W1.2]: read and summarise |
| ViT | https://arxiv.org/abs/2010.11929 | TODO[W1.2]: read and summarise | TODO[W1.2]: read and summarise |
| KernelBench | https://arxiv.org/abs/2502.10517 | TODO[W1.2]: read and summarise | TODO[W1.2]: read and summarise |

**Exercise:** For each paper, write one reading note using `references/reading_notes/README.md`.

### A8. Triton

Triton is a Python-embedded language for writing GPU kernels by blocks instead of threads. H2 asks a language model to write one. KernelBench is a benchmark of language models writing GPU kernels and gives a reference point for how such attempts are judged.

**Exercise:** Run tutorials 01 and 02. Note in the lab notebook what a program ID and a block are.

**Resources:**
- https://triton-lang.org/main/getting-started/tutorials/
- https://arxiv.org/abs/2502.10517

---

## Part B: Practice

### B1. Readers

Load each reader, feed it one rendered page, and read back the transcription and the number of vision tokens. Check the processor did not resize your image.

**Exercise:** For Qwen, log the processor's output image grid for a native-size page and confirm tokens = (H/28) x (W/28).

**Resources:**
- https://huggingface.co/docs/transformers/model_doc/qwen2_vl
- https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ
- https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct (the SmolVLM-2.2B card, read at the pinned revision on 13 Oct: 81 visual tokens per 384 × 384 tile; larger images are split into tiles, with `longest_edge` N × 384 and N = 4 by default; SigLIP so400m-patch14-384 encoder, SmolLM2-1.7B-Instruct decoder, built on Idefics3)
- https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct (the third reader, D7)
- https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct (a smaller sibling; not a reader)

### B2. Rendering

Pillow can load a TrueType font and give you a glyph mask (coverage) at a pixel size. You will composite masks yourself so you control phase, gamma and anti-aliasing.

**Exercise:** Load one font at 12 px, get the mask for "a", print its shape and its min and max values.

**Resource:**
- https://pillow.readthedocs.io/en/stable/reference/ImageFont.html

### B3. Error metrics

CER is the character-level edit distance between reference and hypothesis divided by the reference length. WER is the same at word level. Decide text normalisation (whitespace, line breaks) before the pilot and write it in the preregistration.

**Exercise:** Compute by hand the CER of "teh cat" against "the cat". Then check with `jiwer`.

**Resource:**
- https://jitsi.github.io/jiwer/

### B4. Statistics

The Wilcoxon signed-rank test compares paired samples without assuming normality. The bootstrap resamples pages with replacement to get a confidence interval for any statistic, here the median paired CER difference.

**Exercise:** Make two paired arrays of 30 values where the second is the first plus small noise. Run `scipy.stats.wilcoxon` and a 95% bootstrap interval of the median difference.

**Resources:**
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
- Video: https://www.youtube.com/watch?v=Xz0x-8-cgaQ

### B5. Plotting

- **Figure 1 (phase curve):** x axis is glyph phase within one token cell, in pixels. y axis is CER. One line per reader, each with a shaded bootstrap confidence band. If H1 holds, each line repeats with period equal to that reader's cell size.
- **Figure 2 (trade-off):** x axis is vision tokens per page. y axis is CER. Two series: conventional and patch-aware. Patch-aware should sit lower at equal token count if H1's second clause holds.

**Exercise:** Sketch both figures on paper with made-up shapes (not numbers) before any data exists, so you know what you are looking for.

No link.

### B6. LaTeX

You will write the paper in LaTeX using the MLSys 2027 style.

**Exercise:** Compile `paper/main.tex` once the style file is in place (W5.1).

**Resources:**
- https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes
- https://mlsys.org/Conferences/2027/CallForResearchPapers
