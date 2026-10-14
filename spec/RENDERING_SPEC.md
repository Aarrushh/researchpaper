<!--
PURPOSE: The rendering specification derived from our measurements.
WHAT GOES HERE: Numbered rules, each with its evidence, exact parameters and a test.
FILLED IN AT STEP: W4.1. Tagged spec-v1 at G3.
DO NOT: Add a rule without evidence in reports/. Do not reference code; this must be implementable from text alone.
-->

# Rendering Specification for Machine Readers

This is the document given to GLM-5.3 in H2 (see `h2/README.md`). It is the only input the model receives. It must be implementable from text alone: no references to our code, no undefined terms, every parameter an exact value.

## Scope

This document states how the pages of the paper's experiment are rendered (rules R01 to R06), what the measurements on three machine readers found (rules R07 to R11), and what a GPU kernel must compute to reproduce the pixel stages of that renderer byte for byte (the sections Kernel interface and Conformance).

**Readers.** Three vision-language models read the pages. Each cuts a page into square patches of P × P pixels, each patch becoming one vector, and merges m × m patches into one vision token. One of them also restricts attention to square windows.

| Reader | Patch P | Token cell | Attention window or tile | Native page |
|---|---|---|---|---|
| Qwen2.5-VL-3B-Instruct | 14 px | 28 px (2 × 2 patches) | 112 px windows | 448 × 448 px |
| SmolVLM-2.2B (SmolVLM-Instruct) | 14 px | 42 px (3 × 3 patches) | one 384 px tile | 384 × 384 px |
| SmolVLM-500M-Instruct | 16 px | 64 px (4 × 4 patches) | one 512 px tile | 512 × 512 px |

**Text.** Horizontal, left-to-right text set in Source Sans 3 Regular (an OpenType font with TrueType outlines, licensed under the SIL Open Font License 1.1) at the font sizes R10 lists; the paper measured 10, 12, 14, 16, 20 and 24 px and ran its phase sweeps at the size per reader that R10 names. The characters are the printable ASCII characters, the digits and upper- and lower-case letters of random strings, and the accented letters and symbols that occur in three English novels (such as £, é, œ and ½); every character used has a glyph in the font.

**Out of scope.** Vertical or right-to-left text, other typefaces, hinting, kerning, ligatures, shaping, justification, hyphenation, colour fonts, font sizes outside 10 to 24 px, and pages that are not the native page of one of the three readers.

## Definitions

- **Page.** An H × W array of RGB bytes (0 to 255 per channel), row 0 at the top and column 0 at the left; H and W are the native page of the reader (448, 384 or 512 px, square). A pixel is a point sample at integer coordinates (row y, column x); it is not a little square.
- **Byte and fraction.** A stored colour value is a byte v₈ from 0 to 255; its fraction is v = v₈ / 255. Arithmetic in this document is on fractions unless an integer is named.
- **Patch.** The P × P pixel square the reader's encoder turns into one vector. The patch grid starts at the page's top-left corner, so patch (i, j) covers rows i·P to i·P + P − 1 and columns j·P to j·P + P − 1. P = 14, 14 and 16 px for the three readers.
- **Token cell.** The square of m × m patches merged into one vision token: 28, 42 and 64 px. The cell grid also starts at the page's top-left corner.
- **Attention window.** For Qwen2.5-VL-3B, the 112 px square inside which most of its encoder layers attend; its grid starts at the page's top-left corner. The SmolVLM readers attend over the whole page.
- **Blind strip.** The columns and rows of a page that no patch covers: those at index P·⌊W/P⌋ and beyond. For SmolVLM-2.2B these are columns and rows 378 to 383 (27 × 14 = 378 < 384). The other two readers have none (448 = 32 × 14, 512 = 32 × 16).
- **Rounding.** Every rounding in this document is ⌊v + 0.5⌋, the largest integer not above v + 0.5: halves round up. Nothing rounds halves to even.
- **Font size.** The size s in pixels at 72 dots per inch, so that one point is one pixel. The rasteriser receives it in 1/64 px as ⌊64·s + 0.5⌋ (1024 for 16 px).
- **Glyph bitmap.** The rectangular array of bytes the rasteriser produces for one character at one font size and one fractional phase. Each byte b is the coverage of that pixel by the character's outline, scaled to 0 to 255: coverage = b / 255, 0 outside the outline and 255 fully inside. A character without ink (the space) has an empty bitmap of 0 rows and 0 columns.
- **Bitmap left and top.** Two integers the rasteriser reports with each bitmap: left is the number of columns from the pen position to the bitmap's first column (it may be negative), and top the number of rows from the bitmap's first row down to the baseline (so the bitmap's first row is top rows above the baseline).
- **Linear advance.** The unhinted horizontal advance of a character in pixels at the font size: the font's advance width scaled to the size without any rounding to whole pixels. It is a fraction with at most 16 binary digits (a 16.16 fixed-point number divided by 65536), so sums of advances are exact in double precision.
- **Phase.** The offset of the text block from its home position, in pixels, positive to the right (φ_x) and down (φ_y). A phase is converted to 1/64 px once, X = ⌊64·φ + 0.5⌋, and split into a whole part X ≫ 6 = ⌊X / 64⌋ (whole pixels, which move a bitmap) and a fractional part X & 63 = X mod 64 (64ths of a pixel, which move the outline before it is rasterised).
- **Home position.** Where the text block sits at phase zero: its top-left corner one margin from the page's top edge and one margin from its left edge.
- **Margin.** One patch: 14, 14 and 16 px for the three readers.
- **Text block.** The rectangle inside which lines are laid out. Its width is W − 2·margin − A_x − B and its height H − 2·margin − A_y − B, where A_x and A_y are the phase allowances (the largest phases the pages of a sweep must hold, so that the block stays on the page at every phase; stated in R03) and B the width of the blind strip (6 px for SmolVLM-2.2B, 0 for the others).
- **Pen position.** The horizontal position, in pixels from the block's left edge, at which the next character of a line is placed. It starts at 0 and advances by the linear advance (conventional layout) or by a whole number of slots (patch-aware layout).
- **Line pitch.** The distance between consecutive baselines, in whole pixels.
- **Layouts.** Conventional: each character advances the pen by its linear advance. Patch-aware: each character occupies a whole number of slots one patch wide and is centred in them (R03).
- **Glyph box.** Where a bitmap lands on the page: its first row r and first column q. With the glyph's origin at column x₆₄ and its baseline at row y₆₄, both in 1/64 px, r = (y₆₄ ≫ 6) − top and q = (x₆₄ ≫ 6) + left, where the bitmap was rasterised at the fractional phase (x₆₄ & 63, y₆₄ & 63). A box may extend beyond the page, or lie wholly outside it; the pixels off the page are dropped.
- **Coverage of a pixel.** The fraction c = S / 255 from 0 to 1, where S is the clipped integer sum of the glyph bytes at that pixel (equation (1) in R04). A pixel that no glyph covers has S = 0.
- **sRGB decode D(v).** From a stored fraction v to linear light: D(v) = v / 12.92 when v ≤ 0.04045, otherwise D(v) = ((v + 0.055) / 1.055)^2.4.
- **sRGB encode E(l).** From linear light l to a stored fraction: E(l) = 12.92·l when l ≤ 0.04045 / 12.92 = 0.0031308049535603713, otherwise E(l) = 1.055·l^(1/2.4) − 0.055. The encode threshold is the exact value where the decode's straight piece ends, not the rounded 0.0031308, so that E(D(v)) = v to within 2 × 10⁻¹⁶ over the whole range.
- **Linear light.** The quantity that adds when two lights are mixed; stored sRGB values do not add. Mixing in linear light means decoding both values with D, mixing, and encoding the result with E.
- **Text colour and background colour.** Two RGB byte triples, t₈ and b₈, with fractions t and b per channel. The paper's pages are black (0, 0, 0) on white (255, 255, 255) unless a rule says otherwise.
- **Anti-aliasing mode.** How the coverage c of a pixel sets its output value o per channel (R05): `binary` (o = t where S ≥ 128, otherwise o = b), `coverage_srgb` (o = c·t + (1 − c)·b on the stored fractions) and `coverage_linear` (o = E(c·D(t) + (1 − c)·D(b))).
- **Output byte.** o₈ = ⌊255·clip(o, 0, 1) + 0.5⌋, applied once per channel at the end (R06); clip(o, 0, 1) is o held to the range 0 to 1.

---

## Rule template

```
### R<nn>. <name>
RULE:        one imperative sentence.
WHY:         the physics or the measurement behind it.
EVIDENCE:    figure or table ID in reports/.
PARAMETERS:  exact values.
APPLIES WHEN: conditions.
TEST:        how to check an implementation obeys it.
```

---

### R01. Glyph coverage

- RULE: Rasterise each character from its outline with FreeType 2.13.2, without hinting and without embedded bitmaps, at the font size in 1/64 px at 72 dots per inch, into an 8-bit coverage bitmap, and advance the pen by the linear (unhinted) advance.
- WHY: Hinting snaps stems to the pixel grid for the human eye and changes with every size and position; embedded bitmap strikes replace the outline altogether. Both would make the coverage values depend on something other than the outline and the phase, and the experiment moves the outline by fractions of a pixel to measure exactly that dependence. The 8-bit coverage of a pixel is the area of the pixel inside the outline, which is the box-filtered, point-sampled shape; it is the quantity the anti-aliasing modes of R05 transform.
- EVIDENCE: `reports/tables/contract/contract_checks.csv` (rows whose rule is R01)
- PARAMETERS: FreeType version 2.13.2. Load flags: render to an 8-bit grey bitmap, no hinting, no embedded bitmaps; rendering mode normal (256 grey levels, no sub-pixel colour filtering). Character size ⌊64·s + 0.5⌋ in 1/64 px at 72 × 72 dots per inch. Font: Source Sans 3 Regular. Linear advances at 16 px: `m` 13.264 px and `i` 3.936 px (to 0.001 px). The space has an empty bitmap.
- APPLIES WHEN: every character of every page, in every layout and every anti-aliasing mode.
- TEST: At 16 px, the linear advance of `m` is 13.264 px and of `i` 3.936 px, to within 0.001 px. The bitmap of `o` at 16 px holds grey levels between 1 and 254 along its edges (coverage, not a thresholded mask), and its bytes change when the fractional phase changes (R02); a hinted rendering does not reproduce the advances above at every size.

### R02. Phase

- RULE: Convert a phase φ in pixels to 1/64 px once, X = ⌊64·φ + 0.5⌋; move the bitmap by X ≫ 6 whole pixels and rasterise the outline shifted by (X & 63) / 64 px, positive to the right for φ_x and down for φ_y.
- WHY: A whole-pixel shift changes no pixel value: it only moves the bitmap against the patch grid, so it isolates the reader's encoder. A sub-pixel shift changes the coverage values themselves, because a different part of each pixel is inside the outline. The two kinds of phase are measured separately in the paper, and both must be reproduced exactly, so the phase is rounded once, to the 1/64 px the rasteriser works in, with halves rounded up.
- EVIDENCE: `reports/tables/contract/contract_checks.csv` (rows whose rule is R02)
- PARAMETERS: Resolution 1/64 px. X = ⌊64·φ + 0.5⌋. Whole part X ≫ 6; fractional part X & 63, from 0 to 63. Examples: φ = 3.25 px gives X = 208, whole 3, fraction 16; φ = 13.0078125 px gives 64·φ = 832.5 exactly, X = 833, whole 13, fraction 1 (rounding halves to even would give 832 and the fraction 0, which is wrong). The outline is moved right by (X_x & 63) / 64 px and down by (X_y & 63) / 64 px before rasterisation; the sign of the vertical move is such that the ink moves down the page.
- APPLIES WHEN: every page; the phases of a sweep are listed in R09 and lie between 0 and the allowance A of R03.
- TEST: φ = 3.25 → (3, 16) and φ = 13.0078125 → (13, 1). The 64 bitmaps of `o` at 16 px for the 64 fractional phases along x are pairwise different, and so are the 64 along y. A phase of exactly one patch (14 or 16 px) produces a page identical to the phase-zero page shifted by P pixels, every byte.

### R03. Layout

- RULE: Wrap words greedily inside the text block, place each line's glyphs by linear advances (conventional layout) or in slots one patch wide (patch-aware layout), set the line pitch to ⌊1.25·s + 0.5⌋ px (unaligned) or to the token cell (aligned), keep a margin of one patch, round every position once with ⌊v + 0.5⌋, and apply no kerning, ligatures, shaping, justification or hyphenation.
- WHY: The layout decides where each glyph's ink falls against the patch grid. Linear advances put the glyphs of conventional text at every offset within a patch, so a conventional page has no phase of its own; whole-slot advances put every glyph of one character at the same offset in its slot, so a patch-aware page moves as one against the grid. Everything that could move a glyph by a data-dependent amount (kerning, shaping, justification) is excluded, and every rounding happens once, so that the same text at the same phase always gives the same positions.
- EVIDENCE: `reports/tables/contract/contract_checks.csv` (rows whose rule is R03)
- PARAMETERS: Margin: one patch, 14, 14 and 16 px. Block width W − 2·margin − A_x − B and height H − 2·margin − A_y − B; phase allowances (A_x, A_y) of the paper's sweeps: 105 and 26 px (Qwen2.5-VL-3B), 40 and 40 px (SmolVLM-2.2B), 60 and 60 px (SmolVLM-500M); blind strip B: 0, 6 and 0 px. Line pitch: unaligned ⌊1.25·s + 0.5⌋ (15, 18 and 20 px at 12, 14 and 16 px); aligned: the token cell (28, 42 or 64 px). Word wrap: words are the runs between spaces; a word joins the current line if the line's width plus one space plus the word's width is at most the block width, otherwise it starts a new line; a word wider than the block alone starts a new line and is cut where the line is full. Conventional layout: the pen advances by the linear advance of each character, spaces included. Patch-aware layout: slot width q = P; a character of linear advance a advances the pen by a′ = q·⌈a / q⌉ and its glyph origin is offset from the pen by ⌊(a′ − a) / 2 + 0.5⌋ whole pixels (centred); the space advances exactly one slot and has no offset. Baselines: the first at the font's ascender at that size (as the rasteriser reports it, rounded up to whole pixels: 16 px at 16 px) below the block's top edge, each next one one line pitch lower; a page holds at most ⌊(h − ascender − d) / pitch⌋ + 1 lines, h the block height and d the descender room ⌈−descender / 64⌉ (4, 5 and 6 px at 12, 14 and 16 px). Positions in 1/64 px: a glyph's origin column is x₆₄ = ⌊64·margin + 0.5⌋ + ⌊64·φ_x + 0.5⌋ + ⌊64·(pen + offset) + 0.5⌋ and its baseline row y₆₄ = ⌊64·margin + 0.5⌋ + ⌊64·φ_y + 0.5⌋ + ⌊64·baseline + 0.5⌋, with pen, offset and baseline in pixels from the block's edges; the bitmap is rasterised at the fractional phase (x₆₄ & 63, y₆₄ & 63) and placed at the glyph box of the Definitions.
- APPLIES WHEN: every page of the three readers, in both layouts and both line-pitch modes.
- TEST: The unaligned pitch at 12, 14 and 16 px is 15, 18 and 20 px. On a conventional page, the origin column of every glyph equals ⌊64·margin + 0.5⌋ + ⌊64·φ_x + 0.5⌋ + ⌊64·(the sum of the linear advances of the characters before it on its line, spaces included) + 0.5⌋: no kerning and no justification. On a patch-aware page every origin column is a multiple of 64 when φ_x is whole. The same text laid out at two phases gives the same lines with every position moved by the same amount.

### R04. Accumulate

- RULE: At each pixel, add the bytes of every glyph bitmap that covers it as integers and clip the sum at 255; the pixel's coverage is that sum divided by 255.
- WHY: Integer addition is exact and independent of order, so a parallel implementation may add the glyphs in any order and still match byte for byte. The clip keeps coverage within 0 to 1 where glyphs overlap (two bitmaps that touch, or a box that another glyph's box crosses). Lines never overlap at the pitches of R03, so on the paper's pages the clip changes bytes only where glyphs within a line overlap.
- EVIDENCE: `reports/tables/contract/contract_checks.csv` (rows whose rule is R04)
- PARAMETERS: Equation (1): S(y, x) = min(255, Σᵢ kᵢ(y − rᵢ, x − qᵢ)), c(y, x) = S(y, x) / 255, where kᵢ is the bitmap of glyph i (zero outside its box), rᵢ and qᵢ its first row and column, and the sum runs over all glyphs of the page. Pixels outside the page are dropped; a box of 0 rows or 0 columns contributes nothing.
- APPLIES WHEN: every page; the sum includes every glyph of the page, in every layout and mode.
- TEST: Two glyphs whose bytes add to 127 at a pixel give S = 127 and 128 give S = 128 (R05 decides the output); bytes of 200 and 100 give S = 255, not 300. On a page, accumulating the glyph bytes as integers and dividing by 255 gives, at every pixel, the same coverage as accumulating the fractions b / 255 in double precision with min(1, ·) after each glyph, to within 10⁻¹² and never above 1.

### R05. Composite once

- RULE: Composite the page once, per pixel and per channel, from its coverage c = S / 255, the text colour t and the background colour b: in `binary`, o = t where S ≥ 128 and o = b elsewhere; in `coverage_srgb`, o = c·t + (1 − c)·b on the stored fractions; in `coverage_linear`, o = E(c·D(t) + (1 − c)·D(b)) with the sRGB functions and constants of the Definitions.
- WHY: Source-over compositing of an opaque text colour with alpha c onto an opaque background reduces to a mix of the two colours weighted by c. The three modes differ in what is mixed. `coverage_srgb` mixes the stored values, which is what most text rasterisers do and what human text rendering is tuned for. `coverage_linear` mixes light, which is the intensity a camera would record and which makes a 50 percent coverage pixel visibly lighter (188 against 128 for black on white). `binary` keeps only the thresholded shape. The reader's encoder is linear in the stored pixel value, so which curve maps coverage to value is a free design choice for a machine reader, and the paper measures all three (R11).
- EVIDENCE: `reports/tables/contract/contract_checks.csv` (rows whose rule is R05)
- PARAMETERS: Threshold for `binary`: S ≥ 128 (equivalently c ≥ 0.5 for integer S). sRGB constants: 0.04045, 12.92, 0.055, 1.055, 2.4, and the encode threshold 0.04045 / 12.92 = 0.0031308049535603713. The mix is per channel, with t and b the fractions of the text and background bytes. No rounding takes place in this rule; R06 rounds once.
- APPLIES WHEN: every page; one compositing pass after R04 has finished the whole page, never one pass per glyph.
- TEST: Take the 2 × 2 glyph bitmap [[0, 64], [128, 255]] at row 0, column 0 of a 2 × 2 page; each output entry below is the value of all three channels after R06. Black on white: `binary` [[255, 255], [0, 0]]; `coverage_srgb` [[255, 191], [127, 0]]; `coverage_linear` [[255, 224], [187, 0]]. White on black: `binary` [[0, 0], [255, 255]]; `coverage_srgb` [[0, 64], [128, 255]]; `coverage_linear` [[0, 137], [188, 255]]. The byte 128 in red (200, 30, 60) on cream (250, 245, 230) gives [200, 30, 60] in `binary`, [225, 137, 145] in `coverage_srgb` and [227, 181, 173] in `coverage_linear`. Two glyphs whose bytes add to 127 at a pixel give the background in `binary`, and 128 gives the text colour; bytes of 200 and 100 give the text colour in every mode, because the sum is clipped at 255 (R04). A power of 2.2 in place of the sRGB curve changes 11 of the 27 `coverage_linear` bytes of these examples, by up to 2 levels, and by up to 9 levels over all byte triples; a missing clip in R04 shows only at pixels where overlapping glyph bitmaps sum above 255.

### R06. Round once

- RULE: Convert each output channel to a byte once, at the end, as o₈ = ⌊255·clip(o, 0, 1) + 0.5⌋.
- WHY: Every value before this step is exact (integers) or in double precision, so one rounding at the end lets an independent implementation match the reference byte for byte. Rounding halves up is stated because the usual library rounding (halves to even) differs at exact halves, and the clip removes any excursion outside 0 to 1 that floating-point arithmetic could introduce in `coverage_linear`.
- EVIDENCE: `reports/tables/contract/contract_checks.csv` (rows whose rule is R06)
- PARAMETERS: o₈ = ⌊255·clip(o, 0, 1) + 0.5⌋ per channel; clip(o, 0, 1) = min(1, max(0, o)). No intermediate rounding anywhere in R04 or R05.
- APPLIES WHEN: every page, every channel, once.
- TEST: o = 0.5 gives 128, o = −0.1 gives 0 and o = 1.2 gives 255. On a page in any mode, the output equals the rounding of the composite of the page's coverage, every byte; truncation in place of ⌊· + 0.5⌋ changes bytes of the `coverage_srgb` and `coverage_linear` pages and no byte of the `binary` page, whose values are integers before the rounding.

### R07. Native frame

- RULE: Render each reader's page at its native size, 448 × 448 px for Qwen2.5-VL-3B, 384 × 384 px for SmolVLM-2.2B and 512 × 512 px for SmolVLM-500M, so that the reader's image processor passes it to the encoder without resampling, tiling or padding.
- WHY: Each processor resizes an image whose size it does not accept: Qwen2.5-VL rounds both sides to multiples of 28 px within a pixel budget (448 stays 448; 384 would become 392 and 512 would become 504), and the SmolVLM processors scale any image to their frame and split large images into tiles. A resampled page moves every glyph off the phase it was placed at and changes every coverage value, so the measurement would no longer be of the rendered pixels. At the native size the array that enters the encoder is the rendered array itself, and the vision token count equals ⌊⌊H / P⌋ / m⌋ · ⌊⌊W / P⌋ / m⌋: 256, 81 and 64 tokens.
- EVIDENCE: `reports/tables/readers/reader_facts.csv`
- PARAMETERS: Page sizes 448 × 448, 384 × 384 and 512 × 512 px (height × width), RGB, 8 bits per channel. Vision tokens per page 256, 81 and 64. The processor is run without resizing; the encoder receives one frame of exactly the page's size with no padding.
- APPLIES WHEN: every page of every sweep and of the H2 judge; the page size never changes with the font size or the layout.
- TEST: The page array has the stated height, width and three channels for the reader. Fed to the reader's processor, the array that reaches the encoder has the page's size, and the number of image tokens is 256, 81 or 64 as stated.

### R08. Blind strip

- RULE: For SmolVLM-2.2B, keep ink out of columns 378 to 383 and rows 378 to 383 of the page.
- WHY: SmolVLM-2.2B's encoder cuts its 384 px frame into 14 px patches without padding, so 27 patches cover 378 px and the last 6 columns and rows of the frame belong to no patch. Whatever is drawn there never reaches the encoder: changing those pixels leaves the encoder's output bit-identical, while changing column 377 changes it. Ink in the strip is invisible to the reader and would be counted as an error against the reference text.
- EVIDENCE: `reports/tables/readers/reader_facts.csv`
- PARAMETERS: Strip: columns and rows 378 to 383 inclusive (6 px), on the right and bottom edges of a 384 × 384 px page. The text block of R03 excludes it (B = 6 px). Qwen2.5-VL-3B (448 = 32 × 14) and SmolVLM-500M (512 = 32 × 16) have no strip (B = 0).
- APPLIES WHEN: every SmolVLM-2.2B page, at every phase and in every layout.
- TEST: On a rendered SmolVLM-2.2B page, every pixel with column ≥ 378 or row ≥ 378 has the background colour. Replacing those pixels with noise leaves the reader's encoder output identical; replacing column 377 does not.

### R09. Phase of the text block

- RULE: TODO[W4.1]: for each reader, either "place the text block at the phase φ* = (φ_x, φ_y) px against the patch grid" with the amplitudes the harmonic model found at the periods it tested, or "choose the phase freely: no period reached significance", as the full run's tables decide.
- WHY: A patch embedding is a convolution whose stride equals its kernel, so it commutes with translation only for shifts that are whole multiples of P: shifting the page by any other amount changes every token. The paper's phase sweeps move the text block across one sweep period (112, 42 and 64 px) and fit the character error rate with sinusoids at the periods of the reader's grid (patch, token cell and, for Qwen2.5-VL-3B, attention window). TODO[W4.1]: the measured amplitudes and which groups reached significance after the Holm correction, per reader and per layout, and the sub-pixel result.
- EVIDENCE: `reports/tables/full/harmonics.csv`, `reports/tables/full/paired.csv`, `reports/tables/full/subpixel.csv`; `reports/figures/fig1_phase_full.pdf`
- PARAMETERS: TODO[W4.1]: per reader, the phase φ* in pixels (if any), the periods tested (14, 28 and 112 px; 14 and 42 px; 16 and 64 px), each group's amplitude with its confidence interval, and the layout (conventional or patch-aware) the rule applies to.
- APPLIES WHEN: TODO[W4.1]: the readers, layouts and font size of the full run, at the anti-aliasing mode of R11.
- TEST: The text block's top-left corner lies at (margin + φ_x, margin + φ_y) px from the page's top-left corner, with φ rounded as R02 states; moving the block by one whole patch gives the same page shifted by P pixels, every byte. If the rule says the phase is free, any phase within the allowance of R03 is acceptable.

### R10. Font size

- RULE: TODO[W4.1]: set the font size to the critical size s* for each reader, the smallest size whose character error rate is within δ = 0.01 of the plateau, as the full run's size table decides.
- WHY: The character error rate falls with font size until the strokes are resolved by the patch embedding and then flattens; above that size more pixels buy nothing but cost vision tokens per character. The critical size is where the curve reaches its plateau, and the density cost is the number of characters per vision token lost by rendering larger. TODO[W4.1]: the measured curve per reader and layout from block Z of the full run.
- EVIDENCE: `reports/tables/full/sizes.csv`; `reports/figures/fig2_tradeoff_full.pdf`
- PARAMETERS: TODO[W4.1]: s* in pixels per reader and layout, from the sizes measured (10, 12, 14, 16, 20 and 24 px); the plateau level; δ = 0.01; the characters per vision token at s* and at the next larger size.
- APPLIES WHEN: TODO[W4.1]: the readers and layouts of the full run.
- TEST: The rasteriser receives ⌊64·s* + 0.5⌋ for the font size, the unaligned line pitch equals ⌊1.25·s* + 0.5⌋ px, and the page size and token count stay as R07 states (the size changes the text, not the frame).

### R11. Anti-aliasing

- RULE: TODO[W4.1]: render with the anti-aliasing mode block A of the full run found best (`coverage_srgb`, `coverage_linear` or `binary`), or with any of the three if no paired difference reached significance.
- WHY: The three modes of R05 give the same shape three different coverage-to-value curves. The reader's encoder is linear in the stored value, so the curve changes the input the encoder sees, and whether that changes what it reads is an empirical question answered by the paired comparison of the modes on the same pages. TODO[W4.1]: the measured differences and their confidence intervals.
- EVIDENCE: `reports/tables/full/paired.csv` (the rows of family A)
- PARAMETERS: TODO[W4.1]: the mode; the paired differences in character error rate of `coverage_linear` and `binary` against `coverage_srgb`, with their confidence intervals.
- APPLIES WHEN: TODO[W4.1]: the readers and layouts of the full run, at the phase of R09 and the size of R10.
- TEST: The worked examples of R05 in the chosen mode, byte for byte.

---

## Kernel interface

The reference renderer carries out R01 to R03 on the processor: it rasterises every glyph and decides where each bitmap goes. The kernel receives the result of those rules as its input and carries out the three stages that touch every pixel: accumulate and clip (R04), composite once (R05) and round once (R06). Integer addition makes the order of the glyphs irrelevant, so a parallel kernel may add them in any order and still match byte for byte.

The kernel is a Python function with this signature:

```
def render(bitmaps, offsets, heights, widths, rows, cols, height, width, text_rgb, background_rgb, aa_mode): ...
```

| Argument | Type | Meaning |
|---|---|---|
| `bitmaps` | a one-dimensional tensor of unsigned 8-bit integers (`torch.uint8`) on a CUDA device | Every glyph's bytes, row-major, one glyph after another; glyph i occupies `bitmaps[offsets[i] : offsets[i] + heights[i]·widths[i]]` |
| `offsets` | a tensor of 64-bit integers (`torch.int64`), shape (G,) | Where each glyph starts in `bitmaps` |
| `heights`, `widths` | tensors of 32-bit integers (`torch.int32`), shape (G,) | The box size of each glyph. Either may be 0 (a space), and then the offset may equal the length of `bitmaps` |
| `rows`, `cols` | tensors of 32-bit integers (`torch.int32`), shape (G,) | The page row and column of the box's first pixel (r and q of the Definitions). They may be negative or beyond the page; pixels off the page are dropped |
| `height`, `width` | Python `int`, at least 1 | The page size H and W |
| `text_rgb`, `background_rgb` | tuples of three Python `int` from 0 to 255 | The text and background colours, red, green, blue |
| `aa_mode` | `str` | `"binary"`, `"coverage_srgb"` or `"coverage_linear"`, as R05 defines them |
| Returns | a new tensor of unsigned 8-bit integers, shape (height, width, 3), on the same device as `bitmaps` | The page, RGB, after R04 to R06. G = 0 gives a page of the background colour |

All glyph tensors share the same length G. The glyph bytes are coverage as R01 defines it (b / 255); the kernel never sees the outlines, the font or the layout.

Three further requirements. The function never changes its inputs. It reads and writes no files and opens no network connections and no processes. Every per-pixel operation of R04 to R06 runs inside a function decorated with `@triton.jit`; host code may only allocate, reshape, move and launch.

The environment in which the kernel runs: Windows 11, Python 3.11, PyTorch 2.14.1 built for CUDA 13.0, triton-windows 3.8, and an NVIDIA RTX 5060 Laptop GPU (compute capability 12.0, 8 GB of memory).

## Conformance

The judge renders a page with the reference renderer and with the kernel from the same inputs and compares the two, K the kernel's page and R the reference's, both H × W × 3 bytes:

- Equation (3), the largest difference: e_max = max over y, x and channel c of |K(y, x, c) − R(y, x, c)|.
- Equation (4), the share of differing pixels: f_diff = (1 / (H·W)) · Σ over y, x of 1[max over c of |K(y, x, c) − R(y, x, c)| > 0]; a pixel differs when any of its channels differs.

| `aa_mode` | e_max at most | f_diff at most | Why |
|---|---|---|---|
| `binary` | 0 | 0 | Integers are compared |
| `coverage_srgb` | 0 | 0 | Equation (2): the value before rounding is never within 1/510 of a rounding boundary |
| `coverage_linear` | 1 | 0.001 | Single-precision arithmetic moves 148 of the 16,777,216 possible (S, T, B) triples by one level and none by more, for either way of writing the mix, c·D(t) + (1 − c)·D(b) or D(b) + c·(D(t) − D(b)) |

Why two of the tolerances are zero. In `coverage_srgb`, with T and B one channel of the text and background bytes and S the clipped sum of R04, equation (2): 255·o = (S·T + (255 − S)·B) / 255 = N / 255 with N an integer. The rounding ⌊255·o + 0.5⌋ could go either way only if 255·o lay within floating-point error of a half-integer m + 1/2, which would need 2N = 255·(2m + 1); the left side is even and the right side odd, so 255·o is never closer than 1/510 to a rounding boundary. A single-precision kernel's arithmetic error is about 255 × 2⁻²⁴ ≈ 1.5 × 10⁻⁵, far below 1/510 ≈ 0.002, so a correct kernel in single precision matches byte for byte. `binary` compares integers. Only `coverage_linear` passes through the curved sRGB functions, where single precision can move a result across a boundary; the tolerance of one level on at most one pixel in a thousand allows for that and nothing else. The three facts are computed over all 16,777,216 (S, T, B) triples in `reports/tables/contract/contract_checks.csv` (rows whose rule is Conformance): the smallest distance of 255·o from a half-integer in `coverage_srgb` is exactly 1/510, single precision moves no `coverage_srgb` triple, and it moves 148 `coverage_linear` triples by one level.

A kernel conforms when, for every valid input (every combination of arguments within the types and ranges of the Kernel interface table, including glyph boxes partly or wholly off the page, boxes of zero size, overlapping boxes, any colours and G = 0), its output is within the tolerance of the page's `aa_mode` in both e_max and f_diff. The reference values are exact: R05 and R06 computed in double precision from the integer sums of R04, as the worked examples of R05 show.
