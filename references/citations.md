<!--
PURPOSE: Ledger linking every factual claim in the paper to a source someone actually opened.
WHAT GOES HERE: One row per claim: source, exact location, who verified it and when, where it is used.
FILLED IN AT STEP: Rows added from W1.2; audited at W5.3.
DO NOT: Add a row you have not verified by opening the source. Do not paraphrase numbers from memory.
-->

# Citation ledger

**Rule: nothing enters the paper without a row here.** "Verified by" means that person opened the source and found the claim at the stated location.

| Claim | Source | Location (page or section) | Verified by | Date verified | Used in paper section |
|---|---|---|---|---|---|
| TODO[W1.2]: claim from Glyph | arXiv 2510.17800 | | | | |
| TODO[W1.2]: claim from DeepSeek-OCR | arXiv 2510.18234 | | | | |
| TODO[W1.2]: claim from LensVLM | arXiv 2605.07019 | | | | |
| TODO[W1.2]: claim from ZeroSense | arXiv 2608.01848 | | | | |
| TODO[W1.2]: claim from Grid2Matrix | arXiv 2604.09687 | | | | |
| TODO[W1.2]: claim from Steal the Patch Size | arXiv 2607.00174 | | | | |
| TODO[W1.2]: claim from CodeOCR (title unconfirmed) | arXiv 2602.01785 | | | | |
| TODO[W1.2]: claim from ViT | arXiv 2010.11929 | | | | |
| TODO[W1.2]: claim from KernelBench | arXiv 2502.10517 | | | | |
| sRGB transfer constants: decode is v / 12.92 when v <= 0.04045, otherwise ((v + 0.055) / 1.055)^2.4; encode is 1.055 * l^(1/2.4) - 0.055 when l > 0.0031308, otherwise 12.92 * l. Our encode threshold, 0.04045 / 12.92, is not in this source (D24) | W3C, CSS Color Module Level 4, Candidate Recommendation Draft of 9 October 2026, https://www.w3.org/TR/2026/CRD-css-color-4-20261009/ (the sample code is non-normative; IEC 61966-2-1 itself was not read) | Section 19, "Sample code for Color Conversions" (#color-conversion-code), functions lin_sRGB and gam_sRGB | Claude | 2026-10-12 | TODO[W5.3] |
