# Rendering for Machine Readers

**Text rendering was built for human eyes. Vision-language models read it secondhand. This project asks what text drawn *for the model* would look like, and starts measuring it.**

Python package: `mrr`. Status: experiments finished and frozen (`analysis-freeze-v2`); paper in progress, expected by the end of November 2026.

For the theory behind the study in plain words (how a model sees a page, phase and coherence, every study's question and result, and what the evidence does and does not support), open **[docs/paper_guide.html](docs/paper_guide.html)** in a browser.

---

## 1. The vision

Every font, hinting rule and anti-aliasing filter we use was tuned for the human visual system. Vision-language models (VLMs) see differently. They cut an image into a fixed grid of patches, merge those into vision tokens, and hand the tokens to a language model. Text rendered for people is a lossy, accidental input format for that pipeline.

More and more systems now feed text to models *as images*: screenshots, documents, and visual text compression, where long context is drawn onto a page to spend fewer tokens (DeepSeek-OCR, Glyph). In those systems, rendering decides how much text each vision token can carry before errors creep in.

The long-term goal is a **rendering specification for machine readers**. It would say exactly how to draw text for a VLM so that:

- every pixel is specified, and a GPU kernel can produce it byte for byte;
- the layout is written in terms of the model's own geometry: patch size *P*, merge factor and native frame;
- the rules carry measured, per-model coefficients, so each new model needs a cheap calibration rather than guesswork.

The hypothesis behind the work: if text is drawn to fit how encoders actually read, each vision token carries more text. Fewer tokens per page means less prefill compute and memory on the GPU. **That efficiency link is the motivation, not a result yet.** Section 4 says what has and has not been measured.

## 2. What this repo shows

Three studies on five open VLMs (Qwen2.5-VL-3B, Qwen2.5-VL-7B, InternVL3-2B, SmolVLM-2.2B and SmolVLM-500M): **40,788 transcription rows** plus a 216-row size ladder, all on one laptop GPU. Every hypothesis was preregistered and tagged before data were read. Every table regenerates byte for byte from the frozen rows.

Each result carries its status:

- **[C]** preregistered test
- **[D]** preregistered description
- **[E]** exploratory, frozen
- **[P]** post hoc
- **[T]** deterministic, checked by tests

### What the study achieved

- **A pixel contract for text given to a VLM** [T]. `spec/RENDERING_SPEC.md` has rules R01–R11. R01–R06 define every byte of a page. R07 delivers the page at the reader's native size, so nothing is resized. R08 keeps text out of pixels no patch can see. Contract checks (`make contract`) show the reference renderer meets them.
- **The patch grid matters, but only where glyphs share a phase** [C]. With every glyph at the same offset against the grid, error rises and falls periodically, mostly at the patch period, for all three Study 1 readers. Ordinary text shifted sideways shows no phase effect, because its glyphs already land at every offset (`reports/tables/full/harmonics.csv`).
- **A mechanism, then a preregistered test of it.** Snapping one glyph into each patch-wide slot was predicted to help. It *raised* error on random text, by +0.162, +0.114 and +0.046 CER [C]. The cause was in the decoder: wide letters take two slots and leave white gaps that the model writes as spaces [P]. Study 2 tested the fix, a monospace font whose advance is exactly one patch. It removed the false spaces on all four random-text readers (−0.17 to −0.24 inserted spaces per character, H3) [C] (`reports/tables/study2/paired.csv`).
- **A calibration protocol for any reader.** It costs **16–74 GPU-minutes** on a laptop GPU per reader [E]. It measures how error falls with text size and what one vision token carries at 1% error. Readers with identical patch geometry differ by about 4.4×. At 1% CER on prose, Qwen2.5-VL-3B's vision token carries about 10.4 characters and InternVL3-2B's about 2.4 [E][P] (`reports/tables/load/readoff_tokens.csv`).
- **The first price for visual text compression per reader** [E][P]. At 1% CER on prose, only the Qwen pair's vision tokens beat text tokens: 2.27× and 2.46× the characters of a text token. On random strings no reader reaches 1×; the best is 0.58×.
- **Infrastructure others can reuse.** It includes a byte-exact renderer, a sweep runner, scoring that separates misreading from model behaviour (skipped lines, chat preambles, case changes), and paired and harmonic statistics. A freeze checks **44,989 hashed files** and regenerates **37 tables** byte for byte. The test suite has **846 tests**.

## 3. What the study failed at, and why it matters

- **The rules did not transfer (H7 failed)** [C]. Study 2 preregistered that its layout rules would hold on two new readers. Qwen2.5-VL-7B missed the containment test (H4), so the generalisation claim is withdrawn as worded. Monospace beat ordinary text for only 2 of 4 readers (H5).
  **Why it matters:** this is the strongest argument for the direction of the project. A single fixed rendering does not work for every model. The specification has to be **parameterised**: rules written in terms of *P* and the merge factor, with coefficients measured per reader. The calibration protocol above is the start of that.
- **The size curve measured page load, not acuity** [E][P]. In the calibration, bigger letters also meant fewer characters per page. A check added after the analysis (Amendment 2 of `ANALYSIS_PLAN_3.md`) separated the two. On random text, full pages read 1.7–3.2× worse than half-full pages at about the same size. Above the cliff, a bigger font at a fixed load changed error by only ×0.84–0.97, where the curve predicted ×0.48–0.53.
  **Why it matters:** "how much text fits on a page" is a first-class variable in the spec, not a side effect of font size. Future calibrations must measure size at a fixed load.
- **Patch-aware rendering was not a free win.** Clause (c) of H1 reversed [C]. The largest levers turned out to be which reader, which kind of text and how full the page is, not the fine details of glyph placement.
- **H2 never ran.** The preregistered test of whether a language model can implement the specification as a GPU kernel from the written spec alone was not carried out. It is reported as a deviation.

## 4. Open questions

1. **Compute, not just tokens.** Do token savings become real GPU savings once the encoder's own pass, batching and memory are measured? Nothing here measures throughput, latency or energy. That is the next experiment the vision depends on.
2. **Capacity at a fixed load.** What is a reader's true acuity once page load is held constant, and how should the spec trade size against characters per page?
3. **Generalisation.** Which coefficients can be predicted from architecture (encoder family, *P*, merge, window attention), and which must always be measured? Larger models, other encoder families and closed models are untested.
4. **Beyond the lab page.** The open problems include non-Latin scripts, other typefaces, real documents (columns, tables, figures), resized inputs, and tasks beyond transcription such as question answering and summarisation.
5. **The GPU renderer.** A kernel that renders the spec on the GPU, so pages never leave device memory, and the H2 question of whether a model can write it from the spec.
6. **Co-design.** If rendering is part of the model's input interface, should encoders be trained on spec-rendered text, and should the spec and the encoder be designed together?

## 5. Reproducing the results

Requirements: Python 3.11+, and PyTorch with CUDA for anything that runs a model. Re-running the analysis from the frozen data needs only the CPU.

```bash
make env               # install mrr in editable mode (install PyTorch first)
make test              # the test suite without model tests (what CI runs)
make freeze-check-all  # verify all 44,989 hashes and regenerate every table byte for byte (~35 min, CPU)
make contract          # the spec's contract checks (no model)
make figures-stage0    # redraw Figures 5 and 6 from the frozen tables
make help              # every target
```

`make` is optional: each target is one command, listed in the `Makefile`.

**The freeze tags, in order:**

| Tag | What it fixes |
|---|---|
| `prereg-v1` | Study 1's plan (`PREREGISTRATION.md`), before any data |
| `data-freeze-v1` | Study 1's data |
| `spec-v1` | the rendering specification (`spec/RENDERING_SPEC.md`) |
| `prereg-v2` | Study 2's plan (`PREREGISTRATION_2.md`), before any data |
| `data-freeze-v2` | every run folder, 44,989 files hashed into `results/final/` |
| `data-freeze-s3-stage0` | the calibration's tables |
| `analysis-freeze-v1` | every table and figure regenerates from `results/final/` |
| `analysis-freeze-v2` | the same, plus the load check (Amendment 2) |

**Where things live:**

| Path | Contents |
|---|---|
| `spec/` | the rendering specification, rules R01–R11 |
| `src/mrr/` | rendering, patch grid, readers, scoring, statistics, plotting |
| `scripts/` | numbered entry points, run in order |
| `configs/` | readers, render settings and sweeps (YAML) |
| `results/final/` | the frozen run folders |
| `reports/tables/`, `reports/figures/` | every number and figure in the paper |
| `docs/` | plan, pipeline, data dictionary, decision log, and the theory guide |
| `LAB_NOTEBOOK.md` | dated record of everything that was done |
| `paper/` | the paper's LaTeX skeleton |

The hardware for every run was one RTX 5060 laptop GPU (8 GB) with 32 GB of RAM, using greedy decoding and a single prompt. Qwen2.5-VL-7B ran in 4-bit NF4; the other readers ran in bfloat16.

## 6. The paper

The paper is in progress, **expected by the end of November 2026**. It will report the mechanisms, Study 2's preregistered tests and the calibration method, and every claim will name its table and status. Until it is out, this repository and `docs/paper_guide.html` are the record of the work.

## Licence

- **Code** (`src/`, `scripts/`, `tests/`, `h2/judge/`): [MIT](LICENSE).
- **Data and reports** (`results/`, `reports/`): [CC BY 4.0](LICENSE-DATA.md).
- **Third-party material** keeps its own licence. The fonts in `data/external/fonts/` are SIL OFL 1.1, with each licence next to its font. The prose texts are public-domain Project Gutenberg books. Rendered pages inherit both.
