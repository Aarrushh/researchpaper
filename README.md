<!--
PURPOSE: Front page of the repository. First file anyone opens.
WHAT GOES HERE: Research question, hypotheses, gate status table, folder map, reproduce commands, repo rules.
FILLED IN AT STEP: Scaffolded at setup. Status table updated at G1, G2, G3 and at each freeze tag.
DO NOT: Put results, numbers or paper prose here. Do not un-anonymise before review ends.
-->

# Rendering for Machine Readers: Text Rendering Designed for Vision Encoders

Short name and Python package: `mrr`.

## Research question

Text rendering was designed for the human visual system. In visual text compression (VTC), long text is drawn onto images and read by a vision-language model (VLM) whose encoder cuts the page into fixed patches. We measure how that encoder reads rendered text as a function of where glyphs land on its patch grid, derive a rendering specification from the measurements, and test whether a language model can implement that specification as a GPU kernel.

## Hypotheses

- **H1:** Reading error (character error rate) varies periodically with glyph phase relative to the encoder's token grid, with period equal to that encoder's cell size. Patch-aware rendering lowers reading error at equal token count, most at small font sizes.
- **H2:** A language model (GLM-5.3) given only the written specification can write a Triton kernel whose output matches the reference renderer.

## Status

| Item | Target date | Status | Decision link |
|---|---|---|---|
| G1: readers work, token formula holds, phase renders correct | 2026-10-02 | not started | [docs/05_decisions.md](docs/05_decisions.md) |
| `prereg-v1` tag | 2026-10-03 | not started | [PREREGISTRATION.md](PREREGISTRATION.md) |
| G2: is CER periodic in phase? | 2026-10-09 | not started | [docs/05_decisions.md](docs/05_decisions.md) |
| `data-freeze-v1` tag | 2026-10-16 | not started | [results/README.md](results/README.md) |
| G3: does the H2 kernel match the reference? | 2026-10-23 | not started | [docs/05_decisions.md](docs/05_decisions.md) |
| `spec-v1` tag | 2026-10-23 | not started | [spec/RENDERING_SPEC.md](spec/RENDERING_SPEC.md) |
| `submission` tag | 2026-10-28 | not started | [paper/README.md](paper/README.md) |

## Where things live

```
mrr/
├── README.md, PREREGISTRATION.md, LAB_NOTEBOOK.md, HANDOFF.md   project front matter and governance
├── Makefile, requirements.txt, pyproject.toml                   build, install and run commands
├── .github/workflows/     CI: runs the non-todo tests on every push
├── docs/                  plan, learning guide, pipeline steps, data dictionary, decision log
├── references/            citation ledger, BibTeX, per-paper reading notes
├── configs/               reader, render and sweep configuration (YAML)
├── data/                  fonts and texts (external, raw), rendered pages (interim), processed data; never committed
├── src/mrr/               the Python package: colour, grid, rendering, readers, scoring, stats, plotting, run bookkeeping
├── scripts/               numbered entry points, run in order
├── notebooks/             numbered exploration notebooks; nothing in the paper comes from here
├── results/               per-run folders (runs/) and frozen CSVs (final/)
├── reports/               generated figures and tables
├── spec/                  the rendering specification given to the H2 model
├── h2/                    H2 protocol: prompts, verbatim kernels, judge
├── paper/                 LaTeX skeleton; humans write all prose
└── tests/                 pytest suite
```

## How to reproduce

- `make env`: install the package in editable mode plus requirements (install PyTorch first).
- `make test`: run every test except those marked `todo`.
- `make pilot`: run the pilot sweep from `configs/sweeps/pilot.yaml`.
- `make sweep`: run the full sweep from `configs/sweeps/full.yaml`.
- `make analyze`: run statistics over run CSVs and write `reports/tables/`.
- `make figures`: generate Figures 1 and 2 into `reports/figures/`.

Run `make help` for every target.

## Rules of this repo

1. Raw data is immutable. Nothing in `data/raw/` is edited after download.
2. Results come only from scripts. No CSV is edited by hand.
3. Figures are generated, never hand-edited.
4. `h2/kernels/` is never edited by humans. Model output is saved verbatim.
5. Paper prose is human-written only.

## Start here

Open [docs/00_START_HERE.md](docs/00_START_HERE.md).
