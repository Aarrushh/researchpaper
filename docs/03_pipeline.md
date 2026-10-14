<!--
PURPOSE: The authoritative list of steps and gates. Every TODO in the repo points to a step ID here.
WHAT GOES HERE: One checkbox per step with what to do, files touched and done-when. Gates as boxes.
FILLED IN AT STEP: Ticked as each step completes, W1.1 to W5.5.
DO NOT: Rename or renumber step IDs; TODO markers across the repo depend on them.
-->

# Pipeline

Tick a box only when its "done when" is true. Gates are go/no-go decisions.

Dates follow the re-plan of 12 October 2026 (D21). If G1 has not passed by Wed 14 Oct 12:00, every later date moves by one day. The rule fired on 14 October (D25): every date after Wed 14 Oct 12:00 below is one day later than written. Steps W1.8 to W1.10 and W3.6 to W3.8 were appended that day; no existing ID changed. One laptop runs everything (D20), so a check written for "both laptops" runs once, on that laptop.

## W1 Setup and theory (26 Sep to Wed 14 Oct 2026)

- [ ] **W1.1** Read `docs/00_START_HERE.md` and `docs/01_research_plan.md`.
  - Files: `docs/00_START_HERE.md`, `docs/01_research_plan.md`, `HANDOFF.md`, `LAB_NOTEBOOK.md`
  - Done when: both authors have read both files and written a first HANDOFF and notebook entry.
- [ ] **W1.2** Work through learning topics A1 to A6 in `docs/02_learning_guide.md`, doing every exercise.
  - Files: `LAB_NOTEBOOK.md`, `references/citations.md`, `references/reading_notes/`
  - Done when: every A1 to A6 exercise has an answer in the notebook; prior-art rows in citations.md have a claim.
- [x] **W1.3** Create the Python environment; run `make env` and `make check-env`.
  - Files: `requirements.txt`, `requirements.lock.txt` (new), `LAB_NOTEBOOK.md`
  - Done when: `make check-env` exits 0 on both laptops and `requirements.lock.txt` is committed.
  - Note (Phase 1 review, 14 Oct): since the review added the sysmem-fallback check, `make check-env` exits 1 on this laptop until the NVIDIA setting of build guide 1B.2 is changed; every other check passes.
- [ ] **W1.4** Download fonts and texts into `data/external/` and `data/raw/`; record licences and sources.
  - Files: `data/external/fonts/README.md`, `data/raw/texts/README.md`, `src/mrr/text_source.py`
  - Done when: each font has its licence file beside it; each text has book ID and title recorded; `text_source.py` is implemented.
- [x] **W1.5** Implement `src/mrr/color.py` and `src/mrr/render.py`; make `tests/test_color.py` and `tests/test_render.py` pass.
  - Files: `src/mrr/color.py`, `src/mrr/render.py`, `src/mrr/grid.py`, `configs/render_defaults.yaml`, `scripts/02_render_check.py`, tests
  - Done when: both test files pass without the todo mark and `make render-check` writes overlay images.
- [x] **W1.6** Implement `src/mrr/readers.py`; confirm model IDs, revisions and token formulas; run `scripts/01_check_readers.py`.
  - Files: `src/mrr/readers.py`, `src/mrr/grid.py`, `configs/readers.yaml`, `configs/render_defaults.yaml`
  - Done when: `configs/readers.yaml` has no W1.6 TODOs and `make readers` shows token counts equal to the formula.
- [x] **W1.7** Time one transcription per model; record in `LAB_NOTEBOOK.md`.
  - Files: `LAB_NOTEBOOK.md`, `configs/readers.yaml` (`max_new_tokens`)
  - Done when: latency and peak VRAM for each reader are in the notebook.
- [x] **W1.8** The no-resize test and determinism (B2).
  - Files: `tests/test_no_resize.py`, `tests/test_reader_facts.py` (both marked `model`), `LAB_NOTEBOOK.md`
  - Done when: `tests/test_no_resize.py` passes on both laptops (on the one laptop, under D20) and the summary is in `LAB_NOTEBOOK.md`.
- [x] **W1.9** Shift-equivariance error at the embedding level: paper Figure 0 (B4).
  - Files: `src/mrr/equivariance.py`, `src/mrr/readers.py` (additions), `scripts/03_equivariance.py`, `tests/test_equivariance.py`, `src/mrr/plotting.py`, `results/equivariance/`, `reports/figures/fig0_equivariance.pdf`
  - Done when: `scripts/03_equivariance.py` writes `fig0_equivariance.pdf` and the layer-0 values match the reference values in Appendix C of the build guide (asserted in `tests/test_equivariance.py`).
- [ ] **W1.10** Coherence of the layouts and the font-size ladder.
  - Files: `src/mrr/render.py`, `src/mrr/grid.py` (additions), `scripts/05_coherence.py`, `configs/sweeps/ladder.yaml`, `tests/test_configs.py`, `Makefile` (`ladder`), `results/runs/<run_id>/`, `LAB_NOTEBOOK.md`
  - Done when: the coherence table and the chosen size per reader are in `LAB_NOTEBOOK.md`.

> **GATE G1 (Wed 14 Oct, 12:00)**
> Question: Do the readers see exactly the pages we render, and does paper Figure 0 show the predicted periods?
> - Go: start W2.
> - No-go: fix the failing part; W2 does not start until G1 passes.
>
> Record decision in docs/05_decisions.md

## W2 Pilot sweep (Wed 14 to Fri 16 Oct)

- [ ] **W2.1** Fill `PREREGISTRATION.md` completely; commit; create git tag `prereg-v1`.
  - Files: `PREREGISTRATION.md`
  - Done when: no W2.1 TODOs remain and tag `prereg-v1` exists.
- [ ] **W2.2** Fill `configs/sweeps/pilot.yaml`.
  - Files: `configs/sweeps/pilot.yaml`, `docs/01_research_plan.md`
  - Done when: every level is set and the total run count is written in the file's comment.
- [ ] **W2.3** Run `make pilot`.
  - Files: `scripts/10_run_sweep.py`, `results/runs/<run_id>/`
  - Done when: a run folder exists with `manifest.json` and a complete `results.csv`.
- [ ] **W2.4** Implement `src/mrr/score.py` and `src/mrr/stats.py`; make their tests pass.
  - Files: `src/mrr/score.py`, `src/mrr/stats.py`, `scripts/20_analyze.py`, tests
  - Done when: `tests/test_score.py` and `tests/test_stats.py` pass without the todo mark; `make analyze` writes a table.
- [ ] **W2.5** Produce Figure 1 (phase curve) with `make figures`.
  - Files: `src/mrr/plotting.py`, `scripts/30_make_figures.py`, `reports/figures/`
  - Done when: Figure 1 exists and was generated by the script only.

> **GATE G2 (Fri 16 Oct, 12:00)**
> Question: Is CER periodic in phase, by the criterion written in PREREGISTRATION.md?
> - Go: run W3 as planned.
> - No-go: switch to the negative-result framing described in `docs/01_research_plan.md`; W3 still runs.
>
> Record decision in docs/05_decisions.md

## W3 Full experiment (Fri 16 to Wed 21 Oct)

- [ ] **W3.1** Add the second reader to the sweep.
  - Files: `configs/readers.yaml`, `configs/sweeps/full.yaml`
  - Done when: the second reader passes `make readers` and is listed in `full.yaml`.
- [ ] **W3.2** Fill `configs/sweeps/full.yaml` (all factors).
  - Files: `configs/sweeps/full.yaml`
  - Done when: every level is set and the total run count and estimated hours are in the comment.
- [ ] **W3.3** Run conventional and patch-aware rendering on identical pages (paired).
  - Files: `scripts/10_run_sweep.py`, `results/runs/<run_id>/`
  - Done when: every page has a row for both render modes for every reader.
- [ ] **W3.4** Run all statistics; produce Figure 2 (accuracy versus tokens trade-off).
  - Files: `scripts/20_analyze.py`, `scripts/30_make_figures.py`, `src/mrr/plotting.py`, `reports/`
  - Done when: all preregistered tests have a table in `reports/tables/` and Figure 2 exists.
- [ ] **W3.5** Copy final CSVs to `results/final/`; create git tag `data-freeze-v1`.
  - Files: `scripts/40_freeze.py`, `tests/test_freeze.py`, `Makefile` (`freeze`, `freeze-check`), `results/final/`
  - Done when: tag `data-freeze-v1` exists and every paper number traces to a file in `results/final/`.
- [x] **W3.6** Layer-0 MTF from the weights (B7, stretch). Done 14 October 2026, CPU only, no model loaded.
  - Files: `src/mrr/mtf.py`, `scripts/04_mtf.py`, `tests/test_mtf.py`, `results/mtf/layer0_<reader>.csv` and `manifest.json`, `data/external/weights/README.md` (the fetched tensors, git-ignored), `src/mrr/plotting.py` (`plot_mtf`), `scripts/30_make_figures.py` (`--mtf-dir`), `Makefile` (`mtf`), `reports/figures/fig3a_layer0_mtf.pdf`
  - Done when: `tests/test_mtf.py` passes and paper Figure 3a (`reports/figures/fig3a_layer0_mtf.pdf`) exists.
- [ ] **W3.7** Grating MTF through the whole encoder (B7, stretch).
  - Files: `src/mrr/gratings.py`, `tests/test_gratings.py`, `scripts/06_gratings.py`, `results/mtf/gratings_<reader>.csv`, `scripts/30_make_figures.py`
  - Done when: paper Figure 3b (`reports/figures/fig3b_encoder_mtf.pdf`) exists.
- [ ] **W3.8** Linear probe and bits per token (B7, stretch).
  - Files: `src/mrr/probe.py`, `src/mrr/readers.py` (`token_features`), `scripts/07_probe.py`, `tests/test_probe.py`, `results/probe/probe_<reader>.csv`, `reports/tables/probe.csv`
  - Done when: the probe table `reports/tables/probe.csv` exists.

W3.6 to W3.8 are stretch work and the first to go under the cut order: tick each one, or mark it as cut with the decision that cut it.

> **FREEZE (Wed 21 Oct)**
> Question: Are the final results in `results/final/` and tagged `data-freeze-v1`?
> - Go: run H2 in full (W4.2 to W4.4), under D16.
> - No-go: write the specification (W4.1), skip W4.2 to W4.4 and report H2 as not run, under D16.
>
> Record decision in docs/05_decisions.md (this gate decides D16)

## W4 Specification and H2 (Thu 22 to Sat 24 Oct)

- [ ] **W4.1** Write `spec/RENDERING_SPEC.md`; every rule cites the result that supports it.
  - Files: `spec/RENDERING_SPEC.md`
  - Done when: no W4.1 TODOs remain and every rule's EVIDENCE names a file in `reports/`.
- [ ] **W4.2** Run Triton tutorials 01 and 02; notes in `LAB_NOTEBOOK.md`.
  - Files: `LAB_NOTEBOOK.md`, `requirements.txt` (uncomment triton)
  - Done when: both tutorials ran on one laptop and notes are logged.
- [ ] **W4.3** Give GLM-5.3 only the spec via `h2/prompts/`; save its output verbatim in `h2/kernels/`.
  - Files: `h2/prompts/prompt_vNN.md`, `h2/kernels/kernel_<prompt-version>_<attempt>.py`, `h2/README.md`
  - Done when: prompt and output are committed unedited, with model string, date and settings recorded.
- [ ] **W4.4** Run `h2/judge/compare.py` against the reference renderer.
  - Files: `h2/judge/compare.py`, `reports/tables/`
  - Done when: a judge table exists for every kernel attempt.

> **GATE G3 (Sat 24 Oct)**
> Question: Does the kernel's output match the reference renderer within the preregistered tolerance?
> - Go: report H2 as supported. Tag `spec-v1`.
> - No-go: report H2 as a documented failure with judge numbers. Tag `spec-v1`.
>
> Record decision in docs/05_decisions.md

## W5 Write and submit (Sat 24 to Wed 28 Oct)

- [ ] **W5.1** Humans write sections 3 to 6.
  - Files: `paper/sections/03_related_work.tex` to `paper/sections/06_results.tex`, `paper/main.tex`
  - Done when: sections 3 to 6 are drafted by the humans; MLSys style is in `main.tex`.
- [ ] **W5.2** Humans write sections 1, 2, 7, 8, 9 and the abstract.
  - Files: `paper/sections/00_abstract.tex`, `01`, `02`, `07`, `08`, `09`
  - Done when: a full draft compiles within 10 pages excluding references.
- [ ] **W5.3** Audit every number in the paper against `results/final/` and every claim against `references/citations.md`.
  - Files: `paper/`, `references/citations.md`, `references/references.bib`
  - Done when: every number and citation has a verified row; no W5.3 TODOs remain.
- [ ] **W5.4** Anonymise paper and supplementary material.
  - Files: `paper/`, supplementary archive, `LICENSE_TODO.md`
  - Done when: no names, affiliations, repo URLs or identifying paths in PDF, metadata or supplement.
- [ ] **W5.5** Submit on OpenReview on 28 Oct. 29 and 30 Oct are buffer only. Tag `submission`.
  - Files: `CITATION.cff` (after review), `README.md` status table
  - Done when: OpenReview shows the submission and tag `submission` exists.
