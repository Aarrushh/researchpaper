# PURPOSE: One-word commands for every stage of the pipeline.
# WHAT GOES HERE: Make targets that call scripts/ with the right config.
# FILLED IN AT STEP: Scaffolded at setup. Used from W1.3 onward.
# DO NOT: Put experiment logic here. Targets only call scripts.

PYTHON ?= python
SWEEP  ?= configs/sweeps/full.yaml
READERS ?=                        # e.g. READERS=qwen runs one reader of the ladder, pilot or sweep
DEADLINE ?=                       # e.g. DEADLINE=2026-10-20T12:00 stops the sweep at a page boundary (section 3.6)
RESUME ?=                         # e.g. RESUME=results/runs/<run_id> continues an interrupted run of one reader

.DEFAULT_GOAL := help
.PHONY: help env check-env test test-models test-all readers render-check equivariance coherence
.PHONY: contract reader-facts ladder pilot sweep analyze levels figures figures-stage0 mtf freeze freeze-check freeze-check-all clean-interim

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

env: ## Install mrr in editable mode plus requirements (install PyTorch first)
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -r requirements.txt

check-env: ## Print Python, packages, CUDA and GPU; fail if torch or CUDA is missing
	$(PYTHON) scripts/00_check_env.py

test: ## Run tests except those marked todo or model
	$(PYTHON) -m pytest -m "not todo and not model"

test-models: ## Run tests that need torch, transformers and downloaded weights (D17)
	$(PYTHON) -m pytest -m model -v

test-all: ## Run every test, including todo tests
	$(PYTHON) -m pytest

readers: ## Check reader resize and token counts (W1.6)
	$(PYTHON) scripts/01_check_readers.py --config "configs/readers.yaml"

render-check: ## Render one page per reader at four cell phases and 0.5 px, with grid overlays (W1.5)
	for r in qwen smolvlm smolvlm500m; do $(PYTHON) scripts/02_render_check.py --reader $$r || exit 1; done

equivariance: ## Shift-equivariance error per layer of each vision tower, and Figure 0 (W1.9)
	$(PYTHON) scripts/03_equivariance.py --config "configs/readers.yaml"

coherence: ## Reproduce the D10 coherence table from the layout code, no model (W1.10)
	$(PYTHON) scripts/05_coherence.py --config "configs/readers.yaml"

contract: ## Contract checks behind R01 to R06 and the Conformance table of the spec, into reports/tables/contract (W4.1); no model
	$(PYTHON) scripts/08_contract_checks.py

reader-facts: ## Reader facts table behind R07 and R08 of the spec, into reports/tables/readers (W4.1); GPU
	$(PYTHON) scripts/01_check_readers.py --config "configs/readers.yaml" --out "reports/tables/readers/reader_facts.csv"

ladder: ## Run the font-size ladder, block Z at phase 0 (W1.10); READERS=... to pick readers, RESUME=... to continue a run
	$(PYTHON) scripts/10_run_sweep.py --config "configs/sweeps/ladder.yaml" $(if $(READERS),--readers $(READERS),) $(if $(RESUME),--resume $(RESUME),)

pilot: ## Run the pilot sweep (W2.3); READERS=... to pick readers, RESUME=... to continue a run
	$(PYTHON) scripts/10_run_sweep.py --config "configs/sweeps/pilot.yaml" $(if $(READERS),--readers $(READERS),) $(if $(RESUME),--resume $(RESUME),)

sweep: ## Run the full sweep (W3.3); READERS=..., DEADLINE=... and RESUME=... to pick readers, a stop time or a run to continue
	$(PYTHON) scripts/10_run_sweep.py --config "configs/sweeps/full.yaml" $(if $(READERS),--readers $(READERS),) $(if $(RESUME),--resume $(RESUME),) $(if $(DEADLINE),--deadline $(DEADLINE),)

analyze: ## Run statistics and write reports/tables (W2.4, W3.4); SWEEP=... to pick config
	$(PYTHON) scripts/20_analyze.py --config "$(SWEEP)"

levels: ## Exploratory per-level CER and its parts from the frozen runs, into reports/tables/exploratory (W4.1); SWEEP=... to pick config
	$(PYTHON) scripts/21_levels.py --config "$(SWEEP)"

figures: ## Generate Figures 1, 2 and 3a into reports/figures (W2.5, W3.4, W3.6); SWEEP=... to pick config
	$(PYTHON) scripts/30_make_figures.py --config "$(SWEEP)"

figures-stage0: ## Draw Study 3 stage 0's Figures 5 and 6 from reports/tables/s3_stage0 (ANALYSIS_PLAN_3.md section 2.7)
	$(PYTHON) scripts/31_stage0_figures.py

mtf: ## Layer-0 MTF of each reader from its patch-embedding weights, fetched by byte range; no model, no GPU (W3.6)
	$(PYTHON) scripts/04_mtf.py --config "configs/readers.yaml"

freeze: ## Copy results into results/final with a README of SHA-256 hashes (W3.5); refuses to overwrite a freeze
	$(PYTHON) scripts/40_freeze.py

freeze-check: ## Check results/final: every hash, and the tables regenerate byte for byte (W3.5); SWEEP=... to pick config
	$(PYTHON) scripts/40_freeze.py --check --sweep "$(SWEEP)"

freeze-check-all: ## Check results/final and every paper table: Study 1, Study 2, stage 0 and Study 2's re-scoring (~25 min, CPU)
	$(PYTHON) scripts/40_freeze.py --check --sweep configs/sweeps/full.yaml --sweep configs/sweeps/study2.yaml --fit s3_stage0 --fit study2_rescored

clean-interim: ## Delete regenerable rendered pages in data/interim (keeps README files)
	find data/interim -type f ! -name 'README.md' ! -name '.gitkeep' -delete
