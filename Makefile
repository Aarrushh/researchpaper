# PURPOSE: One-word commands for every stage of the pipeline.
# WHAT GOES HERE: Make targets that call scripts/ with the right config.
# FILLED IN AT STEP: Scaffolded at setup. Used from W1.3 onward.
# DO NOT: Put experiment logic here. Targets only call scripts.

PYTHON ?= python
SWEEP  ?= configs/sweeps/pilot.yaml

.DEFAULT_GOAL := help
.PHONY: help env check-env test test-models test-all readers render-check pilot sweep analyze figures clean-interim

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
	$(PYTHON) scripts/01_check_readers.py --config configs/readers.yaml

render-check: ## Render one page per reader at four cell phases and 0.5 px, with grid overlays (W1.5)
	for r in qwen smolvlm smolvlm500m; do $(PYTHON) scripts/02_render_check.py --reader $$r || exit 1; done

pilot: ## Run the pilot sweep (W2.3)
	$(PYTHON) scripts/10_run_sweep.py --config configs/sweeps/pilot.yaml

sweep: ## Run the full sweep (W3.3)
	$(PYTHON) scripts/10_run_sweep.py --config configs/sweeps/full.yaml

analyze: ## Run statistics and write reports/tables (W2.4, W3.4); SWEEP=... to pick config
	$(PYTHON) scripts/20_analyze.py --config $(SWEEP)

figures: ## Generate Figures 1 and 2 into reports/figures (W2.5, W3.4); SWEEP=... to pick config
	$(PYTHON) scripts/30_make_figures.py --config $(SWEEP)

clean-interim: ## Delete regenerable rendered pages in data/interim (keeps README files)
	find data/interim -type f ! -name 'README.md' ! -name '.gitkeep' -delete
