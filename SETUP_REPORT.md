<!--
PURPOSE: Record of what the setup agent created, tested and skipped.
WHAT GOES HERE: Setup facts only. Written once at scaffold time.
FILLED IN AT STEP: Scaffold time (before W1.1). Not updated afterwards.
DO NOT: Use this as a progress log; use LAB_NOTEBOOK.md.
-->

# Setup report

Written by the setup agent on 2026-09-27.

## What was created

Every file and folder in the requested tree, plus `.venv/` (gitignored). Summary:

- Governance: README, PREREGISTRATION, LAB_NOTEBOOK, HANDOFF, CITATION.cff, LICENSE_TODO.
- Build: Makefile (with `help`), requirements.txt (unpinned, triton commented), pyproject.toml, .gitignore, CI workflow.
- docs/ 00 to 05, references/ (ledger, BibTeX with arXiv IDs only, reading-note template), configs/ (readers, render defaults, pilot and full sweeps).
- data/ READMEs, src/mrr/ stubs, scripts/ skeletons, notebook with a header cell only, results/, reports/, spec/, h2/, paper/ skeleton, tests/.
- Fully implemented: `src/mrr/manifest.py`, `scripts/00_check_env.py`, the Makefile, and the pytest configuration.
- Every other function raises `NotImplementedError` with a step-ID TODO.

## Test result

`pytest -m "not todo"` in a fresh venv with only `pip install -e ".[test]"` (numpy, pyyaml, pytest):

- 13 passed, 11 deselected.
- The smoke tests import every `mrr` module, check the run ID format, write a manifest, and check that `RESULTS_COLUMNS` in `manifest.py` matches the column list in `docs/04_data_dictionary.md`.
- Running all tests (`make test-all`) gives 9 failed and 2 skipped as expected: the todo tests hit `NotImplementedError`, and the render tests skip because no font is downloaded yet.
- `scripts/00_check_env.py` was run. It exited 1 with the "torch is not installed" message, as designed (the setup machine has no torch and no GPU).

## TODO counts

Counted with a grep for the TODO-with-bracket marker, excluding `.venv/`, `.git/` and `__pycache__/`.

| Week | Count |
|---|---|
| W1 | 103 |
| W2 | 49 |
| W3 | 15 |
| W4 | 28 |
| W5 | 15 |
| **Total** | **210** |

TODOs lacking a valid step ID: **none**. Every ID used exists in `docs/03_pipeline.md`.

## Deviations and skips

1. **Location.** The agent ran inside an unrelated, public repository (`ashm-023/sylva`). Committing a double-blind paper repo there would break anonymity. The scaffold was built as a separate, standalone git repository instead. Nothing was pushed anywhere.
2. **`.gitignore`.** Added `data/external/**` (with README and .gitkeep negations) alongside raw, interim and processed. Without it, downloaded font files would have been committable, which contradicts the fonts README.
3. **CI install.** The workflow installs `pip install -e ".[test]"` (numpy, pyyaml, pytest) rather than requirements.txt, so CI needs no GPU or model packages.
4. **Header blocks.** Markdown files use an HTML comment; the notebook carries its header inside the markdown cell (JSON has no comments); `.gitkeep` files carry a `#` comment header.
5. **Decision dates.** D1 to D3 in `docs/05_decisions.md` are dated 2026-09-27 (scaffold date) and marked as decided before it, because the true decision date was not given.
6. **Links in the learning guide.** The header line "All links were verified to exist on 25 Sep 2026" was included as instructed. The setup agent did not itself open or verify those links.
7. **Not run:** no models, no data downloads, no LaTeX build (no style file yet), no torch install. All by design.
