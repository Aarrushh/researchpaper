<!--
PURPOSE: Layout and rules for experiment outputs.
WHAT GOES HERE: One folder per run under runs/, frozen CSVs under final/.
FILLED IN AT STEP: W2.3 onward (runs), W3.5 (final).
DO NOT: Edit any results CSV by hand. Do not commit rendered images.
-->

# results/

## Layout

```
results/
├── runs/<run_id>/
│   ├── manifest.json
│   ├── results.csv
│   ├── transcriptions/<page_id>__<condition_hash>.txt
│   ├── references/<page_id>.txt, coherence.csv, determinism.csv, progress.log
│   └── images/            (gitignored)
├── equivariance/          <reader>_<text_type>.csv and manifest.json (W1.9)
├── mtf/                   layer0_<reader>.csv (profile, f_cyc_per_px, value) and manifest.json (W3.6)
└── final/                 the freeze (W3.5): copies of runs/, equivariance/ and the stretch folders,
                           without images/, plus README.md with the SHA-256 of every file
```

Column definitions and the run ID format are in `docs/04_data_dictionary.md`.

## Rules

- Never edit a results CSV by hand.
- `results/final/` is written by `scripts/40_freeze.py` (`make freeze`) at W3.5 and never by hand; `make freeze-check SWEEP=...` recomputes every hash and regenerates the tables from the frozen runs, which must match byte for byte; then the owner tags `data-freeze-v1`.
- Every number in the paper traces to a file in `results/final/` or a table in `reports/tables/` generated from it.
