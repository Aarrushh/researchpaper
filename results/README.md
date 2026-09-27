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
│   └── images/            (gitignored)
└── final/                 frozen CSVs only, copied at W3.5
```

Column definitions and the run ID format are in `docs/04_data_dictionary.md`.

## Rules

- Never edit a results CSV by hand.
- `results/final/` holds only frozen CSVs copied at W3.5, then tagged `data-freeze-v1`.
- Every number in the paper traces to a file in `results/final/` or a table in `reports/tables/` generated from it.
