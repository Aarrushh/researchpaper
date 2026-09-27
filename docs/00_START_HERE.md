<!--
PURPOSE: Day-one orientation: what to read, in what order, and how TODOs work.
WHAT GOES HERE: Reading order, TODO conventions, the one rule about numbers.
FILLED IN AT STEP: Read at W1.1.
DO NOT: Add results or plans here; they belong in 01_research_plan.md and 03_pipeline.md.
-->

# Start here

## Reading order for day one

1. `README.md` at the repo root.
2. This file.
3. `docs/01_research_plan.md`.
4. `docs/02_learning_guide.md`, topics A1 to A6, doing every exercise.
5. `docs/03_pipeline.md`.

## How TODOs work

Every unfinished item in the repo is marked with a TODO that carries a step ID from `docs/03_pipeline.md`, for example W1.5 or W2.3.

To see this week's work, search the repo for the week prefix:

```
grep -rn "TODO\[W1\." .
```

Change `W1` to `W2`, `W3` and so on as the weeks go by. When you finish a TODO, delete the marker and tick the step in `docs/03_pipeline.md`.

## The one rule

No number enters the paper unless one of you opened its source and logged it in references/citations.md.
