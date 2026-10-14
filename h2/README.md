<!--
PURPOSE: The H2 protocol: can a language model implement the spec as a Triton kernel?
WHAT GOES HERE: Protocol rules and the run record for every attempt.
FILLED IN AT STEP: W4.3 (record), W4.4 (judge results).
DO NOT: Give the model anything other than spec/RENDERING_SPEC.md. Do not edit model outputs.
-->

# H2 protocol

## Rules

1. The model receives only `spec/RENDERING_SPEC.md`, wrapped in a frozen prompt from `prompts/`.
2. No code from `src/`, no test files, no hints beyond the spec.
3. Every attempt records: model string, date, reasoning settings, and the full prompt file used.
4. Outputs are saved verbatim in `kernels/` and never edited by humans. A kernel that does not run is a result, not something to fix.
5. `judge/compare.py` compares each kernel's output with the reference renderer on the same inputs.

## Run record

| Attempt | Prompt file | Model string | Date | Reasoning settings | Kernel file | Judge result |
|---|---|---|---|---|---|---|
| TODO[W4.3]: attempt 1 | TODO[W4.3]: prompt_v01.md | TODO[W4.3]: exact model string | TODO[W4.3]: date | TODO[W4.3]: settings | TODO[W4.3]: kernel file | TODO[W4.4]: judge numbers |

Number of attempts allowed: 3 (`PREREGISTRATION.md`, section 1, H2: an attempt is a reply with HTTP status 200; the first Python code block is judged verbatim).
