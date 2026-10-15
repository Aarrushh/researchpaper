<!--
PURPOSE: Verbatim kernels written by the H2 model.
WHAT GOES HERE: kernel_<prompt-version>_<attempt>.py files, exactly as the model returned them.
FILLED IN AT STEP: W4.3.
DO NOT: Edit, reformat or fix any file here. Humans never touch kernel files.
-->

# h2/kernels/

- Naming: `kernel_<prompt-version>_<attempt>.py`, for example `kernel_v01_1.py`.
- Content: the model's output, verbatim. If the output contains prose around the code, save the full output as `.md` alongside and extract the code block without editing it.
- Never edited by humans.
- Files of one attempt, all written by `h2/call_model.py` and never edited: `kernel_v01_<n>.raw.sse.txt` (or `.raw.json`), the reply as it arrived; `.md`, the answer; `.reasoning.md`, the reasoning; `.py`, the code block, only if the answer has one; `.meta.json`, the request without the key, status, model string, finish reason, usage and times; `.transport-<k>.txt`, a call that failed in transport and is not an attempt.
