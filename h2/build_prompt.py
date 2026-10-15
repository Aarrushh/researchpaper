# PURPOSE: Build the H2 prompt from the specification, so that the frozen prompt is the instruction and the spec and nothing else.
# WHAT GOES HERE: The instruction line of build guide 4.5, the prompt text, and a command that writes prompt_vNN.md once.
# FILLED IN AT STEP: W4.3 (the builder); the prompt itself is written on the day the spec is final (Fri 23 / Sat 24 under D25).
# DO NOT: Overwrite a prompt, add a system message, examples, code, tests or hints, or build from a spec that still has TODO markers.
"""Step W4.3: build h2/prompts/prompt_v01.md from spec/RENDERING_SPEC.md (build guide 4.5).

    python h2/build_prompt.py --spec spec/RENDERING_SPEC.md --out h2/prompts/prompt_v01.md

The prompt is the instruction line, one blank line, then the specification whole, header comment included, so that
`tail -n +3 h2/prompts/prompt_v01.md | diff --strip-trailing-cr - spec/RENDERING_SPEC.md` prints nothing (Gate 3).
Line endings are LF, the bytes git stores for the spec: with core.autocrlf the working copy of the spec may carry
CRLF, and .gitattributes keeps h2/prompts/ byte for byte (-text), so the prompt's SHA-256 is the same on every
checkout. The command refuses to overwrite a prompt and refuses a spec that still holds a TODO marker.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Sequence

INSTRUCTION = ("Implement the Kernel interface section of the specification below as one Python module that uses "
               "Triton. Reply with the complete module in a single python code block.")
TODO = re.compile(r"TODO\[")


def build_prompt(spec_text: str) -> str:
    """The instruction, a blank line, then the specification unchanged apart from CRLF read as LF."""
    return INSTRUCTION + "\n\n" + spec_text.replace("\r\n", "\n")


def write_prompt(spec_path: Path, out_path: Path) -> str:
    """Write the prompt once and return its SHA-256.

    Raises:
        FileExistsError: if out_path exists; a prompt is frozen once written (h2/prompts/README.md).
        ValueError: if the specification still holds a TODO[ marker.
    """
    spec_path, out_path = Path(spec_path), Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; a prompt is never rewritten (make a new version only by amendment)")
    spec_text = spec_path.read_bytes().decode("utf-8")
    todos = [n for n, line in enumerate(spec_text.splitlines(), start=1) if TODO.search(line)]
    if todos:
        raise ValueError(f"{spec_path} still holds TODO markers on lines {todos[:10]}"
                         + (" and more" if len(todos) > 10 else ""))
    data = build_prompt(spec_text).encode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "xb") as fh:
        fh.write(data)
    return hashlib.sha256(data).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec", type=Path, default=Path("spec/RENDERING_SPEC.md"))
    p.add_argument("--out", type=Path, required=True, help="h2/prompts/prompt_v01.md")
    args = p.parse_args(argv)
    try:
        digest = write_prompt(args.spec, args.out)
    except (FileExistsError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"wrote {args.out.as_posix()}; SHA-256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
