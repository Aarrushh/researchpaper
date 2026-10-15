# PURPOSE: Stand in for h2/judge/run_kernel.py in the judge's tests, so that run_attempt and score_page are tested without a GPU.
# WHAT GOES HERE: The four arguments of run_kernel.py and one behaviour per page id prefix: ok (the default), crash, hang, badjson, env, cpu.
# FILLED IN AT STEP: W4.4.
# DO NOT: Import torch, triton or anything from src/. It runs as python -I, from an empty temporary folder, like the real runner.
"""Fake runner for tests/test_h2_judge.py (build guide 4.4, fixtures).

    python -I -X utf8 fake_runner.py <kernel.py> <in_dir> <out_dir> <page_id>

It never loads the kernel. It acts by the page id's first part, before any underscore:
    crash    exits with code 3 and prints nothing
    hang     sleeps for 600 s
    badjson  writes half a result file
    env      an ok result with key_visible: whether ZAI_API_KEY is in its environment
    cpu      an ok result whose device is "cpu"
    anything else (ok)   copies <in_dir>/../reference/<page_id>.npy to <out_dir> and writes an ok result
An ok result carries the fields score_page reads from run_kernel.py's: device "cuda:0", kernel_ms and
repeat_identical; without a device starting with "cuda" score_page fails the page (guide 4.4 does not say this).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path


def main() -> int:
    _kernel, in_dir, out_dir, pid = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    kind = pid.split("_")[0]
    result_path = out_dir / f"{pid}.result.json"
    if kind == "crash":
        return 3
    if kind == "hang":
        time.sleep(600)
        return 0
    if kind == "badjson":
        result_path.write_text('{"ok": true, "dtype": "torch.ui', encoding="utf-8")
        return 0
    shutil.copyfile(in_dir.parent / "reference" / f"{pid}.npy", out_dir / f"{pid}.npy")
    result = {"ok": True, "dtype": "torch.uint8", "device": "cpu" if kind == "cpu" else "cuda:0", "shape": None,
              "kernel_ms": 0.25, "repeat_identical": True}
    if kind == "env":
        result["key_visible"] = "ZAI_API_KEY" in os.environ
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
