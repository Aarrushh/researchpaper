# PURPOSE: Run one H2 kernel on one judge page in a fresh process, for judge.py.
# WHAT GOES HERE: Loading the kernel by path, the first call on the GPU, the input check, the result file, the timing.
# FILLED IN AT STEP: W4.4.
# DO NOT: Import anything from src/. Do not run it by hand on a kernel that both authors have not read.
"""Run one H2 kernel on one judge page; write <page_id>.npy and <page_id>.result.json.

Started only by judge.py (run_attempt), as
    python -I -X utf8 run_kernel.py <kernel.py> <in_dir> <out_dir> <page_id>
with the project venv's python: -I drops the user site-packages, so torch and triton must be in the venv.
The page is judged on the first call; timing comes after the result is written and cannot fail the page.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import triton.testing

KEYS = ("bitmaps", "offsets", "heights", "widths", "rows", "cols")


def main() -> None:
    kernel, in_dir, out_dir, pid = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    result_path = out_dir / f"{pid}.result.json"
    try:
        spec = importlib.util.spec_from_file_location("h2_kernel", kernel)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        meta = json.loads((in_dir / f"{pid}.json").read_text(encoding="utf-8"))
        with np.load(in_dir / f"{pid}.npz") as z:
            before = [z[k].copy() for k in KEYS]
        args = [torch.from_numpy(a.copy()).cuda() for a in before]

        def call():
            return module.render(*args, meta["height"], meta["width"], tuple(meta["text_rgb"]),
                                 tuple(meta["background_rgb"]), meta["aa_mode"])

        out = call()
        torch.cuda.synchronize()
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"render returned {type(out).__name__}, not a torch.Tensor")
        if any(not np.array_equal(a.cpu().numpy(), b) for a, b in zip(args, before)):
            raise RuntimeError("render changed one of its input tensors")
        first = out.detach().cpu().numpy()
        np.save(out_dir / f"{pid}.npy", first)
    except BaseException:
        result_path.write_text(json.dumps({"ok": False, "error": traceback.format_exc()}), encoding="utf-8")
        return
    result = {"ok": True, "dtype": str(out.dtype), "device": str(out.device), "shape": list(out.shape),
              "kernel_ms": None, "repeat_identical": None}
    result_path.write_text(json.dumps(result), encoding="utf-8")          # the page is judged on this first call
    try:
        result["kernel_ms"] = triton.testing.do_bench(call, warmup=25, rep=100)
        result["repeat_identical"] = bool(np.array_equal(call().detach().cpu().numpy(), first))
    except BaseException:
        result["timing_error"] = traceback.format_exc()
    result_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
