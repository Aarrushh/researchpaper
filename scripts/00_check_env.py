# PURPOSE: Check that this machine can run the experiment.
# WHAT GOES HERE: Version and GPU report; non-zero exit if torch or CUDA is missing.
# FILLED IN AT STEP: Fully implemented at setup. Run at W1.3 (make check-env).
# DO NOT: Install anything from here. Do not download models.
"""Step W1.3: environment check."""

from __future__ import annotations

import platform
import sys
from importlib import metadata

PACKAGES = ["torch", "transformers", "accelerate", "qwen-vl-utils", "pillow",
            "numpy", "pandas", "scipy", "jiwer", "matplotlib", "pyyaml", "triton"]


def version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    print(f"Python      {sys.version.split()[0]} ({platform.platform()})")
    print("Packages:")
    for dist in PACKAGES:
        print(f"  {dist:<14} {version(dist)}")

    try:
        import torch
    except ImportError:
        print("\nERROR: torch is not installed. Install PyTorch first using the "
              "official selector for your CUDA version, then rerun `make check-env`.",
              file=sys.stderr)
        return 1

    cuda = torch.cuda.is_available()
    print(f"\nCUDA available  {cuda}")
    print(f"torch CUDA build {torch.version.cuda}")
    if not cuda:
        print("\nERROR: torch cannot see a CUDA GPU. Check the NVIDIA driver and that "
              "you installed a CUDA build of PyTorch.", file=sys.stderr)
        return 1

    props = torch.cuda.get_device_properties(0)
    print(f"GPU             {props.name}")
    print(f"Total VRAM      {props.total_memory / 1024**3:.2f} GiB")
    print("\nOK: environment ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
