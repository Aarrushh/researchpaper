# PURPOSE: Check that this machine can run the experiment.
# WHAT GOES HERE: Version and GPU report; non-zero exit if torch or CUDA is missing.
# FILLED IN AT STEP: Implemented at setup; W1.3 adds the GPU architecture, driver and FreeType checks. Run with make check-env.
# DO NOT: Install anything from here. Do not download models.
"""Step W1.3: environment check.

Exits 1 unless torch imports, sees a CUDA GPU whose architecture (sm_120 on
these laptops) is in the wheel's architecture list, nvidia-smi reports the
driver, and freetype-py bundles FreeType 2.13.2. A missing Triton is only a
warning until W4.2.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata

TRITON = "triton-windows" if sys.platform == "win32" else "triton"
PACKAGES = ["torch", "torchvision", "transformers", "accelerate", "pillow", "numpy",
            "pandas", "scipy", "jiwer", "matplotlib", "freetype-py", "pyyaml", TRITON]
FREETYPE_VERSION = (2, 13, 2)
NVIDIA_SMI = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
              "--format=csv,noheader"]


def version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "not installed"


def fail(message: str) -> int:
    print(f"\nERROR: {message}", file=sys.stderr)
    return 1


def main() -> int:
    print(f"Python      {sys.version.split()[0]} ({platform.platform()})")
    print("Packages:")
    for dist in PACKAGES:
        print(f"  {dist:<14} {version(dist)}")
    if version(TRITON) == "not installed":
        print(f"\nWARNING: {TRITON} is not installed. It is needed from W4.2 only.")

    try:
        import torch
    except ImportError:
        return fail("torch is not installed. Install it from the CUDA 13.0 index "
                    "(first line of requirements.txt), then rerun `make check-env`.")

    cuda = torch.cuda.is_available()
    print(f"\nCUDA available  {cuda}")
    print(f"torch CUDA build {torch.version.cuda}")
    if not cuda:
        return fail("torch cannot see a CUDA GPU. Check the NVIDIA driver and that "
                    "you installed a CUDA build of PyTorch.")

    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    print(f"GPU             {props.name}")
    print(f"Total VRAM      {props.total_memory / 1024**3:.2f} GiB")
    print(f"Architecture    {arch}")
    print(f"Wheel arches    {' '.join(arch_list)}")
    if arch not in arch_list:
        return fail(f"this torch wheel has no {arch} kernels. Install torch from "
                    "https://download.pytorch.org/whl/cu130.")

    try:
        smi = subprocess.run(NVIDIA_SMI, check=True, capture_output=True, text=True,
                             encoding="utf-8")
    except (OSError, subprocess.CalledProcessError) as err:
        return fail(f"nvidia-smi failed ({err}), so the driver version is unknown.")
    print(f"nvidia-smi      {smi.stdout.strip()}")

    try:
        import freetype
    except ImportError:
        return fail("freetype-py is not installed (pip install -r requirements.txt).")
    print(f"FreeType        {freetype.version()}")
    if tuple(freetype.version()) != FREETYPE_VERSION:
        return fail(f"freetype-py must bundle FreeType {FREETYPE_VERSION}; "
                    "install freetype-py 2.5.1.")

    print("\nOK: environment ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
