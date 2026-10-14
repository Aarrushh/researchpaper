# PURPOSE: Check that this machine can run the experiment.
# WHAT GOES HERE: Version and GPU report; non-zero exit if torch or CUDA is missing.
# FILLED IN AT STEP: Implemented at setup; W1.3 adds the GPU architecture, driver and FreeType checks; the Phase 1 review adds the sysmem-fallback check. Run with make check-env.
# DO NOT: Install anything from here. Do not download models.
"""Step W1.3: environment check.

Exits 1 unless freetype-py bundles FreeType 2.13.2 (checked first, on the
CPU, because every pixel test and recorded hash depends on it), torch imports,
sees a CUDA GPU whose architecture (sm_120 on these laptops) is in the wheel's
architecture list, the driver refuses to allocate more than the card's memory,
and nvidia-smi reports the driver. A missing Triton is only a warning until
W4.2.

The memory check is the setting of build guide section 1B.2: with "Prefer No
Sysmem Fallback", running out of GPU memory fails loudly; with the fallback, the
driver moves the overflow to system memory and the run only gets slower. The
check fills the card, so run it with no other GPU job running.
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
BLOCK = 256 * 2**20  # bytes per allocation in the sysmem-fallback check


def version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "not installed"


def fail(message: str) -> int:
    print(f"\nERROR: {message}", file=sys.stderr)
    return 1


def sysmem_fallback(torch, total: int) -> bool:
    """Return True if the driver hands out more than the card's memory: the sysmem fallback is on.

    Allocates and touches 256 MiB blocks until CUDA refuses or the total passes 1.1 times the card's
    memory, then frees them all.
    """
    blocks = []
    try:
        while len(blocks) * BLOCK < 1.1 * total:
            block = torch.empty(BLOCK, dtype=torch.uint8, device="cuda")
            block.fill_(1)  # touch it, so the memory is committed
            blocks.append(block)
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        blocks.clear()
        torch.cuda.empty_cache()


def main() -> int:
    print(f"Python      {sys.version.split()[0]} ({platform.platform()})")
    print("Packages:")
    for dist in PACKAGES:
        print(f"  {dist:<14} {version(dist)}")
    if version(TRITON) == "not installed":
        print(f"\nWARNING: {TRITON} is not installed. It is needed from W4.2 only.")

    try:
        import freetype
    except ImportError:
        return fail("freetype-py is not installed (pip install -r requirements.txt).")
    print(f"FreeType        {freetype.version()}")
    if tuple(freetype.version()) != FREETYPE_VERSION:
        return fail(f"freetype-py must bundle FreeType {FREETYPE_VERSION}; "
                    "install freetype-py 2.5.1.")

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
    if sysmem_fallback(torch, props.total_memory):
        return fail("the driver let this script allocate 1.1 x the card's memory, so the CUDA sysmem "
                    "fallback is on. Set NVIDIA Control Panel > Manage 3D Settings > CUDA - Sysmem "
                    "Fallback Policy > Prefer No Sysmem Fallback (build guide 1B.2), then rerun "
                    "`make check-env`.")
    print("Sysmem fallback off: CUDA refuses 1.1 x the card's memory")

    try:
        smi = subprocess.run(NVIDIA_SMI, check=True, capture_output=True, text=True,
                             encoding="utf-8")
    except (OSError, subprocess.CalledProcessError) as err:
        return fail(f"nvidia-smi failed ({err}), so the driver version is unknown.")
    print(f"nvidia-smi      {smi.stdout.strip()}")

    print("\nOK: environment ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
