# PURPOSE: Shared pytest setup.
# WHAT GOES HERE: Path setup so `mrr` imports without installation; shared fixtures.
# FILLED IN AT STEP: Scaffolded at setup.
# DO NOT: Load models or GPUs in fixtures used by non-todo tests.
"""Pytest configuration."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
