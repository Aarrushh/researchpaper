# PURPOSE: Run bookkeeping: run IDs and manifest.json for every run folder.
# WHAT GOES HERE: Run ID generation, environment capture, the results CSV column list, the code-cleanliness checks the runner makes before it starts.
# FILLED IN AT STEP: Scaffolded at setup; the 33-column schema (D19), CODE_PATHS, code_dirty and code_changed_since at W2.3.
# DO NOT: Write results here. Do not change RESULTS_COLUMNS without updating docs/04_data_dictionary.md.
"""Run bookkeeping.

Every run folder gets a ``manifest.json`` so any row in any CSV can be traced
back to the exact code, packages, GPU and config that produced it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

# Column order of results.csv: the 33 columns of build guide section 2.5 (D19). Documented in docs/04_data_dictionary.md.
RESULTS_COLUMNS: tuple[str, ...] = (
    "run_id",
    "reader",
    "reader_revision",
    "page_id",
    "book_id",
    "text_type",
    "block",
    "condition_hash",
    "repeat",
    "font_file",
    "font_size_px",
    "line_pitch_px",
    "line_pitch_mode",
    "aa_mode",
    "render_mode",
    "phase_x_px",
    "phase_y_px",
    "image_width_px",
    "image_height_px",
    "vision_tokens",
    "reference_chars",
    "transcription_file",
    "output_tokens",
    "hit_limit",
    "cer",
    "wer",
    "hits",
    "substitutions",
    "deletions",
    "insertions",
    "latency_s",
    "timestamp_utc",
    "git_commit",
)
# Study 2 (PREREGISTRATION_2.md section 4): a sweep with ``results_schema: study2`` writes these after the 33, the
# letters-only counts (score.char_counts_letters_only) and the inserted spaces (score.spaces_inserted). Study 1's runs
# keep RESULTS_COLUMNS. Documented in docs/04_data_dictionary.md.
STUDY2_SCORE_COLUMNS: tuple[str, ...] = ("hits_ns", "substitutions_ns", "deletions_ns", "insertions_ns", "spaces_inserted")
RESULTS_SCHEMAS: dict[str, tuple[str, ...]] = {
    "study1": RESULTS_COLUMNS,
    "study2": RESULTS_COLUMNS + STUDY2_SCORE_COLUMNS,
}

# Packages whose versions are recorded, mapped to their distribution names.
TRACKED_PACKAGES: dict[str, str] = {
    "torch": "torch",
    "transformers": "transformers",
    "pillow": "pillow",
    "numpy": "numpy",
    "freetype-py": "freetype-py",  # bundles the FreeType every pixel depends on (section 1A.11)
    "scipy": "scipy",
    "jiwer": "jiwer",
}

NOT_INSTALLED = "not installed"


def _git(*args: str, cwd: str | Path | None = None) -> str | None:
    """Run a git command and return stripped stdout, or None on any failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_commit(short: bool = False, cwd: str | Path | None = None) -> str | None:
    """Return the current commit hash, or None outside a git repository."""
    args = ["rev-parse", "--short=7", "HEAD"] if short else ["rev-parse", "HEAD"]
    return _git(*args, cwd=cwd)


def git_dirty(cwd: str | Path | None = None) -> bool | None:
    """Return True if the working tree has uncommitted changes, None if unknown."""
    status = _git("status", "--porcelain", cwd=cwd)
    if status is None:
        return None
    return status != ""


# The paths a run's results depend on. A run refuses to start while any of them has uncommitted changes (section 2.8).
CODE_PATHS: tuple[str, ...] = ("src", "scripts", "configs", "pyproject.toml", "requirements.lock.txt")


def code_dirty(paths: Sequence[str] = CODE_PATHS, cwd: str | Path | None = None) -> bool | None:
    """Return True if any of ``paths`` has uncommitted changes, untracked files included; None if git cannot answer.

    Narrower than ``git_dirty``: notes, figures and reports may change while a run is going, the code may not.
    """
    status = _git("status", "--porcelain", "--", *paths, cwd=cwd)
    if status is None:
        return None
    return status != ""


def code_changed_since(commit: str, paths: Sequence[str] = CODE_PATHS, cwd: str | Path | None = None) -> bool | None:
    """Return True if any of ``paths`` differs between ``commit`` and HEAD; None if git cannot answer.

    For a resumed run: the manifest's commit must still describe the code (section 2.8). Uncommitted changes are
    ``code_dirty``'s business.
    """
    changed = _git("diff", "--name-only", commit, "HEAD", "--", *paths, cwd=cwd)
    if changed is None:
        return None
    return changed != ""


def package_version(dist_name: str) -> str:
    """Return the installed version of a distribution, or 'not installed'."""
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return NOT_INSTALLED


def gpu_name() -> str | None:
    """Return the CUDA device name if torch is installed and reports CUDA, else None."""
    try:
        import torch  # noqa: PLC0415  (optional, heavy)
    except ImportError:
        return None
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001  (driver errors must not kill bookkeeping)
        return None
    return None


def new_run_id(sweep_name: str, reader_name: str, now: datetime | None = None) -> str:
    """Return a run ID: ``YYYYMMDD-HHMM_<sweep>_<reader>_<short-git-sha>``.

    Time is UTC. The SHA is ``nogit`` outside a git repository.
    """
    now = now or datetime.now(timezone.utc)
    sha = git_commit(short=True) or "nogit"
    return f"{now.strftime('%Y%m%d-%H%M')}_{sweep_name}_{reader_name}_{sha}"


def build_manifest(run_id: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Collect the manifest contents without writing them."""
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: package_version(dist) for name, dist in TRACKED_PACKAGES.items()},
        "gpu_name": gpu_name(),
        "config": dict(cfg),
    }


def write_manifest(run_dir: str | Path, cfg: Mapping[str, Any], run_id: str | None = None) -> Path:
    """Write ``manifest.json`` into run_dir and return its path.

    Args:
        run_dir: the run folder, created if missing. Its name is used as the
            run ID unless run_id is given.
        cfg: the full resolved config for this run.
        run_id: optional explicit run ID.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(run_id or run_dir.name, cfg)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path
