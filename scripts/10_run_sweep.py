# PURPOSE: Run a sweep: render, transcribe, score, append one CSV row per page per condition per repeat.
# WHAT GOES HERE: The command line of mrr.sweep.run_sweep: which sweep, which readers, where to resume, when to stop, which images to keep. The loop over the blocks of the sweep (D10) lives in src/mrr/sweep.py.
# FILLED IN AT STEP: W2.3 (pilot), W3.3 (full).
# DO NOT: Edit results.csv by hand. Do not change a sweep config after its run has started. Do not skip a failed page: fix, then resume.
"""Steps W2.3 and W3.3: run a sweep (build guide section 2.8).

One reader at a time, in the sweep's order; one run folder per reader under the sweep's output_root
(results/runs/<run_id>/) with manifest.json, results.csv, coherence.csv, determinism.csv, progress.log,
references/, transcriptions/ and images/. The runner refuses to start while src, scripts, configs,
pyproject.toml or requirements.lock.txt have uncommitted changes, so that the manifest's commit describes the
code that made every row. An interrupted run resumes from results.csv: --resume <run folder> redoes only the
missing rows of that reader, and an incomplete last line is cut off and logged. --deadline stops at the next
page boundary (section 3.6); a time without a zone is local time.

Run it from the repository root, for example:
    python scripts/10_run_sweep.py --config configs/sweeps/pilot.yaml --readers qwen
    python scripts/10_run_sweep.py --config configs/sweeps/pilot.yaml --resume results/runs/<run_id>
    python scripts/10_run_sweep.py --config configs/sweeps/full.yaml --deadline 2026-10-24T22:00
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import yaml

from mrr import readers
from mrr.sweep import SAVE_IMAGES, readers_config, run_sweep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="configs/sweeps/<name>.yaml")
    p.add_argument("--readers-config", default=None,
                   help="the reader file (default: the sweep's readers_config, else configs/readers.yaml)")
    p.add_argument("--render-config", default="configs/render_defaults.yaml")
    p.add_argument("--readers", default=None, help="comma-separated subset of the sweep's readers (default: all)")
    p.add_argument("--resume", default=None, help="an existing run folder of one reader to continue")
    p.add_argument(
        "--save-images", default="first", choices=SAVE_IMAGES,
        help="keep the rendered pages of every row (all), of the first page only (first, default) or none",
    )
    p.add_argument(
        "--deadline", default=None,
        help="ISO 8601 time to stop at the next page boundary, e.g. 2026-10-24T22:00 (local) or ...T20:00+00:00",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    readers_path = args.readers_config or readers_config(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    readers_cfg = yaml.safe_load(Path(readers_path).read_text(encoding="utf-8"))
    render_defaults = yaml.safe_load(Path(args.render_config).read_text(encoding="utf-8"))
    only = [name.strip() for name in args.readers.split(",") if name.strip()] if args.readers else None
    deadline = datetime.fromisoformat(args.deadline) if args.deadline else None
    run_dirs = run_sweep(
        Path(args.config),
        readers_cfg,
        render_defaults,
        readers,
        only_readers=only,
        resume=Path(args.resume) if args.resume else None,
        save_images=args.save_images,
        require_clean=True,
        deadline=deadline,
    )
    for run_dir in run_dirs:
        print(run_dir)


if __name__ == "__main__":
    main()
