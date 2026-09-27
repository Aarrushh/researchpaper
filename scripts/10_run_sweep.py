# PURPOSE: Run a sweep: render, transcribe, score, append one CSV row per page per condition.
# WHAT GOES HERE: Iteration over the factor product, resume logic, row writing.
# FILLED IN AT STEP: W2.3 (pilot), W3.3 (full).
# DO NOT: Edit results.csv by hand. Do not change a sweep config after its run has started.
"""Steps W2.3 and W3.3: run a sweep.

Must resume safely if interrupted: rows already present in results.csv are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from mrr import readers, render, score, text_source
from mrr.manifest import RESULTS_COLUMNS, new_run_id, write_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="configs/sweeps/<name>.yaml")
    p.add_argument("--readers-config", default="configs/readers.yaml")
    p.add_argument("--render-config", default="configs/render_defaults.yaml")
    p.add_argument("--resume", default=None, help="existing run folder to resume")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sweep_name = Path(args.config).stem

    # TODO[W2.3]: for each reader in sweep["readers"]:
    #   1. Create or reuse the run folder (new_run_id(sweep_name, reader) or --resume).
    #   2. write_manifest(run_dir, resolved_config) on a new run.
    #   3. Load existing results.csv and collect the keys already done (resume).
    #   4. Build pages with text_source (seeded by sweep["seed"], n_pages).
    #   5. Iterate itertools.product over sweep["factors"] and repeats; skip done keys.
    #   6. render.render_page -> readers.transcribe -> score.cer / score.wer.
    #   7. Save transcription to transcriptions/<page_id>__<condition_hash>.txt.
    #   8. Append one row with exactly RESULTS_COLUMNS, flushing after each row.
    raise NotImplementedError("TODO[W2.3]: implement the sweep loop with safe resume")


if __name__ == "__main__":
    main()
