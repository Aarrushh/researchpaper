# PURPOSE: Fit Study 3 stage 0: each reader's acuity curve per text type, with the sizes for 5% and 1% CER and the characters a vision token carries there (design note R7, section 3).
# WHAT GOES HERE: Load the stage-0 sweeps' rows, keep block Z's conventional rows, fit mrr.acuity per reader and text type with passage-bootstrap intervals, write cells.csv and acuity.csv with a manifest.
# FILLED IN AT STEP: Study 3 stage 0 (19 October 2026), written while the stage-0 run goes.
# DO NOT: Pool stage-0 rows with Study 1 or Study 2 in a reported table; --configs with Study 1's ladder and full sweep is a check of the fit only. Read reports/tables/.
"""Study 3 stage 0: turn the run folders of the stage-0 sweeps into each reader's acuity curve.

    python scripts/25_fit_stage0.py [--configs configs/sweeps/s3_stage0_random.yaml configs/sweeps/s3_stage0_prose.yaml
                                               configs/sweeps/s3_stage0_prose_internvl3.yaml]
                                    [--runs-root results/runs] [--out-dir reports/tables] [--name s3_stage0]
                                    [--resamples 2000] [--seed 20261025]

Reads each sweep's run folders through ``sweep.load_results``, drops the pages ``sweep.complete_rows`` excludes, keeps
the rows of block Z in the conventional layout, and writes into ``<out-dir>/<name>/``:

    cells.csv   one row per reader, text type and size: pages, reference characters and errors pooled over the pages,
                pooled CER, mean characters a page, vision tokens a page, characters a vision token, the share of
                pages that hit the token limit, mean output tokens and mean latency
    acuity.csv  one row per reader and text type: the half-error size a (px) and the slope beta of
                CER(s) = 1 / (1 + (s / a) ** beta); the sizes s05 and s01 at which the curve reaches 5% and 1% CER, and
                kappa05 and kappa01, the characters a vision token carries on a full page at those sizes; each with
                its 95% passage-bootstrap interval (_lo, _hi); a / P and a over the token cell; flags; and the check
                fit with a free floor e0 (floor_e0, floor_a, floor_beta, floor_s01, floor_deviance; mrr.acuity says
                why the floor is not in the reported fit)

Every resampling draws passages with replacement and keeps each drawn passage's page at every size (mrr.acuity),
2,000 resamples, seed 20261025.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Sequence

# One BLAS thread before NumPy loads, as scripts/20_analyze.py, so the tables regenerate byte for byte.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import pandas as pd  # noqa: E402

from mrr import acuity  # noqa: E402
from mrr.manifest import new_run_id, write_manifest  # noqa: E402
from mrr.sweep import RunFolder, complete_rows, load_results, sweep_run_dirs  # noqa: E402

_spec = importlib.util.spec_from_file_location("analyze_study1", Path(__file__).with_name("20_analyze.py"))
analyze1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze1)

_log = logging.getLogger(__name__)

DEFAULT_CONFIGS = ("configs/sweeps/s3_stage0_random.yaml", "configs/sweeps/s3_stage0_prose.yaml",
                   "configs/sweeps/s3_stage0_prose_internvl3.yaml")  # InternVL3's prose above 20 px, pooled with the second
RESAMPLES = 2000
SEED = 20261025  # new: 20261012 to 20261024 are taken (sweeps, the bootstrap of Study 2, the font pre-test, stage 0)
CELL_COLUMNS = ("sweep", "reader", "text_type", "font_size_px", "pages", "reference_chars", "errors", "cer",
                "chars_per_page", "vision_tokens", "chars_per_vision_token", "hit_limit_share", "output_tokens_mean",
                "latency_s_mean")
_FITTED = ("a", "beta", "s05", "kappa05", "s01", "kappa01")
ACUITY_COLUMNS = (("sweep", "reader", "text_type", "status", "patch_size", "token_cell_px", "sizes_px", "passages",
                   "pages_min", "vision_tokens")
                  + tuple(c for name in _FITTED for c in (name, f"{name}_lo", f"{name}_hi"))
                  + ("a_over_patch", "a_over_cell", "s05_in_range", "s01_in_range", "chars_c", "chars_gamma",
                     "deviance", "converged", "resamples", "resamples_fitted", "floor_e0", "floor_a", "floor_beta",
                     "floor_s01", "floor_deviance"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS), help="the sweep files to fit")
    p.add_argument("--runs-root", default="results/runs")
    p.add_argument("--out-dir", default="reports/tables")
    p.add_argument("--name", default="s3_stage0", help="the folder under --out-dir")
    p.add_argument("--resamples", type=int, default=RESAMPLES)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


def load_rows(configs: Sequence[str], runs_root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, RunFolder]]:
    """Block Z's conventional rows of every sweep in ``configs``, a ``sweep`` column naming each; the sweeps' facts for
    the manifest; one run folder per reader (for its patch and token cell)."""
    frames, facts, folders_by_reader = [], {}, {}
    for config in configs:
        sweep_path = Path(config)
        run_dirs = sweep_run_dirs(sweep_path, runs_root)
        folders = [RunFolder(d) for d in run_dirs]
        df_all = load_results(sweep_path, runs_root)  # raises on a repeated key, an edited sweep file or another schema
        df, excluded = complete_rows(df_all, folders)
        keep = (df["block"] == "Z") & (df["render_mode"] == "conventional")
        df = df[keep].copy()
        df["sweep"] = sweep_path.stem
        frames.append(df)
        for f in folders:
            folders_by_reader.setdefault(f.reader, f)
        facts[sweep_path.stem] = {
            "sweep_file": str(sweep_path), "sweep_sha1": hashlib.sha1(sweep_path.read_bytes()).hexdigest(),
            "run_dirs": [d.name for d in run_dirs], "rows_loaded": int(len(df_all)), "rows_fitted": int(len(df)),
            "pages_excluded": {Path(k).name: sorted(v) for k, v in excluded.items() if v},
        }
        _log.info("%s: %d rows in %d run folder(s), %d fitted", sweep_path.stem, len(df_all), len(folders), len(df))
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return rows, facts, folders_by_reader


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    runs_root = Path(args.runs_root)
    rows, facts, folders = load_rows(args.configs, runs_root)
    if rows.empty:
        _log.error("no block Z row in %s under %s: run the sweeps first (scripts/11_run_schedule.py)",
                   ", ".join(args.configs), runs_root)
        return 1
    cells, fits = [], []
    for (reader, text_type), group in rows.groupby(["reader", "text_type"], sort=True):
        if group["font_size_px"].nunique() < 3:
            _log.info("  %-13s %-9s %d size(s) so far: not fitted", reader, text_type, group["font_size_px"].nunique())
            continue
        group_cells, fit = acuity.fit_group(group, resamples=args.resamples, seed=args.seed)
        reader_cfg = folders[reader].reader_cfg
        fit.update({"patch_size": int(reader_cfg["patch_size"]), "token_cell_px": int(reader_cfg["token_cell_px"]),
                    "a_over_patch": fit["a"] / int(reader_cfg["patch_size"]),
                    "a_over_cell": fit["a"] / int(reader_cfg["token_cell_px"])})
        cells.extend(group_cells)
        fits.append(fit)
        _log.info("  %-13s %-9s a %.1f px [%.1f, %.1f]  beta %.1f  s05 %.1f  s01 %.1f px  kappa01 %.2f  %s"
                  "  (free floor %.4f)", reader, text_type, fit["a"], fit["a_lo"], fit["a_hi"], fit["beta"],
                  fit["s05"], fit["s01"], fit["kappa01"], fit["status"], fit["floor_e0"])
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"cells": analyze1.write_table(out_dir / "cells.csv", cells, CELL_COLUMNS),
              "acuity": analyze1.write_table(out_dir / "acuity.csv", fits, ACUITY_COLUMNS)}
    for name, n in counts.items():
        _log.info("  %-11s %4d row(s)  %s", name + ".csv", n, out_dir / f"{name}.csv")
    write_manifest(out_dir, {
        "sweeps": facts, "runs_root": str(runs_root), "resamples": args.resamples, "seed": args.seed,
        "alpha": acuity.ALPHA, "epsilons": list(acuity.EPSILONS), "tables": counts,
        "curve": "CER(s) = 1 / (1 + (s / a) ** beta), binomial deviance of capped cell CERs, pages as weights; "
                 "check fit with a free floor e0",
        "analysis_plan": "design note R7, section 3 (stage 0)", "blas_threads": 1,
    }, run_id=new_run_id(args.name, "analysis"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
