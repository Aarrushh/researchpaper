# PURPOSE: Choose each reader's font size from the ladder run (W1.10, build guide section 2.9) and write the table behind the choice.
# WHAT GOES HERE: Per reader, text type and size: the CER pooled over both render modes, each mode's pooled CER, rows, hit_limit share and latencies; the chosen size per reader and its median latency t.
# FILLED IN AT STEP: W1.10 (the ladder), under D26.
# DO NOT: Change the target or the tie rule after seeing the ladder's rows; both are fixed in configs/sweeps/ladder.yaml and section 5 of PREREGISTRATION.md. Do not pool the ladder's rows with any other sweep.
"""Step W1.10: the ladder's size choice (build guide section 2.9, plan Task P2.9 step 2).

For each reader and size, the random62 rows of both render modes are pooled, (S + D + I) / (H + S + D) over the
rows (D13); the size whose pooled CER is closest to the sweep's ``target_cer_random62`` is chosen, and on a tie the
larger size. The natural rows are reported beside it and do not enter the choice. t, the median row latency at the
chosen size over both text types, is the number the budget of section 2.12 uses.

Writes ``<out-dir>/ladder_choice.csv`` (one row per reader, text type and size) and ``<out-dir>/chosen_sizes.csv``
(one row per reader), then a manifest, and prints the choice. A reader of the sweep without a complete run folder
(72 rows) stops the script with exit 1, unless ``--allow-partial``, which writes the tables for what exists and
chooses nothing.
Run it from the repository root: python scripts/09_ladder_choice.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

from mrr.manifest import new_run_id, write_manifest
from mrr.score import CharCounts, pooled_cer
from mrr.sweep import load_results, sweep_run_dirs

TABLE_COLUMNS = (
    "reader", "text_type", "font_size_px", "rows", "cer_pooled", "cer_conventional", "cer_patch_aware",
    "hit_limit_share", "median_latency_s", "mean_latency_s", "median_output_tokens",
)
CHOICE_COLUMNS = ("reader", "target_cer_random62", "chosen_size_px", "cer_random62_at_choice",
                  "distance_to_target", "t_median_latency_s", "t_mean_latency_s", "rows_at_choice")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/sweeps/ladder.yaml")
    p.add_argument("--runs-root", default="results/runs")
    p.add_argument("--out-dir", default="reports/tables/ladder")
    p.add_argument("--allow-partial", action="store_true", help="write the tables for incomplete runs, choose nothing")
    return p.parse_args()


def _pooled(rows) -> float:
    counts = [CharCounts(int(r.hits), int(r.substitutions), int(r.deletions), int(r.insertions))
              for r in rows.itertuples()]
    return pooled_cer(counts) if counts else float("nan")


def choose(cer_by_size: dict[float, float], target: float) -> float:
    """The size whose CER is closest to ``target``; on a tie, the larger size (section 2.9).

    Distances are compared at 12 decimals, so that two CERs equally far from the target in exact arithmetic (0.10
    and 0.20 from 0.15) tie instead of being split by the last bit of a float.
    """
    if not cer_by_size:
        raise ValueError("no sizes to choose from")
    return min(cer_by_size, key=lambda size: (round(abs(cer_by_size[size] - target), 12), -size))


def main() -> int:
    args = parse_args()
    sweep_path = Path(args.config)
    sweep = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    target = float(sweep["target_cer_random62"])
    runs_root = Path(args.runs_root)
    if not sweep_run_dirs(sweep_path, runs_root):
        print(f"no run folder of sweep {sweep_path.stem} under {runs_root}", file=sys.stderr)
        return 1
    df = load_results(sweep_path, runs_root)
    expected = None
    table, choices, problems = [], [], []
    for reader in sweep["readers"]:
        sub = df[df["reader"] == reader]
        sizes = sorted(set(float(s) for s in sub["font_size_px"]))
        if expected is None:
            expected = len(sweep["blocks"]["Z"]["font_size_px"]) * len(sweep["blocks"]["Z"]["render_mode"]) * 2 * int(
                sweep["pages"]["random62"]["per_text_type"])
        if len(sub) != expected:
            problems.append(f"{reader}: {len(sub)} of {expected} rows")
        for text_type in ("natural", "random62"):
            for size in sizes:
                rows = sub[(sub["text_type"] == text_type) & (sub["font_size_px"].astype(float) == size)]
                table.append({
                    "reader": reader, "text_type": text_type, "font_size_px": f"{size:g}", "rows": len(rows),
                    "cer_pooled": _pooled(rows),
                    "cer_conventional": _pooled(rows[rows["render_mode"] == "conventional"]),
                    "cer_patch_aware": _pooled(rows[rows["render_mode"] == "patch_aware"]),
                    "hit_limit_share": float(rows["hit_limit"].mean()) if len(rows) else float("nan"),
                    "median_latency_s": float(np.median(rows["latency_s"])) if len(rows) else float("nan"),
                    "mean_latency_s": float(np.mean(rows["latency_s"])) if len(rows) else float("nan"),
                    "median_output_tokens": float(np.median(rows["output_tokens"])) if len(rows) else float("nan"),
                })
        if len(sub) != expected:
            continue
        random_cer = {size: _pooled(sub[(sub["text_type"] == "random62") & (sub["font_size_px"].astype(float) == size)])
                      for size in sizes}
        size = choose(random_cer, target)
        at = sub[sub["font_size_px"].astype(float) == size]
        choices.append({
            "reader": reader, "target_cer_random62": target, "chosen_size_px": f"{size:g}",
            "cer_random62_at_choice": random_cer[size], "distance_to_target": abs(random_cer[size] - target),
            "t_median_latency_s": float(np.median(at["latency_s"])), "t_mean_latency_s": float(np.mean(at["latency_s"])),
            "rows_at_choice": len(at),
        })
    if problems and not args.allow_partial:
        print("incomplete ladder runs: " + "; ".join(problems), file=sys.stderr)
        return 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out, {"sweep": str(sweep_path).replace("\\", "/"), "target_cer_random62": target,
                         "runs": [p.name for p in sweep_run_dirs(sweep_path, runs_root)],
                         "rows": int(len(df)), "incomplete": problems},
                   run_id=new_run_id(sweep_path.stem, "choice"))
    for name, columns, rows in (("ladder_choice.csv", TABLE_COLUMNS, table), ("chosen_sizes.csv", CHOICE_COLUMNS, choices)):
        with open(out / name, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    for row in table:
        print(f"{row['reader']:12s} {row['text_type']:9s} {row['font_size_px']:>3s} px  rows {row['rows']:2d}  "
              f"CER {row['cer_pooled']:.3f} (conv {row['cer_conventional']:.3f}, pa {row['cer_patch_aware']:.3f})  "
              f"limit {row['hit_limit_share']:.2f}  median {row['median_latency_s']:.1f} s")
    for row in choices:
        print(f"CHOICE {row['reader']}: {row['chosen_size_px']} px, random62 CER {row['cer_random62_at_choice']:.3f} "
              f"(target {target}), t = {row['t_median_latency_s']:.1f} s median, {row['t_mean_latency_s']:.1f} s mean")
    if problems:
        print("incomplete, no choice for: " + "; ".join(problems))
    return 0


if __name__ == "__main__":
    sys.exit(main())
