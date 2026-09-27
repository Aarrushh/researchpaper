# PURPOSE: Verify each reader's resize behaviour and vision token count against the formula.
# WHAT GOES HERE: For each reader, render a test page at three sizes, report shape and tokens.
# FILLED IN AT STEP: W1.6.
# DO NOT: Run sweeps from here. Do not change configs from inside the script.
"""Step W1.6: check readers."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from mrr import grid, readers, render
from mrr.manifest import new_run_id, write_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="configs/readers.yaml")
    p.add_argument("--render-config", default="configs/render_defaults.yaml")
    p.add_argument("--output-root", default="results/runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    render_cfg = yaml.safe_load(Path(args.render_config).read_text(encoding="utf-8"))

    for reader_cfg in cfg["readers"]:
        run_dir = Path(args.output_root) / new_run_id("checkreaders", reader_cfg["name"])
        write_manifest(run_dir, {"readers": reader_cfg, "render": render_cfg})

        cell = grid.token_cell_px(reader_cfg)
        reader = readers.load_reader(reader_cfg)
        # TODO[W1.6]: choose three test sizes; for each, render with render.render_page at
        # grid.native_size(...), call readers.count_vision_tokens, and compare with
        # (height / cell) * (width / cell). Print a table and write it to run_dir/token_check.csv.
        raise NotImplementedError("TODO[W1.6]: implement the three-size token check loop")


if __name__ == "__main__":
    main()
