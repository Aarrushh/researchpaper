# PURPOSE: Run a sweep on the schedule of D26: phases of readers, each phase until its deadline, readers of one phase side by side.
# WHAT GOES HERE: Reading the schedule file, starting or resuming each reader's run folder with scripts/10_run_sweep.py, restarting a run that crashes before its deadline, the schedule log.
# FILLED IN AT STEP: W3.3, under D26 (no pilot; the clock stops the full run).
# DO NOT: Put experiment logic here; every row is written by scripts/10_run_sweep.py. Do not edit the schedule after the sweep has started, except to record a crash in LAB_NOTEBOOK.md.
"""Step W3.3 under D26: run the full sweep on a fixed schedule.

The schedule file lists phases in order. Each phase names the readers that run in it and the time it ends:

    phases:
      - {readers: [qwen], until: "2026-10-16T06:00:00+02:00"}
      - {readers: [smolvlm, smolvlm500m], until: "2026-10-16T22:00:00+02:00"}
    max_restarts: 5

For each phase whose end is still ahead, every reader of the phase is started at once as its own process of
``scripts/10_run_sweep.py --readers <reader> --deadline <until>``. A reader that already has a run folder of this
sweep (manifest with the sweep's name and SHA-1) is continued with ``--resume <folder>``, so each reader keeps one run
folder for the whole sweep. A process that exits with an error before its phase ends is resumed after a pause, up to
``max_restarts`` times per reader over the whole schedule; a process that exits cleanly has reached its deadline or its
page ceiling. The next phase starts when every process of the current one has exited. Every start, exit and restart
is written to the schedule log and printed.

Run it from the repository root, for example:
    python scripts/11_run_schedule.py --config configs/sweeps/full.yaml --schedule configs/schedules/full.yaml
    python scripts/11_run_schedule.py ... --dry-run      # print what would run now, start nothing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from mrr.sweep import sweep_run_dirs

RESTART_PAUSE_S = 30.0  # between a crash and its resume, so that the GPU memory of the dead process is released
POLL_S = 5.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="configs/sweeps/<name>.yaml")
    p.add_argument("--schedule", required=True, help="configs/schedules/<name>.yaml")
    p.add_argument("--runs-root", default=None, help="default: the sweep's output_root")
    p.add_argument("--runner", default="scripts/10_run_sweep.py")
    p.add_argument("--log-dir", default="logs", help="per-reader process logs and schedule.log")
    p.add_argument("--save-images", default="first", choices=("all", "first", "none"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _aware(value: Any, label: str) -> datetime:
    """An ISO 8601 time with its offset; a time without one is refused, so a schedule never depends on the clock's zone."""
    when = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if when.tzinfo is None:
        raise ValueError(f"{label}: {value!r} has no UTC offset; write it as e.g. 2026-10-16T06:00:00+02:00")
    return when.astimezone(timezone.utc)


def load_schedule(path: Path, sweep: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """The phases, with their ``until`` as aware UTC, and ``max_restarts``; refuses readers not in the sweep and
    phases whose ends do not increase."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    phases = raw.get("phases") if isinstance(raw, Mapping) else None
    if not isinstance(phases, list) or not phases:
        raise ValueError(f"{path}: phases must be a non-empty list")
    known = set(sweep["readers"])
    out, previous = [], None
    for i, phase in enumerate(phases):
        readers = list(phase.get("readers") or [])
        if not readers or len(set(readers)) != len(readers):
            raise ValueError(f"{path}: phase {i} must list each of its readers once")
        unknown = [r for r in readers if r not in known]
        if unknown:
            raise ValueError(f"{path}: phase {i} names readers {unknown} that the sweep does not have")
        until = _aware(phase.get("until"), f"phase {i} until")
        if previous is not None and until <= previous:
            raise ValueError(f"{path}: phase {i} ends at {until.isoformat()}, not after the phase before it")
        previous = until
        out.append({"readers": readers, "until": until})
    restarts = raw.get("max_restarts", 5)
    if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts < 0:
        raise ValueError(f"{path}: max_restarts must be a whole number, 0 or more")
    return out, restarts


def reader_run_dir(sweep_path: Path, runs_root: Path, reader: str) -> Path | None:
    """The run folder of ``reader`` for this sweep, or None; more than one is an error, since each reader keeps one."""
    found = []
    for run_dir in sweep_run_dirs(sweep_path, runs_root):
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if str(manifest["config"]["reader"]["name"]) == reader:
            found.append(run_dir)
    if len(found) > 1:
        raise ValueError(f"{reader} has {len(found)} run folders of {sweep_path.stem}: {[p.name for p in found]}")
    return found[0] if found else None


def runner_command(python: str, runner: str, config: str, reader: str, until: datetime, resume: Path | None,
                   save_images: str) -> list[str]:
    cmd = [python, runner, "--config", config, "--readers", reader, "--deadline", until.isoformat(),
           "--save-images", save_images]
    if resume is not None:
        cmd += ["--resume", str(resume)]
    return cmd


def run_schedule(
    sweep_path: Path,
    schedule_path: Path,
    runs_root: Path,
    runner: str,
    log_dir: Path,
    save_images: str = "first",
    *,
    dry_run: bool = False,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
    python: str = sys.executable,
    restart_pause_s: float = RESTART_PAUSE_S,
) -> int:
    """Run every phase still ahead; return 0, or 1 when a reader ran out of restarts."""
    sweep = yaml.safe_load(Path(sweep_path).read_text(encoding="utf-8"))
    phases, max_restarts = load_schedule(schedule_path, sweep)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    schedule_log = log_dir / "schedule.log"
    restarts = {reader: 0 for reader in sweep["readers"]}
    failed = False

    def log(message: str) -> None:
        line = f"{now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        if not dry_run:
            with open(schedule_log, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def start(reader: str, until: datetime) -> subprocess.Popen | None:
        resume = reader_run_dir(Path(sweep_path), runs_root, reader)
        cmd = runner_command(python, runner, str(sweep_path), reader, until, resume, save_images)
        log(f"{'would start' if dry_run else 'start'} {reader} until {until.isoformat()}"
            f"{' resuming ' + resume.name if resume else ' (new run folder)'}: {' '.join(cmd[1:])}")
        if dry_run:
            return None
        out = open(log_dir / f"{Path(sweep_path).stem}_{reader}.log", "a", encoding="utf-8")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)

    for i, phase in enumerate(phases):
        until = phase["until"]
        if now() >= until:
            log(f"phase {i} ({', '.join(phase['readers'])}) ended at {until.isoformat()}; skipped")
            continue
        log(f"phase {i}: {', '.join(phase['readers'])} until {until.isoformat()}")
        procs = {reader: start(reader, until) for reader in phase["readers"]}
        if dry_run:
            continue
        while procs:
            sleep(POLL_S)
            for reader, proc in list(procs.items()):
                code = proc.poll()
                if code is None:
                    continue
                del procs[reader]
                if code == 0:
                    log(f"{reader} exited cleanly (deadline or ceiling)")
                    continue
                if now() >= until:
                    log(f"{reader} exited with {code} after its deadline; not restarted")
                    continue
                if restarts[reader] >= max_restarts:
                    log(f"{reader} exited with {code}; {max_restarts} restarts used, giving up on it")
                    failed = True
                    continue
                restarts[reader] += 1
                log(f"{reader} exited with {code}; restart {restarts[reader]} of {max_restarts} in {restart_pause_s:g} s")
                sleep(restart_pause_s)
                if now() < until:
                    procs[reader] = start(reader, until)
        log(f"phase {i} done")
    log("schedule done")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sweep_path = Path(args.config)
    sweep = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    runs_root = Path(args.runs_root or sweep.get("output_root", "results/runs"))
    return run_schedule(sweep_path, Path(args.schedule), runs_root, args.runner, Path(args.log_dir), args.save_images,
                        dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
