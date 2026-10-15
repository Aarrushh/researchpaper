# PURPOSE: Tests for scripts/11_run_schedule.py, the schedule driver of the full run (D26).
# WHAT GOES HERE: Schedule parsing and refusals, new and resumed run folders, phases in order with their readers side by side, restarts after a crash, skipped past phases, the dry run.
# FILLED IN AT STEP: W3.3, under D26.
# DO NOT: Load a model or start the real runner; a fake runner stands in for scripts/10_run_sweep.py.
"""Tests for the schedule driver."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "11_run_schedule.py"

FAKE_RUNNER = r'''
import argparse, hashlib, json, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--config"); p.add_argument("--readers"); p.add_argument("--deadline")
p.add_argument("--save-images"); p.add_argument("--resume", default=None)
a = p.parse_args()
root = Path(sys.argv[0]).parent
calls = root / "calls.jsonl"
n = sum(1 for line in open(calls, encoding="utf-8") if json.loads(line)["reader"] == a.readers) if calls.exists() else 0
with open(calls, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"reader": a.readers, "deadline": a.deadline, "resume": a.resume}) + "\n")
if a.resume is None:
    sha = hashlib.sha1(Path(a.config).read_bytes()).hexdigest()
    run = root / "runs" / f"run_{a.readers}"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"config": {"sweep_name": Path(a.config).stem, "sweep_sha1": sha,
                                                              "reader": {"name": a.readers}}}), encoding="utf-8")
codes = json.loads((root / "codes.json").read_text(encoding="utf-8")).get(a.readers, [])
sys.exit(codes[n] if n < len(codes) else 0)
'''


def _load_script():
    spec = importlib.util.spec_from_file_location("run_schedule", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_schedule"] = module
    spec.loader.exec_module(module)
    return module


rs = _load_script()
T0 = datetime(2026, 10, 15, 18, 0, tzinfo=timezone.utc)


class Clock:
    """A clock that moves forward only when the driver sleeps."""

    def __init__(self, start: datetime):
        self.t = start

    def now(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        time.sleep(0.02)  # let the fake runner's process make progress in real time
        self.t += timedelta(seconds=seconds)


@pytest.fixture
def setup(tmp_path):
    sweep = tmp_path / "full.yaml"
    sweep.write_text(yaml.safe_dump({"readers": ["qwen", "smolvlm", "smolvlm500m"], "output_root": "runs"}),
                     encoding="utf-8")
    runner = tmp_path / "fake_runner.py"
    runner.write_text(FAKE_RUNNER, encoding="utf-8")

    def make(phases, codes=None, max_restarts=2):
        schedule = tmp_path / "schedule.yaml"
        schedule.write_text(yaml.safe_dump({"phases": phases, "max_restarts": max_restarts}), encoding="utf-8")
        (tmp_path / "codes.json").write_text(json.dumps(codes or {}), encoding="utf-8")
        return sweep, schedule

    return tmp_path, runner, make


def _calls(tmp_path):
    path = tmp_path / "calls.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _run(tmp_path, runner, sweep, schedule, clock, dry_run=False):
    return rs.run_schedule(sweep, schedule, tmp_path / "runs", str(runner), tmp_path / "logs", dry_run=dry_run,
                           now=clock.now, sleep=clock.sleep, restart_pause_s=1.0)


def _iso(dt):
    return dt.isoformat()


def test_phases_run_in_order_side_by_side_and_resume_the_same_folder(setup):
    tmp_path, runner, make = setup
    sweep, schedule = make([
        {"readers": ["qwen"], "until": _iso(T0 + timedelta(hours=2))},
        {"readers": ["smolvlm", "smolvlm500m"], "until": _iso(T0 + timedelta(hours=4))},
        {"readers": ["qwen"], "until": _iso(T0 + timedelta(hours=6))},
    ])
    assert _run(tmp_path, runner, sweep, schedule, Clock(T0)) == 0
    calls = _calls(tmp_path)
    assert [c["reader"] for c in calls[:1]] == ["qwen"]
    assert {c["reader"] for c in calls[1:3]} == {"smolvlm", "smolvlm500m"}
    assert calls[3]["reader"] == "qwen"
    assert calls[0]["resume"] is None and calls[3]["resume"].endswith("run_qwen")  # one folder per reader
    assert [datetime.fromisoformat(c["deadline"]) for c in (calls[0], calls[1], calls[3])] == [
        T0 + timedelta(hours=2), T0 + timedelta(hours=4), T0 + timedelta(hours=6)]
    log = (tmp_path / "logs" / "schedule.log").read_text(encoding="utf-8")
    assert "phase 0: qwen" in log and "phase 2 done" in log and log.rstrip().endswith("schedule done")


def test_a_crash_before_the_deadline_is_resumed_and_restarts_are_counted(setup):
    tmp_path, runner, make = setup
    sweep, schedule = make([{"readers": ["qwen"], "until": _iso(T0 + timedelta(hours=2))}], codes={"qwen": [1, 3]})
    assert _run(tmp_path, runner, sweep, schedule, Clock(T0)) == 0
    calls = _calls(tmp_path)
    assert len(calls) == 3
    assert calls[0]["resume"] is None and all(c["resume"].endswith("run_qwen") for c in calls[1:])
    log = (tmp_path / "logs" / "schedule.log").read_text(encoding="utf-8")
    assert "restart 1 of 2" in log and "restart 2 of 2" in log


def test_a_reader_out_of_restarts_makes_the_schedule_fail_but_the_other_phases_still_run(setup):
    tmp_path, runner, make = setup
    sweep, schedule = make([
        {"readers": ["qwen"], "until": _iso(T0 + timedelta(hours=2))},
        {"readers": ["smolvlm"], "until": _iso(T0 + timedelta(hours=4))},
    ], codes={"qwen": [1, 1, 1, 1]}, max_restarts=1)
    assert _run(tmp_path, runner, sweep, schedule, Clock(T0)) == 1
    readers = [c["reader"] for c in _calls(tmp_path)]
    assert readers == ["qwen", "qwen", "smolvlm"]
    assert "giving up on it" in (tmp_path / "logs" / "schedule.log").read_text(encoding="utf-8")


def test_phases_that_have_ended_are_skipped(setup):
    tmp_path, runner, make = setup
    sweep, schedule = make([
        {"readers": ["qwen"], "until": _iso(T0 - timedelta(minutes=1))},
        {"readers": ["smolvlm"], "until": _iso(T0 + timedelta(hours=1))},
    ])
    assert _run(tmp_path, runner, sweep, schedule, Clock(T0)) == 0
    assert [c["reader"] for c in _calls(tmp_path)] == ["smolvlm"]
    assert "skipped" in (tmp_path / "logs" / "schedule.log").read_text(encoding="utf-8")


def test_dry_run_starts_nothing(setup, capsys):
    tmp_path, runner, make = setup
    sweep, schedule = make([{"readers": ["qwen", "smolvlm"], "until": _iso(T0 + timedelta(hours=1))}])
    assert _run(tmp_path, runner, sweep, schedule, Clock(T0), dry_run=True) == 0
    assert _calls(tmp_path) == [] and not (tmp_path / "logs" / "schedule.log").exists()
    out = capsys.readouterr().out
    assert out.count("would start") == 2 and "--deadline 2026-10-15T19:00:00+00:00" in out


@pytest.mark.parametrize(
    ("phases", "match"),
    [
        ([], "non-empty"),
        ([{"readers": ["qwen"], "until": "2026-10-16T06:00:00"}], "UTC offset"),
        ([{"readers": ["llava"], "until": "2026-10-16T06:00:00+02:00"}], "does not have"),
        ([{"readers": ["qwen", "qwen"], "until": "2026-10-16T06:00:00+02:00"}], "once"),
        ([{"readers": ["qwen"], "until": "2026-10-16T06:00:00+02:00"},
          {"readers": ["smolvlm"], "until": "2026-10-16T05:00:00+02:00"}], "not after"),
    ],
)
def test_bad_schedules_are_refused(tmp_path, phases, match):
    path = tmp_path / "schedule.yaml"
    path.write_text(yaml.safe_dump({"phases": phases}), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        rs.load_schedule(path, {"readers": ["qwen", "smolvlm"]})


def test_two_run_folders_of_one_reader_are_refused(setup):
    tmp_path, runner, make = setup
    sweep, _ = make([{"readers": ["qwen"], "until": _iso(T0 + timedelta(hours=1))}])
    sha = hashlib.sha1(sweep.read_bytes()).hexdigest()
    for name in ("a", "b"):
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps({"config": {"sweep_name": "full", "sweep_sha1": sha,
                                                                  "reader": {"name": "qwen"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="2 run folders"):
        rs.reader_run_dir(sweep, tmp_path / "runs", "qwen")
