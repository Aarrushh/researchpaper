# PURPOSE: Freeze the results: copy them into results/final/ and write the README that names every file's SHA-256 (W3.5).
# WHAT GOES HERE: The copy, the README, and the check that the tables regenerate byte for byte from the frozen copy.
# FILLED IN AT STEP: W3.5 (build guide section 3.8).
# DO NOT: Edit anything under results/final/ by hand. Do not freeze over a previous freeze without --replace. Do not tag here: the tags data-freeze-v1 (Study 1), data-freeze-v2 (Study 2), data-freeze-s3-stage0 (stage 0's tables) and analysis-freeze-v1 (every table, make freeze-check-all) follow the owner's sign-off.
"""Step W3.5: the freeze (build guide 3.8).

    python scripts/40_freeze.py [--results results] [--final results/final] [--replace]
    python scripts/40_freeze.py --check [--final results/final] [--sweep configs/sweeps/full.yaml ...] [--fit s3_stage0 ...]
                                [--tables reports/tables]

The first form copies ``runs/``, ``equivariance/``, ``mtf/`` and ``probe/`` (those that exist) from ``results/`` into
``results/final/``, without the git-ignored ``images/`` folders, and writes ``results/final/README.md``: what each
folder holds, the sweep, reader and commit of every run folder, and the SHA-256 of every file in sha256sum format.
A second freeze is refused unless ``--replace`` removes the first.

The second form checks a freeze: every file's SHA-256 against the README, and, for each ``--sweep``, the tables that
the sweep's analysis script regenerates from ``results/final/runs`` against ``reports/tables/<sweep_name>/``, byte
for byte (only manifest.json may differ): scripts/22_analyze_study2.py for ``study2`` (PREREGISTRATION_2.md section 9,
step 7), scripts/20_analyze.py for every other sweep. Each ``--fit`` names an analysis in ``FITS`` with its own
script (Study 3's stage 0: scripts/25_fit_stage0.py over its three sweeps, ANALYSIS_PLAN_3.md section 2.7; Study 2's
exploratory re-scoring: scripts/26_rescore_study2.py, section 3.2), whose tables are regenerated and compared with
``reports/tables/<fit_name>/`` the same way. Exit 1 on any difference. The tag is the owner's, after sign-off.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mrr.manifest import git_commit, git_dirty

FOLDERS = ("runs", "equivariance", "mtf", "probe")  # copied when present, in this order
SKIPPED = ("images", "__pycache__")  # rendered pages are git-ignored (.gitignore: results/runs/**/images/)
README = "README.md"
WHAT = {
    "runs": "one folder per run, its sweep named in the table below: manifest.json, results.csv, "
            "transcriptions/, references/, coherence.csv, determinism.csv and progress.log (section 2.8); the rendered "
            "pages under images/ are git-ignored and not copied",
    "equivariance": "the shift-equivariance error per layer of each vision tower, <reader>_<text_type>.csv and "
                    "manifest.json (W1.9, Figure 0)",
    "mtf": "the layer-0 modulation transfer per reader (section 3S.1)",
    "probe": "the linear probe for bits per token (section 3S.3)",
}
HASH_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")  # sha256sum format: the hash, two spaces, the path
ROOT = Path(__file__).resolve().parents[1]
ANALYZE = "20_analyze.py"  # under scripts/: Study 1's tables, for every sweep not in ANALYZERS
ANALYZERS = {"study2": "22_analyze_study2.py"}  # by sweep name (PREREGISTRATION_2.md section 9, step 7)
# Analyses written by their own script, by table folder: (script under scripts/, its --configs, further arguments).
# s3_stage0's configs are scripts/25_fit_stage0.py's DEFAULT_CONFIGS, which wrote the committed tables
# (ANALYSIS_PLAN_3.md section 2.7). study2_rescored reads one sweep, so it has no --configs and names its --config
# among the further arguments (ANALYSIS_PLAN_3.md section 3.2).
FITS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "s3_stage0": ("25_fit_stage0.py", ("configs/sweeps/s3_stage0_random.yaml", "configs/sweeps/s3_stage0_prose.yaml",
                                       "configs/sweeps/s3_stage0_prose_internvl3.yaml"), ()),
    "study2_rescored": ("26_rescore_study2.py", (), ("--config", "configs/sweeps/study2.yaml")),
}

_log = logging.getLogger("freeze")


# --------------------------------------------------------------------------- hashes


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(root: Path) -> list[tuple[str, str]]:
    """(path relative to root in POSIX form, SHA-256) of every file under root but the README, sorted by path."""
    out: list[tuple[str, str]] = []
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == README:
            continue
        out.append((rel, sha256_file(path)))
    return out


def read_hashes(readme: Path) -> dict[str, str]:
    """The path -> hash lines of a README written by ``write_readme``."""
    out: dict[str, str] = {}
    for line in Path(readme).read_text(encoding="utf-8").splitlines():
        m = HASH_LINE.match(line)
        if m:
            out[m.group(2)] = m.group(1)
    return out


# --------------------------------------------------------------------------- the README


def run_summary(run_dir: Path) -> dict[str, Any]:
    """What the README says of one run folder, from its manifest and the row count of results.csv."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) or {}
    cfg = manifest.get("config") or {}
    reader = cfg.get("reader")
    reader_name = reader.get("name") if isinstance(reader, Mapping) else reader
    rows = 0
    results = run_dir / "results.csv"
    if results.exists():
        with open(results, encoding="utf-8", newline="") as fh:
            rows = max(sum(1 for _ in csv.reader(fh)) - 1, 0)
    return {
        "run": run_dir.name, "sweep": str(cfg.get("sweep_name") or ""), "reader": str(reader_name or ""),
        "commit": str(manifest.get("git_commit") or ""), "dirty": bool(manifest.get("git_dirty")), "rows": rows,
        "rows_planned": cfg.get("rows_planned"), "timestamp_utc": str(manifest.get("timestamp_utc") or ""),
    }


def write_readme(final: Path, copied: Sequence[str], root: Path | None = None) -> Path:
    """results/final/README.md: the folders, the run folders, and the SHA-256 of every file under ``final``."""
    final = Path(final)
    runs_root = final / "runs"
    runs = []
    if runs_root.is_dir():
        runs = [run_summary(d) for d in sorted(runs_root.iterdir()) if d.is_dir() and (d / "manifest.json").exists()]
    hashes = hash_tree(final)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    commit = git_commit(cwd=root) or "unknown"
    dirty = " with uncommitted changes" if git_dirty(cwd=root) else ""
    lines = [
        "# Frozen results (W3.5)",
        "",
        f"Written by `scripts/40_freeze.py` on {now} at commit `{commit}`{dirty}. Nothing under this folder is edited",
        "by hand. `python scripts/40_freeze.py --check --sweep configs/sweeps/<name>.yaml` recomputes every hash below",
        "and regenerates the tables of `reports/tables/<name>/` from `runs/`, which must match byte for byte on the",
        "machine that wrote them (floats carry 12 significant digits; another BLAS kernel can move a last digit).",
        "",
        "## Folders",
        "",
    ]
    lines += [f"- `{name}/`: {WHAT[name]}." for name in copied]
    lines += [
        "",
        "## Run folders",
        "",
        "| run | sweep | reader | commit | rows | planned | started (UTC) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        planned = "" if r["rows_planned"] is None else str(r["rows_planned"])
        mark = "*" if r["dirty"] else ""
        lines.append(f"| `{r['run']}` | {r['sweep']} | {r['reader']} | `{r['commit'][:7]}`{mark} | {r['rows']} | "
                     f"{planned} | {r['timestamp_utc']} |")
    lines += [
        "",
        "`*`: the manifest records uncommitted changes when the run started.",
        "",
        f"## SHA-256 of every file ({len(hashes)} files)",
        "",
        "`sha256sum` format: the hash, two spaces, the path relative to this folder.",
        "",
        "```",
        *[f"{digest}  {rel}" for rel, digest in hashes],
        "```",
        "",
    ]
    path = final / README
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


# --------------------------------------------------------------------------- the freeze and its check


def freeze(results: Path, final: Path, replace: bool = False, root: Path | None = None) -> list[str]:
    """Copy the result folders into ``final`` and write its README; return the names of the folders copied.

    Raises:
        FileNotFoundError: when ``results`` holds none of ``FOLDERS``.
        FileExistsError: when ``final`` already holds a freeze and ``replace`` is False.
    """
    results, final = Path(results), Path(final)
    present = [name for name in FOLDERS if (results / name).is_dir()]
    if not present:
        raise FileNotFoundError(f"nothing to freeze under {results}: none of {', '.join(FOLDERS)} exists")
    existing = [name for name in FOLDERS if (final / name).exists()]
    if existing and not replace:
        raise FileExistsError(f"{final} already holds {', '.join(existing)}: a freeze is not overwritten; pass "
                              f"--replace to redo it")
    for name in existing:
        shutil.rmtree(final / name)
    if (final / README).exists():
        (final / README).unlink()
    final.mkdir(parents=True, exist_ok=True)
    for name in present:
        shutil.copytree(results / name, final / name, ignore=shutil.ignore_patterns(*SKIPPED))
    write_readme(final, present, root)
    return present


def check_hashes(final: Path) -> list[str]:
    """Every file under ``final`` against the README: the problems found, none when the freeze is intact."""
    final = Path(final)
    readme = final / README
    if not readme.exists():
        return [f"{readme} is missing"]
    recorded = read_hashes(readme)
    actual = dict(hash_tree(final))
    problems: list[str] = []
    for rel in sorted(set(recorded) | set(actual)):
        if rel not in actual:
            problems.append(f"missing: {rel}")
        elif rel not in recorded:
            problems.append(f"not in README: {rel}")
        elif recorded[rel] != actual[rel]:
            problems.append(f"changed: {rel}")
    return problems


def analyzer(sweep_path: Path) -> str:
    """The script under scripts/ that writes a sweep's tables: Study 2's for ``study2``, Study 1's for any other."""
    return ANALYZERS.get(Path(sweep_path).stem, ANALYZE)


def _regenerate_and_compare(name: str, script: str, args: Sequence[str], committed: Path, root: Path,
                            python: str) -> list[str]:
    """Run ``scripts/<script> <args> --out-dir <tmp>`` and compare every CSV it writes under ``<tmp>/<name>/`` with
    ``committed``, byte for byte; manifest.json is not compared. Returns the problems."""
    env = {**os.environ, "PYTHONUTF8": "1",
           "PYTHONPATH": os.pathsep.join(p for p in (str(root / "src"), os.environ.get("PYTHONPATH", "")) if p)}
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="freeze_check_") as tmp:
        cmd = [python, str(root / "scripts" / script), *args, "--out-dir", tmp]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=root)
        if proc.returncode != 0:
            last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
            return [f"{name}: scripts/{script} exited {proc.returncode}: {last}"]
        fresh = Path(tmp) / name
        names = sorted({p.name for p in committed.glob("*.csv")} | {p.name for p in fresh.glob("*.csv")})
        for csv_name in names:
            a, b = committed / csv_name, fresh / csv_name
            if not a.exists():
                problems.append(f"{name}/{csv_name}: regenerated but not in {committed}")
            elif not b.exists():
                problems.append(f"{name}/{csv_name}: in {committed} but not regenerated")
            elif a.read_bytes() != b.read_bytes():
                problems.append(f"{name}/{csv_name}: differs from the regenerated table")
            else:
                _log.info("%s/%s: identical to the regenerated table", name, csv_name)
    return problems


def check_tables(final: Path, sweeps: Sequence[Path], tables_root: Path, root: Path = ROOT,
                 python: str = sys.executable) -> list[str]:
    """Regenerate each sweep's tables from ``final/runs`` and compare them with ``tables_root/<sweep_name>/``.

    Every CSV must be byte-identical (build guide 3.8, step 3); manifest.json is not compared. Returns the problems.
    """
    final, tables_root = Path(final), Path(tables_root)
    problems: list[str] = []
    for sweep_path in sweeps:
        sweep_path = Path(sweep_path)
        name = sweep_path.stem
        committed = tables_root / name
        if not committed.is_dir():
            problems.append(f"{committed} is missing: run make analyze SWEEP={sweep_path} first")
            continue
        args = ["--config", str(sweep_path), "--runs-root", str(final / "runs")]
        problems += _regenerate_and_compare(name, analyzer(sweep_path), args, committed, root, python)
    return problems


def check_fits(final: Path, names: Sequence[str], tables_root: Path, root: Path = ROOT,
               python: str = sys.executable) -> list[str]:
    """Regenerate each fit of ``FITS`` from ``final/runs`` and compare its tables with ``tables_root/<name>/``.

    A fit is an analysis with its own script and ``--name``: over several sweeps (``--configs``), such as Study 3's
    stage 0, or over one sweep named among its further arguments, such as Study 2's re-scoring. Every CSV must be
    byte-identical; manifest.json is not compared. Returns the problems.
    """
    final, tables_root = Path(final), Path(tables_root)
    problems: list[str] = []
    for name in names:
        if name not in FITS:
            problems.append(f"{name}: not a fit this check knows ({', '.join(sorted(FITS))})")
            continue
        script, configs, extra = FITS[name]
        committed = tables_root / name
        if not committed.is_dir():
            problems.append(f"{committed} is missing: run scripts/{script} first")
            continue
        args = [*(("--configs", *configs) if configs else ()), "--name", name, "--runs-root", str(final / "runs"), *extra]
        problems += _regenerate_and_compare(name, script, args, committed, root, python)
    return problems


# --------------------------------------------------------------------------- the command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results", help="the results folder to freeze (default results)")
    p.add_argument("--final", default="results/final", help="where the freeze goes (default results/final)")
    p.add_argument("--replace", action="store_true", help="remove a previous freeze under --final first")
    p.add_argument("--check", action="store_true", help="check the freeze under --final instead of making one")
    p.add_argument("--sweep", action="append", default=[],
                   help="with --check: a sweep whose tables are regenerated from the frozen runs and compared with "
                        "<tables>/<sweep_name>/; repeatable")
    p.add_argument("--fit", action="append", default=[], choices=sorted(FITS),
                   help="with --check: an analysis of several sweeps (FITS) whose tables are regenerated from the "
                        "frozen runs and compared with <tables>/<fit_name>/; repeatable")
    p.add_argument("--tables", default="reports/tables", help="the committed tables (default reports/tables)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    final = Path(args.final)
    if args.check:
        problems = check_hashes(final)
        listed = len(read_hashes(final / README)) if (final / README).exists() else 0
        _log.info("%s: %d file(s) listed in README.md, %d problem(s)", final, listed, len(problems))
        problems += check_tables(final, [Path(s) for s in args.sweep], Path(args.tables))
        problems += check_fits(final, args.fit, Path(args.tables))
        for problem in problems:
            _log.error("  %s", problem)
        if problems:
            _log.error("the freeze does not check out: %d problem(s)", len(problems))
            return 1
        regenerated = args.sweep or args.fit
        _log.info("the freeze checks out%s", " and the tables regenerate byte for byte" if regenerated else "")
        return 0
    try:
        copied = freeze(Path(args.results), final, replace=args.replace, root=ROOT)
    except (FileNotFoundError, FileExistsError) as exc:
        _log.error("%s", exc)
        return 1
    _log.info("%s: %s copied; %d file(s) listed in %s", final, ", ".join(copied), len(read_hashes(final / README)),
              final / README)
    _log.info("next: review the README, run --check, then the owner tags the freeze: data-freeze-v1 for Study 1 "
              "(section 3.8, step 4), data-freeze-v2 for Study 2 (PREREGISTRATION_2.md section 9, step 7)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
