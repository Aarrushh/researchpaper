# PURPOSE: Tests for the run bookkeeping: the results schema (D19) and the code-cleanliness checks of the runner.
# WHAT GOES HERE: The 33 column names in order; code_dirty and code_changed_since on a throwaway repository.
# FILLED IN AT STEP: W2.3.
# DO NOT: Run git in the repository that holds the tests; every git call here runs in tmp_path. Require a GPU or network.
"""Tests for mrr.manifest."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from mrr.manifest import CODE_PATHS, RESULTS_COLUMNS, code_changed_since, code_dirty, git_dirty

# Build guide section 2.5: the 33 columns of results.csv, in order (D19).
SCHEMA = (
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

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


def test_results_columns_are_the_33_names_of_section_2_5_in_order():
    assert RESULTS_COLUMNS == SCHEMA
    assert len(RESULTS_COLUMNS) == 33 == len(set(RESULTS_COLUMNS))


def test_code_paths_name_the_files_a_run_depends_on():
    assert CODE_PATHS == ("src", "scripts", "configs", "pyproject.toml", "requirements.lock.txt")


def _git(repo, *args):
    out = subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A throwaway repository in a folder with a space in its name (plan Review Focus 3), with one commit."""
    root = tmp_path / "repo with space"
    (root / "src").mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (root / "notes.txt").write_text("not code\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "first")
    return root


@needs_git
def test_code_dirty_sees_only_the_code_paths(repo):
    assert code_dirty(cwd=repo) is False
    (repo / "notes.txt").write_text("edited\n", encoding="utf-8")
    assert git_dirty(cwd=repo) is True  # the tree is dirty ...
    assert code_dirty(cwd=repo) is False  # ... but nothing a run depends on changed
    (repo / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert code_dirty(cwd=repo) is True


@needs_git
def test_an_untracked_file_under_a_code_path_counts_as_dirty(repo):
    (repo / "scripts").mkdir()
    (repo / "scripts" / "new.py").write_text("", encoding="utf-8")
    assert code_dirty(cwd=repo) is True


@needs_git
def test_code_changed_since_compares_commits_not_the_working_tree(repo):
    first = _git(repo, "rev-parse", "HEAD")
    assert code_changed_since(first, cwd=repo) is False
    (repo / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert code_changed_since(first, cwd=repo) is False  # uncommitted: that is code_dirty's job
    _git(repo, "commit", "-q", "-am", "second")
    assert code_changed_since(first, cwd=repo) is True
    assert code_changed_since("HEAD", cwd=repo) is False
    second = _git(repo, "rev-parse", "HEAD")
    (repo / "notes.txt").write_text("edited\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "third")
    assert code_changed_since(second, cwd=repo) is False  # notes.txt is outside CODE_PATHS


@needs_git
def test_the_checks_return_none_when_git_cannot_answer(repo, tmp_path):
    assert code_changed_since("0" * 40, cwd=repo) is None
    outside = tmp_path / "not a repo"
    outside.mkdir()
    assert code_dirty(cwd=outside) is None
    assert code_changed_since("HEAD", cwd=outside) is None
