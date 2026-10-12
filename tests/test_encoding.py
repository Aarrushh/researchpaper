# PURPOSE: Check that every text-mode file read or write in our code names its encoding.
# WHAT GOES HERE: One scan of the syntax trees of src/, scripts/, h2/*.py and h2/judge/ for open, read_text and write_text calls that name no text codec.
# FILLED IN AT STEP: W1.5.
# DO NOT: Scan h2/kernels/ (GLM-5.3's unedited output). Do not exempt a file to make this pass; name encoding="utf-8" in the call.
"""Every text-mode open/read_text/write_text in our code names encoding= (Review Focus 1)."""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BINARY_OPENERS = {"Image", "gzip", "tarfile", "zipfile"}   # their .open() is not text I/O


def _mode(node: ast.Call, index: int) -> str | None:
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    if len(node.args) > index and isinstance(node.args[index], ast.Constant):
        return node.args[index].value
    return None


def _offenders():
    # h2/kernels/ holds GLM-5.3's verbatim output, which nobody edits, so it is not scanned.
    files = [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"),
             *ROOT.glob("h2/*.py"), *ROOT.glob("h2/judge/*.py")]
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                mode = _mode(node, 1)
            elif isinstance(func, ast.Attribute) and func.attr in {"read_text", "write_text"}:
                mode = None
            elif isinstance(func, ast.Attribute) and func.attr == "open":
                if isinstance(func.value, ast.Name) and func.value.id in BINARY_OPENERS:
                    continue
                mode = _mode(node, 0)
            else:
                continue
            if mode and "b" in mode:
                continue
            if "encoding" not in {kw.arg for kw in node.keywords}:
                yield f"{path.relative_to(ROOT)}:{node.lineno}"


def test_text_io_names_utf8():
    assert list(_offenders()) == []
