# PURPOSE: Tests of h2/call_model.py and h2/build_prompt.py without the network: the request, the key, the files of a call, code extraction, the prompt.
# WHAT GOES HERE: extract_code cases, build_request, the .env search, call() against fake replies (stream, no stream, transport failures), the prompt builder.
# FILLED IN AT STEP: W4.3.
# DO NOT: Call the real API or read the real .env. Every key here is a fake made up by the test.
"""Tests for h2/call_model.py and h2/build_prompt.py (build guide 4.5). No network, no real key."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import REPO_ROOT

FAKE_KEY = "fake-zai-key-0123456789abcdef"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"h2_{name}", REPO_ROOT / "h2" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves postponed annotations through sys.modules
    spec.loader.exec_module(module)
    return module


cm = _load("call_model")
bp = _load("build_prompt")


@pytest.fixture(autouse=True)
def _never_the_real_env_file(monkeypatch, tmp_path_factory):
    """call() looks for .env under cm.REPO_ROOT; point it at an empty folder so no test can read the owner's key."""
    monkeypatch.setattr(cm, "REPO_ROOT", tmp_path_factory.mktemp("no_env"))


# ---- extract_code ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("info", ["python", "py", "Python"])
def test_extract_code_named_block(info):
    content = f"Here is the module.\n\n```{info}\nimport torch\n\ndef render():\n    return 1\n```\n\nDone.\n"
    assert cm.extract_code(content) == "import torch\n\ndef render():\n    return 1\n"


def test_extract_code_prefers_python_after_unnamed_block():
    content = "```\npip install triton-windows\n```\n\nThen:\n```python\nx = 1\n```\n"
    assert cm.extract_code(content) == "x = 1\n"


def test_extract_code_unnamed_block_alone():
    assert cm.extract_code("text\n```\nx = 1\ny = 2\n```\nmore\n") == "x = 1\ny = 2\n"


def test_extract_code_unclosed_block_runs_to_the_end():
    assert cm.extract_code("intro\n```python\nx = 1\ny = 2\n") == "x = 1\ny = 2\n"
    assert cm.extract_code("intro\n```python\nx = 1\ny = 2") == "x = 1\ny = 2"


def test_extract_code_no_block():
    assert cm.extract_code("No code here, only `inline` code.\n") is None
    assert cm.extract_code("") is None


def test_extract_code_lines_verbatim():
    body = "def f():\r\n\treturn  '```'  # trailing spaces   \r\n\r\n"
    assert cm.extract_code("```python\r\n" + body + "```\r\n") == body
    assert cm.extract_code("~~~py\nx = 1\n```\ny = 2\n~~~\n") == "x = 1\n```\ny = 2\n"
    assert cm.extract_code("````python\nx = '```'\n```\n````\n") == "x = '```'\n```\n"   # a shorter fence stays inside


# ---- build_request and the key --------------------------------------------------------------------------------

def test_build_request_is_the_guide_body_without_a_key(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", FAKE_KEY)
    prompt = "Implement it.\n\n# Spec\n"
    body = cm.build_request(prompt, 131072, True)
    assert body == {"model": "glm-5.3", "messages": [{"role": "user", "content": prompt}],
                    "thinking": {"type": "enabled"}, "reasoning_effort": "max",
                    "temperature": 1.0, "max_tokens": 131072, "stream": True}
    text = json.dumps(body)
    assert FAKE_KEY not in text and "Bearer" not in text and "Authorization" not in text
    assert cm.build_request(prompt, 128000, False)["max_tokens"] == 128000


def _fake_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """A main checkout with a worktree under .claude/worktrees/h2, as on the owner's laptop; paths with spaces."""
    main = tmp_path / "research paper"
    (main / ".git" / "worktrees" / "h2").mkdir(parents=True)
    worktree = main / ".claude" / "worktrees" / "h2"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {(main / '.git' / 'worktrees' / 'h2').as_posix()}\n", encoding="utf-8")
    return main, worktree


def test_env_files_search_the_worktree_then_the_main_checkout(tmp_path):
    main, worktree = _fake_worktree(tmp_path)
    assert cm.env_files(worktree) == [worktree / ".env", main / ".env"]
    assert cm.env_files(main) == [main / ".env"]          # a main checkout has a .git folder, not a file


def test_read_key_order(tmp_path):
    main, worktree = _fake_worktree(tmp_path)
    with pytest.raises(RuntimeError) as e:
        cm.read_key({}, worktree)
    assert str(worktree / ".env") in str(e.value) and str(main / ".env") in str(e.value)
    (main / ".env").write_text(f"# the owner's file\nOTHER=1\nZAI_API_KEY={FAKE_KEY}-main\n", encoding="utf-8")
    assert cm.read_key({}, worktree) == f"{FAKE_KEY}-main"
    (worktree / ".env").write_text(f'﻿ZAI_API_KEY="{FAKE_KEY}-worktree"\r\n', encoding="utf-8")
    assert cm.read_key({}, worktree) == f"{FAKE_KEY}-worktree"
    assert cm.read_key({"ZAI_API_KEY": f"{FAKE_KEY}-env"}, worktree) == f"{FAKE_KEY}-env"
    (worktree / ".env").write_text("ZAI_API_KEY=\n", encoding="utf-8")       # an empty line falls through
    assert cm.read_key({}, worktree) == f"{FAKE_KEY}-main"


# ---- call() against fake replies --------------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, data: bytes, status: int = 200):
        self.status = status
        self._buffer = io.BytesIO(data)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._buffer.readline, b"")

    def read(self):
        return self._buffer.read()


def _sse(*chunks) -> bytes:
    return b"".join(b"data: " + (c if isinstance(c, bytes) else json.dumps(c).encode()) + b"\n\n" for c in chunks)


ANSWER = "Here it is.\n\n```python\nimport triton\n\ndef render(*a):\n    pass\n```\n"
GOOD_STREAM = _sse(
    {"model": "glm-5.3", "choices": [{"index": 0, "delta": {"reasoning_content": "Think "}}]},
    {"model": "glm-5.3", "choices": [{"index": 0, "delta": {"reasoning_content": "hard."}}]},
    {"model": "glm-5.3", "choices": [{"index": 0, "delta": {"content": ANSWER[:20]}}]},
    {"model": "glm-5.3", "choices": [{"index": 0, "delta": {"content": ANSWER[20:]}, "finish_reason": "stop"}],
     "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}},
    b"[DONE]")


@pytest.fixture
def fake_api(monkeypatch, tmp_path):
    """Replace urllib.request.urlopen; record each request; the replies are set by the test."""
    monkeypatch.setenv("ZAI_API_KEY", FAKE_KEY)
    state = {"requests": [], "replies": []}

    def urlopen(request, timeout):
        state["requests"].append({"url": request.full_url, "headers": dict(request.header_items()),
                                  "body": json.loads(request.data), "timeout": timeout})
        reply = state["replies"].pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    prompt = tmp_path / "prompt v01.md"
    prompt.write_bytes("Implement it.\n\nThe spec: £ é\n".encode("utf-8"))
    state["prompt"] = prompt
    state["stem"] = tmp_path / "kernels" / "kernel_v01_1"
    return state


def _files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def _no_key_anywhere(folder: Path) -> bool:
    return all(FAKE_KEY.encode() not in p.read_bytes() for p in folder.rglob("*") if p.is_file())


def test_call_stream_writes_every_file_and_no_key(fake_api, capsys):
    fake_api["replies"].append(FakeResponse(GOOD_STREAM))
    meta = cm.call(fake_api["prompt"], fake_api["stem"], "https://api.example.test/api/paas/v4/", 131072, True, 30)
    folder = fake_api["stem"].parent
    assert _files(folder) == ["kernel_v01_1.md", "kernel_v01_1.meta.json", "kernel_v01_1.py",
                              "kernel_v01_1.raw.sse.txt", "kernel_v01_1.reasoning.md"]
    assert (folder / "kernel_v01_1.raw.sse.txt").read_bytes() == GOOD_STREAM
    assert (folder / "kernel_v01_1.md").read_bytes() == ANSWER.encode()             # \n kept on Windows
    assert (folder / "kernel_v01_1.reasoning.md").read_text(encoding="utf-8") == "Think hard."
    assert (folder / "kernel_v01_1.py").read_bytes() == b"import triton\n\ndef render(*a):\n    pass\n"
    written = json.loads((folder / "kernel_v01_1.meta.json").read_text(encoding="utf-8"))
    assert written == meta
    assert meta["http_status"] == 200 and meta["model"] == "glm-5.3" and meta["finish_reason"] == "stop"
    assert meta["usage"]["completion_tokens"] == 20
    assert meta["request"] == cm.build_request(fake_api["prompt"].read_bytes().decode("utf-8"), 131072, True)
    assert meta["prompt_sha256"] == __import__("hashlib").sha256(fake_api["prompt"].read_bytes()).hexdigest()
    assert meta["start_utc"].endswith("+00:00") and meta["end_utc"].endswith("+00:00")
    sent = fake_api["requests"][0]
    assert sent["url"] == "https://api.example.test/api/paas/v4/chat/completions" and FAKE_KEY not in sent["url"]
    assert sent["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert sent["headers"]["Content-type"] == "application/json" and sent["timeout"] == 30
    assert _no_key_anywhere(folder)
    assert FAKE_KEY not in capsys.readouterr().out
    with pytest.raises(FileExistsError):                                            # never a second reply
        cm.call(fake_api["prompt"], fake_api["stem"], "https://api.example.test", 131072, True)
    assert len(fake_api["requests"]) == 1


def test_call_without_code_block_writes_no_py(fake_api):
    stream = _sse({"model": "glm-5.3", "choices": [{"delta": {"content": "I cannot."}, "finish_reason": "stop"}]})
    fake_api["replies"].append(FakeResponse(stream))
    meta = cm.call(fake_api["prompt"], fake_api["stem"], "https://api.example.test", 131072, True)
    assert "kernel_v01_1.py" not in _files(fake_api["stem"].parent) and meta["code_block"] is False


def test_call_not_streamed(fake_api):
    reply = {"model": "glm-5.3", "choices": [{"message": {"content": ANSWER, "reasoning_content": "R"},
                                              "finish_reason": "length"}], "usage": {"total_tokens": 5}}
    raw = json.dumps(reply).encode()
    fake_api["replies"].append(FakeResponse(raw))
    meta = cm.call(fake_api["prompt"], fake_api["stem"], "https://api.example.test", 131072, False)
    folder = fake_api["stem"].parent
    assert (folder / "kernel_v01_1.raw.json").read_bytes() == raw
    assert meta["finish_reason"] == "length" and meta["request"]["stream"] is False     # a cut-off reply counts
    assert (folder / "kernel_v01_1.py").exists() and _no_key_anywhere(folder)


def test_stream_without_finish_reason_is_a_transport_failure(fake_api):
    broken = _sse({"model": "glm-5.3", "choices": [{"delta": {"content": "```python\nimport tri"}}]})
    fake_api["replies"] += [FakeResponse(broken), FakeResponse(GOOD_STREAM)]
    with pytest.raises(cm.TransportFailure) as e:
        cm.call(fake_api["prompt"], fake_api["stem"], "https://api.example.test", 131072, True)
    folder = fake_api["stem"].parent
    assert _files(folder) == ["kernel_v01_1.transport-1.txt"]
    assert e.value.path == folder / "kernel_v01_1.transport-1.txt"
    assert (folder / "kernel_v01_1.transport-1.txt").read_bytes().startswith(broken)
    meta = cm.call(fake_api["prompt"], fake_api["stem"], "https://api.example.test", 131072, True)   # proceeds
    assert meta["earlier_transport_failures"] == ["kernel_v01_1.transport-1.txt"]
    assert "kernel_v01_1.py" in _files(folder) and "kernel_v01_1.transport-1.txt" in _files(folder)
    assert _no_key_anywhere(folder)


def test_http_error_and_broken_connection_are_transport_failures(fake_api):
    error = urllib.error.HTTPError("https://api.example.test", 500, "Server Error", {},
                                   io.BytesIO(b'{"error": "busy", "echo": "' + FAKE_KEY.encode() + b'"}'))
    fake_api["replies"] += [error, ConnectionResetError("reset by peer")]
    for _ in range(2):
        with pytest.raises(cm.TransportFailure):
            cm.call(fake_api["prompt"], fake_api["stem"], "https://api.example.test", 131072, True)
    folder = fake_api["stem"].parent
    assert _files(folder) == ["kernel_v01_1.transport-1.txt", "kernel_v01_1.transport-2.txt"]
    assert b"HTTP 500" in (folder / "kernel_v01_1.transport-1.txt").read_bytes()
    assert b"ConnectionResetError" in (folder / "kernel_v01_1.transport-2.txt").read_bytes()
    assert _no_key_anywhere(folder)                                                  # an echoed key is redacted


def test_main_falls_back_to_128000_after_a_400(fake_api):
    refused = urllib.error.HTTPError("https://api.example.test", 400, "Bad Request", {}, io.BytesIO(b'{"error": "max_tokens"}'))
    fake_api["replies"] += [refused, FakeResponse(GOOD_STREAM)]
    code = cm.main(["--prompt", str(fake_api["prompt"]), "--out", str(fake_api["stem"]),
                    "--base-url", "https://api.example.test"])
    assert code == 0
    assert [r["body"]["max_tokens"] for r in fake_api["requests"]] == [131072, 128000]
    assert "kernel_v01_1.transport-1.txt" in _files(fake_api["stem"].parent)


def test_main_stops_after_a_transport_failure(fake_api):
    fake_api["replies"] += [ConnectionResetError("reset"), FakeResponse(GOOD_STREAM)]
    args = ["--prompt", str(fake_api["prompt"]), "--out", str(fake_api["stem"]), "--base-url", "https://x.test"]
    assert cm.main(args) == 2 and len(fake_api["requests"]) == 1                   # no paid retry on its own
    assert cm.main(args) == 0 and cm.main(args) == 1                                # again by hand; then refused


def test_call_without_a_key_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setattr(cm, "REPO_ROOT", tmp_path)      # never the real .env
    prompt = tmp_path / "p.md"
    prompt.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no ZAI_API_KEY"):
        cm.call(prompt, tmp_path / "out" / "k", "https://x.test", 131072, True)
    assert not (tmp_path / "out").exists()


# ---- the prompt builder -----------------------------------------------------------------------------------------

def test_build_prompt_is_instruction_blank_line_spec():
    spec = "<!--\nPURPOSE: x\n-->\r\n\r\n# Spec\r\nR01 £\r\n"
    prompt = bp.build_prompt(spec)
    lines = prompt.split("\n")
    assert lines[0] == bp.INSTRUCTION and lines[1] == ""
    assert "\n".join(lines[2:]) == spec.replace("\r\n", "\n")                       # tail -n +3 | diff --strip-trailing-cr
    assert bp.INSTRUCTION == ("Implement the Kernel interface section of the specification below as one Python module "
                              "that uses Triton. Reply with the complete module in a single python code block.")


def test_write_prompt_once_and_never_with_todo(tmp_path):
    spec = tmp_path / "RENDERING_SPEC.md"
    spec.write_bytes("# Spec\r\nR09: TODO[W4.1]: numbers\r\n".encode("utf-8"))
    out = tmp_path / "prompts" / "prompt_v01.md"
    with pytest.raises(ValueError, match="lines \\[2\\]"):
        bp.write_prompt(spec, out)
    assert not out.exists()
    spec.write_bytes("# Spec\r\nR09: phase free é\r\n".encode("utf-8"))
    digest = bp.write_prompt(spec, out)
    assert out.read_bytes() == (bp.INSTRUCTION + "\n\n# Spec\nR09: phase free é\n").encode("utf-8")
    assert digest == __import__("hashlib").sha256(out.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        bp.write_prompt(spec, out)


def test_the_real_prompt_is_not_built_yet():
    """The prompt is frozen only after the spec loses its TODO markers (build guide 4.5; D16, D25)."""
    spec = (REPO_ROOT / "spec" / "RENDERING_SPEC.md").read_text(encoding="utf-8")
    if bp.TODO.search(spec):
        assert not (REPO_ROOT / "h2" / "prompts" / "prompt_v01.md").exists()
