# PURPOSE: Send the frozen H2 prompt to GLM-5.3 once and keep everything that comes back, byte for byte.
# WHAT GOES HERE: The request body, the key lookup, one call with its files, code-block extraction, the command line.
# FILLED IN AT STEP: W4.3.
# DO NOT: Print the key, write it to any file or put it in a URL. Do not edit what the model returns. Do not repeat a call whose reply arrived. Standard library only.
"""Step W4.3: one GLM-5.3 call for H2 (build guide 4.5), standard library only.

    python h2/call_model.py --prompt h2/prompts/prompt_v01.md --out h2/kernels/kernel_v01_<n>
                            [--base-url URL] [--max-tokens N] [--no-stream] [--timeout S]

A call writes, next to <out>:
    <out>.raw.sse.txt (a stream) or <out>.raw.json (no stream)   the bytes as they arrived, saved before parsing
    <out>.md, <out>.reasoning.md                                 the answer and the reasoning
    <out>.py                                                     the code block, verbatim, if the answer has one
    <out>.meta.json                                              the request body (never the key), the HTTP status,
                                                                 the model string, the finish reason, the usage,
                                                                 the UTC times and the prompt's SHA-256
A transport failure (a status other than 200, a broken connection, a reply without a finish reason) is not an
attempt: whatever arrived goes to <out>.transport-<k>.txt, nothing else is written, and the command exits 2 so that
the same command can be run again once the owner says so. Only an HTTP 400 at 131072 tokens is retried at once,
at 128000 (PREREGISTRATION.md, H2: "max_tokens 131072, or 128000 if the API refuses the larger value"); a refused
request generates nothing.

The key comes from the environment variable ZAI_API_KEY, or from a line ZAI_API_KEY=... in the first .env of
env_files(): the repository root, then, when the repository is a git worktree, the root of its main checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

MODEL = "glm-5.3"                                  # never GLM-5.3-Flash
DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"  # provisional: the dummy calls decide (amendment, PREREGISTRATION.md)
ENDPOINT = "/chat/completions"
MAX_TOKENS = 131072
FALLBACK_MAX_TOKENS = 128000
KEY_NAME = "ZAI_API_KEY"
REPO_ROOT = Path(__file__).resolve().parents[1]


class TransportFailure(RuntimeError):
    """The call did not produce an attempt; ``path`` holds whatever arrived and ``status`` the HTTP status, if any."""

    def __init__(self, message: str, path: Path, status: int | None):
        super().__init__(message)
        self.path = path
        self.status = status


def build_request(prompt_text: str, max_tokens: int, stream: bool) -> dict:
    """The JSON body of build guide 4.5: one user message holding the prompt, and never the key."""
    return {"model": MODEL,
            "messages": [{"role": "user", "content": prompt_text}],
            "thinking": {"type": "enabled"}, "reasoning_effort": "max",
            "temperature": 1.0, "max_tokens": max_tokens, "stream": stream}


def env_files(root: Path = REPO_ROOT) -> list[Path]:
    """Where the key may be written, in the order they are searched.

    root/.env first. When root is a git worktree, root/.git is a file "gitdir: <main>/.git/worktrees/<name>", and
    <main>/.env comes second: the owner keeps one .env in the main checkout for every worktree. Both are ignored by
    git (.gitignore, ".env").
    """
    files = [root / ".env"]
    dot_git = root / ".git"
    if dot_git.is_file():
        line = dot_git.read_text(encoding="utf-8").strip()
        if line.startswith("gitdir:"):
            gitdir = Path(line[len("gitdir:"):].strip())
            if not gitdir.is_absolute():
                gitdir = (root / gitdir).resolve()
            if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
                main_env = gitdir.parent.parent.parent / ".env"
                if main_env != files[0]:
                    files.append(main_env)
    return files


def _key_from_file(path: Path) -> str | None:
    """The value of the line ZAI_API_KEY=... in path, without quotes; None when the file or the line is missing."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        if sep and name.strip() == KEY_NAME:
            value = value.strip().strip('"').strip("'")
            return value or None
    return None


def read_key(environ: Any = None, root: Path = REPO_ROOT) -> str:
    """The API key: the environment variable first, then the .env files of env_files(root). Never printed.

    Raises:
        RuntimeError: naming the places searched (never a value) when there is no key.
    """
    environ = os.environ if environ is None else environ
    if environ.get(KEY_NAME):
        return environ[KEY_NAME]
    searched = env_files(root)
    for path in searched:
        key = _key_from_file(path)
        if key:
            return key
    raise RuntimeError(f"no {KEY_NAME}: not in the environment, and no {KEY_NAME}= line in "
                       + ", ".join(str(p) for p in searched))


def extract_code(content: str) -> str | None:
    """The answer's code block, verbatim (build guide 4.5).

    The first fenced block whose info string is python or py, in any case; if there is none, the first fenced block
    of any kind; if there is none, None. A fence is a line of three or more backticks or tildes, indented by at most
    three spaces; it closes at a line of at least as many of the same character and nothing else. A block that is
    never closed runs to the end of the answer. The lines between the fences come back exactly as written, each with
    its newline.
    """
    lines = content.split("\n")
    blocks: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        opening = _fence(lines[i])
        if opening is None:
            i += 1
            continue
        char, length, info = opening
        body: list[str] = []
        j = i + 1
        while j < len(lines):
            closing = _fence(lines[j])
            if closing is not None and closing[0] == char and closing[1] >= length and not closing[2]:
                break
            body.append(lines[j])
            j += 1
        if j < len(lines):            # closed: every line of the block keeps its newline
            text = "".join(line + "\n" for line in body)
        else:                         # never closed: the text to the end, as it ends
            text = "\n".join(body)
        blocks.append((info.split()[0].lower() if info.split() else "", text))
        i = j + 1
    for language, text in blocks:
        if language in ("python", "py"):
            return text
    return blocks[0][1] if blocks else None


def _fence(line: str) -> tuple[str, int, str] | None:
    """(character, length, info string) when line is a code fence, else None."""
    stripped = line.lstrip(" ")
    if len(line) - len(stripped) > 3:
        return None
    for char in ("`", "~"):
        length = len(stripped) - len(stripped.lstrip(char))
        if length >= 3:
            info = stripped[length:].strip()
            if char == "`" and "`" in info:
                return None
            return char, length, info
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _existing_outputs(out_stem: Path) -> list[Path]:
    """Files <out_stem>.* other than the transport files of earlier failures."""
    prefix = out_stem.name + "."
    if not out_stem.parent.is_dir():
        return []
    return sorted(p for p in out_stem.parent.iterdir()
                  if p.name.startswith(prefix) and not _is_transport(p, out_stem))


def _is_transport(path: Path, out_stem: Path) -> bool:
    rest = path.name[len(out_stem.name) + 1:]
    return rest.startswith("transport-") and rest.endswith(".txt") and rest[len("transport-"):-4].isdigit()


def _transport_path(out_stem: Path) -> Path:
    k = 1
    while (out_stem.parent / f"{out_stem.name}.transport-{k}.txt").exists():
        k += 1
    return out_stem.parent / f"{out_stem.name}.transport-{k}.txt"


def _redact(data: bytes, key: str) -> bytes:
    """The key never reaches a file, even if a reply echoed it."""
    return data.replace(key.encode("utf-8"), b"[" + KEY_NAME.encode("ascii") + b"]") if key else data


def _fail(out_stem: Path, arrived: bytes | Path, reason: str, status: int | None, key: str) -> TransportFailure:
    """Move whatever arrived to <out_stem>.transport-<k>.txt, add one line saying why, and return the error."""
    path = _transport_path(out_stem)
    if isinstance(arrived, Path):
        arrived.replace(path)
    else:
        path.write_bytes(_redact(arrived, key))
    with open(path, "ab") as fh:
        fh.write(_redact(f"\n--- transport failure, not an attempt: {reason} ---\n".encode("utf-8"), key))
    return TransportFailure(f"transport failure ({reason}); saved {path.name}; not an attempt", path, status)


def _parse_stream(raw_path: Path) -> dict:
    """Join the chunks of a saved stream: answer, reasoning, finish reason, model, usage."""
    answer, reasoning = [], []
    out: dict[str, Any] = {"finish_reason": None, "model": None, "usage": None}
    for line in raw_path.read_bytes().split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        try:
            chunk = json.loads(line[len(b"data:"):].strip())
        except ValueError:          # [DONE], or anything else that is not JSON
            continue
        if not isinstance(chunk, dict):
            continue
        out["model"] = chunk.get("model") or out["model"]
        out["usage"] = chunk.get("usage") or out["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            answer.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
            out["finish_reason"] = choice.get("finish_reason") or out["finish_reason"]
    out["answer"], out["reasoning"] = "".join(answer), "".join(reasoning)
    return out


def _parse_body(raw: bytes) -> dict:
    """The same fields from a reply that was not streamed (OpenAI-style; the dummy calls confirm where they are)."""
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise ValueError(f"reply is a JSON {type(body).__name__}, not an object")
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {"finish_reason": choice.get("finish_reason"), "model": body.get("model"), "usage": body.get("usage"),
            "answer": message.get("content") or "", "reasoning": message.get("reasoning_content") or ""}


def call(prompt_path: Path, out_stem: Path, base_url: str, max_tokens: int, stream: bool,
         timeout_s: int = 600) -> dict:
    """Send the prompt once and write the files of the module docstring; return the meta.

    Args:
        prompt_path: the frozen prompt; its bytes, decoded as UTF-8, are the one user message.
        out_stem: path without suffix, such as h2/kernels/kernel_v01_1.
        base_url: the API's base, without /chat/completions.
        max_tokens: 131072, or 128000 if the service refuses it.
        stream: whether to ask for server-sent events.
        timeout_s: the socket timeout of each blocking read, not of the whole reply.

    Raises:
        FileExistsError: if any <out_stem>.* exists other than earlier transport files.
        TransportFailure: the reply is not an attempt; whatever arrived is in its path.
        RuntimeError: there is no key.
    """
    prompt_path, out_stem = Path(prompt_path), Path(out_stem)
    existing = _existing_outputs(out_stem)
    if existing:
        raise FileExistsError(f"{existing[0]} exists: this attempt already has a reply; never call again")
    prompt_bytes = prompt_path.read_bytes()
    prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
    body = build_request(prompt_bytes.decode("utf-8"), max_tokens, stream)
    key = read_key(os.environ, REPO_ROOT)
    url = base_url.rstrip("/") + ENDPOINT
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    earlier = sorted(p.name for p in out_stem.parent.glob(out_stem.name + ".transport-*.txt"))
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_stem.parent / (out_stem.name + (".raw.sse.txt" if stream else ".raw.json"))
    start = _utc_now()
    status = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = response.status
            with open(raw_path, "wb") as raw:            # saved before anything is parsed
                if stream:
                    for line in response:                 # flushed line by line as it arrives
                        raw.write(_redact(line, key))
                        raw.flush()
                else:
                    raw.write(_redact(response.read(), key))
    except urllib.error.HTTPError as e:
        try:
            arrived = e.read()
        except Exception:  # noqa: BLE001  (the body of an error reply may itself be broken)
            arrived = b""
        raise _fail(out_stem, arrived, f"HTTP {e.code}", e.code, key) from None
    except (OSError, http.client.HTTPException, ValueError) as e:  # broken connection, timeout, incomplete read
        reason = f"{type(e).__name__}: {e}"
        if raw_path.exists():
            raise _fail(out_stem, raw_path, reason, status, key) from None
        raise _fail(out_stem, b"", reason, status, key) from None
    end = _utc_now()
    if status != 200:
        raise _fail(out_stem, raw_path, f"HTTP {status}", status, key)
    try:
        parsed = _parse_stream(raw_path) if stream else _parse_body(raw_path.read_bytes())
    except ValueError as e:
        raise _fail(out_stem, raw_path, f"reply is not JSON: {e}", status, key) from None
    if not parsed["finish_reason"]:
        raise _fail(out_stem, raw_path, "no finish reason", status, key)

    written = [raw_path.name]
    for suffix, text in ((".md", parsed["answer"]), (".reasoning.md", parsed["reasoning"])):
        path = out_stem.parent / (out_stem.name + suffix)
        with open(path, "w", encoding="utf-8", newline="") as fh:   # newline="": the model's \n stay \n
            fh.write(text)
        written.append(path.name)
    code = extract_code(parsed["answer"])
    if code is not None:
        path = out_stem.parent / (out_stem.name + ".py")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(code)
        written.append(path.name)
    meta = {"request": body, "url": url, "http_status": status, "model": parsed["model"],
            "finish_reason": parsed["finish_reason"], "usage": parsed["usage"], "start_utc": start, "end_utc": end,
            "prompt_file": prompt_path.as_posix(), "prompt_sha256": prompt_sha256, "stream": stream,
            "timeout_s": timeout_s, "code_block": code is not None, "earlier_transport_failures": earlier,
            "files": written + [out_stem.name + ".meta.json"]}
    (out_stem.parent / (out_stem.name + ".meta.json")).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    _print_meta(meta)
    return meta


def _print_meta(meta: dict) -> None:
    """The meta as written, with the prompt shown by its length and hash instead of its text."""
    shown = json.loads(json.dumps(meta))
    for message in shown["request"]["messages"]:
        message["content"] = f"<{len(message['content'])} characters, sha256 {meta['prompt_sha256']}>"
    print(json.dumps(shown, indent=2))


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prompt", required=True, type=Path, help="the frozen prompt, h2/prompts/prompt_v01.md")
    p.add_argument("--out", required=True, type=Path, help="output stem, h2/kernels/kernel_v01_<n>")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="without /chat/completions")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--no-stream", dest="stream", action="store_false", help="ask for one JSON reply instead")
    p.add_argument("--timeout", type=int, default=600, help="socket timeout per read, seconds")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        try:
            call(args.prompt, args.out, args.base_url, args.max_tokens, args.stream, args.timeout)
        except TransportFailure as e:
            if not (e.status == 400 and args.max_tokens == MAX_TOKENS):
                raise
            print(f"{e}\nHTTP 400 at max_tokens {MAX_TOKENS}: the request was refused; once more at {FALLBACK_MAX_TOKENS}")
            call(args.prompt, args.out, args.base_url, FALLBACK_MAX_TOKENS, args.stream, args.timeout)
    except TransportFailure as e:
        print(f"{e}\nRun the same command again when the owner says so.", file=sys.stderr)
        return 2
    except (FileExistsError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
