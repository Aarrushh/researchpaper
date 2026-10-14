# PURPOSE: The layer-0 MTF of each reader from its patch-embedding weights, fetched by byte range at the pinned revision (W3.6).
# WHAT GOES HERE: The range fetch and its SHA-256 check, the saved tensors, results/mtf/layer0_<reader>.csv and the folder's manifest.
# FILLED IN AT STEP: W3.6 (build guide section 3S.1).
# DO NOT: Download a whole checkpoint for one tensor. Fetch from the current revision. Import torch or use a GPU.
"""Step W3.6: the layer-0 MTF from the weights (build guide 3S.1).

    python scripts/04_mtf.py [--config configs/readers.yaml] [--readers qwen smolvlm ...]
                             [--weights-dir data/external/weights] [--out-dir results/mtf] [--offline]

For each reader of configs/readers.yaml: find the shard of its patch-embedding tensor at the pinned revision, read the
shard's safetensors header, fetch exactly the tensor's bytes with an HTTP range request (a reply other than 206 is
refused, so a server that ignores the range never sends the multi-gigabyte file), stop unless their SHA-256 equals the
table below, and fetch preprocessor_config.json at the same revision for sigma_c. The tensor (float32, lossless from
the Hub's BF16) and the config are saved under data/external/weights/, which git ignores, so that the tests and
section 3S.2 reuse them offline; --offline reads them from there, checks the hash again and fetches nothing.

Then results/mtf/layer0_<reader>.csv, columns profile (ring, horizontal, m0_0deg, m0_45deg), f_cyc_per_px and
value: the 32 rings of the radial profile, the 64 horizontal frequencies, and M_0 at 0 and 45 degrees at the ring
centres (k + 0.5)/64, k = 0 to 31; and manifest.json for the folder. Figure 3a is drawn by scripts/30_make_figures.py.

--hub and --table exist for the tests, which serve a fake shard from a local server with their own table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import struct
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from mrr.manifest import new_run_id, write_manifest
from mrr.mtf import grey_kernels, horizontal_profile, m0, radial_profile

HUB = "https://huggingface.co"
UA = {"User-Agent": "mrr-layer0-mtf/1.0"}
NFFT = 128
N_RINGS = 32
M0_ANGLES_DEG = (0.0, 45.0)
# Build guide 3S.1, the tensor table [Verified 10 Oct]; the revisions are the pins of configs/readers.yaml.
TENSORS: dict[str, dict[str, Any]] = {
    "qwen": {
        "name": "visual.patch_embed.proj.weight", "shard": "model-00001-of-00002.safetensors",
        "dtype": "BF16", "shape": [1280, 3, 2, 14, 14], "bytes": 3_010_560,
        "sha256": "a68f46dbc6a020725893197404a0872db63771130d7191c46a7bb343051f757f",
    },
    "smolvlm": {
        "name": "model.vision_model.embeddings.patch_embedding.weight", "shard": "model.safetensors",
        "dtype": "BF16", "shape": [1152, 3, 14, 14], "bytes": 1_354_752,
        "sha256": "0470f18987a02b0fd9f7e3159a6bfdb8142a5f88ff407afb13a29b3ec8336a6f",
    },
    "smolvlm500m": {
        "name": "model.vision_model.embeddings.patch_embedding.weight", "shard": "model.safetensors",
        "dtype": "BF16", "shape": [768, 3, 16, 16], "bytes": 1_179_648,
        "sha256": "ebf6acf8afb0ddbea26f6178d999bbf35ab5a5dde0c01de1670ff54e96caaab6",
    },
}
ROOT = Path(__file__).resolve().parents[1]

_log = logging.getLogger("mtf")


# --------------------------------------------------------------------------- the Hub (notes scripts/fetch_weights.py:36-79)


def http_get(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    """GET url, or the bytes byte_range (inclusive) of it; a range request answered with anything but 206 is refused."""
    headers = dict(UA)
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        if byte_range is not None and resp.status != 206:
            # Never fall through to downloading a multi-GB file.
            raise RuntimeError(f"server ignored Range for {url} (status {resp.status})")
        return resp.read()


def resolve(hub: str, repo: str, sha: str, fname: str) -> str:
    return f"{hub}/{repo}/resolve/{sha}/{fname}"


def weight_map(hub: str, repo: str, sha: str) -> dict[str, str] | None:
    """tensor name -> shard of a sharded checkpoint, or None when the repository has a single model.safetensors."""
    try:
        return json.loads(http_get(resolve(hub, repo, sha, "model.safetensors.index.json")))["weight_map"]
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise


def st_header(url: str) -> tuple[int, dict]:
    """(header length n, header) of a safetensors file: 8 bytes of little-endian length, then n bytes of JSON."""
    n = struct.unpack("<Q", http_get(url, (0, 7)))[0]
    return n, json.loads(http_get(url, (8, 8 + n - 1)))


def to_array(raw: bytes, dtype: str, shape: Sequence[int]) -> np.ndarray:
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return bits.view(np.float32).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    raise ValueError(f"unsupported dtype {dtype}")


def bf16_bytes(arr: np.ndarray) -> bytes:
    """The inverse of to_array for BF16: the raw bytes a float32 array came from (its low 16 bits are zero)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    return (a.view(np.uint32) >> 16).astype("<u2").tobytes()


# --------------------------------------------------------------------------- one reader


def check_sha(raw: bytes, spec: Mapping[str, Any], what: str) -> str:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != spec["sha256"]:
        raise RuntimeError(f"{what}: SHA-256 {digest} is not the table's {spec['sha256']}; stopped, nothing saved")
    return digest


def image_std(pre: Mapping[str, Any], what: str) -> list[float]:
    std = pre.get("image_std")
    if not isinstance(std, list) or len(std) != 3 or not all(isinstance(s, (int, float)) and s > 0 for s in std):
        raise RuntimeError(f"{what}: preprocessor_config.json has no image_std of three positive values: {std!r}")
    return [float(s) for s in std]


def fetch_tensor(hub: str, repo: str, revision: str, spec: Mapping[str, Any]) -> bytes:
    """Exactly the tensor's bytes from its shard at the pinned revision, checked against the table.

    Raises:
        RuntimeError: when the shard, dtype, shape or byte count differs from the table, the server ignores the
            range, or the SHA-256 differs.
    """
    name, what = spec["name"], f"{repo}@{revision[:7]} {spec['name']}"
    wm = weight_map(hub, repo, revision)
    shard = spec["shard"] if wm is None else wm.get(name)
    if shard != spec["shard"]:
        raise RuntimeError(f"{what}: in shard {shard!r}, the table says {spec['shard']!r}")
    url = resolve(hub, repo, revision, shard)
    n, header = st_header(url)
    info = header.get(name)
    if info is None:
        raise RuntimeError(f"{what}: not in the header of {shard}")
    if info["dtype"] != spec["dtype"] or list(info["shape"]) != list(spec["shape"]):
        raise RuntimeError(f"{what}: {info['dtype']} {info['shape']}, the table says {spec['dtype']} {spec['shape']}")
    a, b = info["data_offsets"]
    if b - a != spec["bytes"]:
        raise RuntimeError(f"{what}: {b - a} bytes, the table says {spec['bytes']}")
    raw = http_get(url, (8 + n + a, 8 + n + b - 1))
    if len(raw) != b - a:
        raise RuntimeError(f"{what}: the server sent {len(raw)} bytes of {b - a}")
    check_sha(raw, spec, what)
    return raw


def weight_paths(weights_dir: Path, reader: str) -> tuple[Path, Path, Path]:
    return (weights_dir / f"{reader}__patch_embedding.npy", weights_dir / f"{reader}__preprocessor_config.json",
            weights_dir / f"{reader}__record.json")


def fetch_reader(hub: str, reader_cfg: Mapping[str, Any], spec: Mapping[str, Any],
                 weights_dir: Path) -> tuple[np.ndarray, list[float], dict[str, Any]]:
    """Fetch the reader's tensor and preprocessor config, save both under weights_dir, return (weight, sigma, record)."""
    repo, revision = str(reader_cfg["hf_id"]), str(reader_cfg["revision"])
    raw = fetch_tensor(hub, repo, revision, spec)
    pre_raw = http_get(resolve(hub, repo, revision, "preprocessor_config.json"))
    std = image_std(json.loads(pre_raw), f"{repo}@{revision[:7]}")
    arr = to_array(raw, spec["dtype"], spec["shape"])
    record = {
        "reader": reader_cfg["name"], "hf_id": repo, "revision": revision, "tensor": spec["name"], "shard": spec["shard"],
        "dtype": spec["dtype"], "shape": list(spec["shape"]), "bytes": len(raw), "sha256": spec["sha256"],
        "image_std": std, "hub": hub, "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    npy, pre_path, rec_path = weight_paths(weights_dir, reader_cfg["name"])
    weights_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy, arr)
    pre_path.write_bytes(pre_raw)
    rec_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return arr, std, record


def load_reader(reader_cfg: Mapping[str, Any], spec: Mapping[str, Any],
                weights_dir: Path) -> tuple[np.ndarray, list[float], dict[str, Any]]:
    """The saved tensor, sigma and record of a reader; the tensor's bytes are checked against the table again."""
    npy, pre_path, rec_path = weight_paths(weights_dir, reader_cfg["name"])
    for path in (npy, pre_path, rec_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run without --offline first")
    record = json.loads(rec_path.read_text(encoding="utf-8"))
    if record.get("revision") != str(reader_cfg["revision"]) or record.get("tensor") != spec["name"]:
        raise RuntimeError(f"{rec_path}: records {record.get('tensor')} at {record.get('revision')}, not {spec['name']} at "
                           f"{reader_cfg['revision']}; run without --offline")
    arr = np.load(npy, allow_pickle=False)
    if spec["dtype"] != "BF16":
        raise RuntimeError(f"{npy}: the offline check knows BF16 tensors only, the table says {spec['dtype']}")
    if list(arr.shape) != list(spec["shape"]):
        raise RuntimeError(f"{npy}: shape {list(arr.shape)}, the table says {spec['shape']}")
    check_sha(bf16_bytes(arr), spec, str(npy))
    std = image_std(json.loads(pre_path.read_text(encoding="utf-8")), str(pre_path))
    return arr, std, record


def profiles(weight: np.ndarray, std: Sequence[float]) -> tuple[list[tuple[str, float, float]], np.ndarray]:
    """The CSV rows (profile, f_cyc_per_px, value) of one reader and its grey kernels."""
    K = grey_kernels(weight, std)
    f_ring, ring = radial_profile(K, NFFT, N_RINGS)
    f_h, horizontal = horizontal_profile(K, NFFT)
    rows = [("ring", float(f), float(v)) for f, v in zip(f_ring, ring)]
    rows += [("horizontal", float(f), float(v)) for f, v in zip(f_h, horizontal)]
    for deg in M0_ANGLES_DEG:
        rows += [(f"m0_{deg:g}deg", float(f), m0(K, float(f), np.deg2rad(deg))) for f in f_ring]
    return rows, K


def write_csv(path: Path, rows: Sequence[tuple[str, float, float]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["profile", "f_cyc_per_px", "value"])
        for profile, f, value in rows:
            writer.writerow([profile, f"{f:.10g}", f"{value:.12g}"])


def describe(rows: Sequence[tuple[str, float, float]]) -> str:
    """The ring table of notes A8 for the log: peak ring, lowest ring, the ring at 0.070, the last ring, half-peak."""
    ring = [(f, v) for p, f, v in rows if p == "ring"]
    f = np.array([x for x, _ in ring])
    v = np.array([y for _, y in ring])
    peak = int(np.argmax(v))
    below = peak + np.flatnonzero(v[peak:] < 0.5 * v[peak])  # after the peak, as the A8 table counts it
    half = f"first ring after the peak below half of it {f[below[0]]:.3f}" if below.size else "no ring after the peak below half of it"
    return (f"peak ring {f[peak]:.3f}, lowest ring {v[0]:.3f}, ring at {f[4]:.3f}: {v[4]:.3f}, "
            f"last ring {v[-1]:.3f}, {half} (gain, not a resolution limit)")


# --------------------------------------------------------------------------- the command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/readers.yaml", help="the readers, with hf_id and revision")
    p.add_argument("--readers", nargs="+", default=None, help="a subset of the readers (default: all in --config)")
    p.add_argument("--weights-dir", default="data/external/weights", help="where the tensors are saved (git-ignored)")
    p.add_argument("--out-dir", default="results/mtf", help="where layer0_<reader>.csv and manifest.json go")
    p.add_argument("--offline", action="store_true", help="read the saved tensors under --weights-dir; fetch nothing")
    p.add_argument("--hub", default=HUB, help=argparse.SUPPRESS)  # the tests serve a fake shard locally
    p.add_argument("--table", default=None, help=argparse.SUPPRESS)  # the tests bring their own tensor table (JSON)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    cfg_path, weights_dir, out_dir = Path(args.config), Path(args.weights_dir), Path(args.out_dir)
    readers = {r["name"]: r for r in yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["readers"]}
    table = TENSORS if args.table is None else json.loads(Path(args.table).read_text(encoding="utf-8"))
    names = list(args.readers or readers)
    unknown = [n for n in names if n not in readers]
    if unknown:
        _log.error("not in %s: %s", cfg_path, ", ".join(unknown))
        return 1
    untabled = [n for n in names if n not in table]
    if untabled:
        _log.error("no tensor in the table for %s: add its shard, shape and SHA-256 first (build guide 3S.1)", ", ".join(untabled))
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_readers: dict[str, Any] = {}
    for name in names:
        reader_cfg, spec = readers[name], table[name]
        try:
            if args.offline:
                arr, std, record = load_reader(reader_cfg, spec, weights_dir)
            else:
                arr, std, record = fetch_reader(args.hub, reader_cfg, spec, weights_dir)
        except (RuntimeError, FileNotFoundError, KeyError, ValueError, OSError, urllib.error.URLError) as exc:
            _log.error("%s: %s", name, exc)
            _log.error("stopped before %s: no manifest written", out_dir / "manifest.json")
            return 1
        _log.info("%s: %s from %s at %s, %s bytes, SHA-256 ok (%s); D = %d, P = %d, sigma = %s", name, spec["name"],
                  spec["shard"], str(reader_cfg["revision"])[:7], f"{spec['bytes']:,}",
                  "saved tensor" if args.offline else "fetched", arr.shape[0], arr.shape[-1], std)
        rows, K = profiles(arr, std)
        path = out_dir / f"layer0_{name}.csv"
        write_csv(path, rows)
        _log.info("%s: %s", name, describe(rows))
        _log.info("%s: %d rows in %s", name, len(rows), path)
        manifest_readers[name] = {**record, "filters": int(K.shape[0]), "patch_px": int(K.shape[-1]),
                                  "source": "offline" if args.offline else "fetched", "csv": path.name}
    manifest = write_manifest(out_dir, {
        "readers": manifest_readers, "readers_config": str(cfg_path), "weights_dir": str(weights_dir),
        "nfft": NFFT, "n_rings": N_RINGS, "m0_angles_deg": list(M0_ANGLES_DEG),
    }, run_id=new_run_id("mtf", "layer0"))
    _log.info("manifest: %s", manifest)
    _log.info("next: make figures draws Figure 3a (fig3a_layer0_mtf.pdf) from %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
