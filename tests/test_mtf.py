"""Tests for src/mrr/mtf.py and scripts/04_mtf.py (build guide section 3S.1). CPU only.

The fetch is tested against a local HTTP server that serves a fake safetensors shard with Range support, so the 206
check, the header arithmetic and the SHA-256 stop run without the network. The test marked model needs the real
tensors under data/external/weights/, saved by scripts/04_mtf.py, and checks the ring table of notes A8.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import os
import struct
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from conftest import REPO_ROOT
from mrr import mtf

SCRIPT = REPO_ROOT / "scripts" / "04_mtf.py"
WEIGHTS = REPO_ROOT / "data" / "external" / "weights"
# Notes A8, "Three instruments" [Measured 10 Oct]: the ring with the largest response, the response in the lowest
# ring, in the ring at 0.070 cycles per pixel, in the last ring, and the first ring after the peak below half of it
# (None: none). A gain, never a resolution limit (A8, "Gain is not loss").
RING_TABLE = {
    "qwen": (0.258, 0.198, 0.481, 0.832, None),
    "smolvlm": (0.008, 1.000, 0.555, 0.389, 0.289),
    "smolvlm500m": (0.164, 0.239, 0.787, 0.936, None),
}


# --------------------------------------------------------------------------- the module


def _tokens(K: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Layer 0 on a grey image in normalised units: one (D,) vector per whole patch, valid padding, stride P."""
    D, P, _ = K.shape
    rows, cols = image.shape[0] // P, image.shape[1] // P
    patches = image[:rows * P, :cols * P].reshape(rows, P, cols, P).transpose(0, 2, 1, 3).reshape(rows * cols, P * P)
    return patches @ K.reshape(D, -1).T


def test_grey_kernels_fold_the_colour_slices_with_their_std_and_sum_the_two_frames():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(5, 3, 2, 14, 14))  # Qwen's layout, with two frame slices that differ
    std = [0.2, 0.25, 0.5]
    K = mtf.grey_kernels(w, std)
    assert K.shape == (5, 14, 14) and K.dtype == np.float64
    np.testing.assert_allclose(K, mtf.grey_kernels(w.sum(axis=2), std))
    expected = sum(w.sum(axis=2)[:, c] / std[c] for c in range(3))
    np.testing.assert_allclose(K, expected)
    assert mtf.grey_kernels(np.float32(w[:, :, 0]), std).dtype == np.float64
    with pytest.raises(ValueError, match="image_std"):
        mtf.grey_kernels(w, [0.5, 0.5])
    with pytest.raises(ValueError, match="image_std"):
        mtf.grey_kernels(w, [0.5, 0.0, 0.5])
    with pytest.raises(ValueError, match="weight must be"):
        mtf.grey_kernels(np.zeros((4, 3, 14)), std)
    with pytest.raises(ValueError, match="weight must be"):
        mtf.grey_kernels(np.zeros((4, 3, 14, 16)), std)


def test_kernel_response_is_the_fft_on_its_grid_and_m0_the_rms_over_the_filters():
    rng = np.random.default_rng(1)
    K = rng.normal(size=(6, 14, 14))
    nfft = 64
    F = np.fft.fft2(K, s=(nfft, nfft))  # F[d, k_row, k_col]
    for k_col, k_row in ((3, 0), (0, 5), (4, 4), (7, 2), (1, 9)):
        f = np.hypot(k_col, k_row) / nfft
        theta = np.arctan2(k_row, k_col)
        np.testing.assert_allclose(mtf.kernel_response(K, f, theta), F[:, k_row, k_col], rtol=1e-10, atol=1e-10)
    response = mtf.kernel_response(K, 0.1, 0.3)
    assert response.shape == (6,) and np.iscomplexobj(response)
    assert mtf.m0(K, 0.1, 0.3) == pytest.approx(np.sqrt(np.mean(np.abs(response) ** 2)))
    sums = K.sum(axis=(1, 2))  # at zero frequency every direction sees the kernel sums
    for theta in (0.0, 1.0, np.pi / 2):
        assert mtf.m0(K, 0.0, theta) == pytest.approx(np.sqrt(np.mean(sums ** 2)))
    with pytest.raises(ValueError, match="grey kernels"):
        mtf.kernel_response(K[0], 0.1, 0.0)


@pytest.mark.parametrize("contrast", [0.5, 0.1])
def test_equation_4_on_a_random_bank_with_float_gratings(contrast):
    """G~ of section 3S.2 from float gratings on 6 x 6 patches at four phases equals sqrt(D/8) M_0 (build guide 3S.1)."""
    rng = np.random.default_rng(0)
    D, P, n = 8, 14, 6
    K = mtf.grey_kernels(rng.normal(size=(D, 3, P, P)), [0.5, 0.5, 0.5])
    y, x = np.mgrid[0:n * P, 0:n * P]
    for f in (0.02, 0.1, 0.197, 0.25, 0.4):
        for deg in (0, 30, 45, 90):
            theta = np.deg2rad(deg)
            z = np.stack([
                _tokens(K, 0.5 + 0.5 * contrast * np.sin(2 * np.pi * f * (x * np.cos(theta) + y * np.sin(theta)) + psi))
                for psi in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
            ])  # (4 phases, tokens, D): the grating in x/255 units, as the normaliser sees a byte image
            deviation = z - z.mean(axis=0, keepdims=True)
            g_tilde = np.sqrt(np.mean(np.sum(deviation ** 2, axis=-1))) / contrast
            ratio = g_tilde / (np.sqrt(D / 8) * mtf.m0(K, f, theta))
            assert abs(ratio - 1.0) < 1e-12, (f, deg, ratio)


def test_a_single_delta_kernel_responds_equally_at_every_frequency():
    K = np.zeros((1, 14, 14))
    K[0, 3, 5] = 1.0
    centres, ring = mtf.radial_profile(K)
    frequencies, horizontal = mtf.horizontal_profile(K)
    np.testing.assert_allclose(centres, (np.arange(32) + 0.5) / 64)
    np.testing.assert_allclose(frequencies, np.arange(64) / 128)
    np.testing.assert_allclose(ring, 1.0, atol=1e-12)
    np.testing.assert_allclose(horizontal, 1.0, atol=1e-12)
    centres8, ring8 = mtf.radial_profile(K, nfft=64, n_rings=8)
    assert centres8.shape == ring8.shape == (8,) and centres8[0] == pytest.approx(1 / 32)
    with pytest.raises(ValueError, match="rings"):
        mtf.radial_profile(K, nfft=32, n_rings=32)
    with pytest.raises(ValueError, match="smaller than the patch"):
        mtf.horizontal_profile(K, nfft=8)
    with pytest.raises(ValueError, match="every kernel is zero"):
        mtf.radial_profile(np.zeros((2, 14, 14)))


def test_the_profiles_peak_at_one_and_a_smooth_kernel_falls_with_frequency():
    rng = np.random.default_rng(2)
    K = rng.normal(size=(300, 14, 14))
    centres, ring = mtf.radial_profile(K)
    frequencies, horizontal = mtf.horizontal_profile(K)
    assert ring.max() == pytest.approx(1.0) and horizontal.max() == pytest.approx(1.0)
    assert ring.shape == (32,) and horizontal.shape == (64,) and np.all(ring > 0) and np.all(horizontal > 0)
    u = np.arange(14) - 6.5
    blur = np.exp(-(u[:, None] ** 2 + u[None, :] ** 2) / (2 * 2.0 ** 2))
    # A Gaussian of sigma 2 px falls as exp(-2 pi^2 sigma^2 f^2): 0.017 at 0.227 and 0.007 at 0.25 cycles per pixel; the
    # truncation to 14 px leaves a ripple of about 3e-4 beyond that, so monotonicity is checked where the response is visible.
    _, ring_blur = mtf.radial_profile(blur[None])
    assert np.argmax(ring_blur) == 0 and np.all(np.diff(ring_blur[:14]) < 0) and ring_blur[14:].max() < 0.02
    _, horizontal_blur = mtf.horizontal_profile(blur[None])
    assert np.all(np.diff(horizontal_blur[:32]) < 0) and horizontal_blur[32:].max() < 0.01


def test_csf_mannos_sakrison_peaks_at_7_89_cycles_per_degree():
    f = np.arange(0.0, 60.0, 0.001)
    a = mtf.csf_mannos_sakrison(f)
    assert f[np.argmax(a)] == pytest.approx(7.89, abs=0.005)
    assert a.max() == pytest.approx(0.981, abs=0.001)
    assert mtf.csf_mannos_sakrison(20.0) == pytest.approx(0.50, abs=0.005)  # notes A8: the pixel limit at 40 px per degree
    assert mtf.csf_mannos_sakrison(0.0) == pytest.approx(2.6 * 0.0192)
    assert mtf.csf_mannos_sakrison([1.0, 2.0]).shape == (2,)


# --------------------------------------------------------------------------- the script, against a local Hub


class _RangeHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with Range support; ignore_range makes it answer a range request with the whole file."""

    ignore_range = False

    def log_message(self, *args):  # quiet
        pass

    def do_GET(self):
        header = self.headers.get("Range")
        if header is None or self.ignore_range:
            return super().do_GET()
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        first, last = header.removeprefix("bytes=").split("-")
        chunk = data[int(first):int(last) + 1]
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Range", f"bytes {int(first)}-{int(first) + len(chunk) - 1}/{len(data)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)


def _bf16(rng: np.random.Generator, shape) -> np.ndarray:
    """A float32 array whose values are exactly representable in BF16 (low 16 bits zero)."""
    a = rng.normal(size=shape).astype(np.float32)
    return (a.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def _raw_bf16(arr: np.ndarray) -> bytes:
    return (np.ascontiguousarray(arr).view(np.uint32) >> 16).astype("<u2").tobytes()


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> dict[str, str]:
    """A safetensors file of BF16 tensors; returns the SHA-256 of each tensor's bytes."""
    header: dict = {"__metadata__": {"format": "pt"}}
    blobs, offset, shas = [], 0, {}
    for name, arr in tensors.items():
        raw = _raw_bf16(arr)
        header[name] = {"dtype": "BF16", "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
        shas[name] = hashlib.sha256(raw).hexdigest()
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))
    return shas


@pytest.fixture(scope="module")
def hub(tmp_path_factory):
    """A local Hub: two fake readers, one sharded like Qwen with a 3D weight, one single-file like SmolVLM."""
    root = tmp_path_factory.mktemp("hub")
    rng = np.random.default_rng(3)
    readers = {
        "qwen": {"hf_id": "fake/qwen-vl", "revision": "a" * 40, "shape": (4, 3, 2, 14, 14),
                 "std": [0.26862954, 0.26130258, 0.27577711], "name": "visual.patch_embed.proj.weight",
                 "shard": "model-00001-of-00002.safetensors", "sharded": True},
        "smol": {"hf_id": "fake/smol", "revision": "b" * 40, "shape": (3, 3, 16, 16), "std": [0.5, 0.5, 0.5],
                 "name": "model.vision_model.embeddings.patch_embedding.weight", "shard": "model.safetensors",
                 "sharded": False},
    }
    table, arrays = {}, {}
    for name, r in readers.items():
        folder = root / r["hf_id"] / "resolve" / r["revision"]
        folder.mkdir(parents=True)
        arr = _bf16(rng, r["shape"])
        arrays[name] = arr
        shas = _write_safetensors(folder / r["shard"], {"some.other.weight": _bf16(rng, (5, 7)), r["name"]: arr})
        if r["sharded"]:
            (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                "some.other.weight": r["shard"], r["name"]: r["shard"], "lm_head.weight": "model-00002-of-00002.safetensors",
            }}), encoding="utf-8")
        (folder / "preprocessor_config.json").write_text(
            json.dumps({"image_mean": [0.5, 0.5, 0.5], "image_std": r["std"]}), encoding="utf-8")
        table[name] = {"name": r["name"], "shard": r["shard"], "dtype": "BF16", "shape": list(r["shape"]),
                       "bytes": arr.size * 2, "sha256": shas[r["name"]]}
    config = root / "readers.yaml"
    config.write_text(yaml.safe_dump({"readers": [
        {"name": n, "hf_id": r["hf_id"], "revision": r["revision"], "patch_size": r["shape"][-1]} for n, r in readers.items()
    ]}), encoding="utf-8")
    table_path = root / "table.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_RangeHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_address[1]}", root=root, config=config,
                              table=table, table_path=table_path, arrays=arrays, readers=readers)
    finally:
        server.shutdown()
        server.server_close()


def _mtf(*args) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONUTF8": "1",
           "PYTHONPATH": os.pathsep.join(p for p in (str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")) if p)}
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, env=env,
                          cwd=REPO_ROOT)


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_the_script_fetches_each_tensor_by_range_checks_its_hash_and_writes_the_profiles(hub, tmp_path):
    weights, out = tmp_path / "weights", tmp_path / "mtf"
    proc = _mtf("--config", hub.config, "--table", hub.table_path, "--hub", hub.url, "--weights-dir", weights,
                "--out-dir", out)
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.count("SHA-256 ok (fetched)") == 2 and "Figure 3a" in proc.stderr
    for name, r in hub.readers.items():
        saved = np.load(weights / f"{name}__patch_embedding.npy", allow_pickle=False)
        np.testing.assert_array_equal(saved, hub.arrays[name])
        pre = json.loads((weights / f"{name}__preprocessor_config.json").read_text(encoding="utf-8"))
        assert pre["image_std"] == r["std"]
        record = json.loads((weights / f"{name}__record.json").read_text(encoding="utf-8"))
        assert record["sha256"] == hub.table[name]["sha256"] and record["revision"] == r["revision"]
        assert record["bytes"] == hub.arrays[name].size * 2 and record["shard"] == r["shard"]
        rows = _rows(out / f"layer0_{name}.csv")
        profiles = [row["profile"] for row in rows]
        assert len(rows) == 160 and list(rows[0]) == ["profile", "f_cyc_per_px", "value"]
        assert profiles.count("ring") == 32 and profiles.count("horizontal") == 64
        assert profiles.count("m0_0deg") == 32 and profiles.count("m0_45deg") == 32
        K = mtf.grey_kernels(hub.arrays[name], r["std"])
        f_ring, ring = mtf.radial_profile(K)
        f_h, horizontal = mtf.horizontal_profile(K)
        np.testing.assert_allclose([float(x["value"]) for x in rows if x["profile"] == "ring"], ring, rtol=1e-9)
        np.testing.assert_allclose([float(x["f_cyc_per_px"]) for x in rows if x["profile"] == "ring"], f_ring, rtol=1e-9)
        np.testing.assert_allclose([float(x["value"]) for x in rows if x["profile"] == "horizontal"], horizontal, rtol=1e-9)
        np.testing.assert_allclose([float(x["f_cyc_per_px"]) for x in rows if x["profile"] == "horizontal"], f_h, rtol=1e-9)
        m0_45 = [float(x["value"]) for x in rows if x["profile"] == "m0_45deg"]
        assert m0_45[3] == pytest.approx(mtf.m0(K, f_ring[3], np.deg2rad(45)), rel=1e-9)
        m0_0 = [float(x["value"]) for x in rows if x["profile"] == "m0_0deg"]
        assert m0_0[0] == pytest.approx(mtf.m0(K, f_ring[0], 0.0), rel=1e-9)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    readers = manifest["config"]["readers"]
    assert set(readers) == {"qwen", "smol"} and readers["qwen"]["filters"] == 4 and readers["smol"]["patch_px"] == 16
    assert readers["qwen"]["sha256"] == hub.table["qwen"]["sha256"] and readers["qwen"]["source"] == "fetched"
    assert manifest["config"]["nfft"] == 128 and manifest["config"]["n_rings"] == 32
    assert "_mtf_layer0_" in manifest["run_id"]


def test_the_script_stops_on_a_hash_mismatch_and_saves_nothing_for_that_reader(hub, tmp_path):
    table = {**hub.table, "smol": {**hub.table["smol"], "sha256": "0" * 64}}
    table_path = tmp_path / "table.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")
    proc = _mtf("--config", hub.config, "--table", table_path, "--hub", hub.url, "--weights-dir", tmp_path / "w",
                "--out-dir", tmp_path / "mtf")
    assert proc.returncode == 1 and "smol: " in proc.stderr and "SHA-256" in proc.stderr and "is not the table's" in proc.stderr
    assert (tmp_path / "w" / "qwen__patch_embedding.npy").exists()  # the first reader passed
    assert not (tmp_path / "w" / "smol__patch_embedding.npy").exists()
    assert (tmp_path / "mtf" / "layer0_qwen.csv").exists() and not (tmp_path / "mtf" / "layer0_smol.csv").exists()
    assert not (tmp_path / "mtf" / "manifest.json").exists()


def test_the_script_refuses_a_shard_in_another_place_than_the_table_says(hub, tmp_path):
    table = {**hub.table, "qwen": {**hub.table["qwen"], "shard": "model-00002-of-00002.safetensors"}}
    table_path = tmp_path / "table.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")
    proc = _mtf("--config", hub.config, "--table", table_path, "--hub", hub.url, "--readers", "qwen",
                "--weights-dir", tmp_path / "w", "--out-dir", tmp_path / "mtf")
    assert proc.returncode == 1 and "the table says 'model-00002-of-00002.safetensors'" in proc.stderr
    assert not (tmp_path / "w").exists()


def test_the_script_refuses_a_server_that_ignores_the_range(hub, tmp_path):
    _RangeHandler.ignore_range = True
    try:
        proc = _mtf("--config", hub.config, "--table", hub.table_path, "--hub", hub.url, "--readers", "smol",
                    "--weights-dir", tmp_path / "w", "--out-dir", tmp_path / "mtf")
    finally:
        _RangeHandler.ignore_range = False
    assert proc.returncode == 1 and "ignored Range" in proc.stderr
    assert not (tmp_path / "w").exists() and not (tmp_path / "mtf" / "layer0_smol.csv").exists()


def test_the_script_offline_reuses_the_saved_tensors_and_stops_on_a_changed_file(hub, tmp_path):
    weights, first, second = tmp_path / "weights", tmp_path / "first", tmp_path / "second"
    proc = _mtf("--config", hub.config, "--table", hub.table_path, "--hub", hub.url, "--weights-dir", weights,
                "--out-dir", first)
    assert proc.returncode == 0, proc.stderr
    unreachable = "http://127.0.0.1:9"  # --offline must not touch the network
    proc = _mtf("--config", hub.config, "--table", hub.table_path, "--hub", unreachable, "--offline",
                "--weights-dir", weights, "--out-dir", second)
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.count("SHA-256 ok (saved tensor)") == 2
    for name in hub.readers:
        assert (first / f"layer0_{name}.csv").read_bytes() == (second / f"layer0_{name}.csv").read_bytes()
    assert json.loads((second / "manifest.json").read_text(encoding="utf-8"))["config"]["readers"]["smol"]["source"] == "offline"
    npy = weights / "qwen__patch_embedding.npy"
    arr = np.load(npy, allow_pickle=False)
    arr[0, 0, 0, 0, 0] = -arr[0, 0, 0, 0, 0] + 1.0
    np.save(npy, arr)
    proc = _mtf("--config", hub.config, "--table", hub.table_path, "--hub", unreachable, "--offline",
                "--weights-dir", weights, "--out-dir", tmp_path / "third")
    assert proc.returncode == 1 and "SHA-256" in proc.stderr and "qwen__patch_embedding.npy" in proc.stderr
    proc = _mtf("--config", hub.config, "--table", hub.table_path, "--hub", unreachable, "--offline",
                "--weights-dir", tmp_path / "none", "--out-dir", tmp_path / "fourth")
    assert proc.returncode == 1 and "run without --offline first" in proc.stderr


def test_the_script_refuses_an_unknown_reader_and_a_reader_without_a_table_row(hub, tmp_path):
    proc = _mtf("--config", hub.config, "--table", hub.table_path, "--hub", hub.url, "--readers", "nobody",
                "--weights-dir", tmp_path / "w", "--out-dir", tmp_path / "mtf")
    assert proc.returncode == 1 and "not in" in proc.stderr and "nobody" in proc.stderr
    table_path = tmp_path / "table.json"
    table_path.write_text(json.dumps({"qwen": hub.table["qwen"]}), encoding="utf-8")
    proc = _mtf("--config", hub.config, "--table", table_path, "--hub", hub.url,
                "--weights-dir", tmp_path / "w", "--out-dir", tmp_path / "mtf")
    assert proc.returncode == 1 and "no tensor in the table for smol" in proc.stderr


# --------------------------------------------------------------------------- the real weights (after scripts/04_mtf.py)


@pytest.mark.model
@pytest.mark.parametrize("reader", sorted(RING_TABLE))
def test_the_real_tensors_give_the_ring_table_of_notes_a8(reader):
    npy, pre = WEIGHTS / f"{reader}__patch_embedding.npy", WEIGHTS / f"{reader}__preprocessor_config.json"
    if not (npy.exists() and pre.exists()):
        pytest.skip(f"{npy.name} not present: run scripts/04_mtf.py first (W3.6)")
    std = json.loads(pre.read_text(encoding="utf-8"))["image_std"]
    f, ring = mtf.radial_profile(mtf.grey_kernels(np.load(npy, allow_pickle=False), std))
    peak, lowest, at_070, last, half = RING_TABLE[reader]
    assert f[np.argmax(ring)] == pytest.approx(peak, abs=0.002)
    assert ring[0] == pytest.approx(lowest, abs=0.002)
    assert f[4] == pytest.approx(0.0703, abs=1e-4) and ring[4] == pytest.approx(at_070, abs=0.002)
    assert f[-1] == pytest.approx(0.4921875) and ring[-1] == pytest.approx(last, abs=0.002)
    peak_ring = int(np.argmax(ring))
    below = peak_ring + np.flatnonzero(ring[peak_ring:] < 0.5 * ring[peak_ring])  # after the peak, as A8 counts it
    if half is None:
        assert below.size == 0
    else:
        assert f[below[0]] == pytest.approx(half, abs=0.002)
