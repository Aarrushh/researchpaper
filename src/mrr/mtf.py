# PURPOSE: The layer-0 modulation transfer of a reader from its patch-embedding weights alone (section 3S.1).
# WHAT GOES HERE: Grey kernels, their Fourier response to a grating, the ring and horizontal profiles, and the human CSF.
# FILLED IN AT STEP: W3.6.
# DO NOT: Import torch or load a model. Read the half-response frequency as a resolution limit (notes A8, "Gain is not loss").
"""The layer-0 MTF from the weights (build guide 3S.1; notes A8, equations (4) to (9)).

Layer 0 of every reader is a strided convolution after the processor's normaliser (x/255 - mu_c)/sigma_c, so its
response to a grey grating follows from the patch-embedding weights alone. Fold each filter's colour slices into
one grey kernel, equation (1); take the kernel's Fourier transform at the grating's frequency and direction,
equation (2); average the power over the filters, equation (3). Equation (4) ties M_0 to the gain of section 3S.2:
for a grating of any contrast averaged over four phases a quarter cycle apart, G~_0(f, theta) = sqrt(D/8) M_0(f, theta).
NumPy only; nothing here needs a model, an image or a GPU.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def grey_kernels(weight: np.ndarray, image_std: Sequence[float]) -> np.ndarray:
    """Equation (1): one grey kernel per filter, (D, P, P), from a (D, C, P, P) or (D, C, 2, P, P) weight.

    The weight is cast to float64. A five-dimensional weight (Qwen's 3D convolution over two frames) is summed over
    its frame axis first, because a still image enters the encoder as two identical frames. Each colour slice is then
    divided by that channel's standard deviation from the processor's preprocessor_config.json and the slices are
    summed: a grey pixel puts the same value into R, G and B, so the three slices act as one kernel.

    Raises:
        ValueError: when the weight is not (D, C, P, P) or (D, C, 2, P, P) with square patches, or image_std does not
            hold one positive value per colour channel.
    """
    w = np.asarray(weight, dtype=np.float64)
    if w.ndim == 5:
        w = w.sum(axis=2)
    if w.ndim != 4 or w.shape[-1] != w.shape[-2]:
        raise ValueError(f"weight must be (D, C, P, P) or (D, C, 2, P, P) with square patches, got {np.shape(weight)}")
    std = np.asarray(image_std, dtype=np.float64)
    if std.shape != (w.shape[1],) or not np.all(std > 0):
        raise ValueError(f"image_std must hold {w.shape[1]} positive values, one per colour channel, got {list(image_std)!r}")
    return (w / std[None, :, None, None]).sum(axis=1)


def _kernels(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 3:
        raise ValueError(f"K must be (D, P, P) grey kernels, got shape {K.shape}")
    return K


def kernel_response(K: np.ndarray, f: float, theta_rad: float) -> np.ndarray:
    """Equation (2): each kernel's Fourier transform at frequency f (cycles per pixel) and direction theta, (D,) complex.

    K is (D, P, P) with K[d, v, u], v the row and u the column of the pixel, the layout of a convolution weight. The
    exponent is -2 pi i f (u cos theta + v sin theta): theta = 0 is a grating whose wave runs along the columns, so
    its bars are vertical; theta = pi/2 runs down the rows.
    """
    K = _kernels(K)
    u = np.arange(K.shape[-1], dtype=np.float64)[None, :]
    v = np.arange(K.shape[-2], dtype=np.float64)[:, None]
    phase = 2.0 * np.pi * float(f) * (u * np.cos(theta_rad) + v * np.sin(theta_rad))  # (P, P), indexed [v, u]
    return np.tensordot(K, np.exp(-1j * phase), axes=([1, 2], [0, 1]))


def m0(K: np.ndarray, f: float, theta_rad: float) -> float:
    """Equation (3): the root mean square over the filters of |K_d^(f, theta)|, the layer-0 response to a grating."""
    response = kernel_response(K, f, theta_rad)
    return float(np.sqrt(np.mean(np.abs(response) ** 2)))


def _power_spectrum(K: np.ndarray, nfft: int, chunk: int = 128) -> np.ndarray:
    """The mean over the filters of |FFT|^2 of each kernel zero-padded to nfft x nfft, zero frequency at the centre."""
    K = _kernels(K)
    if nfft < K.shape[-1] or nfft < K.shape[-2]:
        raise ValueError(f"nfft = {nfft} is smaller than the patch {K.shape[-2]} x {K.shape[-1]}: the kernel would be cut")
    power = np.zeros((nfft, nfft))
    for i in range(0, K.shape[0], chunk):  # chunked: the full (D, nfft, nfft) array is 0.3 GB for Qwen
        power += (np.abs(np.fft.fft2(K[i:i + chunk], s=(nfft, nfft))) ** 2).sum(axis=0)
    power /= K.shape[0]
    if not power.max() > 0:
        raise ValueError("every kernel is zero: no response to normalise")
    return np.fft.fftshift(power)


def radial_profile(K: np.ndarray, nfft: int = 128, n_rings: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """The response averaged over direction: (ring centres, profile), each of length n_rings.

    Each grey kernel is zero-padded to nfft x nfft, |FFT|^2 is summed over the filters, and the power is averaged
    over n_rings rings of equal width in |f| from 0 to half a cycle per pixel (1/64 cycle per pixel each by default).
    The profile is the square root of the ring averages divided by the largest, so its peak is 1; the centres are
    (k + 0.5) / (2 n_rings), k = 0 to n_rings - 1.

    Raises:
        ValueError: when nfft is smaller than the patch or than 2 n_rings (a ring narrower than one FFT bin could be
            empty), or when every kernel is zero.
    """
    if nfft < 2 * n_rings:
        raise ValueError(f"nfft = {nfft} is smaller than 2 x {n_rings} rings: a ring would be narrower than an FFT bin")
    power = _power_spectrum(K, nfft)
    f = np.fft.fftshift(np.fft.fftfreq(nfft))
    fx, fy = np.meshgrid(f, f)
    radius = np.hypot(fx, fy)
    edges = np.linspace(0.0, 0.5, n_rings + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    rings = np.array([power[(radius >= a) & (radius < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    return centres, np.sqrt(rings / rings.max())


def horizontal_profile(K: np.ndarray, nfft: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """The response to horizontal frequencies alone, which vertical strokes excite: (frequencies, profile).

    The row of zero vertical frequency of the mean power spectrum, at the nfft/2 non-negative frequencies
    k / nfft, k = 0 to nfft/2 - 1; the square root, divided by the largest value.
    """
    power = _power_spectrum(K, nfft)
    f = np.fft.fftshift(np.fft.fftfreq(nfft))
    row = power[nfft // 2, nfft // 2:]
    return f[nfft // 2:], np.sqrt(row / row.max())


def csf_mannos_sakrison(f_cyc_per_deg) -> np.ndarray:
    """Notes A8, equation (7): the Mannos and Sakrison (1974) contrast sensitivity, f in cycles per degree, f >= 0.

    A(f) = 2.6 (0.0192 + 0.114 f) exp(-(0.114 f)^1.1); it peaks at 7.89 cycles per degree with A = 0.981. A model of a
    whole human visual system, not of one layer: the figure places it on the encoder's axis at an assumed number of
    pixels per degree, f_px = f_deg / rho (equation (9)), and never calls the comparison a match.
    """
    f = np.asarray(f_cyc_per_deg, dtype=np.float64)
    return 2.6 * (0.0192 + 0.114 * f) * np.exp(-((0.114 * f) ** 1.1))
