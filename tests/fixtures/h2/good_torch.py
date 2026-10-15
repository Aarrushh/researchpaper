# PURPOSE: A correct H2 kernel in plain PyTorch, to prove that the judge passes a correct implementation.
# WHAT GOES HERE: render() with the kernel interface of the specification; no Triton.
# FILLED IN AT STEP: W4.4.
# DO NOT: Show it to the H2 model or put it in a prompt.
import torch


def _L(v):
    return torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _V(l):
    return torch.where(l <= 0.04045 / 12.92, 12.92 * l, 1.055 * l.clamp(min=0) ** (1 / 2.4) - 0.055)


def render(bitmaps, offsets, heights, widths, rows, cols, height, width, text_rgb, background_rgb, aa_mode):
    k = torch.zeros((height, width), dtype=torch.int32, device=bitmaps.device)
    for o, h, w, r, c in zip(offsets.tolist(), heights.tolist(), widths.tolist(), rows.tolist(), cols.tolist()):
        r0, c0, r1, c1 = max(r, 0), max(c, 0), min(r + h, height), min(c + w, width)
        if r1 > r0 and c1 > c0:
            g = bitmaps[o:o + h * w].view(h, w).to(torch.int32)
            k[r0:r1, c0:c1] += g[r0 - r:r1 - r, c0 - c:c1 - c]
    cov = (k.clamp(max=255).double() / 255.0)[..., None]
    t = torch.tensor(text_rgb, dtype=torch.float64, device=bitmaps.device) / 255.0
    b = torch.tensor(background_rgb, dtype=torch.float64, device=bitmaps.device) / 255.0
    if aa_mode == "binary":
        a = (cov >= 0.5).double()
        out = a * t + (1 - a) * b
    elif aa_mode == "coverage_srgb":
        out = cov * t + (1 - cov) * b
    else:
        out = _V(cov * _L(t) + (1 - cov) * _L(b))
    return torch.floor(255.0 * out.clamp(0, 1) + 0.5).to(torch.uint8)
