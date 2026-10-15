# PURPOSE: A correct H2 kernel in Triton, to prove that the judge passes a correct Triton implementation (build guide 4.4).
# WHAT GOES HERE: render() with the kernel interface of the specification; R04 in one @triton.jit function (tl.atomic_add into an int32 page), R05 and R06 in a second, in float32.
# FILLED IN AT STEP: W4.4.
# DO NOT: Show it to the H2 model or put it in a prompt. Do not import anything outside torch and triton.
import torch
import triton
import triton.language as tl

BLOCK = 1024
MODES = {"binary": 0, "coverage_srgb": 1, "coverage_linear": 2}


@triton.jit
def _accumulate(bitmaps, offsets, heights, widths, rows, cols, acc, height, width, BLOCK: tl.constexpr):
    # R04, the sum: one program per glyph adds each of its bytes into the int32 page acc; pixels off the page are
    # dropped. Integer addition makes the order of the glyphs irrelevant.
    g = tl.program_id(0)
    start = tl.load(offsets + g)
    h = tl.load(heights + g)
    w = tl.load(widths + g)
    r = tl.load(rows + g)
    c = tl.load(cols + g)
    n = h * w
    w1 = tl.maximum(w, 1)
    for first in range(0, n, BLOCK):
        i = first + tl.arange(0, BLOCK)
        y = r + i // w1
        x = c + i % w1
        keep = (i < n) & (y >= 0) & (y < height) & (x >= 0) & (x < width)
        v = tl.load(bitmaps + start + i, mask=keep, other=0)
        tl.atomic_add(acc + y * width + x, v.to(tl.int32), mask=keep)


@triton.jit
def _decode(v):
    # sRGB decode D(v); x ** 2.4 as exp2(2.4 log2 x), x > 0
    return tl.where(v <= 0.04045, v / 12.92, tl.exp2(2.4 * tl.log2((v + 0.055) / 1.055)))


@triton.jit
def _encode(lin):
    # sRGB encode E(l), with the threshold where the decode's straight piece ends
    return tl.where(lin <= 0.04045 / 12.92, 12.92 * lin, 1.055 * tl.exp2(tl.log2(lin) / 2.4) - 0.055)


@triton.jit
def _channel(out, p, keep, s, c, t8, b8, CH: tl.constexpr, MODE: tl.constexpr):
    # R05 and R06 for one channel of the pixels p: composite once, round once with floor(255 o + 0.5)
    zero = s * 0
    t = zero + t8
    b = zero + b8
    if MODE == 0:
        v = tl.where(s >= 128, t, b)
    elif MODE == 1:
        v = tl.floor((s * t + (255 - s) * b).to(tl.float32) / 255.0 + 0.5)
    else:
        lt = _decode(t.to(tl.float32) / 255.0)
        lb = _decode(b.to(tl.float32) / 255.0)
        o = _encode(c * lt + (1.0 - c) * lb)
        o = tl.minimum(tl.maximum(o, 0.0), 1.0)
        v = tl.floor(255.0 * o + 0.5)
    tl.store(out + p * 3 + CH, v.to(tl.uint8), mask=keep)


@triton.jit
def _composite(acc, out, n_pixels, t0, t1, t2, b0, b1, b2, MODE: tl.constexpr, BLOCK: tl.constexpr):
    # R04, the clip, then R05 and R06 for each of the three channels
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = p < n_pixels
    s = tl.minimum(tl.load(acc + p, mask=keep, other=0), 255)
    c = s.to(tl.float32) / 255.0
    _channel(out, p, keep, s, c, t0, b0, 0, MODE)
    _channel(out, p, keep, s, c, t1, b1, 1, MODE)
    _channel(out, p, keep, s, c, t2, b2, 2, MODE)


def render(bitmaps, offsets, heights, widths, rows, cols, height, width, text_rgb, background_rgb, aa_mode):
    mode = MODES[aa_mode]
    acc = torch.zeros(height * width, dtype=torch.int32, device=bitmaps.device)
    if offsets.numel() > 0 and bitmaps.numel() > 0:
        _accumulate[(offsets.numel(),)](bitmaps, offsets, heights, widths, rows, cols, acc, height, width,
                                        BLOCK=BLOCK)
    out = torch.empty((height, width, 3), dtype=torch.uint8, device=bitmaps.device)
    n = height * width
    _composite[(triton.cdiv(n, BLOCK),)](acc, out, n, *text_rgb, *background_rgb, MODE=mode, BLOCK=BLOCK)
    return out
