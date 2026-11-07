#!/usr/bin/env python3
"""B14 -- does do_bench's write-flush bias (B12) distort the measured speedup of weight-quantized
decode GEMVs, whose candidates read DIFFERENT numbers of bytes?

y = dequant(W) x for x [K] fp16 (batch 1) and three weight formats of the same matrix:
  fp16  W [N, K] fp16                                  (2 B / weight)   + cuBLAS F.linear
  w8    W int8 [N, K], per-row fp16 scale (W8A16)      (1 B / weight)
  w4    W int4 packed 2 per uint8 [N, K/2], value = nibble - 8, per-row fp16 scale (W4A16)
                                                       (0.5 B / weight)
Each format has 12 Triton configs (BLOCK_N {8,16,32} x BLOCK_K {128,256} x num_warps {4,8}); 37
candidates per shape. Every candidate is checked against an fp32 reference of its own dequantized
weights. Each is timed with the REAL triton.testing.do_bench (median) in two modes, alternating order:
  zero   as shipped (clear_cache = cache.zero_(), a 256 MiB write); read  clear_cache patched to a read
  python3 quant_bias_probe.py --meta OUT.json [--smoke] [--run-seed N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing as tt

SHAPES = [(4096, 4096), (14336, 4096), (4096, 14336)]
CONFIGS = [(bn, bk, nw) for bn in (8, 16, 32) for bk in (128, 256) for nw in (4, 8)]
FORMATS = ("fp16", "w8", "w4")
WARMUP_MS, REP_MS = 10, 50
REL_TOL = 2e-2
SEED = 1234


@triton.jit
def gemv_fp16(w_ptr, s_ptr, x_ptr, y_ptr, N, K, stride_wn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + rn[:, None] * stride_wn + rk[None, :], mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0)
        xv = tl.load(x_ptr + rk, mask=rk < K, other=0.0)
        acc += tl.sum(w.to(tl.float32) * xv.to(tl.float32)[None, :], axis=1)
    tl.store(y_ptr + rn, acc.to(tl.float16), mask=rn < N)


@triton.jit
def gemv_w8(w_ptr, s_ptr, x_ptr, y_ptr, N, K, stride_wn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + rn[:, None] * stride_wn + rk[None, :], mask=(rn[:, None] < N) & (rk[None, :] < K), other=0)
        xv = tl.load(x_ptr + rk, mask=rk < K, other=0.0)
        acc += tl.sum(w.to(tl.float32) * xv.to(tl.float32)[None, :], axis=1)
    s = tl.load(s_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    tl.store(y_ptr + rn, (acc * s).to(tl.float16), mask=rn < N)


@triton.jit
def gemv_w4(w_ptr, s_ptr, x_ptr, y_ptr, N, K, stride_wn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # byte j of a row holds k = 2j (low nibble) and k = 2j + 1 (high nibble); value = nibble - 8
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    KH = K // 2
    for j0 in range(0, KH, BLOCK_K // 2):
        rj = j0 + tl.arange(0, BLOCK_K // 2)
        b = tl.load(w_ptr + rn[:, None] * stride_wn + rj[None, :], mask=(rn[:, None] < N) & (rj[None, :] < KH), other=136)
        lo = (b & 15).to(tl.float32) - 8.0
        hi = (b >> 4).to(tl.float32) - 8.0
        xe = tl.load(x_ptr + 2 * rj, mask=rj < KH, other=0.0).to(tl.float32)
        xo = tl.load(x_ptr + 2 * rj + 1, mask=rj < KH, other=0.0).to(tl.float32)
        acc += tl.sum(lo * xe[None, :] + hi * xo[None, :], axis=1)
    s = tl.load(s_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    tl.store(y_ptr + rn, (acc * s).to(tl.float16), mask=rn < N)


KERNELS = {"fp16": gemv_fp16, "w8": gemv_w8, "w4": gemv_w4}


def quantize(Wf: torch.Tensor) -> dict:
    """Returns per format: (stored weight, per-row scale, fp32 dequantized reference weight)."""
    amax = Wf.abs().amax(dim=1).clamp_min(1e-8)
    s8 = (amax / 127).half()
    q8 = torch.round(Wf / s8.float()[:, None]).clamp(-127, 127).to(torch.int8)
    s4 = (amax / 7).half()
    q4 = (torch.round(Wf / s4.float()[:, None]).clamp(-8, 7) + 8).to(torch.uint8)
    packed = (q4[:, 0::2] | (q4[:, 1::2] << 4)).contiguous()
    w16 = Wf.half()
    ones = torch.ones(Wf.shape[0], device=Wf.device, dtype=torch.float16)
    return {"fp16": (w16, ones, w16.float()),
            "w8": (q8, s8, q8.float() * s8.float()[:, None]),
            "w4": (packed, s4, (q4.float() - 8) * s4.float()[:, None])}


def name(c) -> str:
    return c if c == "cublas" else "{}_bn{}_bk{}_w{}".format(*c)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--smoke", action="store_true", help="first shape, 1 config per format + cuBLAS")
    ap.add_argument("--run-seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(SEED)
    prop = torch.cuda.get_device_properties(0)
    shapes = SHAPES[:1] if a.smoke else SHAPES
    cfgs = CONFIGS[:1] if a.smoke else CONFIGS
    cands = ["cublas"] + [(f, *c) for f in FORMATS for c in cfgs]
    drv = triton.runtime.driver.active
    orig = drv.clear_cache
    read_clear = lambda cache: cache.view(torch.float32).sum()  # noqa: E731
    rows, invalid = [], []
    rng = random.Random(a.run_seed)
    for N, K in shapes:
        Wf = torch.randn(N, K, device="cuda") * 0.02
        fm = quantize(Wf)
        del Wf
        x = (torch.randn(K, device="cuda") * 0.5).half()
        y = torch.empty(N, device="cuda", dtype=torch.float16)
        refs = {f: fm[f][2] @ x.float() for f in FORMATS}
        for f in FORMATS:
            fm[f] = fm[f][:2]
        torch.cuda.empty_cache()

        def make_fn(c):
            if c == "cublas":
                w16, x2 = fm["fp16"][0], x.view(1, -1)
                return lambda: F.linear(x2, w16)
            f, bn, bk, nw = c
            w, s = fm[f]
            grid = (triton.cdiv(N, bn),)
            return lambda: KERNELS[f][grid](w, s, x, y, N, K, w.stride(0), BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)

        valid = []
        for c in cands:
            ref = refs["fp16" if c == "cublas" else c[0]]
            try:
                out = make_fn(c)()
                out = (out if c == "cublas" else y).float().view(-1)
                torch.cuda.synchronize()
                err = ((out - ref).abs().max() / ref.abs().max()).item()
            except Exception as ex:  # noqa: BLE001
                invalid.append({"shape": [N, K], "cand": name(c), "error": repr(ex)[:200]})
                continue
            if err <= REL_TOL:
                valid.append(c)
            else:
                invalid.append({"shape": [N, K], "cand": name(c), "rel_err": err})
        order = valid[:]
        rng.shuffle(order)
        for i, c in enumerate(order):
            fn = make_fn(c)
            res = {}
            for mode in (("zero", "read") if i % 2 == 0 else ("read", "zero")):
                drv.clear_cache = orig if mode == "zero" else read_clear
                try:
                    res[mode] = tt.do_bench(fn, warmup=WARMUP_MS, rep=REP_MS, return_mode="median")
                finally:
                    drv.clear_cache = orig
            rows.append({"shape": [N, K], "cand": name(c), "format": "fp16" if c == "cublas" else c[0],
                         "zero_ms": res["zero"], "read_ms": res["read"]})
        del fm, refs
        torch.cuda.empty_cache()
    meta = {"benchmark": "quant_bias_probe", "device": prop.name, "l2_bytes": prop.L2_cache_size,
            "torch": torch.__version__, "triton": triton.__version__, "smoke": a.smoke, "run_seed": a.run_seed,
            "warmup_ms": WARMUP_MS, "rep_ms": REP_MS, "rows": rows, "invalid": invalid, "n_candidates": len(cands)}
    json.dump(meta, open(a.meta, "w"), indent=1)
    print(f"{len(rows)} rows, {len(invalid)} invalid candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
