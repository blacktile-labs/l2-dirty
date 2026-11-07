#!/usr/bin/env python3
"""B13 -- does do_bench's write-flush bias (B12) change which kernel an autotuner picks for an LLM
decode GEMV?

Candidates for y = W x (W [N, K] fp16, x [K] fp16, batch 1; LLM linear-layer shapes):
  36 configs of one Triton GEMV kernel (BLOCK_N x BLOCK_K x num_warps) + cuBLAS (F.linear),
  the same kind of choice Triton's autotuner and Inductor's max-autotune make.
Each candidate is timed with the REAL triton.testing.do_bench (median) in two modes, alternating
order per candidate:
  zero   do_bench as shipped: clear_cache = cache.zero_()  (a 256 MiB write)
  read   the same call with clear_cache patched to cache.view(float32).sum()  (read only)
Every candidate is checked against an fp32 reference first; failing candidates are excluded.
  python3 autotune_bias_probe.py --meta OUT.json [--smoke] [--run-seed N]
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

SHAPES = [(4096, 4096), (14336, 4096), (4096, 14336)]       # (N, K): attention proj, MLP up, MLP down
CONFIGS = [(bn, bk, nw) for bn in (8, 16, 32, 64) for bk in (128, 256, 512) for nw in (2, 4, 8)]
WARMUP_MS, REP_MS = 10, 50
REL_TOL = 2e-2
SEED = 1234


@triton.jit
def gemv_kernel(w_ptr, x_ptr, y_ptr, N, K, stride_wn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + rn[:, None] * stride_wn + rk[None, :],
                    mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0)
        xv = tl.load(x_ptr + rk, mask=rk < K, other=0.0)
        acc += tl.sum(w.to(tl.float32) * xv.to(tl.float32)[None, :], axis=1)
    tl.store(y_ptr + rn, acc.to(tl.float16), mask=rn < N)


def make_fn(cand, W, x, y):
    if cand == "cublas":
        x2 = x.view(1, -1)
        return lambda: F.linear(x2, W)
    bn, bk, nw = cand
    N, K = W.shape
    grid = (triton.cdiv(N, bn),)
    return lambda: gemv_kernel[grid](W, x, y, N, K, W.stride(0), BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)


def name(cand) -> str:
    return cand if cand == "cublas" else "bn{}_bk{}_w{}".format(*cand)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--smoke", action="store_true", help="first shape, 4 configs + cuBLAS")
    ap.add_argument("--run-seed", type=int, default=0, help="shuffles the candidate order of this run")
    a = ap.parse_args()
    torch.manual_seed(SEED)
    prop = torch.cuda.get_device_properties(0)
    shapes = SHAPES[:1] if a.smoke else SHAPES
    cands = ["cublas"] + (CONFIGS[::9] if a.smoke else CONFIGS)
    drv = triton.runtime.driver.active
    orig = drv.clear_cache
    read_clear = lambda cache: cache.view(torch.float32).sum()  # noqa: E731
    rows, invalid = [], []
    rng = random.Random(a.run_seed)
    for N, K in shapes:
        W = (torch.randn(N, K, device="cuda") * 0.02).half()
        x = (torch.randn(K, device="cuda") * 0.5).half()
        y = torch.empty(N, device="cuda", dtype=torch.float16)
        ref = (W.float() @ x.float())
        valid = []
        for c in cands:
            try:
                out = make_fn(c, W, x, y)()
                out = (out if c == "cublas" else y).float().view(-1)
                torch.cuda.synchronize()
                err = ((out - ref).abs().max() / ref.abs().max()).item()
            except Exception as ex:  # noqa: BLE001 -- e.g. a config that does not compile
                invalid.append({"shape": [N, K], "cand": name(c), "error": repr(ex)[:200]})
                continue
            if err <= REL_TOL:
                valid.append(c)
            else:
                invalid.append({"shape": [N, K], "cand": name(c), "rel_err": err})
        order = valid[:]
        rng.shuffle(order)
        for i, c in enumerate(order):
            fn = make_fn(c, W, x, y)
            res = {}
            for mode in (("zero", "read") if i % 2 == 0 else ("read", "zero")):
                drv.clear_cache = orig if mode == "zero" else read_clear
                try:
                    res[mode] = tt.do_bench(fn, warmup=WARMUP_MS, rep=REP_MS, return_mode="median")
                finally:
                    drv.clear_cache = orig
            rows.append({"shape": [N, K], "cand": name(c), "zero_ms": res["zero"], "read_ms": res["read"]})
        del W
        torch.cuda.empty_cache()
    meta = {"benchmark": "autotune_bias_probe", "device": prop.name, "l2_bytes": prop.L2_cache_size,
            "torch": torch.__version__, "triton": triton.__version__, "smoke": a.smoke, "run_seed": a.run_seed,
            "warmup_ms": WARMUP_MS, "rep_ms": REP_MS, "rows": rows, "invalid": invalid,
            "n_candidates": len(cands)}
    json.dump(meta, open(a.meta, "w"), indent=1)
    print(f"{len(rows)} rows, {len(invalid)} invalid candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
