#!/usr/bin/env python3
"""B16 -- do_bench's write-flush bias on PRODUCTION 4-bit decode kernels (not our own Triton kernels).

Candidates for y = x W^T, x [1, K] (batch-1 LLM decode), on the B13/B14 LLM shapes:
  cublas_fp16      F.linear, fp16                                          (baseline, 2 B / weight)
  tinygemm_k{2,4,8} torch._weight_int4pack_mm (PyTorch's tinygemm, used by gpt-fast / torchao int4
                   weight-only), bf16 activations, group 128, innerKTiles 2 / 4 / 8; weights built with
                   torch.testing._internal.common_quantization._group_quantize_tensor as in PyTorch's test
  bnb_nf4          bitsandbytes.matmul_4bit with an NF4 weight (blocksize 64) -- the kernel behind
                   HuggingFace load_in_4bit; batch 1 dispatches to its 4-bit GEMV
Every candidate is checked first against an fp32 product with its own (dequantized) weights:
max err / max |ref| <= 2e-2 (tinygemm: (q - 8) * scale + zero, the kernel's reconstruction). Failing or unavailable candidates are
excluded and recorded. Each is timed with the REAL triton.testing.do_bench (median) in two modes,
alternating order: zero (as shipped: clear_cache = cache.zero_()) and read (patched to a read).
  python3 prod_quant_probe.py --meta OUT.json [--smoke] [--run-seed N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import torch
import torch.nn.functional as F
import triton
import triton.testing as tt

SHAPES = [(4096, 4096), (14336, 4096), (4096, 14336)]
GROUP = 128
INNER_K = (2, 4, 8)
BNB_BLOCK = 64
WARMUP_MS, REP_MS = 10, 50
TOL = 2e-2
SEED = 1234


def group_quantize_int4(w: torch.Tensor, group: int):
    """w [N, K] bf16 -> (uint8 [N, K/2], scales_and_zeros bf16 [K/group, N, 2]).
    Same steps as torch.testing._internal.common_quantization._group_quantize_tensor (PyTorch 2.7.1),
    re-implemented to avoid importing PyTorch's test utilities: unsigned q in [0, 15],
    scale = (max - min) / 15, zero = min + 8 scale, byte j = (q[2j] << 4) | q[2j + 1]."""
    N, K = w.shape
    g = w.reshape(-1, group)                       # arithmetic in w's dtype (bf16), as the helper does
    mn, mx = g.amin(dim=1, keepdim=True), g.amax(dim=1, keepdim=True)
    scales = (mx - mn).clamp(min=1e-6) / 15
    zeros = mn + scales * 8
    q = g.sub(mn).div(scales).round().clamp_(0, 15).to(torch.int32).reshape(N, K)
    packed = (q[:, ::2] << 4 | q[:, 1::2]).to(torch.uint8).contiguous()
    sz = torch.cat([scales.reshape(N, -1, 1), zeros.reshape(N, -1, 1)], 2).transpose(0, 1).contiguous()
    return packed, sz


def dequant_int4(packed: torch.Tensor, sz: torch.Tensor, group: int) -> torch.Tensor:
    """fp32 [N, K] weights as the tinygemm kernel reconstructs them: (q - 8) * scale + zero."""
    N = packed.shape[0]
    q = torch.stack([packed >> 4, packed & 15], 2).reshape(N, -1).float()
    sc = sz[..., 0].float().t().repeat_interleave(group, 1)
    ze = sz[..., 1].float().t().repeat_interleave(group, 1)
    return (q - 8) * sc + ze


def build(N: int, K: int):
    """Returns {name: (fn, tolerance, reference)} for one shape."""
    W = torch.randn(N, K, device="cuda") * 0.02
    x = torch.randn(1, K, device="cuda") * 0.5
    c = {}
    w16, x16 = W.half(), x.half()
    c["cublas_fp16"] = (lambda: F.linear(x16, w16), TOL, (x16.float() @ w16.float().t()))
    try:
        wq, sz = group_quantize_int4(W.bfloat16(), GROUP)
        xbf = x.bfloat16()
        ref = xbf.float() @ dequant_int4(wq, sz, GROUP).t()
        for kt in INNER_K:
            wp = torch._convert_weight_to_int4pack(wq, kt)
            c[f"tinygemm_k{kt}"] = (lambda wp=wp: torch._weight_int4pack_mm(xbf, wp, GROUP, sz), TOL, ref)
    except Exception as ex:  # noqa: BLE001
        c["tinygemm_error"] = repr(ex)[:300]
    try:
        import bitsandbytes as bnb
        import bitsandbytes.functional as bf
        wq4, qs = bf.quantize_4bit(w16, quant_type="nf4", blocksize=BNB_BLOCK, compress_statistics=False)
        wdq = bf.dequantize_4bit(wq4, qs).float()
        c["bnb_nf4"] = (lambda: bnb.matmul_4bit(x16, wq4.t(), quant_state=qs), TOL, (x16.float() @ wdq.t()))
        c["_bnb_version"] = bnb.__version__
    except Exception as ex:  # noqa: BLE001
        c["bnb_error"] = repr(ex)[:300]
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--smoke", action="store_true", help="first shape only")
    ap.add_argument("--run-seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(SEED)
    prop = torch.cuda.get_device_properties(0)
    drv = triton.runtime.driver.active
    orig = drv.clear_cache
    read_clear = lambda cache: cache.view(torch.float32).sum()  # noqa: E731
    rows, invalid, notes = [], [], {}
    rng = random.Random(a.run_seed)
    for N, K in (SHAPES[:1] if a.smoke else SHAPES):
        cands = build(N, K)
        for k in [k for k in cands if k.endswith("_error") or k.startswith("_")]:
            notes[f"{N}x{K} {k}"] = cands.pop(k)
        valid = []
        for name, (fn, tol, ref) in cands.items():
            try:
                out = fn().float().view(-1)[: ref.numel()]
                torch.cuda.synchronize()
                err = ((out - ref.view(-1)).abs().max() / ref.abs().max()).item()
            except Exception as ex:  # noqa: BLE001
                invalid.append({"shape": [N, K], "cand": name, "error": repr(ex)[:300]})
                continue
            (valid.append(name) if err <= tol else
             invalid.append({"shape": [N, K], "cand": name, "rel_err": err}))
        order = valid[:]
        rng.shuffle(order)
        for i, name in enumerate(order):
            fn = cands[name][0]
            res = {}
            for mode in (("zero", "read") if i % 2 == 0 else ("read", "zero")):
                drv.clear_cache = orig if mode == "zero" else read_clear
                try:
                    res[mode] = tt.do_bench(fn, warmup=WARMUP_MS, rep=REP_MS, return_mode="median")
                finally:
                    drv.clear_cache = orig
            rows.append({"shape": [N, K], "cand": name, "zero_ms": res["zero"], "read_ms": res["read"]})
        del cands
        torch.cuda.empty_cache()
    meta = {"benchmark": "prod_quant_probe", "device": prop.name, "l2_bytes": prop.L2_cache_size,
            "cc": f"{prop.major}.{prop.minor}", "torch": torch.__version__, "triton": triton.__version__,
            "smoke": a.smoke, "run_seed": a.run_seed, "rows": rows, "invalid": invalid, "notes": notes}
    json.dump(meta, open(a.meta, "w"), indent=1)
    print(f"{len(rows)} rows, {len(invalid)} invalid, notes {notes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
