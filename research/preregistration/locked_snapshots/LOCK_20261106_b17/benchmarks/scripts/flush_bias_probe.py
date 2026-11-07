#!/usr/bin/env python3
"""B17 -- how much does triton.testing.do_bench's write flush bias memory-bound LLM kernels, measured
against the fix proposed in triton-lang/triton PR #12556, on GPUs people actually benchmark on?

Flush modes (do_bench's clear_cache patched per mode; everything else is the real do_bench, median):
  zero  as shipped up to Triton 3.9: cache.zero_() over the 256 MiB buffer (a write)
  pr    PR #12556 head ba27327b: a Triton kernel doing a masked volatile load of the buffer, no store
  sum   cache.view(float32).sum() -- the read flush used in our B12-B16, for linkage
  none  no flush (only for the E-0 control case: cuBLAS fp16 GEMV 4096x4096)
Kernels (batch-1 decode unless noted; every candidate is checked against an fp32 reference first,
max err / max |ref| <= 2e-2; failing or unavailable candidates are excluded and recorded):
  gemv     y = x W^T, W [N, K] for (N, K) in (4096, 4096), (14336, 4096), (4096, 14336):
           cublas_fp16 (F.linear), tinygemm_k{2,4,8} (torch._weight_int4pack_mm, bf16 act, group 128),
           bnb_nf4 (bitsandbytes.matmul_4bit, NF4 blocksize 64)               -- as in B16
  attn     F.scaled_dot_product_attention, q [1, 32, 1, 128], k/v [1, 8, L, 128] bf16 (GQA, Llama-3-8B
           heads), L in 4096 / 16384 / 65536 (KV cache 16 / 64 / 256 MiB): sdpa_bf16
  rmsnorm  prefill x [T, 4096] bf16, T in 2048 / 8192 (16 / 64 MiB): liger_rmsnorm (Liger-Kernel
           LigerRMSNormFunction), torch_rmsnorm (F.rms_norm, eager)
  swiglu   prefill a, b [T, 14336] bf16, T in 512 / 2048 (14 / 56 MiB each): liger_swiglu
           (LigerSiLUMulFunction), torch_swiglu (F.silu(a) * b, eager)
Mode order is shuffled per case and run. The PR flush's PTX is inspected (volatile global load,
no global store) and the CUDA kernels each candidate launches are recorded with torch.profiler.
  python3 flush_bias_probe.py --meta OUT.json [--smoke] [--run-seed N]
"""


# no "from __future__ import annotations": Triton must see tl.constexpr as an object, not a string
import argparse
import itertools
import json
import random
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing as tt

GEMV_SHAPES = [(4096, 4096), (14336, 4096), (4096, 14336)]
ATTN_L = [4096, 16384, 65536]
NORM_T = [2048, 8192]
SWIGLU_T = [512, 2048]
HQ, HKV, HD, HID, FFN = 32, 8, 128, 4096, 14336
GROUP, INNER_K, BNB_BLOCK = 128, (2, 4, 8), 64
WARMUP_MS, REP_MS = 10, 50
TOL = 2e-2
SEED = 1234
PR_HEAD = "ba27327b9a4667b58e89ae5e1f0d7efe453a2021"


# ---- PR #12556 flush, copied from the PR head (python/triton/testing.py), MIT licensed ----------
@triton.jit
def _read_flush_kernel(X, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Volatile keeps the read even when its result is unused, without a store.
    tl.load(X + off, mask=off < n, volatile=True)


def pr_clear(cache):
    x = cache.view(torch.float32)
    block = 1024
    grid = ((x.numel() + block - 1) // block, )
    return _read_flush_kernel[grid](x, x.numel(), block)


def ptx_check() -> dict:
    """Compile the PR flush once and count loads/stores in its PTX."""
    try:
        buf = torch.zeros(1 << 20, dtype=torch.int32, device="cuda")
        before = buf.clone()
        k = pr_clear(buf)
        torch.cuda.synchronize()
        ptx = k.asm["ptx"]
        return {"ld_volatile_global": ptx.count("ld.volatile.global"), "ld_global": ptx.count("ld.global"),
                "st_global": ptx.count("st.global"), "buffer_unchanged": bool(torch.equal(buf, before))}
    except Exception as ex:  # noqa: BLE001
        return {"error": repr(ex)[:300]}


# ---- int4 helpers (identical to B16 prod_quant_probe.py) ----------------------------------------
def group_quantize_int4(w: torch.Tensor, group: int):
    N, K = w.shape
    g = w.reshape(-1, group)
    mn, mx = g.amin(dim=1, keepdim=True), g.amax(dim=1, keepdim=True)
    scales = (mx - mn).clamp(min=1e-6) / 15
    zeros = mn + scales * 8
    q = g.sub(mn).div(scales).round().clamp_(0, 15).to(torch.int32).reshape(N, K)
    packed = (q[:, ::2] << 4 | q[:, 1::2]).to(torch.uint8).contiguous()
    sz = torch.cat([scales.reshape(N, -1, 1), zeros.reshape(N, -1, 1)], 2).transpose(0, 1).contiguous()
    return packed, sz


def dequant_int4(packed: torch.Tensor, sz: torch.Tensor, group: int) -> torch.Tensor:
    N = packed.shape[0]
    q = torch.stack([packed >> 4, packed & 15], 2).reshape(N, -1).float()
    sc = sz[..., 0].float().t().repeat_interleave(group, 1)
    ze = sz[..., 1].float().t().repeat_interleave(group, 1)
    return (q - 8) * sc + ze


# ---- case builders: each returns ({cand: (fn, reference)}, notes) --------------------------------
def build_gemv(N: int, K: int):
    W = torch.randn(N, K, device="cuda") * 0.02
    x = torch.randn(1, K, device="cuda") * 0.5
    c, notes = {}, {}
    w16, x16 = W.half(), x.half()
    c["cublas_fp16"] = (lambda: F.linear(x16, w16), x16.float() @ w16.float().t())
    try:
        wq, sz = group_quantize_int4(W.bfloat16(), GROUP)
        xbf = x.bfloat16()
        ref = xbf.float() @ dequant_int4(wq, sz, GROUP).t()
        for kt in INNER_K:
            wp = torch._convert_weight_to_int4pack(wq, kt)
            c[f"tinygemm_k{kt}"] = (lambda wp=wp: torch._weight_int4pack_mm(xbf, wp, GROUP, sz), ref)
    except Exception as ex:  # noqa: BLE001
        notes["tinygemm_error"] = repr(ex)[:300]
    try:
        import bitsandbytes as bnb
        import bitsandbytes.functional as bf
        wq4, qs = bf.quantize_4bit(w16, quant_type="nf4", blocksize=BNB_BLOCK, compress_statistics=False)
        wdq = bf.dequantize_4bit(wq4, qs).float()
        c["bnb_nf4"] = (lambda: bnb.matmul_4bit(x16, wq4.t(), quant_state=qs), x16.float() @ wdq.t())
    except Exception as ex:  # noqa: BLE001
        notes["bnb_error"] = repr(ex)[:300]
    return c, notes, N * K * 2


def build_attn(L: int):
    q = torch.randn(1, HQ, 1, HD, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, HKV, L, HD, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, HKV, L, HD, device="cuda", dtype=torch.bfloat16)
    rep = HQ // HKV
    ref = torch.empty(1, HQ, 1, HD, device="cuda")
    for h in range(HKV):                       # fp32 reference without materializing expanded K/V
        qh = q[0, h * rep:(h + 1) * rep, 0].float()               # [rep, D]
        s = (qh @ k[0, h].float().t()) / HD ** 0.5                # [rep, L]
        ref[0, h * rep:(h + 1) * rep, 0] = torch.softmax(s, -1) @ v[0, h].float()
    fn = lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True)  # noqa: E731
    return {"sdpa_bf16": (fn, ref)}, {}, 2 * HKV * L * HD * 2


def build_rmsnorm(T: int):
    x = torch.randn(T, HID, device="cuda", dtype=torch.bfloat16)
    w = (1 + 0.1 * torch.randn(HID, device="cuda")).bfloat16()
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    c, notes = {"torch_rmsnorm": (lambda: F.rms_norm(x, (HID,), w, 1e-6), ref)}, {}
    try:
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        c["liger_rmsnorm"] = (lambda: LigerRMSNormFunction.apply(x, w, 1e-6), ref)
    except Exception as ex:  # noqa: BLE001
        notes["liger_error"] = repr(ex)[:300]
    return c, notes, T * HID * 2


def build_swiglu(T: int):
    a = torch.randn(T, FFN, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(T, FFN, device="cuda", dtype=torch.bfloat16)
    ref = F.silu(a.float()) * b.float()
    c, notes = {"torch_swiglu": (lambda: F.silu(a) * b, ref)}, {}
    try:
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction
        c["liger_swiglu"] = (lambda: LigerSiLUMulFunction.apply(a, b), ref)
    except Exception as ex:  # noqa: BLE001
        notes["liger_error"] = repr(ex)[:300]
    return c, notes, T * FFN * 2


def cases(smoke: bool):
    pick = (lambda xs: xs[:1]) if smoke else (lambda xs: xs)
    for N, K in pick(GEMV_SHAPES):
        yield "gemv", f"{N}x{K}", lambda N=N, K=K: build_gemv(N, K)
    for L in pick(ATTN_L):
        yield "attn", f"L{L}", lambda L=L: build_attn(L)
    for T in pick(NORM_T):
        yield "rmsnorm", f"T{T}", lambda T=T: build_rmsnorm(T)
    for T in pick(SWIGLU_T):
        yield "swiglu", f"T{T}", lambda T=T: build_swiglu(T)


def kernel_names(fn) -> list[str]:
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            fn()
            torch.cuda.synchronize()
        return sorted({e.name[:120] for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA})
    except Exception as ex:  # noqa: BLE001
        return [f"profiler error: {ex!r}"[:120]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--smoke", action="store_true", help="first size of each family only")
    ap.add_argument("--run-seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(SEED)
    prop = torch.cuda.get_device_properties(0)
    drv = triton.runtime.driver.active
    orig = drv.clear_cache
    flush = {"zero": orig, "pr": pr_clear, "sum": lambda cache: cache.view(torch.float32).sum(),
             "none": lambda cache: None}
    perms = list(itertools.permutations(("zero", "pr", "sum")))
    rng = random.Random(a.run_seed)
    rows, invalid, notes, kernels = [], [], {}, {}
    ptx = ptx_check()
    torch.set_grad_enabled(False)
    for fam, label, builder in cases(a.smoke):
        try:
            cands, n, size = builder()
        except Exception as ex:  # noqa: BLE001
            invalid.append({"family": fam, "case": label, "cand": "*", "error": repr(ex)[:300]})
            continue
        for k, v in n.items():
            notes[f"{fam} {label} {k}"] = v
        valid = []
        for name, (fn, ref) in cands.items():
            try:
                out = fn().float().reshape(-1)[: ref.numel()]
                torch.cuda.synchronize()
                err = ((out - ref.reshape(-1)).abs().max() / ref.abs().max()).item()
            except Exception as ex:  # noqa: BLE001
                invalid.append({"family": fam, "case": label, "cand": name, "error": repr(ex)[:300]})
                continue
            if err <= TOL:
                valid.append(name)
                if a.run_seed <= 1:
                    kernels[f"{fam} {label} {name}"] = kernel_names(fn)
            else:
                invalid.append({"family": fam, "case": label, "cand": name, "rel_err": err})
        order = valid[:]
        rng.shuffle(order)
        for name in order:
            fn = cands[name][0]
            modes = list(rng.choice(perms))
            if fam == "gemv" and label == "4096x4096" and name == "cublas_fp16":
                modes.insert(rng.randrange(4), "none")
            t = {}
            for mode in modes:
                drv.clear_cache = flush[mode]
                try:
                    t[mode] = tt.do_bench(fn, warmup=WARMUP_MS, rep=REP_MS, return_mode="median")
                finally:
                    drv.clear_cache = orig
            rows.append({"family": fam, "case": label, "bytes": size, "cand": name, "ms": t, "order": modes})
        del cands
        torch.cuda.empty_cache()
    vers = {"torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__}
    for mod in ("bitsandbytes", "liger-kernel"):
        try:
            import importlib.metadata as md
            vers[mod] = md.version(mod)
        except Exception:  # noqa: BLE001
            vers[mod] = None
    meta = {"benchmark": "flush_bias_probe", "device": prop.name, "l2_bytes": prop.L2_cache_size,
            "cc": f"{prop.major}.{prop.minor}", "sm_count": prop.multi_processor_count, "versions": vers,
            "pr_head": PR_HEAD, "ptx_check": ptx, "smoke": a.smoke, "run_seed": a.run_seed, "rows": rows,
            "invalid": invalid, "notes": notes, "kernels": kernels}
    json.dump(meta, open(a.meta, "w"), indent=1)
    print(f"{len(rows)} rows, {len(invalid)} invalid, ptx {ptx}, notes {notes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
