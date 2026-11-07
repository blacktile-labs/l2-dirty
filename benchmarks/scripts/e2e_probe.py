#!/usr/bin/env python3
"""B21 -- which per-kernel timing predicts an in-model decode step: a write flush, a read flush, or steady state?

Rules U-0..U-3 are pre-registered in research/HYPOTHESES.md, section "B21 pre-registration", before any B21
measurement.

Decode proxy: L = 4 layers of Llama-2-7B shapes (H = 4096, FFN = 11008), batch 1, the four weight GEMVs of
each layer in order: fused QKV [3H, H], O [H, H], fused gate+up [2 FFN, H], down [H, FFN]. Every layer has its
own weights (as in a model), so the weights a GEMV reads were not touched since the previous decode step.
Variants: fp16 (cuBLAS F.linear), int4 (PyTorch tinygemm, group 128, innerKTiles 8, bf16 activations) and nf4
(bitsandbytes matmul_4bit, blocksize 64; only where bitsandbytes is installed).
  "truth"   T_step: the whole proxy step captured in one CUDA graph, replayed back to back (as decode runs with
            CUDA graphs), CUDA events around each replay, median of 300 replays.
  harnesses per unique GEMV shape, summed over the step (x L layers):
            write  triton.testing.do_bench as shipped (zero_() of 256 MiB, median)
            cold   cleanbench cold (read flush >= max(256 MiB, 4 x L2) + GPU busy-wait, median)
            steady cleanbench steady (rotation over the L layers' weights, median)
            l2w    do_bench with clear_cache = zero_() of an L2-sized buffer, as Inductor (descriptive)
"""

import argparse
import json
import statistics
import sys

import torch
import torch.nn.functional as F
import triton
import triton.testing as tt

import cleanbench as cb

H, FFN, LAYERS = 4096, 11008, 4
GROUP, INNER_K, BNB_BLOCK = 128, 8, 64
SHAPES = {"qkv": (3 * H, H), "o": (H, H), "gate_up": (2 * FFN, H), "down": (H, FFN)}
STEP_REPS, WARM_REPS, BENCH_REPS = 300, 20, 200
SEED = 2100
TOL = 2e-2


def group_quantize_int4(w: torch.Tensor, group: int):
    """Same as flush_bias_probe.py (B16/B17)."""
    N, K = w.shape
    g = w.reshape(-1, group)
    mn, mx = g.amin(dim=1, keepdim=True), g.amax(dim=1, keepdim=True)
    scales = (mx - mn).clamp(min=1e-6) / 15
    zeros = mn + scales * 8
    q = g.sub(mn).div(scales).round().clamp_(0, 15).to(torch.int32).reshape(N, K)
    packed = (q[:, ::2] << 4 | q[:, 1::2]).to(torch.uint8).contiguous()
    sz = torch.cat([scales.reshape(N, -1, 1), zeros.reshape(N, -1, 1)], 2).transpose(0, 1).contiguous()
    return packed, sz


def dequant_int4(packed, sz, group):
    N = packed.shape[0]
    q = torch.stack([packed >> 4, packed & 15], 2).reshape(N, -1).float()
    sc = sz[..., 0].float().t().repeat_interleave(group, 1)
    ze = sz[..., 1].float().t().repeat_interleave(group, 1)
    return (q - 8) * sc + ze


def build(variants):
    """variant -> {shape: [callable per layer]}, plus correctness failures and notes."""
    xs = {K: torch.randn(1, K, device="cuda") * 0.5 for K in (H, FFN)}
    ops = {v: {s: [] for s in SHAPES} for v in variants}
    fails, notes = 0, {}
    bnb = None
    if "nf4" in variants:
        try:
            import bitsandbytes
            import bitsandbytes.functional as bf
            bnb = (bitsandbytes, bf)
            notes["bitsandbytes"] = bitsandbytes.__version__
        except Exception as ex:  # noqa: BLE001
            notes["nf4_error"] = repr(ex)[:200]
            del ops["nf4"]
    for layer in range(LAYERS):
        for s, (N, K) in SHAPES.items():
            W = torch.randn(N, K, device="cuda") * 0.02
            x = xs[K]
            w16, x16 = W.half(), x.half()
            ops["fp16"][s].append(lambda w16=w16, x16=x16: F.linear(x16, w16))
            if layer == 0:
                ref = x16.float() @ w16.float().t()
                fails += ((F.linear(x16, w16).float() - ref).abs().max() / ref.abs().max()).item() > TOL
            if "int4" in ops:
                wq, sz = group_quantize_int4(W.bfloat16(), GROUP)
                wp = torch._convert_weight_to_int4pack(wq, INNER_K)
                xbf = x.bfloat16()
                ops["int4"][s].append(lambda wp=wp, sz=sz, xbf=xbf: torch._weight_int4pack_mm(xbf, wp, GROUP, sz))
                if layer == 0:
                    ref = xbf.float() @ dequant_int4(wq, sz, GROUP).t()
                    y = torch._weight_int4pack_mm(xbf, wp, GROUP, sz).float()
                    fails += ((y - ref).abs().max() / ref.abs().max()).item() > TOL
                del wq
            if "nf4" in ops:
                bitsandbytes, bf = bnb
                wq4, qs = bf.quantize_4bit(w16, quant_type="nf4", blocksize=BNB_BLOCK, compress_statistics=False)
                ops["nf4"][s].append(lambda wq4=wq4, qs=qs, x16=x16: bitsandbytes.matmul_4bit(x16, wq4.t(), quant_state=qs))
                if layer == 0:
                    ref = x16.float() @ bf.dequantize_4bit(wq4, qs).float().t()
                    y = bitsandbytes.matmul_4bit(x16, wq4.t(), quant_state=qs).float()
                    fails += ((y - ref).abs().max() / ref.abs().max()).item() > TOL
            del W, w16
        torch.cuda.empty_cache()
    return ops, int(fails), notes


def step_fns(ops_v):
    return [ops_v[s][layer] for layer in range(LAYERS) for s in SHAPES]


def time_step(fns, notes, v):
    """CUDA graph of the whole step, replayed back to back; returns per-replay times (us) or None."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                for f in fns:
                    f()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for f in fns:
                f()
        for _ in range(WARM_REPS):
            g.replay()
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(STEP_REPS)]
        for st, en in ev:
            st.record()
            g.replay()
            en.record()
        torch.cuda.synchronize()
        return [st.elapsed_time(en) * 1e3 for st, en in ev]
    except Exception as ex:  # noqa: BLE001 -- recorded; the variant is then excluded
        notes[f"graph_error_{v}"] = repr(ex)[:300]
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="20 replays / 20 reps (pipeline check)")
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()
    torch.manual_seed(SEED)
    global STEP_REPS, BENCH_REPS
    if a.smoke:
        STEP_REPS, BENCH_REPS = 20, 20
    prop = torch.cuda.get_device_properties(0)
    l2 = prop.L2_cache_size
    ops, fails, notes = build(("fp16", "int4", "nf4"))
    cL2 = torch.zeros(l2 // 4, dtype=torch.int32, device="cuda")
    drv = triton.runtime.driver.active
    orig = drv.clear_cache
    events = []
    for v, ops_v in ops.items():
        ts = time_step(step_fns(ops_v), notes, v)
        if ts is not None:
            events.append({"kind": "step", "variant": v, "times_us": ts, "median_us": statistics.median(ts)})
        for s in SHAPES:
            fn, copies = ops_v[s][0], ops_v[s]
            res = {"write": tt.do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3,
                   "cold": cb.bench(fn, mode="cold", reps=BENCH_REPS).median_us,
                   "steady": cb.bench(rotate=copies, mode="steady", reps=BENCH_REPS).median_us}
            drv.clear_cache = lambda c: cL2.zero_()
            try:
                res["l2w"] = tt.do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3
            finally:
                drv.clear_cache = orig
            events.append({"kind": "kernel", "variant": v, "shape": s, **{f"{h}_us": x for h, x in res.items()}})
    meta = {"benchmark": "e2e_probe", "device": prop.name, "l2_bytes": l2, "torch": torch.__version__,
            "triton": triton.__version__, "layers": LAYERS, "H": H, "FFN": FFN, "shapes": SHAPES, "notes": notes,
            "smoke": a.smoke, "seed": SEED, "events": events, "correctness_failures": fails}
    json.dump(meta, open(a.meta, "w"), indent=1)
    print(f"{len(events)} events, variants={list(ops)}, correctness_failures={fails}, notes={notes}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
