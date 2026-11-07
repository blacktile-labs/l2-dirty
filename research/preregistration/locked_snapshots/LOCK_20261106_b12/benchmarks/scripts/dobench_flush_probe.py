#!/usr/bin/env python3
"""B12 -- how much does Triton's do_bench L2 flush bias the timing of an LLM decode GEMV?

triton.testing.do_bench (every tag v2.0.0..v3.9.0; also PyTorch Inductor's benchmarker) clears the L2
before each timed call with `cache.zero_()` on a 256 MiB int32 buffer: a WRITE. Our dirty-cap results
(C5n, B6b/B9b/B11) say the L2 then still holds a dirty residue of about R x L2 bytes, which the timed
kernel must write back to DRAM when it evicts it. A read-only flush (`cache.view(float32).sum()`)
leaves no dirty lines (triton-lang/triton#12554 measured this on an A100 but left the ~1/2-L2 size of
the effect unexplained).

Timed op: y = F.linear(x, W) with x [1, K] fp16, W [N, K] fp16, K = 4096 (a batch-1 decode linear
layer; memory bound: it reads W once). W is sized w_read x L2 for w_read in {2, 4, 8}, so the GEMV
evicts every line of the L2. ONE variable changes: the flush ("zero" or "read") before the call.
  --mode range   (ncu --replay-mode app-range): per repetition: flush; sync; [range: GEMV; sync]
                 + 2 control ranges (empty, after a read flush)
  --mode timing  (no profiler): per repetition: flush; start event; GEMV; end event (as do_bench:
                 no sync between flush and start); variants interleaved; plus EXPLORATORY rows:
                 a fixed 4096 x 4096 weight (32 MiB) and triton.testing.do_bench with its own flush
                 vs the same call with clear_cache patched to a read
Correctness: every GEMV size is checked once against an fp32 reference before measuring.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch
import torch.nn.functional as F

K = 4096
W_READ = (2, 4, 8)
FLUSH_BYTES = 256 << 20          # == triton do_bench's cache size
VARIANTS = ("zero", "read")
RANGE_REPS = 5
TIMING_REPS = 40
SEED = 1234
REL_TOL = 2e-2


def n_rows(w_read: float, l2: int) -> int:
    return max(128, int(w_read * l2 / (2 * K)) // 128 * 128)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("range", "timing"), required=True)
    ap.add_argument("--smoke", action="store_true", help="w_read 2 only, 2 repetitions")
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()
    torch.manual_seed(SEED)
    prop = torch.cuda.get_device_properties(0)
    l2 = prop.L2_cache_size
    ws = (2,) if a.smoke else W_READ
    cache = torch.empty(FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    flushes = {"zero": lambda: cache.zero_(), "read": lambda: cache.view(torch.float32).sum()}
    x = (torch.randn(1, K, device="cuda") * 0.5).half()
    mats = {w: (torch.randn(n_rows(w, l2), K, device="cuda") * 0.02).half() for w in ws}
    fixed = (torch.randn(4096, K, device="cuda") * 0.02).half()
    fails = 0
    for w, m in list(mats.items()) + [("fixed", fixed)]:
        y = F.linear(x, m).float()
        ref = x.float() @ m.float().t()
        err = ((y - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
        fails += err > REL_TOL
    torch.cuda.synchronize()
    events: list[dict] = []
    reps = 2 if a.smoke else (RANGE_REPS if a.mode == "range" else TIMING_REPS)

    if a.mode == "range":
        for _ in range(2):                                   # control: empty ranges
            flushes["read"]()
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            events.append({"kind": "control"})
        for w in ws:
            m = mats[w]
            for rep in range(reps):
                for var in VARIANTS:
                    flushes[var]()
                    torch.cuda.synchronize()
                    torch.cuda.profiler.start()
                    F.linear(x, m)
                    torch.cuda.synchronize()
                    torch.cuda.profiler.stop()
                    events.append({"kind": "gemv", "variant": var, "w_read": w, "rep": rep,
                                   "weight_bytes": m.numel() * 2, "out_bytes": m.shape[0] * 2})
    else:
        def timed(m, var, n):
            out = []
            for _ in range(n):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                flushes[var]()
                s.record()
                F.linear(x, m)
                e.record()
                out.append((s, e))
            torch.cuda.synchronize()
            return [s.elapsed_time(e) * 1e3 for s, e in out]       # microseconds
        for m in list(mats.values()) + [fixed]:                       # warm up kernels and clocks
            for var in VARIANTS:
                timed(m, var, 5)
        times = {(w, var): [] for w in list(ws) + ["fixed"] for var in VARIANTS}
        for rep in range(reps):
            for w, m in list(mats.items()) + [("fixed", fixed)]:
                for var in VARIANTS:
                    times[(w, var)] += timed(m, var, 1)
        for (w, var), ts in times.items():
            m = fixed if w == "fixed" else mats[w]
            events.append({"kind": "timing", "w_read": w, "variant": var, "weight_bytes": m.numel() * 2,
                           "times_us": ts, "median_us": statistics.median(ts)})
        try:                                                           # EXPLORATORY: real do_bench
            import triton
            import triton.testing as tt
            drv = triton.runtime.driver.active
            orig = drv.clear_cache
            for w, m in list(mats.items()) + [("fixed", fixed)]:
                fn = lambda m=m: F.linear(x, m)
                d0 = tt.do_bench(fn, warmup=25, rep=200, return_mode="median")
                drv.clear_cache = lambda c: c.view(torch.float32).sum()
                try:
                    d1 = tt.do_bench(fn, warmup=25, rep=200, return_mode="median")
                finally:
                    drv.clear_cache = orig
                events.append({"kind": "do_bench", "w_read": w, "triton": triton.__version__,
                               "default_us": d0 * 1e3, "read_flush_us": d1 * 1e3})
        except Exception as ex:  # noqa: BLE001 -- exploratory only, never fatal
            events.append({"kind": "do_bench_error", "error": repr(ex)[:300]})
    meta = {"benchmark": "dobench_flush_probe", "mode": a.mode, "device": prop.name, "l2_bytes": l2,
            "sm_count": prop.multi_processor_count, "torch": torch.__version__, "K": K,
            "flush_bytes": FLUSH_BYTES, "smoke": a.smoke, "reps": reps, "seed": SEED,
            "events": events, "correctness_failures": int(fails)}
    json.dump(meta, open(a.meta, "w"), indent=1)
    print(f"{a.mode}: {len(events)} events, correctness_failures={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
