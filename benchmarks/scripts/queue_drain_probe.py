#!/usr/bin/env python3
"""B18b -- does a short L2 flush make CUDA-event timing include host launch latency ("queue drain")?

Rules Q-0..Q-4 are pre-registered in research/HYPOTHESES.md, section "B18b pre-registration", before any
B18b measurement. One GPU (the lab's RTX 3060 Laptop, GA106).

Loop per timed call, as triton.testing.do_bench: flush; [pad]; start.record(); fn(); end.record(); no sync.
  cond "plain"  no pad (as do_bench)
  cond "pad"    torch.cuda._sleep(cycles) of about PAD_US on the GPU right before start.record(), so the GPU is
                still busy while the host enqueues start, fn and end; the pad runs before the start event and
                is not timed
Flush modes: pr256, prL2, pr2L2 (PR #12556 read flush on 256 MiB / L2 / 2 x L2), zero256, zeroL2 (zero_()),
zero256_wait (zero256, synchronize, sleep 10 ms).
Kernels: gemv4   F.linear(x[1, 4096], W) fp16, W = 4 x L2 (cuBLAS)
         e_large torch.add(x, 1, out=y), x and y float32 of 2 x L2 each
         t_small a Triton kernel y = 2 x, x and y float32 of L2 / 2 each (short kernel, Python launch)
Blocks of BLOCK calls per (kernel, cond, mode), the first BLOCK_DROP discarded; order shuffled per rep.
Also measured (descriptive): GPU time of each flush alone; host time to enqueue start + fn + end while the
GPU is busy; real triton.testing.do_bench with clear_cache patched (exploratory).
"""

import argparse
import json
import random
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing as tt

K = 4096
FLUSH_BYTES = 256 << 20
WAIT_S = 0.010
PAD_US = 300.0
MODES = ("pr256", "prL2", "pr2L2", "zero256", "zeroL2", "zero256_wait")
CONDS = ("plain", "pad")
KERNELS = ("gemv4", "e_large", "t_small")
REPS, BLOCK, BLOCK_DROP = 20, 6, 2
SEED = 1819
TOL = 2e-2
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


@triton.jit
def _scale_kernel(X, Y, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(Y + off, tl.load(X + off, mask=m) * 2.0, mask=m)


def build_kernels(l2: int):
    ks = {}
    n = max(128, int(4 * l2 / (2 * K)) // 128 * 128)
    x = (torch.randn(1, K, device="cuda") * 0.5).half()
    W = (torch.randn(n, K, device="cuda") * 0.02).half()
    ks["gemv4"] = (lambda: F.linear(x, W),
                   lambda: ((F.linear(x, W).float() - x.float() @ W.float().t()).abs().max()
                            / (x.float() @ W.float().t()).abs().max()).item() <= TOL)
    m = int(2 * l2) // 4
    a, b = torch.randn(m, device="cuda"), torch.empty(m, device="cuda")
    ks["e_large"] = (lambda: torch.add(a, 1.0, out=b), lambda: (torch.add(a, 1.0, out=b), torch.equal(b, a + 1.0))[1])
    s = int(l2 // 2) // 4
    c, d = torch.randn(s, device="cuda"), torch.empty(s, device="cuda")
    grid = ((s + 1023) // 1024, )
    ks["t_small"] = (lambda: _scale_kernel[grid](c, d, s, 1024),
                     lambda: (_scale_kernel[grid](c, d, s, 1024), torch.equal(d, c * 2.0))[1])
    return ks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="2 repetitions (pipeline check)")
    ap.add_argument("--check", action="store_true", help="correctness and pad calibration only; no timing")
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    prop = torch.cuda.get_device_properties(0)
    l2 = prop.L2_cache_size
    c256 = torch.zeros(FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    cL2 = torch.zeros(l2 // 4, dtype=torch.int32, device="cuda")
    c2L2 = torch.zeros(2 * l2 // 4, dtype=torch.int32, device="cuda")

    def flush(mode: str):
        if mode in ("zero256", "zero256_wait"):
            c256.zero_()
        if mode == "zero256_wait":
            torch.cuda.synchronize()
            time.sleep(WAIT_S)
        elif mode == "pr256":
            pr_clear(c256)
        elif mode == "prL2":
            pr_clear(cL2)
        elif mode == "pr2L2":
            pr_clear(c2L2)
        elif mode == "zeroL2":
            cL2.zero_()

    ks = build_kernels(l2)
    fails = sum(int(not chk()) for _, chk in ks.values())
    before = c2L2.clone()
    pr_clear(c2L2)
    torch.cuda.synchronize()
    fails += int(not torch.equal(before, c2L2))

    def gpu_us(fn, n=20):                                         # GPU time of fn alone (synchronized)
        out = []
        for _ in range(n):
            torch.cuda._sleep(200000)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            out.append(s.elapsed_time(e) * 1e3)
        return statistics.median(out)

    probe = 2_000_000
    us_per_cycle = gpu_us(lambda: torch.cuda._sleep(probe), 10) / probe
    pad_cycles = int(PAD_US / us_per_cycle)
    pad_us = gpu_us(lambda: torch.cuda._sleep(pad_cycles), 10)
    for fn, _ in ks.values():                                     # warm up kernels and flushes
        for mode in MODES:
            flush(mode)
            fn()
    torch.cuda.synchronize()
    events: list[dict] = []
    if not a.check:
        reps = 2 if a.smoke else REPS
        times = {(k, c, m): [] for k in KERNELS for c in CONDS for m in MODES}
        for _ in range(reps):
            for k in KERNELS:
                fn = ks[k][0]
                cells = [(c, m) for c in CONDS for m in MODES]
                rng.shuffle(cells)
                for c, m in cells:
                    ev = []
                    for _ in range(BLOCK):
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        flush(m)
                        if c == "pad":
                            torch.cuda._sleep(pad_cycles)
                        s.record()
                        fn()
                        e.record()
                        ev.append((s, e))
                    torch.cuda.synchronize()
                    times[(k, c, m)] += [s.elapsed_time(e) * 1e3 for s, e in ev[BLOCK_DROP:]]
        for (k, c, m), ts in times.items():
            events.append({"kind": "timing", "kernel": k, "cond": c, "mode": m, "times_us": ts,
                           "median_us": statistics.median(ts), "mean_us": statistics.fmean(ts)})
        flush_us = {m: gpu_us(lambda m=m: flush(m)) for m in MODES if m != "zero256_wait"}
        host_us = {}
        for k in KERNELS:                                         # host enqueue time with the GPU busy
            fn, hs = ks[k][0], []
            for _ in range(50):
                torch.cuda._sleep(pad_cycles * 4)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t0 = time.perf_counter()
                s.record()
                fn()
                e.record()
                hs.append((time.perf_counter() - t0) * 1e6)
                torch.cuda.synchronize()
            host_us[k] = statistics.median(hs)
        events.append({"kind": "aux", "flush_gpu_us": flush_us, "host_enqueue_us": host_us})
        try:                                                      # EXPLORATORY: real do_bench, patched flush
            drv = triton.runtime.driver.active
            orig = drv.clear_cache
            variants = {"zero256 (shipped)": None, "zeroL2": lambda c: cL2.zero_(), "pr256": lambda c: pr_clear(c256),
                        "prL2": lambda c: pr_clear(cL2),
                        "zeroL2 + pad": lambda c: (cL2.zero_(), torch.cuda._sleep(pad_cycles)),
                        "prL2 + pad": lambda c: (pr_clear(cL2), torch.cuda._sleep(pad_cycles))}
            for k in ("t_small", "gemv4"):
                res = {}
                for name, cc in variants.items():
                    if cc is not None:
                        drv.clear_cache = cc
                    try:
                        res[name] = tt.do_bench(ks[k][0], warmup=25, rep=100, return_mode="median") * 1e3
                    finally:
                        drv.clear_cache = orig
                events.append({"kind": "do_bench", "kernel": k, "triton": triton.__version__, "median_us": res})
        except Exception as ex:  # noqa: BLE001 -- exploratory only, never fatal
            events.append({"kind": "do_bench_error", "error": repr(ex)[:300]})
    meta = {"benchmark": "queue_drain_probe", "device": prop.name, "l2_bytes": l2, "torch": torch.__version__,
            "triton": triton.__version__, "pr_head": PR_HEAD, "pad_target_us": PAD_US, "pad_cycles": pad_cycles,
            "pad_us": pad_us, "smoke": a.smoke, "check": a.check, "seed": SEED, "events": events,
            "correctness_failures": int(fails)}
    json.dump(meta, open(a.meta, "w"), indent=1)
    print(f"{len(events)} events, correctness_failures={fails}" + (f", pad_us={pad_us:.0f}" if a.check else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
