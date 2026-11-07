#!/usr/bin/env python3
"""B18 -- which L2-flush fixes work, and who pays for a kernel's own writes?

Rules H-0..H-4 are pre-registered in research/HYPOTHESES.md, section "B18 pre-registration", before any
B18 measurement. One GPU (the lab's RTX 3060 Laptop, GA106, L2 3 MiB, counters enabled).

Every measured call is preceded, as in triton.testing.do_bench, by a call of the same kernel and then a
flush ("mode"); only the flush changes:
  none          no flush (warm)
  zero256       cache.zero_() on 256 MiB, as do_bench ships (a write)
  zero256_wait  zero256, then synchronize and sleep 10 ms
  zero256_read  zero256, then the PR #12556 read flush on the same 256 MiB
  zeroL2        zero_() on an L2-sized int32 buffer, as PyTorch Inductor's benchmarker, nvbench, CUTLASS
  pr256         PR #12556 read flush (masked volatile load, no store) on 256 MiB: the reference
  sum256        cache.view(float32).sum() on 256 MiB
  prL2          PR #12556 read flush on an L2-sized buffer
  pr2L2         PR #12556 read flush on a 2 x L2 buffer
  rotate        no flush; the kernel cycles over enough copies of its inputs and outputs (>= 4 x L2)
Timed kernels: gemv4   y = F.linear(x[1, 4096], W) fp16, W = 4 x L2 (output 3 KB)
               e_tiny  y = x + 1, float32, x and y = L2 / 128 each (footprint L2 / 64)
               e_small y = x + 1, x and y = L2 / 8 each (footprint 0.25 L2)
               e_large y = x + 1, x and y = 2 x L2 each
  --mode range   (ncu --replay-mode app-range): [previous call; flush; sync; range: call; sync]; for rotate
                 [copies + 1 rotated calls; sync; range: call on the next copy; sync]; 2 empty control ranges
  --mode timing  (no profiler; gemv4 and e_large): CUDA events per call, flush before each call with no
                 sync (as do_bench); blocks of 5 calls per mode, the first 2 of each block discarded
  --mode check   correctness and the PR flush's PTX only; prints no measurement
"""

import argparse
import json
import math
import random
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

K = 4096
FLUSH_BYTES = 256 << 20
WAIT_S = 0.010
ROT_FOOTPRINT_L2 = 4
MODES = ("none", "zero256", "zero256_wait", "zero256_read", "zeroL2", "pr256", "sum256", "prL2", "pr2L2",
         "rotate")
KERNELS = ("gemv4", "e_tiny", "e_small", "e_large")
E_FRAC = {"e_tiny": 1 / 128, "e_small": 1 / 8, "e_large": 2.0}
TIMING_KERNELS = ("gemv4", "e_large")
RANGE_REPS, TIMING_REPS, BLOCK, BLOCK_DROP = 5, 20, 5, 2
SEED = 1818
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


class Kernel:
    """One timed op with `copies` independent input/output sets (copy 0 is used by every flush mode)."""

    def __init__(self, name: str, l2: int, copies_for_rotation: bool):
        self.name, self.ptr = name, 0
        if name == "gemv4":
            n = max(128, int(4 * l2 / (2 * K)) // 128 * 128)
            self.in_bytes, self.out_bytes = n * K * 2, n * 2
            per_copy = self.in_bytes / l2
            self.copies = max(2, math.ceil(ROT_FOOTPRINT_L2 / per_copy)) if copies_for_rotation else 1
            self.x = (torch.randn(1, K, device="cuda") * 0.5).half()
            self.data = [(torch.randn(n, K, device="cuda") * 0.02).half() for _ in range(self.copies)]
            self._call = lambda i: F.linear(self.x, self.data[i])
        else:
            m = int(E_FRAC[name] * l2) // 4 // 1024 * 1024        # float32 elements, multiple of 1024
            self.in_bytes = self.out_bytes = m * 4
            per_copy = 2 * self.in_bytes / l2
            self.copies = max(2, math.ceil(ROT_FOOTPRINT_L2 / per_copy)) if copies_for_rotation else 1
            self.data = [(torch.randn(m, device="cuda"), torch.empty(m, device="cuda")) for _ in range(self.copies)]
            self._call = lambda i: torch.add(self.data[i][0], 1.0, out=self.data[i][1])

    def call(self, i: int = 0):
        return self._call(i)

    def next(self) -> int:
        self.ptr = (self.ptr + 1) % self.copies
        return self.ptr

    def check(self) -> int:
        """Correctness of every copy against a reference; returns the number of failures."""
        fails = 0
        for i in range(self.copies):
            if self.name == "gemv4":
                y = self._call(i).float()
                ref = self.x.float() @ self.data[i].float().t()
                fails += ((y - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item() > TOL
            else:
                self._call(i)
                x, y = self.data[i]
                fails += not torch.equal(y, x + 1.0)
        return int(fails)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("range", "timing", "check"), required=True)
    ap.add_argument("--smoke", action="store_true", help="1 repetition per cell (pipeline check)")
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
        if mode in ("zero256", "zero256_wait", "zero256_read"):
            c256.zero_()
        if mode == "zero256_wait":
            torch.cuda.synchronize()
            time.sleep(WAIT_S)
        elif mode in ("zero256_read", "pr256"):
            pr_clear(c256)
        elif mode == "zeroL2":
            cL2.zero_()
        elif mode == "sum256":
            c256.view(torch.float32).sum()
        elif mode == "prL2":
            pr_clear(cL2)
        elif mode == "pr2L2":
            pr_clear(c2L2)

    names = TIMING_KERNELS if a.mode == "timing" else KERNELS
    kernels = {n: Kernel(n, l2, copies_for_rotation=True) for n in names}
    fails = sum(k.check() for k in kernels.values())
    before = c2L2.clone()
    kp = pr_clear(c2L2)
    torch.cuda.synchronize()
    ptx = kp.asm["ptx"]
    pr_check = {"ld_volatile_global": ptx.count("ld.volatile.global"), "st_global": ptx.count("st.global"),
                "buffer_unchanged": bool(torch.equal(c2L2, before))}
    fails += int(not pr_check["buffer_unchanged"]) + int(pr_check["st_global"] > 0)
    for k in kernels.values():                                   # warm up every kernel and flush once
        for m in MODES:
            flush(m)
            k.call(0)
    torch.cuda.synchronize()
    events: list[dict] = []
    reps = 1 if a.smoke else (RANGE_REPS if a.mode == "range" else TIMING_REPS)

    if a.mode == "range":
        for _ in range(2):                                       # control: empty ranges after a read flush
            pr_clear(c256)
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            events.append({"kind": "control"})
        for name in names:
            k = kernels[name]
            for rep in range(reps):
                order = list(MODES)
                rng.shuffle(order)
                for mode in order:
                    if mode == "rotate":
                        for _ in range(k.copies + 1):
                            k.call(k.next())
                        i = k.next()
                    else:
                        k.call(0)                               # the previous call of the same kernel
                        flush(mode)
                        i = 0
                    torch.cuda.synchronize()
                    torch.cuda.profiler.start()
                    k.call(i)
                    torch.cuda.synchronize()
                    torch.cuda.profiler.stop()
                    events.append({"kind": "call", "kernel": name, "mode": mode, "rep": rep,
                                   "in_bytes": k.in_bytes, "out_bytes": k.out_bytes, "copies": k.copies})
    elif a.mode == "timing":
        times = {(n, m): [] for n in names for m in MODES}
        for rep in range(reps):
            for name in names:
                k = kernels[name]
                order = list(MODES)
                rng.shuffle(order)
                for mode in order:
                    if mode == "rotate":
                        for _ in range(k.copies + 1):
                            k.call(k.next())
                    ev = []
                    for _ in range(BLOCK):
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        if mode == "rotate":
                            i = k.next()
                        else:
                            flush(mode)
                            i = 0
                        s.record()
                        k.call(i)
                        e.record()
                        ev.append((s, e))
                    torch.cuda.synchronize()
                    times[(name, mode)] += [s.elapsed_time(e) * 1e3 for s, e in ev[BLOCK_DROP:]]
        for (name, mode), ts in times.items():
            events.append({"kind": "timing", "kernel": name, "mode": mode, "times_us": ts,
                           "median_us": statistics.median(ts), "mean_us": statistics.fmean(ts)})
    meta = {"benchmark": "flush_fix_probe", "mode": a.mode, "device": prop.name, "l2_bytes": l2,
            "sm_count": prop.multi_processor_count, "torch": torch.__version__, "triton": triton.__version__,
            "pr_head": PR_HEAD, "pr_flush_check": pr_check, "flush_bytes": FLUSH_BYTES, "wait_s": WAIT_S,
            "kernels": {n: {"in_bytes": k.in_bytes, "out_bytes": k.out_bytes, "copies": k.copies}
                        for n, k in kernels.items()},
            "smoke": a.smoke, "reps": reps, "seed": SEED, "events": events, "correctness_failures": int(fails)}
    json.dump(meta, open(a.meta, "w"), indent=1)
    print(f"{a.mode}: {len(events)} events, correctness_failures={fails}, pr_flush_check={pr_check}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
