#!/usr/bin/env python3
"""B9 -- does the L2 write dirty lines back to DRAM while the GPU is idle?

Why: B7r/B7r2 (D-049, D-050; exploratory, both failed a validity check) suggest that ncu's
one-kernel-at-a-time replay under-counts the DRAM writes of eager PyTorch chains near L2 capacity,
as if dirty lines were written back in the profiler's gaps between kernels, where no kernel is
measured. B9 measures that mechanism directly, with DRAM WRITES only (the device's background
traffic on this laptop is reads only: D-048).

Cell (w, idle, T): after a flush (16 MiB read: the L2 holds only clean lines), per repetition
  range A: one write kernel stores w x L2 bytes (full 128-B lines, below the dirty cap 1/2 L2),
           synchronize, then stay idle for T:
             idle = "host": the CPU sleeps, no kernel runs (like ncu's gaps between kernels)
             idle = "spin": a spin kernel (torch.cuda._sleep, no memory access) keeps one SM busy
  range B: the flush kernel again; it evicts every line, so the lines still dirty are written back
  A = writes(range A) / bytes written, B = writes(range B) / bytes written; A + B ~ 1 if nothing is lost.
Control ranges: a flush, then a range holding only a host sleep of 10 ms (background writes; and
proof that a range counts idle time: the background READS appear).
D is checked after every repetition (outside the ranges): D[i] == i ^ tag.
Run under: ncu --replay-mode app-range (see idle_wb_probe.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/dmkl-inductor"))

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_opchain import Flusher  # noqa: E402  (16 MiB read-only flush, Triton)

LINE = 128
W_FRACS = (0.125, 0.25, 0.375)
T_HOST_MS = (0.0, 0.1, 1.0, 10.0)
SPIN_CYCLES = 15_000_000   # ~10 ms of spinning on this laptop (spin kernels ran ~1.1-2.2 ms per 1M cycles in B7r)
CONTROL_MS = 10.0
REPS = 5


@triton.jit
def write_lines(d_ptr, n, tag, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(d_ptr + offs, offs ^ tag, mask=offs < n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="w 0.1875, T = 0 only, 2 repetitions, plus the control")
    ap.add_argument("--meta", default="idle_wb_meta.json")
    a = ap.parse_args()
    prop = torch.cuda.get_device_properties(0)
    l2 = prop.L2_cache_size
    flush = Flusher(l2)
    cells = ([(0.1875, "host", 0.0)] if a.smoke else
             [(w, "host", t) for w in W_FRACS for t in T_HOST_MS] + [(w, "spin", 10.0) for w in W_FRACS])
    reps = 2 if a.smoke else REPS
    n_max = max(int(w * l2) // LINE * LINE // 4 for w, _, _ in cells)
    d = torch.zeros(n_max, dtype=torch.int32, device="cuda")
    idx = torch.arange(n_max, dtype=torch.int32, device="cuda")
    # warm up every kernel outside the ranges (Triton compile/load, spin kernel module)
    write_lines[(triton.cdiv(n_max, 1024),)](d, n_max, 1, BLOCK=1024)
    flush()
    torch.cuda._sleep(1000)
    torch.cuda.synchronize()

    ranges, fails, tag = [], 0, 7
    meta = {"benchmark": "idle_wb_probe", "device": prop.name, "l2_bytes": l2, "torch": torch.__version__,
            "triton": triton.__version__, "smoke": a.smoke, "reps": reps, "spin_cycles": SPIN_CYCLES,
            "control_ms": CONTROL_MS, "flush_bytes": flush.n * 4}

    def rng_start():
        torch.cuda.synchronize()
        torch.cuda.profiler.start()

    def rng_stop():
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()

    for k in range(2):  # control ranges at the start
        flush()
        rng_start()
        time.sleep(CONTROL_MS / 1e3)
        rng_stop()
        ranges.append({"kind": "control", "idle_ms": CONTROL_MS})
    for w, idle, t_ms in cells:
        n = int(w * l2) // LINE * LINE // 4
        for rep in range(reps):
            tag += 1
            flush()
            rng_start()
            write_lines[(triton.cdiv(n, 1024),)](d, n, tag, BLOCK=1024)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if idle == "host" and t_ms > 0:
                time.sleep(t_ms / 1e3)
            elif idle == "spin":
                torch.cuda._sleep(SPIN_CYCLES)
                torch.cuda.synchronize()
            held = (time.perf_counter() - t0) * 1e3
            rng_stop()
            ranges.append({"kind": "A", "w": w, "idle": idle, "T_ms": t_ms, "rep": rep, "bytes": n * 4,
                           "held_ms": round(held, 3)})
            rng_start()
            flush()
            rng_stop()
            ranges.append({"kind": "B", "w": w, "idle": idle, "T_ms": t_ms, "rep": rep, "bytes": n * 4})
            ok = bool(torch.equal(d[:n], idx[:n] ^ tag))
            fails += not ok
            torch.cuda.synchronize()
    meta["ranges"], meta["correctness_failures"] = ranges, fails
    json.dump(meta, open(a.meta, "w"), indent=1)
    print(f"{len(ranges)} ranges, correctness_failures={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
