#!/usr/bin/env python3
"""B18c -- validate cleanbench (read flush + GPU pad + steady mode) on the lab's RTX 3060 Laptop (GA106).

Rules T-0..T-2 are pre-registered in research/HYPOTHESES.md, section "B18c pre-registration", before any B18c
measurement. Per run, for every kernel and config, 5 bench() calls of 100 timed reps each, interleaved in a
shuffled order; a "cell value" is the median over bench() calls of the per-call mean.
Kernels: gemv4   F.linear(x[1, 4096], W) fp16, W = 4 x L2 (copies: 2)
         e_large torch.add(x, 1, out=y), x and y 2 x L2 each (copies: 2)
         e_mid   same with x and y 0.4 x L2 each (output below the dirty cap; copies: 5)
         t_small Triton y = 2x, x and y L2 / 2 each (copies: 4)
Configs: cold_2L2 (flush_bytes = 2 x L2), cold_256 (flush_bytes = 256 MiB), steady (rotation, no flush)
Descriptive: triton.testing.do_bench as shipped, host dispatch time and the pad cleanbench chose.
"""

import argparse
import json
import random
import statistics
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing as tt

import cleanbench as cb

K = 4096
CALLS, REPS = 5, 100
SEED = 1820
TOL = 2e-2
CONFIGS = ("cold_2L2", "cold_256", "steady")


@triton.jit
def _scale_kernel(X, Y, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(Y + off, tl.load(X + off, mask=m) * 2.0, mask=m)


def build(l2: int):
    """name -> (list of callables over independent copies, correctness check)"""
    ks = {}
    n = max(128, int(4 * l2 / (2 * K)) // 128 * 128)
    x = (torch.randn(1, K, device="cuda") * 0.5).half()
    Ws = [(torch.randn(n, K, device="cuda") * 0.02).half() for _ in range(2)]
    ks["gemv4"] = ([lambda W=W: F.linear(x, W) for W in Ws],
                   lambda: all(((F.linear(x, W).float() - x.float() @ W.float().t()).abs().max()
                                / (x.float() @ W.float().t()).abs().max()).item() <= TOL for W in Ws))

    def adds(frac, copies):
        m = int(frac * l2) // 4 // 1024 * 1024
        bufs = [(torch.randn(m, device="cuda"), torch.empty(m, device="cuda")) for _ in range(copies)]
        calls = [lambda a=a, b=b: torch.add(a, 1.0, out=b) for a, b in bufs]
        return calls, lambda: all((torch.add(a, 1.0, out=b), torch.equal(b, a + 1.0))[1] for a, b in bufs)
    ks["e_large"] = adds(2.0, 2)
    ks["e_mid"] = adds(0.4, 5)
    s = int(l2 // 2) // 4 // 1024 * 1024
    bufs = [(torch.randn(s, device="cuda"), torch.empty(s, device="cuda")) for _ in range(4)]
    grid = ((s + 1023) // 1024, )
    ks["t_small"] = ([lambda c=c, d=d: _scale_kernel[grid](c, d, s, 1024) for c, d in bufs],
                     lambda: all((_scale_kernel[grid](c, d, s, 1024), torch.equal(d, c * 2.0))[1] for c, d in bufs))
    return ks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="1 bench() call per cell, 10 reps (pipeline check)")
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    prop = torch.cuda.get_device_properties(0)
    l2 = prop.L2_cache_size
    ks = build(l2)
    fails = sum(int(not chk()) for _, chk in ks.values())
    buf = cb.flush_buffer(2 * l2)
    before = buf.clone()
    cb.read_flush(buf)
    torch.cuda.synchronize()
    fails += int(not torch.equal(before, buf))
    calls_n, reps = (1, 10) if a.smoke else (CALLS, REPS)
    cells = [(k, c) for k in ks for c in CONFIGS] * calls_n
    rng.shuffle(cells)
    events = []
    for k, c in cells:
        calls = ks[k][0]
        if c == "steady":
            r = cb.bench(rotate=calls, mode="steady", reps=reps)
        else:
            r = cb.bench(calls[0], mode="cold", reps=reps, flush_bytes=(2 * l2 if c == "cold_2L2" else 256 << 20))
        events.append({"kind": "bench", "kernel": k, "config": c, "mean_us": r.mean_us, "median_us": r.median_us,
                       "p25_us": r.p25_us, "p75_us": r.p75_us, "pad_us": r.pad_us, "host_dispatch_us": r.host_dispatch_us,
                       "flush_bytes": r.flush_bytes, "warnings": r.warnings,
                       "tick_fraction": cb.tick_fraction(r.times_us)})
    for k in ks:                                                    # descriptive: do_bench as shipped
        events.append({"kind": "do_bench", "kernel": k,
                       "median_us": tt.do_bench(ks[k][0][0], warmup=25, rep=100, return_mode="median") * 1e3})
    meta = {"benchmark": "cleanbench_validate", "device": prop.name, "l2_bytes": l2, "torch": torch.__version__,
            "triton": triton.__version__, "smoke": a.smoke, "seed": SEED, "calls_per_cell": calls_n, "reps": reps,
            "events": events, "correctness_failures": int(fails)}
    json.dump(meta, open(a.meta, "w"), indent=1)
    print(f"{len(events)} events, correctness_failures={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
