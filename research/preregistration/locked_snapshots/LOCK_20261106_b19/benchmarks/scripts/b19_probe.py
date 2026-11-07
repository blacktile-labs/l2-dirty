#!/usr/bin/env python3
"""B19 -- on large-L2 GPUs: is B17's L2-sized-flush anomaly "queue drain", and how large is the gap between
cleanbench's cold and steady modes?

Rules R-0..R-4 are pre-registered in research/HYPOTHESES.md, section "B19 pre-registration", before any B19
GPU is rented. Timing only (rented containers have no counter access).
Kernels: rms16  Liger-Kernel RMSNorm bf16 [2048, 4096] (16 MiB in, 16 MiB out), the B17 anomaly kernel
         rms64  the same with [8192, 4096] (64 MiB in, 64 MiB out)
         gemv   F.linear(x[1, 4096], W[14336, 4096]) fp16 (112 MiB), the B17 reference
         gemv16 F.linear(x[1, 2048], W[4096, 2048]) fp16 (16 MiB, below half of every L2 here)
         e_q    torch.add(x, 1, out=y) float32, x and y 0.25 x L2 each (output below the dirty cap)
Part A (as B18b): loop flush; [pad]; start; fn; end; blocks of 6 calls, first 2 dropped, 20 reps, shuffled.
        Kernels rms16, gemv16, gemv; flush modes zero256, zeroL2, pr256, none; conds plain / pad, where
        pad = max(300 us, 3 x host dispatch time of the kernel) of torch.cuda._sleep before the start event.
Part B: cleanbench.bench cold (default flush) and steady (copies covering >= 4 x L2), 5 calls x 100 reps,
        shuffled; kernels rms64, rms16, gemv, e_q.
Part C: triton.testing.do_bench as shipped, every kernel (descriptive).
"""

import argparse
import json
import math
import random
import statistics
import sys

import torch
import torch.nn.functional as F
import triton
import triton.testing as tt

import cleanbench as cb

FLUSH_BYTES = 256 << 20
MODES = ("zero256", "zeroL2", "pr256", "none")
CONDS = ("plain", "pad")
A_KERNELS = ("rms16", "gemv16", "gemv")
B_KERNELS = ("rms64", "rms16", "gemv", "e_q")
REPS, BLOCK, DROP = 20, 6, 2
B_CALLS, B_REPS = 5, 100
SEED = 1900
TOL = 2e-2
HID, EPS = 4096, 1e-6


def copies_for(per_copy_bytes: int, l2: int) -> int:
    return max(2, math.ceil(4 * l2 / per_copy_bytes))


def build(l2: int):
    """name -> (callables over independent copies, correctness check, notes)"""
    ks, notes = {}, {}
    try:
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        import importlib.metadata as md
        notes["liger"] = md.version("liger-kernel")
        w = (1 + 0.1 * torch.randn(HID, device="cuda")).to(torch.bfloat16)
        for name, T in (("rms16", 2048), ("rms64", 8192)):
            n = copies_for(2 * T * HID * 2, l2)
            xs = [torch.randn(T, HID, device="cuda", dtype=torch.bfloat16) for _ in range(n)]
            calls = [lambda x=x: LigerRMSNormFunction.apply(x, w, EPS) for x in xs]

            def chk(xs=xs):
                x = xs[0]
                y = LigerRMSNormFunction.apply(x, w, EPS).float()
                xf = x.float()
                ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
                return ((y - ref).abs().max() / ref.abs().max()).item() <= TOL
            ks[name] = (calls, chk)
    except Exception as ex:  # noqa: BLE001 -- recorded; R-0 fails without Liger
        notes["liger_error"] = repr(ex)[:300]
    for name, (N, Kd) in (("gemv", (14336, 4096)), ("gemv16", (4096, 2048))):
        n = copies_for(N * Kd * 2, l2)
        x = (torch.randn(1, Kd, device="cuda") * 0.5).half()
        Ws = [(torch.randn(N, Kd, device="cuda") * 0.02).half() for _ in range(n)]
        calls = [lambda W=W, x=x: F.linear(x, W) for W in Ws]

        def chk(x=x, W=Ws[0]):
            ref = x.float() @ W.float().t()
            return ((F.linear(x, W).float() - ref).abs().max() / ref.abs().max()).item() <= TOL
        ks[name] = (calls, chk)
    m = int(0.25 * l2) // 4 // 1024 * 1024
    bufs = [(torch.randn(m, device="cuda"), torch.empty(m, device="cuda")) for _ in range(copies_for(8 * m, l2))]
    ks["e_q"] = ([lambda a=a, b=b: torch.add(a, 1.0, out=b) for a, b in bufs],
                 lambda: all((torch.add(a, 1.0, out=b), torch.equal(b, a + 1.0))[1] for a, b in bufs))
    return ks, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="2 reps in part A, 1 call x 10 reps in part B")
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    prop = torch.cuda.get_device_properties(0)
    l2 = prop.L2_cache_size
    ks, notes = build(l2)
    fails = sum(int(not chk()) for _, chk in ks.values())
    c256 = torch.zeros(FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    cL2 = torch.zeros(l2 // 4, dtype=torch.int32, device="cuda")
    before = c256.clone()
    cb.read_flush(c256)
    torch.cuda.synchronize()
    fails += int(not torch.equal(before, c256))
    del before

    def flush(mode):
        if mode == "zero256":
            c256.zero_()
        elif mode == "zeroL2":
            cL2.zero_()
        elif mode == "pr256":
            cb.read_flush(c256)

    host = {k: cb.host_dispatch_us(ks[k][0][0]) for k in ks}
    pads = {k: max(300.0, 3.0 * host[k]) for k in ks}
    events = []
    reps = 2 if a.smoke else REPS
    times = {}
    for _ in range(reps):
        for k in [k for k in A_KERNELS if k in ks]:
            fn = ks[k][0][0]
            cells = [(c, m) for c in CONDS for m in MODES]
            rng.shuffle(cells)
            for c, m in cells:
                ev = []
                for _ in range(BLOCK):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    flush(m)
                    if c == "pad":
                        cb.pad(pads[k])
                    s.record()
                    fn()
                    e.record()
                    ev.append((s, e))
                torch.cuda.synchronize()
                times.setdefault((k, c, m), []).extend(s.elapsed_time(e) * 1e3 for s, e in ev[DROP:])
    for (k, c, m), ts in times.items():
        events.append({"kind": "loop", "kernel": k, "cond": c, "mode": m, "times_us": ts})
    bcalls, breps = (1, 10) if a.smoke else (B_CALLS, B_REPS)
    cells = [(k, cfg) for k in B_KERNELS if k in ks for cfg in ("cold", "steady")] * bcalls
    rng.shuffle(cells)
    for k, cfg in cells:
        calls = ks[k][0]
        r = (cb.bench(rotate=calls, mode="steady", reps=breps) if cfg == "steady"
             else cb.bench(calls[0], mode="cold", reps=breps))
        events.append({"kind": "bench", "kernel": k, "config": cfg, "copies": len(calls), "mean_us": r.mean_us,
                       "median_us": r.median_us, "p25_us": r.p25_us, "p75_us": r.p75_us, "pad_us": r.pad_us,
                       "host_dispatch_us": r.host_dispatch_us, "flush_bytes": r.flush_bytes})
    for k in ks:
        events.append({"kind": "do_bench", "kernel": k,
                       "median_us": tt.do_bench(ks[k][0][0], warmup=25, rep=100, return_mode="median") * 1e3})
    meta = {"benchmark": "b19_probe", "device": prop.name, "l2_bytes": l2, "cc": f"{prop.major}.{prop.minor}",
            "torch": torch.__version__, "triton": triton.__version__, "notes": notes, "host_dispatch_us": host,
            "pads_us": pads, "smoke": a.smoke, "seed": SEED, "kernels": sorted(ks), "events": events,
            "correctness_failures": int(fails)}
    json.dump(meta, open(a.meta, "w"), indent=1)
    print(f"{len(events)} events, kernels={sorted(ks)}, correctness_failures={fails}, notes={notes}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
