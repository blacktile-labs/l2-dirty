#!/usr/bin/env python3
"""B7 -- a real PyTorch op chain: eager (unfused, 3 kernels) vs TorchInductor (fused, 1 kernel).

Chain (transformer-MLP epilogue):  y = gelu(x + b) + r,   x, r, y: [M, 4096] fp32,  b: [4096]
  eager     add(x, b) -> t1;  gelu(t1) -> t2;  add(t2, r) -> y    (PyTorch's caching allocator
            places t1, t2, y; t1 is freed before y is allocated, exactly as in `gelu(x + b) + r`)
  compiled  torch.compile(chain): one Triton kernel reading x, b, r and writing y
Sizes are given as multiples of the device L2 (bytes of ONE [M, 4096] tensor / L2).

--mode counters  (run under ncu --replay-mode application; bytes are read from the profiler)
  per size: flush, eager | flush, compiled | flush, eager x 3 | flush, compiled x 3
  and writes --plan JSON: every measured launch with the byte ranges it reads and writes (real
  tensor addresses), so counter_b7.py can replay the exact address stream through the simulators.
--mode counters --profiler-ranges  (B7r; run under ncu --replay-mode app-range)
  the same launches and plan, but each segment (the chain calls after one flush) is bracketed by
  cudaProfilerStart/Stop, so ncu measures the segment's launches back to back as ONE range instead
  of one kernel at a time with profiler gaps in between. Each range starts with a spin kernel
  (torch.cuda._sleep, no memory access) so the CPU has enqueued every launch of the segment before
  the GPU reaches them: no CPU-dispatch gaps either. After the first flush of every size an extra
  CONTROL range holds only the spin kernel: it measures the device's background DRAM traffic (on this
  WSL2 laptop ~12 KB/ms of reads, bursty, no writes; scratch probe 2026-11-05). Spin kernels are not
  in the plan.
  --range-gate flag (B7r2): instead of a fixed-length spin, each range starts with a GATE kernel
  that polls a flag word until the CPU has enqueued the whole segment and releases it (a 1-element
  fill on a side stream). The launches then run back to back however slow the CPU was (B7r failed
  its back-to-back check with the spin). The flag words are a view into the flush buffer: no new
  allocation, so tensor addresses and the plan are unchanged. Control ranges hold the gate for
  CONTROL_HOLD_S. gate_iters per range is recorded; GATE_MAX_ITER means the gate timed out.
--mode time      (no profiler) CUDA-event timing of the same chains:
  graph   : each chain captured once in a CUDA graph -> GPU-side time without Python dispatch
            isolated = flush then 1 replay per event pair; steady = 20 replays per event pair
  eager-e2e / compiled-e2e : plain Python calls, 20 back-to-back calls per event pair (includes
            CPU dispatch; this is what a user's loop sees)
Correctness (every size, before anything is measured): compiled == eager (assert_close,
rtol = atol = 1e-5; erf implementations differ so not bit-exact) and eager vs a float64 CPU
reference on the first 8 rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time

# Inductor's on-disk cache must be warm before profiling: ncu application replay reruns the whole
# process per pass and needs an identical launch sequence in every pass (no compile-time launches).
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/dmkl-inductor"))
# Compile in-process: Inductor's subprocess worker pool does not exit under ncu (hangs the replay).
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

H = 4096
FLUSH_BYTES = 16 << 20
GATE_MAX_ITER = 2_000_000  # ~0.3 s of polling at ~0.15 us per volatile L2 load (prototype 2026-11-05)
CONTROL_HOLD_S = 0.0015    # ~ the median segment enqueue time in B7r (1.1 ms)
SLEEP_CYCLES = 2_000_000  # ~1.1-1.5 ms at the laptop's 1.3-1.8 GHz SM clock: longer than enqueueing the
                          # longest segment (9 eager launches), short to limit background traffic


@triton.jit
def flush_l2_read(buf_ptr, n, sink_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(buf_ptr + offs, mask=offs < n, other=0.0)
    s = tl.sum(v, axis=0)
    if s == 123456789.0:  # the buffer is all zeros: never stores, but the loads cannot be elided
        tl.store(sink_ptr, s)


@triton.jit
def gate_wait(flag_ptr, max_iter):
    i = 0
    v = tl.load(flag_ptr, volatile=True)
    while (v == 0) & (i < max_iter):
        v = tl.load(flag_ptr, volatile=True)
        i += 1
    tl.store(flag_ptr + 1, i)


class Gate:
    """Holds the stream until release(); flag words live in the (all-zero) flush buffer."""

    def __init__(self, flusher):
        self.flag = flusher.buf.view(torch.int32)[:2]
        self.side = torch.cuda.Stream()
        with torch.cuda.stream(self.side):  # warm up: load the fill kernel and compile the gate
            self.flag[0:1].fill_(1)
        torch.cuda.synchronize()
        gate_wait[(1,)](self.flag, GATE_MAX_ITER)
        torch.cuda.synchronize()
        self.flag.zero_()
        torch.cuda.synchronize()

    def hold(self):
        gate_wait[(1,)](self.flag, GATE_MAX_ITER)

    def release(self):
        with torch.cuda.stream(self.side):
            self.flag[0:1].fill_(1)

    def iters_and_reset(self) -> int:
        n = int(self.flag[1].item())
        self.flag.zero_()
        torch.cuda.synchronize()
        return n


class Flusher:
    def __init__(self, l2_bytes: int):
        self.n = max(4 * l2_bytes, FLUSH_BYTES) // 4
        self.buf = torch.zeros(self.n, device="cuda", dtype=torch.float32)
        self.sink = torch.zeros(1, device="cuda", dtype=torch.float32)

    def __call__(self):
        grid = (triton.cdiv(self.n, 4096),)
        flush_l2_read[grid](self.buf, self.n, self.sink, BLOCK=4096)

    def range(self):
        return [self.buf.data_ptr(), self.n * 4]


def chain(x, b, r):
    return F.gelu(x + b) + r


def eager_traced(x, b, r, log):
    """== chain(x, b, r) step by step (same kernels, same allocation order) + address log."""
    B = x.numel() * 4
    t1 = x + b
    log.append({"kernel": "eager_add1", "reads": [[x.data_ptr(), B], [b.data_ptr(), b.numel() * 4]],
                "writes": [[t1.data_ptr(), B]]})
    t2 = F.gelu(t1)
    log.append({"kernel": "eager_gelu", "reads": [[t1.data_ptr(), B]], "writes": [[t2.data_ptr(), B]]})
    del t1
    y = t2 + r
    log.append({"kernel": "eager_add2", "reads": [[t2.data_ptr(), B], [r.data_ptr(), B]],
                "writes": [[y.data_ptr(), B]]})
    return y


def compiled_traced(cfn, x, b, r, log):
    B = x.numel() * 4
    y = cfn(x, b, r)
    log.append({"kernel": "inductor_fused", "reads": [[x.data_ptr(), B], [r.data_ptr(), B],
                                                      [b.data_ptr(), b.numel() * 4]],
                "writes": [[y.data_ptr(), B]]})
    return y


def rows_for(f: float, l2: int) -> int:
    return max(1, int(round(f * l2 / (4 * H))))


def make_inputs(m: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, H, device="cuda", generator=g)
    r = torch.randn(m, H, device="cuda", generator=g)
    b = torch.randn(H, device="cuda", generator=g)
    return x, b, r


def check(cfn, x, b, r) -> str | None:
    ye = chain(x, b, r)
    yc = cfn(x, b, r)
    try:
        torch.testing.assert_close(yc, ye, rtol=1e-5, atol=1e-5)
        xs, bs, rs = (t.double().cpu() for t in (x[:8], b, r[:8]))
        ref = F.gelu(xs + bs) + rs
        torch.testing.assert_close(ye[:8].double().cpu(), ref, rtol=1e-5, atol=1e-5)
    except AssertionError as e:
        return str(e).splitlines()[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("counters", "time"), required=True)
    ap.add_argument("--sizes-l2x", default="0.125,0.25,0.5,1,2,4")
    ap.add_argument("--smoke", action="store_true", help="sizes 0.0625,1.5; time: reps 5")
    ap.add_argument("--profiler-ranges", action="store_true",
                    help="counters mode: one cudaProfilerStart/Stop range per segment (B7r)")
    ap.add_argument("--range-gate", choices=("spin", "flag"), default="spin",
                    help="how a range holds the GPU until the segment is enqueued (B7r spin, B7r2 flag)")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steady-iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--plan", default="opchain_plan.json")
    ap.add_argument("--csv", default="opchain_time.csv")
    ap.add_argument("--meta", default="opchain_meta.json")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 3
    torch.backends.cuda.matmul.allow_tf32 = False
    # dynamic=False compiles one graph per shape; dynamo's default recompile_limit (8) silently falls
    # back to eager at the 9th shape (B7 run 20261105-085346, D-044). Raise it and fail loudly instead.
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    sizes = [0.0625, 1.5] if a.smoke else [float(s) for s in a.sizes_l2x.split(",")]
    reps = 5 if a.smoke else a.reps
    prop = torch.cuda.get_device_properties(0)
    l2 = prop.L2_cache_size
    flush = Flusher(l2)
    cfn = torch.compile(chain, dynamic=False)
    meta = {"benchmark": "torch_opchain", "mode": a.mode, "smoke": a.smoke, "device": prop.name,
            "l2_bytes": l2, "torch": torch.__version__, "triton": triton.__version__,
            "cuda_runtime": torch.version.cuda, "H": H, "sizes_l2x": sizes, "seed": a.seed,
            "flush_bytes": flush.n * 4, "steady_calls_counters": 3, "correctness": {},
            "dynamo_recompile_limit": torch._dynamo.config.recompile_limit}

    # ---- compile + correctness for every size, before anything is measured ----
    inputs = {}
    for f in sizes:
        m = rows_for(f, l2)
        x, b, r = make_inputs(m, a.seed)
        err = check(cfn, x, b, r)
        meta["correctness"][f"l2x={f:g};M={m}"] = err or "ok"
        if err:
            print(f"FAIL l2x={f:g}: {err}", file=sys.stderr)
        inputs[f] = (m, x, b, r)
    torch.cuda.synchronize()
    fails = sum(v != "ok" for v in meta["correctness"].values())
    meta["correctness_failures"] = fails
    if fails:
        json.dump(meta, open(a.meta, "w"), indent=1)
        return 1

    if a.mode == "counters":
        plan = []
        ranges = []  # B7r: one entry per profiler range, in issue order

        gate = Gate(flush) if a.profiler_ranges and a.range_gate == "flag" else None

        def range_begin():
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            if gate:
                gate.hold()
            else:
                torch.cuda._sleep(SLEEP_CYCLES)

        def range_end():
            if gate:
                gate.release()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            return gate.iters_and_reset() if gate else None

        for f in sizes:
            m, x, b, r = inputs[f]
            seq = [("eager", "isolated", 1), ("compiled", "isolated", 1),
                   ("eager", "steady", 3), ("compiled", "steady", 3)]
            for variant, mode, calls in seq:
                flush()
                plan.append({"l2x": f, "M": m, "variant": variant, "mode": mode, "kernel": "flush",
                             "reads": [flush.range()], "writes": []})
                if a.profiler_ranges and (variant, mode) == seq[0][:2]:
                    range_begin()
                    if gate:
                        time.sleep(CONTROL_HOLD_S)
                    gi = range_end()
                    ranges.append({"kind": "control", "l2x": f, **({"gate_iters": gi} if gate else {})})
                if a.profiler_ranges:
                    range_begin()
                    t0 = time.perf_counter()
                log: list = []
                y = None
                for _ in range(calls):
                    y = eager_traced(x, b, r, log) if variant == "eager" else compiled_traced(cfn, x, b, r, log)
                if a.profiler_ranges:
                    enq = (time.perf_counter() - t0) * 1e6  # CPU time to enqueue the segment; must be < spin
                    gi = range_end()
                    ranges.append({"kind": "segment", "l2x": f, "variant": variant, "mode": mode,
                                   "first_launch": len(plan), "n_launches": len(log), "enqueue_us": round(enq, 1),
                                   **({"gate_iters": gi} if gate else {})})
                del y
                for L in log:
                    plan.append({"l2x": f, "M": m, "variant": variant, "mode": mode, **L})
            torch.cuda.synchronize()
        meta["profiler_ranges"] = a.profiler_ranges
        if a.profiler_ranges:
            meta["range_gate"], meta["ranges"] = a.range_gate, ranges
            if gate:
                meta["gate_max_iter"], meta["control_hold_s"] = GATE_MAX_ITER, CONTROL_HOLD_S
            else:
                meta["sleep_cycles"] = SLEEP_CYCLES
        json.dump({"meta": meta, "launches": plan}, open(a.plan, "w"))
        print(f"counters: {len(plan)} measured launches over {len(sizes)} sizes; plan -> {a.plan}"
              + (f"; {len(ranges)} profiler ranges" if a.profiler_ranges else ""))
    else:
        rows = []
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def timed(fn, iters):
            out = []
            for _ in range(reps):
                e0.record()
                for _ in range(iters):
                    fn()
                e1.record()
                e1.synchronize()
                out.append(e0.elapsed_time(e1) * 1000 / iters)  # us per call
            return out

        # One CUDA-graph replay costs ~50 us of CPU time on WSL2 (smoke, 2026-11-05), more than a small
        # kernel; so each graph holds many chains and the replay cost is amortized:
        #   graph_steady   : one graph = GRAPH_STEADY back-to-back chains           -> us per chain
        #   graph_isolated : one graph = GRAPH_ISO x (flush, chain) minus one graph = GRAPH_ISO x flush
        GRAPH_STEADY, GRAPH_ISO = 50, 20

        def capture(body):
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
            torch.cuda.synchronize()
            return g

        def replay_us(g, per):
            for _ in range(a.warmup):
                g.replay()
            out = []
            for _ in range(reps):
                e0.record()
                g.replay()
                e1.record()
                e1.synchronize()
                out.append(e0.elapsed_time(e1) * 1000 / per)
            return out

        def flush_only():
            for _ in range(GRAPH_ISO):
                flush()

        for f in sizes:
            m, x, b, r = inputs[f]
            g_flush = capture(flush_only)
            t_flush = replay_us(g_flush, GRAPH_ISO)
            rows += [(f, m, "flush", "graph_flush_only", GRAPH_ISO, i, t) for i, t in enumerate(t_flush)]
            for name, fn in (("eager", chain), ("compiled", cfn)):
                for _ in range(3):  # warm up outside capture
                    fn(x, b, r)

                def steady_body(fn=fn):
                    for _ in range(GRAPH_STEADY):
                        fn(x, b, r)

                def iso_body(fn=fn):
                    for _ in range(GRAPH_ISO):
                        flush()
                        fn(x, b, r)

                st = replay_us(capture(steady_body), GRAPH_STEADY)
                rows += [(f, m, name, "graph_steady", GRAPH_STEADY, i, t) for i, t in enumerate(st)]
                fc = replay_us(capture(iso_body), GRAPH_ISO)
                ft = statistics.median(t_flush)
                rows += [(f, m, name, "graph_isolated", GRAPH_ISO, i, t - ft) for i, t in enumerate(fc)]
            for name, fn in (("eager", chain), ("compiled", cfn)):
                for _ in range(a.warmup):
                    fn(x, b, r)
                st = timed(lambda: fn(x, b, r), a.steady_iters)
                rows += [(f, m, name, "e2e_steady", a.steady_iters, i, t) for i, t in enumerate(st)]
            med = {(n, md): statistics.median(t for (ff, _, nn, mm, _, _, t) in rows if ff == f and nn == n and mm == md)
                   for n in ("eager", "compiled") for md in ("graph_isolated", "graph_steady", "e2e_steady")}
            print(f"l2x {f:<7g} M {m:<5} " + "  ".join(
                f"{md}: {med[('eager', md)]:7.1f}/{med[('compiled', md)]:7.1f} us ({med[('eager', md)] / med[('compiled', md)]:.2f}x)"
                for md in ("graph_isolated", "graph_steady", "e2e_steady")))
            torch.cuda.synchronize()
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["l2x", "M", "variant", "timing", "iters", "rep", "us_per_call"])
            w.writerows(rows)
        meta["reps"], meta["warmup"], meta["steady_iters"] = reps, a.warmup, a.steady_iters
    json.dump(meta, open(a.meta, "w"), indent=1)
    print(f"correctness_failures={fails}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
