#!/usr/bin/env python3
"""Minimal repro for the PyTorch issue draft (reports/outreach/02_pytorch_inductor_issue.md).

InductorBenchmarker on CUDA flushes the L2 with zero_() on an L2-sized int32 buffer before every timed call.
Two separate effects follow, and this script shows both on one GPU in about 20 s:
  1. the flush WRITES, so the L2 is left with dirty lines that the timed kernel has to write back to DRAM;
  2. an L2-sized flush is short; if it finishes before Python has enqueued the timed kernel, the GPU queue
     drains, the start event is timestamped early and host dispatch time is timed. A GPU busy-wait right
     after the flush (torch.cuda._sleep) keeps the queue busy and removes this part.
Kernels: a batch-1 fp16 GEMV (112 MiB of weights, memory bound) and a short elementwise kernel (L2/2 in, L2/2
out). Effect 2 shows only when host dispatch takes longer than the flush, i.e. on fast GPUs or slow hosts.

Usage: python repro_inductor_flush_bias.py      (one NVIDIA GPU, ~1.5 GB free)
"""

import random
import statistics

import torch
import torch.nn.functional as F


def time_calls(fn, flush, iters=200, warmup=20):
    """Per-call CUDA-event timing with a flush before every call (the pattern InductorBenchmarker uses)."""
    for _ in range(warmup):
        flush()
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for start, end in ev:
        flush()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    t = [s.elapsed_time(e) * 1e3 for s, e in ev]  # microseconds
    return min(t), statistics.median(t)


def busy_wait_cycles(us: float) -> int:
    """Clock cycles of torch.cuda._sleep that keep the GPU busy for about `us` microseconds."""
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    torch.cuda._sleep(1_000_000)
    e.record()
    torch.cuda.synchronize()
    return int(us / (s.elapsed_time(e) * 1e3 / 1_000_000))


def main():
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    l2 = props.L2_cache_size
    W = (torch.randn(14336, 4096, device="cuda") * 0.02).half()   # 112 MiB of weights, larger than any L2
    x = torch.randn(1, 4096, device="cuda").half()                 # batch-1 decode
    a = torch.randn(l2 // 8, device="cuda")                        # L2/2 bytes of float32
    b = torch.empty_like(a)
    kernels = {"GEMV 112 MiB": lambda: F.linear(x, W), "elementwise L2/2": lambda: torch.mul(a, 2.0, out=b)}
    l2_buf = torch.empty(l2 // 4, dtype=torch.int, device="cuda")                  # as InductorBenchmarker allocates it
    big_buf = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")  # as triton.testing.do_bench
    pad = busy_wait_cycles(300.0)
    ref = "read, 256 MiB torch.sum (stand-in for triton#12556)"
    flushes = {
        "write, L2-sized zero_() (InductorBenchmarker)": lambda: l2_buf.zero_(),
        "write, L2-sized zero_() + GPU busy-wait": lambda: (l2_buf.zero_(), torch.cuda._sleep(pad)),
        "write, 256 MiB zero_() (triton.testing.do_bench)": lambda: big_buf.zero_(),
        ref: lambda: big_buf.view(torch.float32).sum(),
    }
    print(f"{props.name}, L2 {l2 >> 20} MiB, torch {torch.__version__}, CUDA {torch.version.cuda}")
    rng = random.Random(0)
    for kname, fn in kernels.items():
        res = {}
        for _ in range(5):  # interleave the modes in a shuffled order to avoid order effects
            order = list(flushes.items())
            rng.shuffle(order)
            for name, flush in order:
                res.setdefault(name, []).append(time_calls(fn, flush))
        base_min = statistics.median(r[0] for r in res[ref])
        base_med = statistics.median(r[1] for r in res[ref])
        print(f" {kname}:")
        for name, r in res.items():
            mn, md = statistics.median(q for q, _ in r), statistics.median(q for _, q in r)
            print(f"  {name:50s} min {mn:8.2f} us ({mn / base_min - 1:+6.1%})   median {md:8.2f} us ({md / base_med - 1:+6.1%})")
        try:
            from torch._inductor.runtime.benchmarking import benchmarker
            ms = benchmarker.benchmark_gpu(fn)
            print(f"  {type(benchmarker).__name__}.benchmark_gpu(fn) = {ms * 1e3:.2f} us "
                  f"({ms * 1e3 / base_min - 1:+.1%} vs the read-flush min)")
        except Exception as ex:  # noqa: BLE001
            print("  Inductor benchmarker not callable here:", repr(ex)[:200])


if __name__ == "__main__":
    main()
