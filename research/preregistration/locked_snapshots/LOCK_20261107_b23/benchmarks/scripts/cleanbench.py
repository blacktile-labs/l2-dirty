"""cleanbench: CUDA-event timing of a GPU kernel with a clean L2 and a busy GPU queue.

Three timing pitfalls measured in this repository, and what this module does about each:
  1. Write-based L2 flushes (`cache.zero_()` in triton.testing.do_bench and most tools) leave dirty lines that
     the timed kernel must write back (B12-B17).                    -> flush by READING (volatile loads, no store)
  2. A flush that finishes before the host has enqueued the kernel lets the GPU queue drain, so the start
     event fires early and host dispatch time is timed (B18b).      -> a GPU busy-wait ("pad") right before the
                                                                       start event, sized from the measured
                                                                       host dispatch time
  3. A read flush keeps up to about L2/2 of a kernel's OWN write-back out of the timed window (B18).
                                                                     -> mode="steady" rotates over copies of the
                                                                       inputs/outputs, so each call pays the
                                                                       write-back of an earlier call, as in a
                                                                       steady stream of calls
Usage:
    r = bench(lambda: f(x, y))                          # cold: clean, evicted L2
    r = bench(rotate=[lambda i=i: f(xs[i], ys[i]) for i in range(n)], mode="steady")
    print(r.median_us, r.p25_us, r.p75_us, r.warnings)
The read-flush kernel is copied from triton-lang/triton PR #12556 (head ba27327b, MIT licensed).
"""

import statistics
import time
from dataclasses import dataclass, field

TICK_US = 1.024            # CUDA event timestamps on the GPUs we measured fall on this grid
MIN_FLUSH = 256 << 20

try:                       # the statistics helpers below work without torch / triton
    import torch
    import triton
    import triton.language as tl

    # Copied from triton-lang/triton PR #12556 (python/triton/testing.py, head ba27327b), MIT licensed.
    # No `from __future__ import annotations` in this module: Triton must see tl.constexpr, not a string.
    @triton.jit
    def _read_flush(X, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        # Volatile keeps the read even when its result is unused, without a store.
        tl.load(X + off, mask=off < n, volatile=True)
except ImportError:
    torch = None

_US_PER_CYCLE = None
_BUFFERS: dict = {}


def read_flush(buf) -> None:
    """Evict the L2 by reading `buf` (int32 or float32 view) without writing anything."""
    x = buf.view(torch.float32)
    _read_flush[((x.numel() + 1023) // 1024, )](x, x.numel(), 1024)


def flush_buffer(nbytes: int | None = None):
    """A cached int32 buffer of max(256 MiB, 4 x L2) bytes (or `nbytes`) on the current device."""
    l2 = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    nbytes = nbytes or max(MIN_FLUSH, 4 * l2)
    key = (torch.cuda.current_device(), nbytes)
    if key not in _BUFFERS:
        _BUFFERS[key] = torch.zeros(nbytes // 4, dtype=torch.int32, device="cuda")
    return _BUFFERS[key]


def _us_per_cycle() -> float:
    global _US_PER_CYCLE
    if _US_PER_CYCLE is None:
        cyc, out = 2_000_000, []
        for _ in range(5):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            torch.cuda._sleep(cyc)
            e.record()
            torch.cuda.synchronize()
            out.append(s.elapsed_time(e) * 1e3 / cyc)
        _US_PER_CYCLE = statistics.median(out)
    return _US_PER_CYCLE


def pad(us: float) -> None:
    """Keep the GPU busy for about `us` microseconds (enqueued, returns immediately)."""
    torch.cuda._sleep(max(1, int(us / _us_per_cycle())))


def host_dispatch_us(fn, n: int = 30) -> float:
    """Median host time to enqueue start event + fn + end event while the GPU is busy."""
    out = []
    for _ in range(n):
        pad(2000)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        s.record()
        fn()
        e.record()
        out.append((time.perf_counter() - t0) * 1e6)
        torch.cuda.synchronize()
    return statistics.median(out)


def tick_fraction(times_us: list[float], tick: float = TICK_US, tol: float = 0.02) -> float:
    """Fraction of samples within `tol` microseconds of a multiple of the timer tick."""
    if not times_us:
        return 0.0
    return sum(1 for t in times_us if abs(t / tick - round(t / tick)) * tick <= tol) / len(times_us)


def quantiles(times_us: list[float]) -> tuple[float, float, float, float]:
    """(p25, median, p75, p95) with linear interpolation."""
    xs = sorted(times_us)

    def q(p):
        k = (len(xs) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
    return q(0.25), q(0.5), q(0.75), q(0.95)


def warnings_for(times_us: list[float], mode: str) -> list[str]:
    w = []
    p25, p50, p75, _ = quantiles(times_us)
    if p50 < 30 * TICK_US and tick_fraction(times_us) > 0.5:
        w.append(f"median {p50:.1f} us is within ~30 timer ticks ({TICK_US} us): differences under ~5 % are not resolved")
    if p50 > 0 and (p75 - p25) / p50 > 0.10:
        w.append(f"IQR is {(p75 - p25) / p50:.0%} of the median: noisy, increase reps or check clocks")
    if mode == "cold":
        w.append("cold mode: up to ~L2/2 of this kernel's own write-back is not timed (see mode='steady')")
    return w


@dataclass
class Result:
    mode: str
    times_us: list[float]
    median_us: float
    mean_us: float
    p25_us: float
    p75_us: float
    p95_us: float
    pad_us: float
    host_dispatch_us: float
    flush_bytes: int
    warnings: list[str] = field(default_factory=list)


def bench(fn=None, *, rotate=None, mode: str = "cold", reps: int = 100, warmup: int = 10,
          flush_bytes: int | None = None, pad_us: float | None = None) -> Result:
    """Time `fn` (mode="cold": read flush + pad before every call) or the callables in `rotate`
    (mode="steady": no flush, cycle over copies whose total footprint should be >= 4 x L2; pad before every call)."""
    if mode not in ("cold", "steady"):
        raise ValueError("mode must be 'cold' or 'steady'")
    calls = list(rotate) if mode == "steady" else [fn]
    if mode == "steady" and len(calls) < 2:
        raise ValueError("steady mode needs >= 2 rotated callables covering >= 4 x L2 in total")
    if mode == "cold" and fn is None:
        raise ValueError("cold mode needs fn")
    buf = flush_buffer(flush_bytes) if mode == "cold" else None
    h = host_dispatch_us(calls[0])
    pad_us = pad_us if pad_us is not None else min(5000.0, max(100.0, 3.0 * h))
    j = 0                                                       # one running index: rotation never restarts
    for _ in range(warmup + len(calls)):
        if buf is not None:
            read_flush(buf)
        calls[j % len(calls)]()
        j += 1
    torch.cuda.synchronize()
    ev = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if buf is not None:
            read_flush(buf)
        pad(pad_us)
        s.record()
        calls[j % len(calls)]()
        e.record()
        j += 1
        ev.append((s, e))
    torch.cuda.synchronize()
    ts = [s.elapsed_time(e) * 1e3 for s, e in ev]
    p25, p50, p75, p95 = quantiles(ts)
    return Result(mode, ts, p50, statistics.fmean(ts), p25, p75, p95, pad_us, h,
                  buf.numel() * 4 if buf is not None else 0, warnings_for(ts, mode))
