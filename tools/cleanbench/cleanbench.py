"""cleanbench: CUDA-event timing of a GPU kernel with a clean L2 and a busy GPU queue.

Three timing pitfalls, and what this module does about each:
  1. Write-based L2 flushes (`cache.zero_()` in triton.testing.do_bench and most tools) leave dirty lines that the
     timed kernel must write back.                                  -> flush by READING (volatile loads, no store)
  2. A flush that finishes before the host has enqueued the kernel lets the GPU queue drain, so the start event
     fires early and host dispatch time is timed. This is a known CUDA-event pitfall: NVIDIA's nvbench holds the
     stream with a `blocking_kernel` for the same reason.         -> a GPU busy-wait ("pad") right before the start
                                                                       event, sized from the measured host dispatch
                                                                       time (a simpler, fixed-length stand-in)
  3. A read flush leaves a kernel's OWN output dirty, so its write-back happens during the next flush and is never
     timed (about L2/2 of it on an RTX 3060 Laptop; nearly a 64 MiB output on an L40S).
                                                                    -> mode="steady" rotates over copies of the
                                                                       inputs/outputs, so each call pays the
                                                                       write-back of an earlier call
Usage:
    r = bench(lambda: f(x, y))                          # cold: clean, evicted L2
    r = bench(rotate=[lambda i=i: f(xs[i], ys[i]) for i in range(n)], mode="steady")
    r = bench_graph([lambda i=i: f(xs[i], ys[i]) for i in range(n)])     # kernel time from CUPTI, see section 4
    rs = compare({"a": [...], "b": [...]})                                # candidates in interleaved rounds
    print(r.median_us, r.p25_us, r.p75_us, r.warnings)
The read-flush kernel is copied from triton-lang/triton PR #12556 (head ba27327b, MIT licensed).
This is the packaged copy; benchmarks/scripts/cleanbench.py is the version locked for experiments B18c-B21.
"""

import math
import re
import statistics
import subprocess
import time
from contextlib import contextmanager
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

    @triton.jit
    def _cb_marker(X):
        # One tiny kernel with a unique name: CUPTI records before it (warm-up) are dropped.
        tl.load(X, volatile=True)
except ImportError:
    torch = None

_US_PER_CYCLE: dict = {}       # device -> us per GPU cycle of torch.cuda._sleep (recalibrate with recalibrate_pad())
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
    """Calibrated per device. The busy-wait counts SM cycles, so its length follows the SM clock: if the clock has
    changed a lot since calibration, call recalibrate_pad(). The pad only needs to be long enough, not exact."""
    dev = torch.cuda.current_device()
    if dev not in _US_PER_CYCLE:
        cyc, out = 2_000_000, []
        for _ in range(5):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            torch.cuda._sleep(cyc)
            e.record()
            torch.cuda.synchronize()
            out.append(s.elapsed_time(e) * 1e3 / cyc)
        _US_PER_CYCLE[dev] = statistics.median(out)
    return _US_PER_CYCLE[dev]


def recalibrate_pad() -> None:
    """Forget the busy-wait calibration of the current device (e.g. after a clock change)."""
    _US_PER_CYCLE.pop(torch.cuda.current_device(), None)


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
    quality: dict = field(default_factory=dict)          # bench_graph / compare: the gates of quality_gates()


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



# ---------------------------------------------------------------------------------------------------------------------
# 4. CUDA events time more than the kernel: they add a fixed 2-6 us per call (launch and event work), which is up to
#    +40 % for a 10 us kernel. bench_graph() times the kernel itself from the GPU's own start/end timestamps (CUPTI
#    activity records via torch.profiler; no counter permission needed), over a CUDA graph that rotates through copies
#    of the inputs (so inputs are cold, as in a model step) after >= warm_s of sustained replays (so clocks are in their
#    loaded state). Validated in B23b (RTX 3060 Laptop, SM clock locked): within 0.79 % of the same kernel's time inside
#    a CUDA-graph decode step for all 12 kernels tested; event timing (bench cold) was +1.3 to +13.3 %.
#    The clock must be steady for 1 %: lock it (locked_sm_clock) or check the clock-event counters (clock_events).
# Prior art: FlashInfer bench_gpu_time_with_cupti (CUPTI timing), Triton testing.set_gpu_clock (clock locking),
# nvbench (throttle detection).
# ---------------------------------------------------------------------------------------------------------------------
MARK_NAME, NAME_LEN = "_cb_marker", 160
_MARK = None
SLOWDOWN_COUNTERS = ("SW Thermal Slowdown", "HW Thermal Slowdown", "HW Power Braking", "SW Power Capping")


def after_marker(recs, mark=MARK_NAME):
    """CUPTI records (name, start ns, duration ns) after the last marker record; None if there is no marker."""
    idx = [i for i, r in enumerate(recs) if mark in r[0]]
    return recs[idx[-1] + 1:] if idx else None


def records_ok(recs, expect, reps) -> bool:
    """True if recs are exactly `reps` repetitions of the expected kernel-name sequence `expect`."""
    if not expect or recs is None or len(recs) != reps * len(expect):
        return False
    return all(r[0][:NAME_LEN] == expect[i % len(expect)] for i, r in enumerate(recs))


def call_times(recs, k):
    """Durations (us) of consecutive calls of k kernels each (summed over the call's kernels)."""
    if k <= 0 or len(recs) % k:
        return []
    return [sum(r[2] for r in recs[i:i + k]) / 1e3 for i in range(0, len(recs), k)]


def _mark():
    global _MARK
    if _MARK is None:
        _MARK = torch.zeros(1, device="cuda")
    _cb_marker[(1, )](_MARK)


def _records(prof):
    out = [(e.name(), e.start_ns(), e.duration_ns()) for e in prof.profiler.kineto_results.events()
           if e.device_type().name == "CUDA"]
    return sorted(out, key=lambda r: r[1])


def _kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        _mark()
        fn()
        torch.cuda.synchronize()
    return [r[0][:NAME_LEN] for r in (after_marker(_records(p)) or [])]


def _capture(fns):
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
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(5):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return g, st.elapsed_time(en) * 1e3 / 5


def _keep_busy(g, t_rep, seconds, chunk_s=0.02):
    """Replay g for >= `seconds`, in chunks of ~chunk_s of GPU work, never more than ~2 chunks queued."""
    per, t0, prev = max(1, math.ceil(chunk_s * 1e6 / t_rep)), time.perf_counter(), None
    while True:
        for _ in range(per):
            g.replay()
        ev = torch.cuda.Event()
        ev.record()
        if prev is not None:
            prev.synchronize()
        prev = ev
        if time.perf_counter() - t0 >= seconds:
            return


def clock_events(device: int = 0) -> dict:
    """Cumulative clock-event counters (us) from `nvidia-smi -q -d PERFORMANCE` ({} if unavailable)."""
    try:
        text = subprocess.run(["nvidia-smi", "-i", str(device), "-q", "-d", "PERFORMANCE"], capture_output=True,
                              text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    sec = text.split("Clocks Event Reasons Counters", 1)
    out = {}
    for line in (sec[1].splitlines()[1:] if len(sec) == 2 else []):
        m = re.match(r"\s+([A-Za-z ]+?)\s+:\s+(\d+) us", line)
        if not m:
            break
        out[m.group(1)] = int(m.group(2))
    return out


@contextmanager
def locked_sm_clock(mhz: int, device: int = 0):
    """Try to lock the SM clock at `mhz` (`nvidia-smi -lgc`; needs root / Administrator). Yields True if locked, False
    if not permitted; always resets (`nvidia-smi -rgc`) on exit if it locked. Pick a clock the GPU can hold under
    sustained load (below where it throttles). Same idea as triton.testing.set_gpu_clock."""
    r = subprocess.run(["nvidia-smi", "-i", str(device), "-lgc", f"{mhz},{mhz}"], capture_output=True, text=True)
    ok = r.returncode == 0 and "clocks set" in (r.stdout + r.stderr).lower()
    try:
        yield ok
    finally:
        if ok:
            subprocess.run(["nvidia-smi", "-i", str(device), "-rgc"], capture_output=True, text=True)


def quality_gates(records_ok: bool, tracing_effect: float | None, clock_before: dict, clock_after: dict,
                  footprint_bytes: int | None, l2_bytes: int, trace_tol: float = 0.005) -> dict:
    """The conditions under which bench_graph was validated to 1 % (report section 9.1, one clock-locked GPU):
    every CUPTI segment held exactly the expected kernels; tracing changed the graph's replay time by <= trace_tol;
    the driver's clock-event counters were readable and did not grow during the measurement; the rotated inputs
    covered >= 4 x L2. `validated` is True only if all hold; otherwise treat the number as exploratory."""
    slowed = {k: clock_after.get(k, 0) - clock_before.get(k, 0) for k in SLOWDOWN_COUNTERS
              if clock_after.get(k, 0) > clock_before.get(k, 0)}
    g = {"records_ok": bool(records_ok),
         "tracing_effect": tracing_effect,
         "tracing_ok": tracing_effect is not None and abs(tracing_effect) <= trace_tol,
         "clock_counters_available": bool(clock_before) and bool(clock_after),
         "clock_slowdown_us": slowed,
         "footprint_ok": None if footprint_bytes is None else footprint_bytes >= 4 * l2_bytes}
    g["clock_ok"] = g["clock_counters_available"] and not slowed
    g["validated"] = g["records_ok"] and g["tracing_ok"] and g["clock_ok"] and g["footprint_ok"] is True
    return g


def bench_graph(rotate, *, warm_s: float = 1.0, reps: int = 40, rounds: int = 3,
                footprint_bytes: int | None = None, seed: int = 0) -> Result:
    """Kernel time per call from CUPTI timestamps, over a CUDA graph that calls the callables in `rotate` (copies of the
    inputs whose total footprint is >= 4 x L2; pass footprint_bytes so this is checked) once each, replayed back to
    back after >= warm_s of sustained replays. See compare() for the gates in Result.quality."""
    return compare({"_": rotate}, warm_s=warm_s, reps=reps, rounds=rounds,
                   footprints={"_": footprint_bytes}, seed=seed)["_"]


def compare(candidates: dict, *, warm_s: float = 1.0, reps: int = 40, rounds: int = 3, footprints: dict | None = None,
            seed: int = 0, trace_tol: float = 0.005) -> dict:
    """{name: rotate list} -> {name: Result}. Each round times every candidate once, in a seeded random order (so a
    drift in clocks or temperature does not always favour the same position). Per candidate and round: >= warm_s of
    untraced replays, a CUPTI-traced segment of `reps` replays (CUDA events around each replay), then the same number
    of untraced replays with events, which gives the tracing effect. Assumes each callable runs its kernels on the
    current stream without overlap; its time is the sum of its kernels' durations. Result.quality holds the gates of
    quality_gates(); a result with quality["validated"] False is exploratory."""
    import random

    from torch.profiler import ProfilerActivity, profile
    footprints = footprints or {}
    l2 = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    prep = {}
    for name, rot in candidates.items():
        rot = list(rot)
        if len(rot) < 2:
            raise ValueError(f"{name}: rotate needs >= 2 callables over copies covering >= 4 x L2")
        names = _kernel_names(rot[0])
        if not names:
            raise RuntimeError(f"{name}: no GPU kernel recorded for one call")
        g, t_rep = _capture(rot)
        prep[name] = {"g": g, "t": t_rep, "names": names, "n": len(rot), "times": [], "ok": True, "traced": [],
                      "untraced": [], "round_medians": [], "positions": []}
    rng = random.Random(seed)
    before = clock_events()
    for _ in range(rounds):
        order = list(prep)
        rng.shuffle(order)
        for pos, name in enumerate(order):
            c = prep[name]
            _keep_busy(c["g"], c["t"], warm_s)
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                _keep_busy(c["g"], c["t"], 0.05)
                _mark()
                for st, en in ev:
                    st.record()
                    c["g"].replay()
                    en.record()
                torch.cuda.synchronize()
            recs = after_marker(_records(p))
            ok = records_ok(recs, c["names"] * c["n"], reps)
            c["ok"] &= ok
            ts = call_times(recs, len(c["names"])) if ok else []
            c["times"] += ts
            c["round_medians"].append(statistics.median(ts) if ts else None)
            c["positions"].append(pos)
            c["traced"] += [st.elapsed_time(en) * 1e3 for st, en in ev]
            _keep_busy(c["g"], c["t"], 0.05)
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
            for st, en in ev:
                st.record()
                c["g"].replay()
                en.record()
            torch.cuda.synchronize()
            c["untraced"] += [st.elapsed_time(en) * 1e3 for st, en in ev]
    after = clock_events()
    out = {}
    for name, c in prep.items():
        te = statistics.median(c["traced"]) / statistics.median(c["untraced"]) - 1 if c["untraced"] else None
        q = quality_gates(c["ok"], te, before, after, footprints.get(name), l2, trace_tol)
        q.update({"round_medians_us": c["round_medians"], "positions": c["positions"]})
        w = []
        if not q["records_ok"]:
            w.append("CUPTI records did not match the expected kernels in some rounds; those rounds were dropped")
        if not q["tracing_ok"]:
            w.append(f"tracing changed the replay time by {te:+.2%} (limit {trace_tol:.1%}): CUPTI perturbs this graph")
        if not q["clock_counters_available"]:
            w.append("clock-event counters unavailable: clock stability not checked")
        elif not q["clock_ok"]:
            w.append(f"the GPU slowed its clock during the measurement ({q['clock_slowdown_us']} us): timings are not "
                     "steady; lock the SM clock (locked_sm_clock) or cool the GPU")
        if q["footprint_ok"] is None:
            w.append("footprint not given: pass footprint_bytes to check that the rotation covers >= 4 x L2")
        elif not q["footprint_ok"]:
            w.append("the rotated inputs cover < 4 x L2: calls may find their inputs in L2")
        if not q["validated"]:
            w.append("exploratory: not all conditions of the 1 % validation hold (see quality)")
        ts = c["times"]
        if not ts:
            out[name] = Result("graph", [], float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                               0.0, 0.0, 0, w, q)
            continue
        p25, p50, p75, p95 = quantiles(ts)
        if p50 > 0 and (p75 - p25) / p50 > 0.10:
            w.append(f"IQR is {(p75 - p25) / p50:.0%} of the median: noisy, check clocks")
        out[name] = Result("graph", ts, p50, statistics.fmean(ts), p25, p75, p95, 0.0, 0.0, 0, w, q)
    return out
