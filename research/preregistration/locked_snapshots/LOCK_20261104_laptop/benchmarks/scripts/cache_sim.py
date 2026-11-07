"""Tiny L2 simulator for the B4 fusion pipeline (model v2) -- produces ESTIMATES only.

Why: the closed-form residency models of v1 (lru_cliff / proportional) ignored two effects the
2026-11-04 smoke run exposed: (1) dirty write-backs of the producer's output are paid by
whichever kernel evicts them, and (2) in steady state what matters is each variant's *whole*
working set (fused x+y vs. materialized x+t+y), not only t.

Model (assumptions are deliberate simplifications, to be falsified):
  S1  one fully-associative cache of `l2_bytes`, 128-B lines (all B4 accesses are full lines)
  S2  replacement policy: "lru" or "random" (two competing hypotheses about the real L2)
  S3  write-allocate without fetch (each warp writes whole lines), write-back on eviction
  S4  kernels stream lines in increasing index order: for line j: read src[j]; write dst[j]
  S5  "isolated" = cache emptied before the window (the read-flush evicts everything; dirty
      lines it evicts are written back *outside* the timed window)
  S6  "steady"   = the call sequence repeated; the 3rd repetition is reported
DRAM bytes in a window = (read misses + dirty evictions during the window) x 128.
"""

from __future__ import annotations

import random
from collections import OrderedDict

LINE = 128


class Cache:
    def __init__(self, capacity_lines: int, policy: str, seed: int = 1):
        if capacity_lines <= 0:
            raise ValueError("capacity must be > 0")
        if policy not in ("lru", "random"):
            raise ValueError(f"unknown policy {policy}")
        self.cap = capacity_lines
        self.policy = policy
        self.reads = 0
        self.writebacks = 0
        if policy == "lru":
            self.lru: OrderedDict[int, bool] = OrderedDict()
        else:
            self.rng = random.Random(seed)
            self.lines: list[int] = []
            self.pos: dict[int, int] = {}
            self.dirty: dict[int, bool] = {}

    def clear(self) -> None:
        """Evict everything (dirty write-backs happen outside the window, not counted)."""
        if self.policy == "lru":
            self.lru.clear()
        else:
            self.lines.clear()
            self.pos.clear()
            self.dirty.clear()

    def reset_counters(self) -> None:
        self.reads = 0
        self.writebacks = 0

    def access(self, line: int, write: bool) -> None:
        if self.policy == "lru":
            d = self.lru
            if line in d:
                d.move_to_end(line)
                if write:
                    d[line] = True
                return
            if not write:
                self.reads += 1
            d[line] = write
            if len(d) > self.cap:
                _, was_dirty = d.popitem(last=False)
                self.writebacks += was_dirty
            return
        # random replacement
        if line in self.pos:
            if write:
                self.dirty[line] = True
            return
        if not write:
            self.reads += 1
        if len(self.lines) >= self.cap:
            r = self.rng.randrange(len(self.lines))
            victim = self.lines[r]
            last = self.lines[-1]
            self.lines[r] = last
            self.pos[last] = r
            self.lines.pop()
            del self.pos[victim]
            self.writebacks += self.dirty.pop(victim)
        self.pos[line] = len(self.lines)
        self.lines.append(line)
        self.dirty[line] = write


def _kernel(cache: Cache, src: int, dst: int, n_lines: int) -> None:
    for j in range(n_lines):
        cache.access(src + j, False)
        cache.access(dst + j, True)


VARIANTS = ("fused", "materialized", "stage2_cold", "stage2_warm")


def simulate_fusion(n_elems: int, l2_bytes: int, policy: str = "lru") -> dict[tuple[str, str], int]:
    """ESTIMATED DRAM bytes per timed window, keyed by (variant, mode).

    mode is "isolated" for all variants, and also "steady" for fused / materialized
    (matching what fusion.cu times).
    """
    n_lines = max(1, (4 * n_elems + LINE - 1) // LINE)
    X, T, Y = 0, n_lines, 2 * n_lines  # disjoint line ranges for x, t, y
    cap = max(1, l2_bytes // LINE)
    seq = {
        "fused": [(X, Y)],
        "materialized": [(X, T), (T, Y)],
        "stage2_cold": [(T, Y)],
    }
    out: dict[tuple[str, str], int] = {}

    def run(calls):
        for src, dst in calls:
            _kernel(c, src, dst, n_lines)

    for v in ("fused", "materialized", "stage2_cold"):
        c = Cache(cap, policy)
        run(seq[v])
        out[(v, "isolated")] = (c.reads + c.writebacks) * LINE
    # stage2_warm: flush, untimed stage1, timed stage2
    c = Cache(cap, policy)
    run([(X, T)])
    c.reset_counters()
    run([(T, Y)])
    out[("stage2_warm", "isolated")] = (c.reads + c.writebacks) * LINE
    # steady: repeat the call sequence; report the 3rd repetition
    for v in ("fused", "materialized"):
        c = Cache(cap, policy)
        for _ in range(2):
            run(seq[v])
        c.reset_counters()
        run(seq[v])
        out[(v, "steady")] = (c.reads + c.writebacks) * LINE
    return out


def ratios(sim: dict[tuple[str, str], int]) -> dict[str, float]:
    """The three DRAM-byte ratios the B4 predictions are stated in."""
    def r(a, b):
        return sim[a] / sim[b] if sim[b] else float("inf") if sim[a] else float("nan")
    return {
        "mat/fused_iso": r(("materialized", "isolated"), ("fused", "isolated")),
        "mat/fused_steady": r(("materialized", "steady"), ("fused", "steady")),
        "warm/cold": r(("stage2_warm", "isolated"), ("stage2_cold", "isolated")),
    }


# ---------------------------------------------------------------------------
# Time model on top of the simulated bytes (ESTIMATE):
#   t(window) = k * tau + DRAM_bytes / BW
# k = kernels in the window (materialized: 2, others: 1); tau = per-kernel overhead (launch,
# ramp-up/tail) that differs between isolated and steady timing; BW = measured memcpy BW.
# ---------------------------------------------------------------------------
KERNELS = {"fused": 1, "materialized": 2, "stage2_cold": 1, "stage2_warm": 1}


def predict_times_us(sim: dict[tuple[str, str], int], tau_iso_us: float, tau_steady_us: float,
                     bw_gbps: float) -> dict[tuple[str, str], float]:
    out = {}
    for (v, mode), b in sim.items():
        tau = tau_iso_us if mode == "isolated" else tau_steady_us
        out[(v, mode)] = KERNELS[v] * tau + b / (bw_gbps * 1e3)
    return out


def calibrate_tau(points: list[tuple[str, str, float, int]], bw_gbps: float) -> dict[str, float]:
    """points: (variant, mode, measured_us, simulated_bytes). Returns the median per-kernel
    residual (measured - bytes/BW) / k for each mode."""
    import statistics
    res: dict[str, list[float]] = {"isolated": [], "steady": []}
    for v, mode, us, b in points:
        res[mode].append((us - b / (bw_gbps * 1e3)) / KERNELS[v])
    return {m: statistics.median(r) for m, r in res.items() if r}
