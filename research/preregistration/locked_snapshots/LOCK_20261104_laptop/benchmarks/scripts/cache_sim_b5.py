"""B5 (multi-sweep stencil) on top of the *unchanged* v2 cache simulator -- ESTIMATES only.

cache_sim.py is locked by earlier pre-registrations, so this module only imports from it.

Line-level model of a stencil sweep (assumption T5): rows r-2..r+2 of the source are reused
within a few KiB, far below L2, so at 128-B line granularity a sweep behaves like a streaming
kernel: read src line j, write dst line j (same as the B4 pointwise kernels). The periodic wrap
(row 0 reads row H-1) touches 1-2 extra rows and is ignored.

Schedules (buffers X, P, Q of B bytes each):
  mat4        X->P, P->Q, Q->P, P->Q      (4 kernels of the 5-point sweep, ping-pong)
  fused2      X->P, P->Q                   (2 kernels of the 13-point composite)
  sweep_cold  P->Q                         (after flush)
  sweep_warm  P->Q                         (after untimed X->P)
"""

from __future__ import annotations

import statistics

from cache_sim import LINE, Cache, _kernel

W = 1024  # must match stencil.cu
BY = 8

# variant -> (kernel type, number of kernels in the timed window)
KERNELS = {"mat4": ("5pt", 4), "fused2": ("13pt", 2), "sweep_cold": ("5pt", 1), "sweep_warm": ("5pt", 1)}


def n_for(l2x: float, l2_bytes: int) -> int:
    """Element count exactly as stencil.cu rounds it (whole 8-row blocks, >= 8 rows)."""
    h = int(l2x * l2_bytes / 4 / W)
    h = max(8, h // BY * BY)
    return h * W


def simulate_stencil(n_elems: int, l2_bytes: int, policy: str = "lru") -> dict[tuple[str, str], int]:
    n_lines = max(1, (4 * n_elems + LINE - 1) // LINE)
    X, P, Q = 0, n_lines, 2 * n_lines
    cap = max(1, l2_bytes // LINE)
    sched = {"mat4": [(X, P), (P, Q), (Q, P), (P, Q)], "fused2": [(X, P), (P, Q)], "sweep_cold": [(P, Q)]}
    out: dict[tuple[str, str], int] = {}

    def run(c, calls):
        for s, d in calls:
            _kernel(c, s, d, n_lines)

    for v, calls in sched.items():
        c = Cache(cap, policy)
        run(c, calls)
        out[(v, "isolated")] = (c.reads + c.writebacks) * LINE
    c = Cache(cap, policy)
    run(c, [(X, P)])
    c.reset_counters()
    run(c, [(P, Q)])
    out[("sweep_warm", "isolated")] = (c.reads + c.writebacks) * LINE
    for v in ("mat4", "fused2"):
        c = Cache(cap, policy)
        for _ in range(2):
            run(c, sched[v])
        c.reset_counters()
        run(c, sched[v])
        out[(v, "steady")] = (c.reads + c.writebacks) * LINE
    return out


def predict_times_us(sim: dict[tuple[str, str], int], tau: dict[tuple[str, str], float],
                     bw_gbps: float) -> dict[tuple[str, str], float]:
    """t = k * tau[(kernel type, mode)] + DRAM bytes / BW."""
    out = {}
    for (v, mode), b in sim.items():
        kt, k = KERNELS[v]
        out[(v, mode)] = k * tau[(kt, mode)] + b / (bw_gbps * 1e3)
    return out


def calibrate_tau(points: list[tuple[str, str, float, int]], bw_gbps: float) -> dict[tuple[str, str], float]:
    """points: (variant, mode, measured_us, simulated_bytes) -> median per-kernel residual per
    (kernel type, mode). Same rule as cache_sim.calibrate_tau, but keyed by kernel type because
    the 5-point and 13-point kernels may stream at different efficiencies."""
    res: dict[tuple[str, str], list[float]] = {}
    for v, mode, us, b in points:
        kt, k = KERNELS[v]
        res.setdefault((kt, mode), []).append((us - b / (bw_gbps * 1e3)) / k)
    return {key: statistics.median(r) for key, r in res.items()}


def ratios(t: dict[tuple[str, str], float]) -> dict[str, float]:
    return {
        "mat4/fused2_iso": t[("mat4", "isolated")] / t[("fused2", "isolated")],
        "mat4/fused2_steady": t[("mat4", "steady")] / t[("fused2", "steady")],
        "warm/cold": t[("sweep_warm", "isolated")] / t[("sweep_cold", "isolated")],
    }
