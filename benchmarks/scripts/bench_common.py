"""Dependency-free helpers for the data-move-kernel-lab harness.

Three kinds of numbers flow through this module and must never be mixed up:

* MEASURED  -- read from benchmark CSV files written by a CUDA binary.
* ESTIMATED -- produced by the ``model_*`` functions below from stated
               assumptions. They are predictions to be falsified, not results.
* DERIVED   -- arithmetic on measured numbers (e.g. bytes / median time).

Python >= 3.10, standard library only.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Statistics (on MEASURED samples)
# ---------------------------------------------------------------------------


def percentile(samples: list[float], q: float) -> float:
    """Percentile with linear interpolation between closest ranks.

    Same definition as numpy.percentile(method="linear"). q in [0, 100].
    """
    if not samples:
        raise ValueError("percentile of empty sample")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    s = sorted(samples)
    pos = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


@dataclass(frozen=True)
class Summary:
    n: int
    median: float
    p95: float
    min: float
    max: float
    mean: float
    stdev: float  # sample standard deviation; 0.0 when n == 1

    @property
    def cv(self) -> float:
        """Coefficient of variation (stdev / mean)."""
        return self.stdev / self.mean if self.mean > 0 else float("nan")


def summarize(samples: list[float]) -> Summary:
    if not samples:
        raise ValueError("summarize of empty sample")
    return Summary(
        n=len(samples),
        median=statistics.median(samples),
        p95=percentile(samples, 95.0),
        min=min(samples),
        max=max(samples),
        mean=statistics.fmean(samples),
        stdev=statistics.stdev(samples) if len(samples) > 1 else 0.0,
    )


# ---------------------------------------------------------------------------
# Traffic model for copy_stride (ESTIMATES)
#
# Kernel: dst[i] = src[i * stride], i in [0, n_out), n_out = n_src // stride.
# Assumptions (stated so they can be falsified):
#   A1  src and dst are aligned to the chunk size g (cudaMalloc: 256 B).
#   A2  No reuse across repetitions: L2 is flushed before every timed rep.
#   A3  Every g-byte chunk that contains at least one accessed element is
#       transferred exactly once (no thrashing, no extra prefetch beyond g).
#   A4  elem_bytes <= g and g % elem_bytes == 0, so no element straddles chunks.
#   A5  Writes are contiguous and fully cover their chunks (no read-for-ownership).
# g = 32 B models L1->L2 *sector* traffic. The DRAM fetch granularity on a given
# GPU may be 32, 64 or 128 B (cudaLimitMaxL2FetchGranularity is a hint), so the
# DRAM estimate is reported for several g and the measurement decides.
# ---------------------------------------------------------------------------


def _touched_bytes(n_elems: int, stride: int, elem_bytes: int, g: int) -> int:
    if n_elems <= 0:
        return 0
    if stride < 1 or elem_bytes < 1 or g < elem_bytes or g % elem_bytes:
        raise ValueError("model assumptions A4 violated")
    span = stride * elem_bytes
    if span >= g:
        return n_elems * g  # every element lives in its own chunk
    last_byte = (n_elems - 1) * span + elem_bytes  # exclusive end of last element
    return math.ceil(last_byte / g) * g  # every chunk in range holds an element


def model_strided_read_bytes(n_src: int, stride: int, elem_bytes: int = 4, g: int = 32) -> int:
    """ESTIMATED read bytes at chunk granularity g for copy_strided_read."""
    return _touched_bytes(n_src // stride, stride, elem_bytes, g)


def model_contig_write_bytes(n_out: int, elem_bytes: int = 4, g: int = 32) -> int:
    """ESTIMATED write bytes for a contiguous output of n_out elements."""
    return _touched_bytes(n_out, 1, elem_bytes, g)


def model_copy_stride_bytes(n_src: int, stride: int, elem_bytes: int = 4, g: int = 32) -> int:
    """ESTIMATED total (read + write) bytes for one copy_stride call."""
    n_out = n_src // stride
    return model_strided_read_bytes(n_src, stride, elem_bytes, g) + model_contig_write_bytes(
        n_out, elem_bytes, g
    )


def logical_bytes(n_out: int, elem_bytes: int = 4) -> int:
    """Bytes the algorithm fundamentally needs: read n_out + write n_out elements."""
    return 2 * n_out * elem_bytes


def amplification(n_src: int, stride: int, elem_bytes: int = 4, g: int = 32) -> float:
    """ESTIMATED moved bytes / logical bytes."""
    lb = logical_bytes(n_src // stride, elem_bytes)
    return model_copy_stride_bytes(n_src, stride, elem_bytes, g) / lb if lb else float("nan")


def theoretical_dram_bw_gbps(mem_clock_khz: int, bus_width_bits: int, ddr_factor: int = 2) -> float:
    """ESTIMATED theoretical peak from device attributes.

    Assumes the reported memory clock needs a x2 (DDR) factor, which holds for
    GDDR6 as reported by cudaDevAttrMemoryClockRate on Ampere, but must be
    cross-checked against the vendor spec sheet for each GPU.
    """
    if mem_clock_khz <= 0 or bus_width_bits <= 0:
        return float("nan")
    return ddr_factor * mem_clock_khz * 1e3 * bus_width_bits / 8 / 1e9


# ---------------------------------------------------------------------------
# CSV loading (MEASURED)
# ---------------------------------------------------------------------------

REPS_COLUMNS = (
    "variant", "kernel", "elem_bytes", "n_src", "stride", "n_out", "logical_read_bytes",
    "logical_write_bytes", "block", "grid", "flush_l2", "rep", "time_ms",
)


@dataclass
class VariantSamples:
    variant: str
    kernel: str
    elem_bytes: int
    n_src: int
    stride: int
    n_out: int
    logical_bytes: int
    times_ms: list[float]


def load_reps_csv(path: Path) -> list[VariantSamples]:
    """Group per-repetition rows by variant, preserving first-seen order."""
    groups: dict[str, VariantSamples] = {}
    order: list[str] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = set(REPS_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            v = row["variant"]
            if v not in groups:
                groups[v] = VariantSamples(
                    variant=v,
                    kernel=row["kernel"],
                    elem_bytes=int(row["elem_bytes"]),
                    n_src=int(row["n_src"]),
                    stride=int(row["stride"]),
                    n_out=int(row["n_out"]),
                    logical_bytes=int(row["logical_read_bytes"]) + int(row["logical_write_bytes"]),
                    times_ms=[],
                )
                order.append(v)
            groups[v].times_ms.append(float(row["time_ms"]))
    return [groups[v] for v in order]


# ===========================================================================
# B2-B4: generic CSV schema (written by common.cuh) and traffic models (ESTIMATES)
# ===========================================================================

GENERIC_COLUMNS = (
    "benchmark", "variant", "params", "n_elems", "logical_read_bytes", "logical_write_bytes",
    "timing_mode", "iters_per_rep", "flush_l2", "rep", "time_ms",
)


@dataclass
class Series:
    """All repetitions of one (variant, params, timing_mode)."""

    benchmark: str
    variant: str
    params: dict[str, str]
    params_raw: str
    timing_mode: str
    n_elems: int
    logical_bytes: int
    times_ms: list[float]

    def param(self, key: str, cast=str):
        return cast(self.params[key])


def parse_params(raw: str) -> dict[str, str]:
    """'R=4096;C=4096' -> {'R': '4096', 'C': '4096'}."""
    out: dict[str, str] = {}
    for part in filter(None, raw.split(";")):
        k, sep, v = part.partition("=")
        if not sep:
            raise ValueError(f"bad params field: {raw!r}")
        out[k] = v
    return out


def load_generic_csv(path: Path) -> list[Series]:
    groups: dict[tuple, Series] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = set(GENERIC_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            key = (row["variant"], row["params"], row["timing_mode"])
            if key not in groups:
                groups[key] = Series(
                    benchmark=row["benchmark"], variant=row["variant"],
                    params=parse_params(row["params"]), params_raw=row["params"],
                    timing_mode=row["timing_mode"], n_elems=int(row["n_elems"]),
                    logical_bytes=int(row["logical_read_bytes"]) + int(row["logical_write_bytes"]),
                    times_ms=[],
                )
            groups[key].times_ms.append(float(row["time_ms"]))
    return list(groups.values())


SECTOR = 32  # bytes; B1 on RTX 3060 behaved as 32-B DRAM granularity for isolated sectors


# ---- B2 transpose ----------------------------------------------------------
# Assumptions: T1 4-byte elements, 32x32 tiles, 32x8 threads. T2 the naive kernel's partial-
# sector writes are completed inside L2 within one block's lifetime (each block writes 32
# consecutive out-elements per out-row), so DRAM traffic equals copy traffic. T3 no L2 reuse
# across repetitions (flushed).

def model_transpose(R: int, C: int, variant: str) -> dict[str, float]:
    n = R * C
    if variant not in ("copy_2d", "naive", "tiled", "tiled_pad", "cublas_geam"):
        raise ValueError(f"unknown transpose variant {variant}")
    est = {
        "dram_bytes": 8.0 * n,                       # T2: same for every variant
        "l1_load_sectors": n / 8.0,                  # coalesced reads everywhere
        "l2_store_sectors": n * 1.0 if variant == "naive" else n / 8.0,  # naive: 1 sector/element
        "smem_conflict_ways": 32.0 if variant == "tiled" else 1.0,
    }
    if variant == "cublas_geam":
        est["l2_store_sectors"] = float("nan")       # implementation unknown
        est["smem_conflict_ways"] = float("nan")
    return est


# ---- B3 gather / scatter_add ------------------------------------------------
# All in bytes. "compulsory" is a lower bound (every distinct sector fetched once at SECTOR
# granularity), "no_reuse" an upper bound (every lookup misses), "hot_set" a point estimate
# assuming an ideal frequency-based L2 that keeps the K = L2/32 hottest sectors.

def model_gather_table_bytes(m: int, t: int, dist: str, l2_bytes: int,
                             distinct_sectors: int | None = None,
                             hot_sector_share: float | None = None) -> dict[str, float]:
    if dist == "seq":
        # idx[i] = i mod t: contiguous; passes beyond the first re-read the table, which hits
        # in L2 only if the whole table fits.
        repeat = 0.0 if m <= t or 4 * t <= l2_bytes else 4.0 * (m - t)
        return {"compulsory": 4.0 * min(m, t), "hot_set": 4.0 * min(m, t) + repeat, "no_reuse": 4.0 * m}
    if distinct_sectors is None or hot_sector_share is None:
        raise ValueError("random distributions need index statistics from meta.json")
    k = min(distinct_sectors, l2_bytes // SECTOR)
    return {
        "compulsory": float(SECTOR * distinct_sectors),
        "hot_set": float(SECTOR * (m * (1.0 - hot_sector_share) + k)),
        "no_reuse": float(SECTOR * m),
    }


def model_gather_bytes(m: int, table: dict[str, float]) -> dict[str, float]:
    """idx read + out write + table traffic, for each table estimate."""
    return {k: 8.0 * m + v for k, v in table.items()}


def model_scatter_add_bytes(m: int, table: dict[str, float]) -> dict[str, float]:
    """idx + val reads, and each touched acc sector is read and later written back."""
    return {k: 8.0 * m + 2.0 * v for k, v in table.items()}


# ---- B4 fusion ----------------------------------------------------------------
# B = bytes per array. L2-tier traffic is exact for these kernels (no L1 reuse); DRAM traffic
# depends on whether t survives in L2 between the two kernels, for which we keep two competing
# residency models so the measurement can decide:
#   lru_cliff:    t survives only if x and t together fit (2B <= L2), else ~nothing survives
#                 (a cyclic scan larger than an LRU cache gets no hits)
#   proportional: the surviving fraction is min(1, L2 / (2B))

def fusion_residency(b: int, l2_bytes: int, model: str) -> float:
    if model == "none":
        return 0.0
    if model == "lru_cliff":
        return 1.0 if 2 * b <= l2_bytes else 0.0
    if model == "proportional":
        return min(1.0, l2_bytes / (2.0 * b))
    raise ValueError(f"unknown residency model {model}")


def model_fusion_bytes(n: int, l2_bytes: int, variant: str, model: str = "none") -> dict[str, float]:
    b = 4 * n
    if variant == "fused":
        return {"l2_tier": 2.0 * b, "dram": 2.0 * b}
    if variant == "materialized":
        r = fusion_residency(b, l2_bytes, model)
        # x read + y write always; t write + t read reach DRAM only for the non-resident part
        return {"l2_tier": 4.0 * b, "dram": 2.0 * b + 2.0 * b * (1.0 - r)}
    if variant in ("stage2_cold", "stage2_warm"):
        r = 0.0 if variant == "stage2_cold" else fusion_residency(b, l2_bytes, model)
        return {"l2_tier": 2.0 * b, "dram": b * (1.0 - r) + b}
    raise ValueError(f"unknown fusion variant {variant}")


def fusion_flops(n: int, k: int, variant: str) -> float:
    """2 FLOP per FMA; f and g each do K FMAs per element."""
    per = {"fused": 4 * k, "materialized": 4 * k, "stage2_cold": 2 * k, "stage2_warm": 2 * k}[variant]
    return float(per) * n


def fp32_peak_gflops(sm_count: int, sm_clock_khz: int, lanes_per_sm: int = 128) -> float:
    """ESTIMATE: lanes x 2 FLOP (FMA) x clock. 128 FP32 lanes/SM holds for cc 8.6 per NVIDIA's
    documentation; the attribute clock is the nominal boost, not the clock actually sustained."""
    if sm_count <= 0 or sm_clock_khz <= 0:
        return float("nan")
    return sm_count * lanes_per_sm * 2 * sm_clock_khz * 1e3 / 1e9


# ---- B3 fetch-granularity probe (gather_pair64 / gather_pair128) -------------------------
# A lookup of {j, j ^ flip} touches one DRAM chunk of size g if the flipped bit lies inside the
# chunk (g > 4*flip bytes), else two. With no reuse and the same cost per byte, the time ratio
# pair/single is (other + pair_bytes) / (other + g), other = idx read + out write = 8 B.
PAIR_FLIPS = {"gather_pair64": 8, "gather_pair128": 16}


def model_pair_bytes_per_lookup(flip_elems: int, g: int, elem_bytes: int = 4) -> int:
    return g if g > flip_elems * elem_bytes else 2 * g


def predicted_pair_ratio(variant: str, g: int, other_bytes: int = 8) -> float:
    pair = model_pair_bytes_per_lookup(PAIR_FLIPS[variant], g)
    return (other_bytes + pair) / (other_bytes + g)
