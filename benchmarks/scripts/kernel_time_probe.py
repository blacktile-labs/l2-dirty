#!/usr/bin/env python3
"""B23 -- can a single-kernel benchmark time each kernel within 1 % of the same kernel's time inside the model?

Rules W-0..W-3 are pre-registered in research/HYPOTHESES.md, section "B23 pre-registration", before any B23
measurement. Same decode proxy and weight copies as B22 (graph_time_probe.build_copies, imported unchanged):
4 layers of Llama-2-7B shapes, batch 1, four weight GEMVs per layer, fp16 (cuBLAS) / int4 (tinygemm) /
nf4 (bitsandbytes).

Kernel time = the GPU's own start and end timestamps of the op's kernels (CUPTI activity records, read through
torch.profiler), summed over the op's kernels. No CUDA events, so launch and event costs are not included.
Per variant, ROUNDS rounds, each: the step segment, then one rotation segment per shape (truth and the primary
harness interleave, so slow drift in clocks or temperature hits both). A segment = >= WARM_S of untraced replays
(queued in ~CHUNK_S chunks, so the GPU is still busy while the profiler starts); REWARM_S of traced replays; a marker kernel;
the traced replays with CUDA events; then REWARM_S and the same number of untraced replays with CUDA events.
  truth           the 16-op step in one CUDA graph; per op, its CUPTI duration inside the step
                  (4 layers x STEP_TRACED replays x ROUNDS)
  cupti_rot       (primary) the op over its N weight copies (N = max(4, ceil(4 L2 / weight bytes)), as B22) in
                  one CUDA graph; CUPTI per call (N x ROT_REPS x ROUNDS)
  event_graph     the untraced half of the rotation segments: CUDA events around replays / N (B22's harness)
After the rounds, per shape:
  event_cold      cleanbench cold as shipped (CUDA events), as in B21 / B22
  cupti_cold      cleanbench's cold loop (read flush >= max(256 MiB, 4 x L2), GPU busy-wait, call); CUPTI
  cupti_rot_idle  the rotation graph after IDLE_S of idle GPU and no warm-up (effect of warm-up / clock state)
Gate: the step's CUDA-event time traced vs untraced (paired halves of the same segments).
Every measured segment is checked for the exact number and the expected names of kernel records.
"""

import argparse
import json
import math
import statistics
import sys
import time

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

import cleanbench as cb
import e2e_probe as E
import graph_time_probe as G
from cupti_records import NAME_LEN, after_marker, op_times, records_ok

ROUNDS, STEP_UNTRACED, STEP_TRACED, ROT_REPS, BENCH_REPS = 3, 350, 100, 40, 200
WARM_S, REWARM_S, IDLE_S, CHUNK_S = 1.0, 0.05, 1.0, 0.02
SEED = 2300
FLUSH_NAME, PAD_NAME = "_read_flush", "spin_kernel"


@triton.jit
def _b23_marker(X):
    # One tiny kernel with a unique name: records before it (warm-up) are dropped.
    tl.load(X, volatile=True)


_MARK = None


def mark():
    global _MARK
    if _MARK is None:
        _MARK = torch.zeros(1, device="cuda")
    _b23_marker[(1, )](_MARK)


def gpu_records(prof):
    """GPU activity records (name, start ns, duration ns) of a profiler session, in start order."""
    out = [(e.name(), e.start_ns(), e.duration_ns()) for e in prof.profiler.kineto_results.events()
           if e.device_type().name == "CUDA"]
    return sorted(out, key=lambda r: r[1])


def qstats(xs):
    if not xs:
        return None
    p25, p50, p75, p95 = cb.quantiles(xs)
    return {"median": p50, "p25": p25, "p75": p75, "p95": p95, "n": len(xs)}


def kernels_of(fn):
    """Names of the GPU records one call of fn produces."""
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        mark()
        fn()
        torch.cuda.synchronize()
    return [r[0][:NAME_LEN] for r in (after_marker(gpu_records(p)) or [])]


def capture(fns):
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


def warm(g, t_rep, seconds):
    """Keep the GPU busy replaying g for >= `seconds`: chunks of ~CHUNK_S of GPU work, waiting for the previous chunk
    before queueing the next, so the queue never holds more than ~2 chunks (the host can enqueue far faster than the
    GPU runs, which would otherwise leave seconds of queued work). Returns (replays, wall seconds)."""
    per = max(1, math.ceil(CHUNK_S * 1e6 / t_rep))
    t0, n, prev = time.perf_counter(), 0, None
    while True:
        for _ in range(per):
            g.replay()
        n += per
        ev = torch.cuda.Event()
        ev.record()
        if prev is not None:
            prev.synchronize()
        prev = ev
        if time.perf_counter() - t0 >= seconds:
            return n, time.perf_counter() - t0


def event_replays(g, reps):
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for st, en in ev:
        st.record()
        g.replay()
        en.record()
    torch.cuda.synchronize()
    return [st.elapsed_time(en) * 1e3 for st, en in ev]


def segment(g, t_rep, reps, expect, warm_s, untraced=True):
    """See the module docstring. Returns records after the marker, the record check, traced and untraced replay
    times (us; untraced None if not asked) and the warm-up actually done."""
    info = {"warm_n": 0, "warm_s": 0.0}
    if warm_s > 0:
        info["warm_n"], info["warm_s"] = warm(g, t_rep, warm_s)
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        if warm_s > 0:
            warm(g, t_rep, REWARM_S)
        mark()
        for st, en in ev:
            st.record()
            g.replay()
            en.record()
        torch.cuda.synchronize()
    recs = after_marker(gpu_records(p)) or []
    traced = [st.elapsed_time(en) * 1e3 for st, en in ev]
    un = None
    if untraced:
        warm(g, t_rep, REWARM_S)
        un = event_replays(g, reps)
    return recs, records_ok(recs, expect, reps), traced, un, info


def cupti_cold(fn, names, reps):
    """cleanbench's cold loop (read flush, busy-wait, call), timed by CUPTI instead of events."""
    buf = cb.flush_buffer()
    pad_us = min(5000.0, max(100.0, 3.0 * cb.host_dispatch_us(fn)))
    for _ in range(11):
        cb.read_flush(buf)
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        mark()
        for _ in range(reps):
            cb.read_flush(buf)
            cb.pad(pad_us)
            fn()
        torch.cuda.synchronize()
    recs = after_marker(gpu_records(p)) or []
    mine = [r for r in recs if FLUSH_NAME not in r[0] and PAD_NAME not in r[0]]
    ok = len(recs) == reps * (len(names) + 2) and records_ok(mine, names, reps)
    return (op_times(mine, [len(names)])[0] if ok else []), ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="2 rounds, few replays, short warm-up (pipeline check)")
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()
    global ROUNDS, STEP_UNTRACED, STEP_TRACED, ROT_REPS, BENCH_REPS, WARM_S, REWARM_S, IDLE_S
    if a.smoke:
        ROUNDS, STEP_UNTRACED, STEP_TRACED, ROT_REPS, BENCH_REPS = 2, 20, 10, 5, 20
        WARM_S, REWARM_S, IDLE_S = 0.1, 0.01, 0.1
    torch.manual_seed(SEED)
    prop = torch.cuda.get_device_properties(0)
    l2 = prop.L2_cache_size
    ops, fails, notes, ncopies = G.build_copies(l2)
    phases, events = [], []
    for v, ops_v in ops.items():
        names = {s: kernels_of(ops_v[s][0]) for s in E.SHAPES}
        order = [s for _ in range(E.LAYERS) for s in E.SHAPES]               # op slots of one step replay
        expect = [n for s in order for n in names[s]]
        try:
            g, t_step = capture([ops_v[s][layer] for layer in range(E.LAYERS) for s in E.SHAPES])
        except Exception as ex:  # noqa: BLE001 -- recorded; the variant then has no truth
            notes[f"step_graph_error_{v}"] = repr(ex)[:300]
            continue
        grs = {s: capture(ops_v[s]) for s in E.SHAPES}                     # shape -> (graph, replay us)
        step = {"kind": "step", "variant": v, "traced_us": [], "untraced_us": [], "count_ok": [], "warm": []}
        rows = {s: {"kind": "row", "variant": v, "shape": s, "names": names[s], "copies": len(ops_v[s]),
                    "truth": {"samples": [], "round_medians": []}, "rot": [], "rot_round_medians": [],
                    "rot_traced_us": [], "rot_untraced_us": [], "count_ok": {"cupti_rot": []}, "warm": []}
                for s in E.SHAPES}
        for rnd in range(ROUNDS):
            t0 = time.time()
            recs, ok, tr, un, info = segment(g, t_step, STEP_TRACED, expect, WARM_S)
            step["traced_us"] += tr
            step["untraced_us"] += un
            step["count_ok"].append(ok)
            step["warm"].append(info)
            slots = op_times(recs, [len(names[s]) for s in order]) if ok else [[] for _ in order]
            for s in E.SHAPES:
                xs = [x for j, o in enumerate(order) if o == s for x in slots[j]]
                rows[s]["truth"]["samples"] += xs
                rows[s]["truth"]["round_medians"].append(statistics.median(xs) if xs else None)
            phases.append([f"{v} r{rnd} step", t0, time.time()])
            for s in E.SHAPES:
                t0 = time.time()
                r = rows[s]
                n = len(ops_v[s])
                recs, ok, tr, un, info = segment(*grs[s], ROT_REPS, names[s] * n, WARM_S)
                xs = [x for col in op_times(recs, [len(names[s])] * n) for x in col] if ok else []
                r["rot"] += xs
                r["rot_round_medians"].append(statistics.median(xs) if xs else None)
                r["rot_traced_us"] += tr
                r["rot_untraced_us"] += un
                r["count_ok"]["cupti_rot"].append(ok)
                r["warm"].append(info)
                phases.append([f"{v} r{rnd} rot {s}", t0, time.time()])
        del g
        for s in E.SHAPES:
            r, fn, n = rows[s], ops_v[s][0], len(ops_v[s])
            t0 = time.time()
            r["event_cold"] = cb.bench(fn, mode="cold", reps=BENCH_REPS).median_us
            d, r["count_ok"]["cupti_cold"] = cupti_cold(fn, names[s], BENCH_REPS)
            r["cupti_cold"] = qstats(d)
            phases.append([f"{v} cold {s}", t0, time.time()])
            torch.cuda.synchronize()
            time.sleep(IDLE_S)
            t0 = time.time()
            recs, ok, _, _, _ = segment(*grs[s], ROT_REPS, names[s] * n, 0, untraced=False)
            r["count_ok"]["cupti_rot_idle"] = ok
            r["cupti_rot_idle"] = qstats([x for col in op_times(recs, [len(names[s])] * n) for x in col] if ok else [])
            phases.append([f"{v} rot idle {s}", t0, time.time()])
            r["cupti_rot"] = qstats(r.pop("rot"))
            r["event_graph"] = statistics.median(r["rot_untraced_us"]) / n
            r["rot_tracing_effect"] = statistics.median(r["rot_traced_us"]) / statistics.median(r["rot_untraced_us"]) - 1
            r["count_ok"]["cupti_rot"] = all(r["count_ok"]["cupti_rot"])
            for k in ("rot_traced_us", "rot_untraced_us"):
                r[k] = qstats(r[k])
            events.append(r)
        step["count_ok"] = all(step["count_ok"])
        events.append(step)
        del grs
    meta = {"benchmark": "kernel_time_probe", "device": prop.name, "l2_bytes": l2, "torch": torch.__version__,
            "triton": triton.__version__, "layers": E.LAYERS, "ncopies": ncopies, "notes": notes, "smoke": a.smoke,
            "seed": SEED, "rounds": ROUNDS, "warm_s": WARM_S, "idle_s": IDLE_S,
            "utc_offset_s": time.localtime().tm_gmtoff, "phases": phases, "events": events,
            "correctness_failures": fails}
    json.dump(meta, open(a.meta, "w"), indent=1)
    rows = [e for e in events if e["kind"] == "row"]
    oks = [e["count_ok"] for e in events if e["kind"] == "step"] + [x for r in rows for x in r["count_ok"].values()]
    print(f"{len(rows)} rows, variants={list(ops)}, record checks ok {sum(oks)}/{len(oks)}, "
          f"kernels per op {sorted({len(r['names']) for r in rows})}, correctness_failures={fails}, notes={notes}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
