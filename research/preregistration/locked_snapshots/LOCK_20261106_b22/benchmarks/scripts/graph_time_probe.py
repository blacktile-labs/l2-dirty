#!/usr/bin/env python3
"""B22 -- can a per-kernel timing predict an in-model decode step within 1 %?

Rules V-0..V-3 are pre-registered in research/HYPOTHESES.md, section "B22 pre-registration", before any B22
measurement. Same decode proxy as B21 (e2e_probe.py, imported unchanged): 4 layers of Llama-2-7B shapes, batch 1,
four weight GEMVs per layer, fp16 / int4 (tinygemm) / nf4 (bitsandbytes).
  truth  T_step: the whole step in one CUDA graph, replayed back to back, events around each replay, median of 1000
  graph  (new) per shape: N copies of the weights (N = max(4, ceil(4 L2 / weight bytes)); the step's 4 layers are
         the first 4 copies), one CUDA graph calling the kernel once per copy in turn, replayed back to back;
         per-call time = median replay time / N. Each call reads weights untouched for N - 1 calls (>= 4 x L2), as in
         the step, and there is no per-launch event or host cost.
  cold   cleanbench cold, as in B21 (reference)
  write  triton.testing.do_bench as shipped, as in B21 (reference)
"""

import argparse
import json
import math
import statistics
import sys

import torch
import torch.nn.functional as F
import triton
import triton.testing as tt

import cleanbench as cb
import e2e_probe as E

STEP_REPS, GRAPH_REPS, BENCH_REPS = 1000, 100, 200
SEED = 2200
TOL = 2e-2


def build_copies(l2: int):
    """variant -> shape -> list of callables over N copies; the first LAYERS copies form the decode step."""
    xs = {K: torch.randn(1, K, device="cuda") * 0.5 for K in (E.H, E.FFN)}
    ops, fails, notes, ncopies = {"fp16": {}, "int4": {}}, 0, {}, {}
    try:
        import bitsandbytes
        import bitsandbytes.functional as bf
        ops["nf4"] = {}
        notes["bitsandbytes"] = bitsandbytes.__version__
    except Exception as ex:  # noqa: BLE001
        notes["nf4_error"] = repr(ex)[:200]
    for s, (N, K) in E.SHAPES.items():
        x = xs[K]
        x16, xbf = x.half(), x.bfloat16()
        need = {"fp16": N * K * 2, "int4": N * K // 2, "nf4": N * K // 2}
        for v in ops:
            ops[v][s] = []
            ncopies[f"{v} {s}"] = max(E.LAYERS, math.ceil(4 * l2 / need[v]))
        for c in range(max(ncopies[f"{v} {s}"] for v in ops)):
            W = torch.randn(N, K, device="cuda") * 0.02
            w16 = W.half()
            if c < ncopies[f"fp16 {s}"]:
                ops["fp16"][s].append(lambda w16=w16, x16=x16: F.linear(x16, w16))
                if c == 0:
                    ref = x16.float() @ w16.float().t()
                    fails += ((F.linear(x16, w16).float() - ref).abs().max() / ref.abs().max()).item() > TOL
            if c < ncopies[f"int4 {s}"]:
                wq, sz = E.group_quantize_int4(W.bfloat16(), E.GROUP)
                wp = torch._convert_weight_to_int4pack(wq, E.INNER_K)
                ops["int4"][s].append(lambda wp=wp, sz=sz, xbf=xbf: torch._weight_int4pack_mm(xbf, wp, E.GROUP, sz))
                if c == 0:
                    ref = xbf.float() @ E.dequant_int4(wq, sz, E.GROUP).t()
                    y = torch._weight_int4pack_mm(xbf, wp, E.GROUP, sz).float()
                    fails += ((y - ref).abs().max() / ref.abs().max()).item() > TOL
                del wq
            if "nf4" in ops and c < ncopies[f"nf4 {s}"]:
                wq4, qs = bf.quantize_4bit(w16, quant_type="nf4", blocksize=E.BNB_BLOCK, compress_statistics=False)
                ops["nf4"][s].append(lambda wq4=wq4, qs=qs, x16=x16: bitsandbytes.matmul_4bit(x16, wq4.t(), quant_state=qs))
                if c == 0:
                    ref = x16.float() @ bf.dequantize_4bit(wq4, qs).float().t()
                    y = bitsandbytes.matmul_4bit(x16, wq4.t(), quant_state=qs).float()
                    fails += ((y - ref).abs().max() / ref.abs().max()).item() > TOL
            del W
            if c >= ncopies[f"fp16 {s}"]:
                del w16
        torch.cuda.empty_cache()
    return ops, int(fails), notes, ncopies


def graph_times(fns, reps):
    """Capture fns once into a CUDA graph and replay it back to back; per-replay times in us."""
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
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for st, en in ev:
        st.record()
        g.replay()
        en.record()
    torch.cuda.synchronize()
    return [st.elapsed_time(en) * 1e3 for st, en in ev]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="few replays (pipeline check)")
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()
    torch.manual_seed(SEED)
    step_reps, graph_reps, bench_reps = (30, 10, 20) if a.smoke else (STEP_REPS, GRAPH_REPS, BENCH_REPS)
    prop = torch.cuda.get_device_properties(0)
    l2 = prop.L2_cache_size
    ops, fails, notes, ncopies = build_copies(l2)
    events = []
    for v, ops_v in ops.items():
        step = [ops_v[s][layer] for layer in range(E.LAYERS) for s in E.SHAPES]
        try:
            ts = graph_times(step, step_reps)
            events.append({"kind": "step", "variant": v, "times_us": ts, "median_us": statistics.median(ts)})
        except Exception as ex:  # noqa: BLE001
            notes[f"step_graph_error_{v}"] = repr(ex)[:300]
            continue
        for s in E.SHAPES:
            copies = ops_v[s]
            gt = graph_times(copies, graph_reps)
            events.append({"kind": "kernel", "variant": v, "shape": s, "copies": len(copies),
                           "graph_us": statistics.median(gt) / len(copies),
                           "cold_us": cb.bench(copies[0], mode="cold", reps=bench_reps).median_us,
                           "write_us": tt.do_bench(copies[0], warmup=25, rep=100, return_mode="median") * 1e3})
    meta = {"benchmark": "graph_time_probe", "device": prop.name, "l2_bytes": l2, "torch": torch.__version__,
            "triton": triton.__version__, "layers": E.LAYERS, "ncopies": ncopies, "notes": notes, "smoke": a.smoke,
            "seed": SEED, "events": events, "correctness_failures": fails}
    json.dump(meta, open(a.meta, "w"), indent=1)
    print(f"{len(events)} events, variants={list(ops)}, correctness_failures={fails}, notes={notes}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
