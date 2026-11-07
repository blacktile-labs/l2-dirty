#!/usr/bin/env python3
"""B15 -- why does do_bench's write-flush penalty differ between configs of the same decode GEMV?

Post-hoc hypothesis from B14 (RTX 4080S, W4 at 4096 x 14336: one config +7 us, another +21.5 us):
the dirty residue costs the same DRAM write-back bytes for every config, but a config that leaves
DRAM bandwidth idle can hide those writes in its slack, while a bandwidth-saturating config pays.
Per config of the B14 kernels (fp16 / W8A16 / W4A16, 12 configs each; no cuBLAS) and LLM shape:
  --mode range   (ncu --replay-mode app-range): per repetition and flush mode (zero = do_bench's
                 zero_(), read = read-only flush): flush; sync; [range: kernel; sync]
                 -> DRAM read and write bytes of the kernel after each flush
  --mode timing  (no profiler): the REAL triton.testing.do_bench (median) with the shipped flush and
                 with clear_cache patched to a read, alternating order
Every config is checked against the fp32 reference of its dequantized weights first.
  python3 absorb_probe.py --mode {range,timing} --meta OUT.json [--smoke] [--run-seed N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import triton
import triton.testing as tt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_bias_probe import CONFIGS, FORMATS, KERNELS, REL_TOL, quantize  # noqa: E402

SHAPES = [(4096, 14336), (14336, 4096)]      # the MLP shapes where B14 found the distortion
FLUSH_BYTES = 256 << 20
RANGE_REPS = 3
WARMUP_MS, REP_MS = 10, 50
SEED = 1234


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("range", "timing"), required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--smoke", action="store_true", help="first shape, 2 configs per format")
    ap.add_argument("--run-seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(SEED)
    prop = torch.cuda.get_device_properties(0)
    shapes = SHAPES[:1] if a.smoke else SHAPES
    cfgs = CONFIGS[:2] if a.smoke else CONFIGS
    cands = [(f, *c) for f in FORMATS for c in cfgs]
    cache = torch.empty(FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    flushes = {"zero": lambda: cache.zero_(), "read": lambda: cache.view(torch.float32).sum()}
    drv = triton.runtime.driver.active
    orig = drv.clear_cache
    read_clear = lambda c: c.view(torch.float32).sum()  # noqa: E731
    rows, invalid = [], []
    rng = random.Random(a.run_seed)
    for N, K in shapes:
        Wf = torch.randn(N, K, device="cuda") * 0.02
        fm = quantize(Wf)
        del Wf
        x = (torch.randn(K, device="cuda") * 0.5).half()
        y = torch.empty(N, device="cuda", dtype=torch.float16)
        refs = {f: fm[f][2] @ x.float() for f in FORMATS}
        for f in FORMATS:
            fm[f] = fm[f][:2]
        torch.cuda.empty_cache()

        def make_fn(c):
            f, bn, bk, nw = c
            w, s = fm[f]
            grid = (triton.cdiv(N, bn),)
            return lambda: KERNELS[f][grid](w, s, x, y, N, K, w.stride(0), BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)

        valid = []
        for c in cands:
            try:
                make_fn(c)()
                torch.cuda.synchronize()
                err = ((y.float() - refs[c[0]]).abs().max() / refs[c[0]].abs().max()).item()
            except Exception as ex:  # noqa: BLE001
                invalid.append({"shape": [N, K], "cand": "{}_bn{}_bk{}_w{}".format(*c), "error": repr(ex)[:200]})
                continue
            (valid.append(c) if err <= REL_TOL else
             invalid.append({"shape": [N, K], "cand": "{}_bn{}_bk{}_w{}".format(*c), "rel_err": err}))
        order = valid[:]
        rng.shuffle(order)
        for i, c in enumerate(order):
            fn, cname = make_fn(c), "{}_bn{}_bk{}_w{}".format(*c)
            w = fm[c[0]][0]
            base = {"shape": [N, K], "cand": cname, "format": c[0], "weight_bytes": w.numel() * w.element_size()}
            if a.mode == "range":
                for rep in range(1 if a.smoke else RANGE_REPS):
                    for mode in (("zero", "read") if (i + rep) % 2 == 0 else ("read", "zero")):
                        flushes[mode]()
                        torch.cuda.synchronize()
                        torch.cuda.profiler.start()
                        fn()
                        torch.cuda.synchronize()
                        torch.cuda.profiler.stop()
                        rows.append({**base, "kind": "range", "mode": mode, "rep": rep})
            else:
                res = {}
                for mode in (("zero", "read") if i % 2 == 0 else ("read", "zero")):
                    drv.clear_cache = orig if mode == "zero" else read_clear
                    try:
                        res[mode] = tt.do_bench(fn, warmup=WARMUP_MS, rep=REP_MS, return_mode="median")
                    finally:
                        drv.clear_cache = orig
                rows.append({**base, "kind": "timing", "zero_ms": res["zero"], "read_ms": res["read"]})
        del fm, refs
        torch.cuda.empty_cache()
    meta = {"benchmark": "absorb_probe", "mode": a.mode, "device": prop.name, "l2_bytes": prop.L2_cache_size,
            "torch": torch.__version__, "triton": triton.__version__, "smoke": a.smoke, "run_seed": a.run_seed,
            "rows": rows, "invalid": invalid, "n_candidates": len(cands)}
    json.dump(meta, open(a.meta, "w"), indent=1)
    print(f"{a.mode}: {len(rows)} rows, {len(invalid)} invalid candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
