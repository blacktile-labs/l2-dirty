#!/usr/bin/env python3
"""B9b -- does ncu's one-kernel-at-a-time replay lose DRAM write-backs above the dirty cap?

B9 (D-052): dirty lines resting below the cap are not written back while the GPU is idle. The
remaining explanation of the B7r/B7r2 pattern (eager chains near capacity: fewer DRAM writes per
kernel than back to back) is write-back work still in flight when a kernel ends, which then
completes in the profiler's gap and is attributed to no kernel.

Per repetition, after a flush (the L2 holds only clean lines), one write kernel stores w x L2 bytes
(full lines), then a flush evicts everything; the same pair is done twice:
  --mode range  (ncu --replay-mode app-range)
      split:  range A = [write; sync]   range B = [flush; sync]      (a range boundary between them)
      joint:  range C = [write; flush; sync]                         (back to back: the reference)
  --mode kernel (ncu --replay-mode application): the same two pairs, measured per kernel
      W = writes of the write kernel, F = writes of the following flush
Control ranges (range mode only): a flush, then a range holding a 10 ms host sleep.
D is checked after every pair, outside the ranges.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/dmkl-inductor"))

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idle_wb_probe import LINE, write_lines  # noqa: E402
from torch_opchain import Flusher  # noqa: E402

W_FRACS = (0.375, 0.75, 1.0, 1.5)
REPS = 5
CONTROL_MS = 10.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("range", "kernel"), required=True)
    ap.add_argument("--smoke", action="store_true", help="w 0.5625 only, 2 repetitions")
    ap.add_argument("--meta", default="queued_wb_meta.json")
    a = ap.parse_args()
    prop = torch.cuda.get_device_properties(0)
    l2 = prop.L2_cache_size
    flush = Flusher(l2)
    ws = (0.5625,) if a.smoke else W_FRACS
    reps = 2 if a.smoke else REPS
    n_of = {w: int(w * l2) // LINE * LINE // 4 for w in ws}
    n_max = max(n_of.values())
    d = torch.zeros(n_max, dtype=torch.int32, device="cuda")
    idx = torch.arange(n_max, dtype=torch.int32, device="cuda")
    write_lines[(triton.cdiv(n_max, 1024),)](d, n_max, 1, BLOCK=1024)  # warm up / compile
    flush()
    torch.cuda.synchronize()
    rng = a.mode == "range"

    def start():
        if rng:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()

    def stop():
        torch.cuda.synchronize()
        if rng:
            torch.cuda.profiler.stop()

    events, fails, tag = [], 0, 7  # range mode: one entry per range; kernel mode: one per measured pair
    if rng:
        for _ in range(2):
            flush()
            start()
            time.sleep(CONTROL_MS / 1e3)
            stop()
            events.append({"kind": "control"})
    for w in ws:
        n = n_of[w]
        grid = (triton.cdiv(n, 1024),)
        for rep in range(reps):
            for pair in ("split", "joint"):
                tag += 1
                flush()
                if pair == "split":
                    start()
                    write_lines[grid](d, n, tag, BLOCK=1024)
                    stop()
                    start()
                    flush()
                    stop()
                    events += ([{"kind": "A", "w": w, "rep": rep, "bytes": n * 4},
                                {"kind": "B", "w": w, "rep": rep, "bytes": n * 4}] if rng else
                               [{"kind": "pair", "pair": pair, "w": w, "rep": rep, "bytes": n * 4}])
                else:
                    start()
                    write_lines[grid](d, n, tag, BLOCK=1024)
                    flush()
                    stop()
                    events.append({"kind": "C", "w": w, "rep": rep, "bytes": n * 4} if rng else
                                  {"kind": "pair", "pair": pair, "w": w, "rep": rep, "bytes": n * 4})
                fails += not bool(torch.equal(d[:n], idx[:n] ^ tag))
                torch.cuda.synchronize()
    meta = {"benchmark": "queued_wb_probe", "mode": a.mode, "device": prop.name, "l2_bytes": l2,
            "torch": torch.__version__, "triton": triton.__version__, "smoke": a.smoke, "reps": reps,
            "flush_bytes": flush.n * 4, "events": events, "correctness_failures": fails}
    json.dump(meta, open(a.meta, "w"), indent=1)
    print(f"{len(events)} events, correctness_failures={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
