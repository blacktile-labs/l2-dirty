"""B20 scorer: does the latest Nsight Compute still lose per-kernel DRAM write-backs (C6)?
Rules N-0..N-2 pre-registered in research/HYPOTHESES.md, section "B20 pre-registration", before any B20 run.
Inputs: the score.json of queued_wb_probe.sh (B9b protocol) run once with each Nsight Compute version.
loss(w) = S_c - S_k (fraction of the stored bytes that per-kernel replay does not count), as in B9b.

  N-0  gate: B9b rule Q-0 passes for both versions
  N-1  (primary) new version, w in {0.75, 1}: all loss <= 0.02 "fixed in the new version";
       all loss >= 0.05 "persists in the new version"; else "partial"
  N-2  |loss_new - loss_old| <= 0.03 at both w "unchanged between versions"; else "changed"
"""

import argparse
import json
import sys
from pathlib import Path

WS = ("0.75", "1")
FIXED, PERSISTS, SAME = 0.02, 0.05, 0.03


def score(old: dict, new: dict, log=print) -> dict:
    v = {"N-0": bool(old.get("Q-0")) and bool(new.get("Q-0"))}
    lo, ln = old["Q-1 loss"], new["Q-1 loss"]
    v["loss old"], v["loss new"] = {w: lo[w] for w in lo}, {w: ln[w] for w in ln}
    if not v["N-0"]:
        v["N-1"] = v["N-2"] = "not scored (N-0 failed)"
    else:
        xs = [ln[w] for w in WS]
        v["N-1"] = ("fixed in the new version" if all(x <= FIXED for x in xs)
                    else "persists in the new version" if all(x >= PERSISTS for x in xs) else "partial")
        v["N-2"] = "unchanged between versions" if all(abs(ln[w] - lo[w]) <= SAME for w in WS) else "changed"
    log(f"N-0 {v['N-0']} · loss old {lo} · loss new {ln} -> N-1 {v['N-1']} · N-2 {v['N-2']}")
    return v


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--old", type=Path, required=True, help="score.json, old Nsight Compute")
    p.add_argument("--new", type=Path, required=True, help="score.json, new Nsight Compute")
    p.add_argument("--json-out", type=Path)
    a = p.parse_args()
    v = score(json.loads(a.old.read_text()), json.loads(a.new.read_text()))
    if a.json_out:
        a.json_out.write_text(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
