"""B23b clock gate (research/HYPOTHESES.md, "B23b amendment"): was the SM clock steady during a kernel_time_probe.sh run?
Reads the run directory's nvsmi.csv (1 s samples) and perf_before_r*/perf_after_r* (nvidia-smi -q -d PERFORMANCE).

  busy samples: utilization.gpu >= 90 %;  c = median SM clock of busy samples
  C-1  >= 99 % of busy samples within +-1 % of c, and c within +-2 % of the locked clock (--locked-mhz)
  C-2  no thermal or hardware slowdown during the runs: the increase of the cumulative counters
       "SW Thermal Slowdown", "HW Thermal Slowdown" and "HW Power Braking" summed over runs is 0
       (a counter lower after a run than before counts its after value, i.e. a reset)
  descriptive: SW power capping increase, temperature range, clock-event reason counts
Prints counts and the verdict only with --check-only; full numbers otherwise.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

GATED = ("SW Thermal Slowdown", "HW Thermal Slowdown", "HW Power Braking")
DESCR = ("SW Power Capping",)


def counters(text: str) -> dict:
    """Cumulative clock-event counters (us) from the 'Clocks Event Reasons Counters' section."""
    sec = text.split("Clocks Event Reasons Counters", 1)
    if len(sec) < 2:
        return {}
    out = {}
    for line in sec[1].splitlines()[1:]:
        m = re.match(r"\s+([A-Za-z ]+?)\s+:\s+(\d+) us", line)
        if not m:
            break
        out[m.group(1)] = int(m.group(2))
    return out


def increase(before: dict, after: dict, key: str) -> int:
    a, b = after.get(key, 0), before.get(key, 0)
    return a - b if a >= b else a


def gate(rows: list[dict], perf: list[tuple[str, str]], locked_mhz: float) -> dict:
    num = lambda s: float(s.strip().split()[0])  # noqa: E731
    busy = [r for r in rows if num(r["utilization.gpu [%]"]) >= 90]
    sm = [num(r["clocks.current.sm [MHz]"]) for r in busy]
    out = {"busy samples": len(sm)}
    if not sm:
        out.update({"C-1": False, "C-2": False})
        return out
    c = statistics.median(sm)
    within = sum(abs(x / c - 1) <= 0.01 for x in sm) / len(sm)
    out.update({"busy median SM MHz": c, "busy within 1 %": round(within, 4),
                "SM MHz range": [min(sm), max(sm)],
                "temperature C": [min(num(r["temperature.gpu"]) for r in busy), max(num(r["temperature.gpu"]) for r in busy)],
                "reasons": dict(Counter(r.get("clocks_event_reasons.active", "").strip() for r in busy))})
    out["C-1"] = within >= 0.99 and abs(c / locked_mhz - 1) <= 0.02
    inc = {k: 0 for k in GATED + DESCR}
    for b, a in perf:
        cb, ca = counters(b), counters(a)
        for k in inc:
            inc[k] += increase(cb, ca, k)
    out["counter increase us"] = inc
    out["C-2"] = bool(perf) and all(inc[k] == 0 for k in GATED)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--locked-mhz", type=float, required=True)
    p.add_argument("--json-out", type=Path)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    with open(a.run_dir / "nvsmi.csv") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f, skipinitialspace=True)
                if r.get(" utilization.gpu [%]", r.get("utilization.gpu [%]"))]
    perf = [((a.run_dir / f"perf_before_r{i}.txt").read_text(), (a.run_dir / f"perf_after_r{i}.txt").read_text())
            for i in range(1, 10) if (a.run_dir / f"perf_after_r{i}.txt").exists()]
    out = gate(rows, perf, a.locked_mhz)
    out["clock gate"] = out["C-1"] and out["C-2"]
    if a.check_only:
        print(f"busy samples {out['busy samples']}, runs {len(perf)}, C-1 {out['C-1']}, C-2 {out['C-2']}")
    else:
        print(json.dumps(out, indent=1))
    if a.json_out:
        a.json_out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
