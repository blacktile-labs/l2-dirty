"""Post-hoc (D-107): share s of a kernel's in-step time that scales with the SM clock, from two laptop runs of B23.

Model t(f) = a/f + b (b: DRAM-bound part at a fixed memory clock); s = (a/f0)/t(f0) at the unlocked clock f0.
With t1 at the locked clock f1:  t1/t0 - 1 = s * (f0/f1 - 1)  =>  s = (t1/t0 - 1) / (f0/f1 - 1).
Two clock states determine the two parameters exactly, so this is an estimate, not a validated fit. f0 is summarized
from 1 s nvidia-smi samples taken during each variant's step phases of the unlocked run; the result depends on how
f0 is summarized (median vs mean), so both are printed.

usage: python3 clock_sensitivity.py <unlocked run dir> <locked run dir> [--locked-mhz 1005]
"""

import argparse
import csv
import datetime as dt
import glob
import json
import statistics
from pathlib import Path


def step_clocks(run: Path) -> dict:
    samp = []
    for r in list(csv.reader(open(run / "nvsmi.csv")))[1:]:
        try:
            samp.append((dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f"), int(r[1].split()[0])))
        except (ValueError, IndexError):
            pass
    out: dict = {}
    for f in sorted(glob.glob(str(run / "run_r*_meta.json"))):
        m = json.load(open(f))
        tz = dt.timezone(dt.timedelta(seconds=m.get("utc_offset_s", 0)))
        for label, t0, t1 in m["phases"]:
            if "step" not in label:
                continue
            a = dt.datetime.fromtimestamp(t0, tz).replace(tzinfo=None)
            b = dt.datetime.fromtimestamp(t1, tz).replace(tzinfo=None)
            out.setdefault(label.split()[0], []).extend(c for t, c in samp if a <= t <= b)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("unlocked", type=Path)
    p.add_argument("locked", type=Path)
    p.add_argument("--locked-mhz", type=float, default=1005.0, help="observed SM clock of the locked run")
    a = p.parse_args()
    clocks = step_clocks(a.unlocked)
    t0 = json.load(open(a.unlocked / "score.json"))["truth us"]
    t1 = json.load(open(a.locked / "score.json"))["truth us"]
    print(f"{'kernel':14s} {'t unlocked':>10s} {'t locked':>9s} {'ratio':>6s} | f0 median  s | f0 mean    s")
    for k in sorted(t0):
        v = k.split()[0]
        cs = clocks.get(v, [])
        if not cs:
            continue
        r = t1[k] / t0[k]
        row = [f"{k:14s} {t0[k]:10.1f} {t1[k]:9.1f} {r:6.3f}"]
        for f0 in (statistics.median(cs), statistics.fmean(cs)):
            row.append(f"{f0:7.0f} {((r - 1) / (f0 / a.locked_mhz - 1)):5.2f}")
        print(" | ".join(row), f"(n={len(cs)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
