"""B11: cross-architecture summary of B6b / B8 / B9b (+ B10b first-cell) verdicts.
Rules pre-registered in research/HYPOTHESES.md, section "B11 pre-registration", before any B11 GPU
was rented. The per-protocol scorers are unchanged; this file only compares their verdicts with the
GA106 reference and applies two cross-architecture rules.

Per GPU:
  P-0   counters readable (B6b and B8 score.json exist) and correctness_failures = 0 in every run meta
  match number of the 9 primary verdicts equal to GA106's (V-2, V-3, V-4, V-5, Z-1, Z-2, Z-2b, Q-1, Q-3)
  C5n   narrow dirty cap, from the B6b full set (range mode):
          WBR(0:0.4375) <= 0.10 and WBR(0:0.625) >= 0.80  -> "present (transition in 0.44-0.63 L2)"
          WBR(0:0.6875) < 0.50                              -> "absent up to 0.69 L2"
          otherwise                                         -> "other"
  C6    per-kernel ncu loss, from B9b: L = S_c - S_k at w = 0.75 L2
          >= 0.05 -> "present"; <= 0.02 -> "absent"; otherwise "inconclusive"
  FC    first-cell artifact, combining the sector control 0:0.375 as FIRST cell (B6b, WBR c1) and as
        SECOND cell (B10b F-1, WBR c2) on the same GPU:
          c1 < 0.05                      -> "nothing to test (control passes as first cell)"
          c1 >= 0.05 and c2 < 0.05       -> "first-cell artifact confirmed"
          c1 >= 0.05 and c2 >= 0.10      -> "not a first-cell artifact"
          otherwise                      -> "inconclusive"
Across the NEW GPUs whose P-0 passed (GA106 and GA102 are not counted):
  C5n-X, C6-X: "generalizes" if present on >= 2; "Ampere-specific" if present on 0 (with >= 2 valid);
               "mixed" otherwise; "insufficient" if fewer than 2 valid GPUs
  python3 b11_summary.py gpu --name ... --b6b DIR --b8 DIR --b9b DIR [--firstcell DIR] --json-out F
  python3 b11_summary.py aggregate --gpu-json F1 F2 ...
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REF_GA106 = {
    "V-2": "H-D confirmed (dirty footprint only)", "V-3": "sharp step", "V-4": "counts lines",
    "V-5": "resident", "Z-1": "dvsc supported", "Z-2": "effective capacity >= 0.94 L2",
    "Z-2b": "reads retained just above capacity (not LRU-sharp)", "Q-1": "mixed",
    "Q-3": "v4 predicts one write kernel's write-backs",
}
C5N_LO_W, C5N_LO_MAX = 0.4375, 0.10
C5N_HI_W, C5N_HI_MIN = 0.625, 0.80
C5N_ABSENT_W, C5N_ABSENT_MAX = 0.6875, 0.50
C6_W, C6_PRESENT, C6_ABSENT = "0.75", 0.05, 0.02
FC_CTRL_W, FC_PASS, FC_ACTIVE = 0.375, 0.05, 0.10
X_MIN_VALID, X_MIN_PRESENT = 2, 2


def first_cell(sector_cells: list[dict], fc_verdict: dict) -> tuple[str, dict]:
    c1, c2 = wbr(sector_cells, 0, FC_CTRL_W), fc_verdict["F-1 control WBR"]
    vals = {"c1 (first)": c1, "c2 (second)": c2}
    if c1 is None:
        return "not measured", vals
    if c1 < FC_PASS:
        return "nothing to test (control passes as first cell)", vals
    if c2 < FC_PASS:
        return "first-cell artifact confirmed", vals
    if c2 >= FC_ACTIVE:
        return "not a first-cell artifact", vals
    return "inconclusive", vals


def load(d: Path | None) -> dict | None:
    if d is None or not (d / "score.json").exists():
        return None
    return json.loads((d / "score.json").read_text())


def failures(dirs: list[Path]) -> int:
    return sum(json.loads(Path(m).read_text())["correctness_failures"]
               for d in dirs for m in glob.glob(str(d / "*_meta.json")))


def wbr(cells: list[dict], r: float, w: float) -> float | None:
    for c in cells:
        if abs(c["r"] - r) < 1e-9 and abs(c["w"] - w) < 1e-9:
            return c["WBR"]
    return None


def c5n(full_cells: list[dict]) -> tuple[str, dict]:
    lo, hi = wbr(full_cells, 0, C5N_LO_W), wbr(full_cells, 0, C5N_HI_W)
    ab = wbr(full_cells, 0, C5N_ABSENT_W)
    vals = {"WBR(0:0.4375)": lo, "WBR(0:0.625)": hi, "WBR(0:0.6875)": ab}
    if lo is not None and hi is not None and lo <= C5N_LO_MAX and hi >= C5N_HI_MIN:
        return "present (transition in 0.44-0.63 L2)", vals
    if ab is not None and ab < C5N_ABSENT_MAX:
        return "absent up to 0.69 L2", vals
    return "other", vals


def c6(b9b: dict | None) -> tuple[str, float | None]:
    if b9b is None:
        return "not measured", None
    loss = b9b["verdict"]["Q-1 loss"][C6_W]
    if loss >= C6_PRESENT:
        return "present", loss
    if loss <= C6_ABSENT:
        return "absent", loss
    return "inconclusive", loss


def gpu(name: str, b6b_dir: Path, b8_dir: Path, b9b_dir: Path | None, fc_dir: Path | None) -> dict:
    b6b, b8, b9b, fc = load(b6b_dir), load(b8_dir), load(b9b_dir), load(fc_dir)
    out: dict = {"gpu": name}
    dirs = [d for d in (b6b_dir, b8_dir, fc_dir) if d is not None]
    out["P-0"] = b6b is not None and b8 is not None and failures(dirs) == 0
    got = {}
    if b6b:
        got.update({x: b6b["verdict"].get(x) for x in ("V-2", "V-3", "V-4", "V-5")})
    if b8:
        got.update({x: b8["verdict"].get(x) for x in ("Z-1", "Z-2", "Z-2b")})
    if b9b:
        got.update({x: b9b["verdict"].get(x) for x in ("Q-1", "Q-3")})
    out["verdicts"] = got
    out["match"] = {x: got.get(x) == ref for x, ref in REF_GA106.items()}
    out["n_match"] = sum(out["match"].values())
    out["C5n"], out["C5n values"] = c5n(b6b["full"]) if b6b else ("not measured", {})
    out["C6"], out["C6 loss"] = c6(b9b)
    if fc:
        out["first-cell"] = {x: fc["verdict"][x] for x in ("F-0", "F-1", "F-2", "V-4'")}
        if b6b:
            out["FC"], out["FC values"] = first_cell(b6b["sector"], fc["verdict"])
    return out


def cross(rows: list[dict], key: str, present_prefix: str) -> str:
    valid = [r for r in rows if r["P-0"] and not r[key].startswith("not measured")]
    if len(valid) < X_MIN_VALID:
        return f"insufficient ({len(valid)} valid GPUs)"
    n = sum(r[key].startswith(present_prefix) for r in valid)
    if n >= X_MIN_PRESENT:
        return f"generalizes (present on {n}/{len(valid)})"
    if n == 0:
        return f"Ampere-specific (present on 0/{len(valid)})"
    return f"mixed (present on {n}/{len(valid)})"


def aggregate(rows: list[dict], log=print) -> dict:
    for r in rows:
        log(f"{r['gpu']}: P-0 {'PASS' if r['P-0'] else 'FAIL'} | match GA106 {r['n_match']}/9 "
            f"(miss: {', '.join(x for x, ok in r['match'].items() if not ok) or '-'}) | C5n {r['C5n']} "
            f"{r['C5n values']} | C6 {r['C6']} (L={r['C6 loss']})"
            + (f" | first-cell {r['first-cell']}" if "first-cell" in r else "")
            + (f" | FC {r['FC']} {r['FC values']}" if "FC" in r else ""))
    v = {"C5n-X": cross(rows, "C5n", "present"), "C6-X": cross(rows, "C6", "present")}
    log("cross-architecture verdict: " + json.dumps(v))
    return v


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gpu")
    g.add_argument("--name", required=True)
    g.add_argument("--b6b", type=Path, required=True)
    g.add_argument("--b8", type=Path, required=True)
    g.add_argument("--b9b", type=Path)
    g.add_argument("--firstcell", type=Path)
    g.add_argument("--json-out", type=Path, required=True)
    ag = sub.add_parser("aggregate")
    ag.add_argument("--gpu-json", type=Path, nargs="+", required=True)
    ag.add_argument("--json-out", type=Path)
    a = p.parse_args()
    if a.cmd == "gpu":
        row = gpu(a.name, a.b6b, a.b8, a.b9b, a.firstcell)
        a.json_out.write_text(json.dumps(row, indent=1))
        aggregate([row])
        return 0
    rows = [json.loads(f.read_text()) for f in a.gpu_json]
    v = aggregate(rows)
    if a.json_out:
        a.json_out.write_text(json.dumps({"gpus": rows, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
