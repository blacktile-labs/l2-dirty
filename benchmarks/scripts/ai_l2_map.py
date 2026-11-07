#!/usr/bin/env python3
"""Where do transformer-block intermediates sit relative to the L2? (analysis only, no measurement)

For a generic pre-norm decoder block (RMSNorm, GQA attention, SwiGLU MLP; fp16/bf16 = 2 bytes) and
a range of token counts T (decode: T = batch; prefill: T = batch x sequence), list the tensors one
eager kernel hands to the next and classify each against a GPU's L2 with the rules MEASURED in this
project (C5n, B6b/B8/B11; Turing, Ampere, Ada):
  "L2"    size <= 0.5 L2        written, then read by the next kernel without DRAM traffic
                                (below the dirty cap): fusing saves launches and L2 traffic only
  "edge"  0.5 L2 < size <= L2   partly written back (dirty cap) and partly re-read from the L2:
                                the zone where our model is least accurate (B7 R-1)
  "DRAM"  size > L2             a DRAM round trip (write + read): fusing saves ~2 x size bytes
The shapes are generic (d_model, heads, KV heads, d_ff), not tied to a specific model.
  python3 ai_l2_map.py [--json OUT] [--markdown OUT]
"""

from __future__ import annotations

import argparse
import json
import sys

MiB = 1 << 20
BYTES = 2
# L2 sizes: "measured" = cudaDeviceProp.l2CacheSize read by our probes on that GPU; "spec" = vendor
# documentation, NOT measured here.
GPUS = [
    ("RTX 3060 Laptop (GA106)", 3 * MiB, "measured"),
    ("RTX 2060S (TU106)", 4 * MiB, "measured"),
    ("RTX 3090 (GA102)", 6 * MiB, "measured"),
    ("RTX 4060 (AD107)", 24 * MiB, "measured"),
    ("A100 (GA100)", 40 * MiB, "spec"),
    ("H100 SXM (GH100)", 50 * MiB, "spec"),
    ("RTX 4080S (AD103)", 64 * MiB, "measured"),
]
CONFIGS = [  # name, d_model, n_heads, n_kv_heads, d_ff
    ("d2048", 2048, 16, 16, 5632),
    ("d4096", 4096, 32, 8, 14336),
    ("d8192", 8192, 64, 8, 28672),
]
DECODE_T = (1, 8, 32, 128)
PREFILL_T = (512, 2048, 8192)
CAP, FULL = 0.5, 1.0


def tensors(d: int, h: int, kv: int, ff: int, T: int) -> dict[str, int]:
    dkv = d * kv // h
    return {"T x d (norm, attn-out, residual)": T * d * BYTES,
            "qkv": T * (d + 2 * dkv) * BYTES,
            "gate+up (T x 2 d_ff)": T * 2 * ff * BYTES,
            "act (T x d_ff)": T * ff * BYTES}


def zone(size: int, l2: int) -> str:
    return "L2" if size <= CAP * l2 else "edge" if size <= FULL * l2 else "DRAM"


def build() -> list[dict]:
    rows = []
    for name, d, h, kv, ff in CONFIGS:
        for phase, ts in (("decode", DECODE_T), ("prefill", PREFILL_T)):
            for T in ts:
                for tname, size in tensors(d, h, kv, ff, T).items():
                    rows.append({"config": name, "phase": phase, "T": T, "tensor": tname, "bytes": size,
                                 "zones": {g: zone(size, l2) for g, l2, _ in GPUS}})
    return rows


def markdown(rows: list[dict]) -> str:
    sym = {"L2": "L2", "edge": "**edge**", "DRAM": "DRAM"}
    out = ["| config | phase | T | tensor | MiB | " + " | ".join(f"{g} ({l2 // MiB} MiB, {src})"
                                                          for g, l2, src in GPUS) + " |",
           "|---" * (5 + len(GPUS)) + "|"]
    for r in rows:
        out.append(f"| {r['config']} | {r['phase']} | {r['T']} | {r['tensor']} | {r['bytes'] / MiB:.3g} | "
                   + " | ".join(sym[r["zones"][g]] for g, _, _ in GPUS) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json")
    ap.add_argument("--markdown")
    a = ap.parse_args()
    rows = build()
    if a.json:
        json.dump({"gpus": GPUS, "configs": CONFIGS, "rows": rows}, open(a.json, "w"), indent=1)
    md = markdown(rows)
    if a.markdown:
        open(a.markdown, "w").write(md + "\n")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
