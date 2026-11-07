#!/usr/bin/env python3
"""Figure 1 of the technical report: write-flush timing bias of a batch-1 fp16 decode GEMV vs L2 size,
nine GPUs (B17, research/HYPOTHESES.md section B17). Reads the committed local rescore
benchmarks/results/b17_summary_20261106/b17_*.json; writes a PDF (and PNG for review).
  python3 make_report_figures.py --out reports/outreach/arxiv/fig_bias_vs_l2.pdf
Palette: dataviz reference categorical slots 1-4 (validated: CVD adjacent dE >= 9.1; slots 3-4 below 3:1
contrast on white, so every point is direct-labelled, architectures also differ by marker shape, and
the numbers are in Table 2 of the report).
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARCH = {"8.9": ("Ada", "#2a78d6", "o"), "8.0": ("Ampere", "#eb6834", "s"),
        "9.0": ("Hopper", "#1baf7a", "D"), "12.0": ("Blackwell", "#eda100", "^")}
SHORT = {"NVIDIA GeForce RTX 4060": "4060", "NVIDIA GeForce RTX 4060 Ti": "4060 Ti",
         "NVIDIA GeForce RTX 4080 SUPER": "4080 SUPER", "NVIDIA GeForce RTX 4090": "4090", "NVIDIA L40S": "L40S",
         "NVIDIA GeForce RTX 5060 Ti": "5060 Ti", "NVIDIA GeForce RTX 5090": "5090",
         "NVIDIA A100-PCIE-40GB": "A100", "NVIDIA H100 NVL": "H100 NVL"}
# label offsets (points) chosen to avoid overlaps at (32, ~0.07) and (96, 0.11 / 0.25)
OFF = {"4060": (6, -10), "4060 Ti": (6, 4), "5060 Ti": (6, -12), "A100": (6, 4), "H100 NVL": (6, 4),
       "4080 SUPER": (-24, -14), "4090": (6, -10), "L40S": (-30, 4), "5090": (-26, -4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="benchmarks/results/b17_summary_20261106")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pts = []
    for f in sorted(glob.glob(f"{a.summary}/b17_*.json")):
        v = json.load(open(f))
        pts.append((v["l2_bytes"] / 2**20, 100 * v["B_ref"], v["cc"], SHORT.get(v["device"], v["device"])))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for cc, (name, col, mk) in ARCH.items():
        p = [q for q in pts if q[2] == cc]
        if p:
            ax.scatter([q[0] for q in p], [q[1] for q in p], s=60 if mk == "^" else 44, c=col, marker=mk, label=name,
                       edgecolors="#fcfcfb", linewidths=1.5, zorder=3)
    for x, y, _, lab in pts:
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=OFF.get(lab, (6, 4)), fontsize=7.5,
                    color="#0b0b0b")
    ax.set_xlabel("L2 cache size (MiB)")
    ax.set_ylabel("Timing bias of write flush (%)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 30)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"wrote {out} ({len(pts)} GPUs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
