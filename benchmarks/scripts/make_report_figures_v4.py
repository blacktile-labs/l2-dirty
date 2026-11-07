#!/usr/bin/env python3
"""Figures 2-4 of the technical report (v4, D-108).
  fig_dirty_l2.pdf        schematic: who pays for dirty lines under a write flush, a read flush, and the kernel's own output
  fig_queue_drain.pdf     schematic: host/GPU timeline of queue drain, and the busy-wait that removes it
  fig_kernel_errors.pdf   data: per-kernel error vs the same kernel inside the decode proxy (B23b laptop locked,
                          B23c RTX 4060), CUPTI rotation vs CUDA events in the cold loop, with a +-1 % band
  python3 make_report_figures_v4.py --outdir reports/outreach/arxiv
Palette: Blacktile "Redline" design tokens on white paper: neutrals from the material greys (kernel #26262c,
write flush #6b6b75, read flush #a1a1aa, launch gap a light tint), and the accent red #e63946 only for what the
measurement should not include (dirty write-back, the drained queue, the biased timer). Identity also by lightness,
hatch, marker shape and direct labels, never colour alone.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

KERNEL, WFLUSH, RFLUSH, RED = "#26262c", "#6b6b75", "#a1a1aa", "#e63946"   # --bt-material, -edge, text-secondary, accent
INK, INK2, GRID, IDLE = "#0b0b0d", "#6b6b75", "#ececee", "#d9d9dd"
RED_SOFT = "#fce7e9"   # --bt-accent-soft (12 % red) over white
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK})


def block(ax, x0, x1, y, h, color, label=None, hatch=None, tcolor=INK, fs=7.5, ec="white", ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.06", linewidth=0.8,
                                edgecolor=ec, facecolor=color, hatch=hatch, linestyle=ls))
    if label:
        ax.text((x0 + x1) / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tcolor)


def bracket(ax, x0, x1, y, text):
    ax.annotate("", xy=(x0, y), xytext=(x1, y), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0))
    ax.plot([x0, x0], [y - 0.12, y + 0.12], color=INK, lw=1.0)
    ax.plot([x1, x1], [y - 0.12, y + 0.12], color=INK, lw=1.0)
    ax.text((x0 + x1) / 2, y + 0.18, text, ha="center", va="bottom", fontsize=7.5, color=INK)


def fig_dirty(out: Path):
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(3, 1, figsize=(6.3, 3.4))
    rows = [
        ("(a) Write flush", [(0.2, 3.0, WFLUSH, None), (3.0, 3.3, IDLE, None), (3.3, 5.1, KERNEL, None),
                             (5.1, 6.9, RED, "///")],
         (3.3, 6.9, "timed: kernel + the flush's write-back")),
        ("(b) Read flush", [(0.2, 3.0, RFLUSH, None), (3.0, 3.3, IDLE, None), (3.3, 5.1, KERNEL, None)],
         (3.3, 5.1, "timed: kernel only")),
        ("(c) Read flush,\nnext call", [(0.2, 1.4, KERNEL, None), (1.4, 2.9, RED, "///"), (2.9, 4.6, RFLUSH, None),
                                        (4.6, 4.9, IDLE, None), (4.9, 6.7, KERNEL, None)],
         None),
    ]
    for ax, (title, blocks, timed) in zip(axes, rows):
        ax.set_xlim(-1.6, 7.1)
        ax.set_ylim(-0.1, 1.25)
        ax.axis("off")
        ax.text(-1.55, 0.3, title, ha="left", va="center", fontsize=8, color=INK, fontweight="bold")
        for x0, x1, c, h in blocks:
            block(ax, x0, x1, 0.05, 0.5, c, None, h)
        if timed:
            bracket(ax, timed[0], timed[1], 0.78, timed[2])
    axes[2].text(2.15, 0.82, "own output written back during the next flush: not timed", fontsize=7.5, color=INK,
                 ha="center")
    axes[2].annotate("time →", xy=(7.05, -0.05), ha="right", fontsize=7.5, color=INK2)
    handles = [Patch(facecolor=WFLUSH, edgecolor="white", label="write flush"),
               Patch(facecolor=RFLUSH, edgecolor="white", label="read flush"),
               Patch(facecolor=KERNEL, edgecolor="white", label="kernel"),
               Patch(facecolor=RED, edgecolor="white", hatch="///", label="write-back of dirty lines"),
               Patch(facecolor=IDLE, edgecolor="white", label="launch gap")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7.5, handlelength=1.6,
               columnspacing=1.2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(h_pad=0.2, rect=(0, 0.07, 1, 1))
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)


def fig_queue(out: Path):
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 3.0))
    for ax, padded in zip(axes, (False, True)):
        ax.set_xlim(-1.5, 6.4)
        ax.set_ylim(-0.25, 2.3)
        ax.axis("off")
        ax.text(-1.45, 1.55, "host", fontsize=8, color=INK2, va="center")
        ax.text(-1.45, 0.55, "GPU", fontsize=8, color=INK2, va="center")
        # host: enqueue flush, (pad), start, dispatch fn (slow, Python), end
        block(ax, -0.3, 0.3, 1.35, 0.4, IDLE, "flush", fs=6.5)
        hx = 0.3
        if padded:
            block(ax, hx, hx + 0.55, 1.35, 0.4, IDLE, "wait", fs=6.5)
            hx += 0.55
        block(ax, hx, hx + 0.55, 1.35, 0.4, IDLE, "start", fs=6.5)
        block(ax, hx + 0.55, hx + 2.9, 1.35, 0.4, IDLE, "dispatch fn (Python, launcher)", fs=6.5)
        block(ax, hx + 2.9, hx + 3.35, 1.35, 0.4, IDLE, "end", fs=6.5)
        launch = hx + 2.9
        # GPU
        block(ax, 0.3, 1.2, 0.35, 0.4, RFLUSH, "flush", fs=7)
        if padded:
            block(ax, 1.2, launch + 0.05, 0.35, 0.4, WFLUSH, "GPU busy-wait (not timed)", tcolor="white", fs=7)
            t0 = launch + 0.05
        else:
            block(ax, 1.22, launch + 0.03, 0.35, 0.4, RED_SOFT, "idle: queue drained", tcolor=INK, fs=7, ec=RED, ls="--")
            t0 = 1.2
        block(ax, launch + 0.05, launch + 1.6, 0.35, 0.4, KERNEL, "kernel", tcolor="white", fs=7)
        ax.plot([t0, t0], [0.25, 0.85], color=INK, lw=1.0)
        ax.plot([launch + 1.6, launch + 1.6], [0.25, 0.85], color=INK, lw=1.0)
        lab = "event interval = kernel" if padded else "event interval = idle gap + kernel"
        ax.annotate("", xy=(t0, 0.05), xytext=(launch + 1.6, 0.05), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0))
        ax.text((t0 + launch + 1.6) / 2, -0.2, lab, ha="center", va="center", fontsize=7.5, color=INK)
        ax.text(-1.45, 2.15, "(b) busy-wait before the start event" if padded else "(a) short flush: start event fires early",
                fontsize=8, fontweight="bold", color=INK, va="center")
    fig.tight_layout(h_pad=0.4)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)


def fig_errors(out: Path, laptop: Path, rtx4060: Path):
    panels = [("RTX 3060 Laptop, clock locked (gates passed)", json.load(open(laptop))),
              ("RTX 4060, rented (W-0 passed, clock gate failed)", json.load(open(rtx4060)))]
    order = [f"{v} {s}" for v in ("fp16", "int4", "nf4") for s in ("qkv", "o", "gate_up", "down")]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.3), sharey=True)
    for ax, (title, d) in zip(axes, panels):
        ax.axvspan(-1, 1, color=GRID, zorder=0)
        ax.axvline(0, color=INK2, lw=0.6, zorder=1)
        ys = list(range(len(order)))[::-1]
        for k, y in zip(order, ys):
            r = 100 * d["E"]["cupti_rot"][k]
            e = 100 * d["E"]["event_cold"][k]
            ax.plot([r, e], [y, y], color=IDLE, lw=1.0, zorder=1)
            ax.scatter([e], [y], marker="s", s=22, facecolor="white", edgecolor=RED, linewidths=1.3, zorder=3)
            ax.scatter([r], [y], marker="o", s=26, color=KERNEL, edgecolor="white", linewidths=0.8, zorder=4)
        ax.set_yticks(ys)
        ax.set_yticklabels(order, fontsize=7.5)
        ax.set_xlim(-16, 16)
        ax.set_xlabel("error vs the same kernel in the proxy (%)")
        ax.set_title(title, fontsize=8, color=INK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text(0, -0.95, "±1 %", ha="center", va="center", fontsize=7, color=INK2)
        ax.set_ylim(-1.3, len(order) - 0.5)
    axes[0].scatter([], [], marker="o", s=26, color=KERNEL, label="CUPTI rotation in a CUDA graph")
    axes[0].scatter([], [], marker="s", s=22, facecolor="white", edgecolor=RED, linewidths=1.3,
                    label="CUDA events, cold loop (cleanbench bench)")
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.55, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--laptop", default="benchmarks/results/kernel_time_20261106-180845/score.json")
    ap.add_argument("--rtx4060", default="benchmarks/results/b23c_4060_20261107/score_local.json")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dirty(out / "fig_dirty_l2.pdf")
    fig_queue(out / "fig_queue_drain.pdf")
    fig_errors(out / "fig_kernel_errors.pdf", Path(a.laptop), Path(a.rtx4060))
    print("wrote fig_dirty_l2, fig_queue_drain, fig_kernel_errors (.pdf, .png) in", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
