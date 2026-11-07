#!/usr/bin/env python3
"""Build the arXiv LaTeX source (main.tex) from reports/TECH_REPORT_DRAFT.md so the two never diverge.
Needs pandoc. Output is pure ASCII LaTeX that compiles with pdflatex (arXiv's default) and XeTeX.
  python3 reports/outreach/arxiv/make_tex.py && tectonic reports/outreach/arxiv/main.tex
"""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "TECH_REPORT_DRAFT.md"

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,longtable,array,calc}
\usepackage{graphicx}
\usepackage{flafter}
\usepackage{fancyvrb}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor={blue!45!black},citecolor={blue!45!black},urlcolor={blue!45!black}]{hyperref}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small\hyphenpenalty=10000\exhyphenpenalty=10000}
\setlength{\emergencystretch}{3em}
\linespread{1.06}
\usepackage{xurl}
\usepackage[skip=0.55\baselineskip plus 2pt]{parskip}
\usepackage{tcolorbox}
\newtcolorbox{codebox}{colback=black!3,colframe=black!15,boxrule=0.4pt,arc=2pt,left=7pt,right=7pt,top=2pt,bottom=2pt,
  before skip=6pt,after skip=8pt}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setlength{\LTcapwidth}{\textwidth}
\renewcommand{\arraystretch}{1.1}
\title{%(title)s}
\author{
\begin{tabular}{@{}c@{\hspace{2.2em}}c@{}}
Rungrod Thongjampa$^{*}$ & Chanchada Wathanathonnan \\[1pt]
{\small\texttt{rungrod.th@blacktile.dev}} & {\small\texttt{chanchada.wa@blacktile.dev}}
\end{tabular}\\[14pt]
Blacktile Research\\[4pt]
{\small $^{*}$Corresponding author}}
\date{Technical report, November 2026}
\begin{document}
\maketitle
\begin{abstract}
\noindent %(abstract)s
\end{abstract}
"""

FIGURE = r"""
\begin{figure}[tbp]
\centering
\includegraphics[width=0.62\textwidth]{fig_bias_vs_l2.pdf}
\caption{Timing bias of the shipped write flush relative to a read flush, for a batch-1 fp16 decode GEMV
(112\,MiB of weights), on nine GPUs. Within Ada the bias rises with L2 size; A100 and H100 sit above the Ada
trend, Blackwell below. Values are in the table of Section~5; the four GPUs with $\ge$64\,MiB of L2 are scored
under the amended validity check (Section~13).}
\label{fig:bias}
\end{figure}
"""

FIG_DIRTY = r"""
\begin{figure}[tbp]
\centering
\includegraphics[width=0.92\textwidth]{fig_dirty_l2.pdf}
\caption{Who pays for dirty L2 lines (schematic; in practice the write-back overlaps the kernel). (a) A write flush
leaves dirty lines that the timed kernel must write back. (b) A read flush leaves the L2 clean. (c) Under a read
flush, a kernel's own output stays dirty and is written back during the next flush, outside any timed window.}
\label{fig:dirty}
\end{figure}
"""

FIG_QUEUE = r"""
\begin{figure}[tbp]
\centering
\includegraphics[width=0.92\textwidth]{fig_queue_drain.pdf}
\caption{Queue drain (schematic). (a) When the flush ends before the host has enqueued the kernel, the GPU reaches
the start event early and the event interval includes the idle gap. (b) A GPU busy-wait enqueued before the start
event keeps the queue busy; the busy-wait itself is not timed.}
\label{fig:queue}
\end{figure}
"""

FIG_ERRORS = r"""
\begin{figure}[tbp]
\centering
\includegraphics[width=0.95\textwidth]{fig_kernel_errors.pdf}
\caption{Per-kernel error against the same kernel inside the GEMV-only decode proxy (ratio of medians over three
runs; shaded band $\pm$1\,\%). Left: laptop GPU with the SM clock locked, all gates passed. Right: rented RTX 4060,
clock not locked (power-capped); the NF4 rows show a context effect of unknown cause.}
\label{fig:errors}
\end{figure}
"""

BIB = r"""
\begin{thebibliography}{99}\footnotesize\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}
\bibitem{tr12554} triton-lang/triton issue \#12554, write flush in \texttt{do\_bench}, \url{https://github.com/triton-lang/triton/issues/12554}.
\bibitem{tr12556} triton-lang/triton PR \#12556, read the L2 flush buffer instead of writing it, \url{https://github.com/triton-lang/triton/pull/12556}.
\bibitem{pt203681} pytorch/pytorch PR \#203681 (ROCm), \url{https://github.com/pytorch/pytorch/pull/203681}.
\bibitem{tileops2822} tile-ai/TileOPs PR \#2822, \url{https://github.com/tile-ai/TileOPs/pull/2822}.
\bibitem{tileops1990} tile-ai/TileOPs issue \#1990, \url{https://github.com/tile-ai/TileOPs/issues/1990}.
\bibitem{tilelang3534} tile-ai/tilelang \#3534, \url{https://github.com/tile-ai/tilelang/pull/3534}.
\bibitem{patent} NVIDIA Corporation, \emph{Control mechanism for fine-tuned cache to backing-store synchronization}, US Patent 9,639,466 B2, filed 2012, granted 2017.
\bibitem{finn} Finn, \emph{What happens when a GPU writes memory}, Doubleword blog, 2026.
\bibitem{tr1252} triton-lang/triton issue \#1252, cache flush for benchmarking (2023), \url{https://github.com/triton-lang/triton/issues/1252}.
\bibitem{helion4126} pytorch/helion PR \#4126, L2 flush regimes for the autotuner's benchmarks, \url{https://github.com/pytorch/helion/pull/4126}.
\bibitem{fi8076} flashinfer-ai/flashinfer PR \#8076, evict L2 by reading in cold-L2 profiling, \url{https://github.com/flashinfer-ai/flashinfer/pull/8076}.
\bibitem{hipep1238} ROCm/hip-ep PR \#1238, time kernel unit tests on cold caches, \url{https://github.com/ROCm/hip-ep/pull/1238}.
\bibitem{deepep17} deepseek-ai/DeepEP-Ascend issue \#17, flush\_l2 does not take effect as expected, \url{https://github.com/deepseek-ai/DeepEP-Ascend/issues/17}.
\bibitem{nvbench} NVIDIA nvbench, \url{https://github.com/NVIDIA/nvbench}.
\bibitem{speechmatics} L.~Atkins, D.~MacLeod, \emph{How to Accurately Time CUDA Kernels in Pytorch}, Speechmatics, 2023, \url{https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch}.
\bibitem{sinha} P.~Sinha, A.~Guliani, R.~Jain, B.~Tran, M.~D.~Sinclair, S.~Venkataraman, \emph{Not All GPUs Are Created Equal: Characterizing Variability in Large-Scale, Accelerator-Rich Systems}, SC22, arXiv:2208.11035.
\bibitem{he} H.~He, \emph{Strangely, Matrix Multiplications on GPUs Run Faster When Given ``Predictable'' Data!}, 2024, \url{https://www.thonking.ai/p/strangely-matrix-multiplications}.
\bibitem{crisp} R.~Nath, D.~Tullsen, \emph{The CRISP Performance Model for Dynamic Voltage and Frequency Scaling in a GPGPU}, MICRO-48, 2015, doi:10.1145/2830772.2830826.
\bibitem{flashinfer} FlashInfer, \texttt{flashinfer.testing.bench\_gpu\_time\_with\_cupti}, \url{https://docs.flashinfer.ai/generated/flashinfer.testing.bench_gpu_time_with_cupti.html}.
\bibitem{ernst} D.~Ernst, G.~Hager, M.~Holzer, M.~Knorr, G.~Wellein, \emph{Opening the Black Box: Performance Estimation during Code Generation for GPUs}, arXiv:2107.01143, 2021.
\end{thebibliography}
\end{document}
"""

UNI = {"≈": r"\ensuremath{\approx}", "×": r"\ensuremath{\times}", "µ": r"\ensuremath{\mu}", "ρ": r"\ensuremath{\rho}",
       "−": r"\ensuremath{-}", "≤": r"\ensuremath{\le}", "≥": r"\ensuremath{\ge}", "²": r"\ensuremath{^2}",
       "→": r"\ensuremath{\rightarrow}", "±": r"\ensuremath{\pm}", "–": "--", "—": "---", "“": "``", "”": "''",
       "‘": "`", "’": "'", "§": r"\S{}", "…": r"\ldots{}", "Δ": r"\ensuremath{\Delta}", "÷": r"\ensuremath{\div}", "°": r"\ensuremath{^\circ}", "₀": r"\ensuremath{_0}", "₁": r"\ensuremath{_1}", "\u00a0": "~"}


def ascii_tex(s: str) -> str:
    for k, v in UNI.items():
        s = s.replace(k, v)
    bad = sorted({c for c in s if ord(c) > 127})
    if bad:
        sys.exit(f"non-ASCII characters left: {bad}")
    return s


def wrap_tables(body: str) -> str:
    """pandoc emits {@{}ll..l@{}} for gfm tables (no wrapping): give each column a ragged-right p{} width
    proportional to its longest cell (crudely de-TeXed), so wide tables fit the text width."""
    def fix(m):
        spec = m.group(1)
        n = len(spec)
        rows = [r for r in m.group(2).split("\\\\") if " & " in r]
        lens = [7] * n
        for ri, r in enumerate(rows):
            for k, c in enumerate(r.split(" & ")[:n]):
                c = re.sub(r"\\(?:hyperref|cite)\[[^\]]*\]|\\cite\{[^}]*\}", "", c)      # link targets take no width
                txt = re.sub(r"\\[a-zA-Z]+|[{}]|\\noalign|toprule|midrule|bottomrule", "", c).strip()
                # header row: only its longest word must fit; data rows: whole cell, capped
                wid = lambda t: sum(1.35 if ch.isupper() or ch.isdigit() else 1.0 for ch in t) + 1  # noqa: E731
                w = max((wid(t) for t in txt.split()), default=0) if ri == 0 else min(wid(txt), 45)
                lens[k] = max(lens[k], w)
        tot = sum(lens)
        cols = "".join(rf">{{\{'raggedleft' if a == 'r' else 'raggedright'}\arraybackslash}}"
                       rf"p{{(\linewidth - {2 * n}\tabcolsep) * \real{{{x / tot:.3f}}}}}" for a, x in zip(spec, lens))
        return rf"\begin{{longtable}}[]{{@{{}}{cols}@{{}}}}" + m.group(2)
    return re.sub(r"\\begin\{longtable\}\[\]\{@\{\}([lr]+)@\{\}\}(.*?)(?=\\end\{longtable\})", fix, body, flags=re.S)


CITE_NUM = {"12556": "tr12556", "12554": "tr12554", "1252": "tr1252", "203681": "pt203681", "2822": "tileops2822",
            "1990": "tileops1990", "3534": "tilelang3534", "4126": "helion4126", "8076": "fi8076", "1238": "hipep1238",
            "17": "deepep17"}
CITE_TEXT = [  # (LaTeX text in the related-work section, bib key): cited once, where the work is credited
    ("NVIDIA US 9,639,466 B2", "patent"), ("\\emph{What happens when a GPU writes memory} (2026)", "finn"),
    ("NVIDIA nvbench\\textquotesingle s", "nvbench"), ("(Atkins and MacLeod, 2023)", "speechmatics"),
    ("\\texttt{bench\\_\\allowbreak{}gpu\\_\\allowbreak{}time\\_\\allowbreak{}with\\_\\allowbreak{}cupti}", "flashinfer"),
    ("Sinha et al. (SC22; arXiv 2208.11035)", "sinha"), ("H. He (2024)", "he"),
    ("CRISP (Nath and Tullsen, MICRO 2015)", "crisp"), ("Ernst et al. 2021", "ernst")]


def space_arch_groups(body: str) -> str:
    """In tables with an "Arch" column, add a little space where the architecture changes, so groups read as groups."""
    def fix(m):
        lines, out, prev = m.group(0).split("\n"), [], None
        for ln in lines:
            cells = ln.split(" & ")
            if len(cells) > 2 and ln.rstrip().endswith("\\\\") and "Arch" not in ln:
                arch = cells[1].strip()
                if prev is not None and arch != prev:
                    out.append(r"\addlinespace[4pt]")
                prev = arch
            out.append(ln)
        return "\n".join(out)
    return re.sub(r"\\begin\{longtable\}(?:(?!\\end\{longtable\}).)*?\bArch\b.*?\\end\{longtable\}", fix, body, flags=re.S)


def add_citations(body: str) -> str:
    """Every '#<number>' of a PR or issue we credit gets a \\cite to its bibliography entry (clickable); named works
    are cited once, in the related-work section."""
    body = re.sub(r"\\#(\d+)(?!\d)", lambda m: m.group(0) + (f"~\\cite{{{CITE_NUM[m.group(1)]}}}"
                                                          if m.group(1) in CITE_NUM else ""), body)
    i = body.find(r"\section{Related work and credit}")
    head, tail = body[:i], body[i:]
    for text, key in CITE_TEXT:
        if text not in tail:
            sys.exit(f"related-work citation anchor not found: {text}")
        tail = tail.replace(text, text + f"~\\cite{{{key}}}", 1)
    return head + tail


def add_section_links(body: str) -> str:
    """Make every "§N" / "§N.M" a link to that section (numbered as LaTeX numbers them). Run on pandoc's output,
    before anything is inserted between a heading and its \\label."""
    labels, sec, sub = {}, 0, 0
    head = r"\\(section|subsection)(\*?)\{(?:[^{}]|\{[^{}]*\})*\}\s*\\label\{([^}]+)\}"
    for m in re.finditer(head, body):
        if m.group(2) or m.group(3) == "artifact":
            continue
        if m.group(1) == "section":
            sec, sub = sec + 1, 0
            labels[str(sec)] = m.group(3)
        else:
            sub += 1
            labels[f"{sec}.{sub}"] = m.group(3)
    missing = set()

    def link(m):
        k = m.group(1)
        if k not in labels:
            missing.add(k)
            return m.group(0)
        return rf"\hyperref[{labels[k]}]{{§{k}}}"
    body = re.sub(r"§(\d+(?:\.\d+)?)(?![\d.]*\d)", link, body)
    if missing:
        sys.exit(f"section references without a section: {sorted(missing)}")
    return body


def pandoc(md: str) -> str:
    r = subprocess.run(["pandoc", "-f", "gfm+tex_math_dollars", "-t", "latex", "--no-highlight", "--columns=72",
                        "--shift-heading-level-by=-1", "--wrap=preserve"],
                       input=md, capture_output=True, text=True, check=True)
    return r.stdout


def main() -> int:
    md = SRC.read_text(encoding="utf-8")
    md = re.sub(r"<!--.*?-->\s*", "", md, flags=re.S)
    title = re.search(r"^# (.+)$", md, flags=re.M).group(1).strip()
    md = re.sub(r"^# .+\n", "", md, count=1, flags=re.M)
    md = re.sub(r"^\*Draft technical report.*?\*\n", "", md, flags=re.M | re.S)
    abs_m = re.search(r"^## Abstract\n(.*?)(?=^## )", md, flags=re.S | re.M)
    abstract, md = abs_m.group(1).strip(), md.replace(abs_m.group(0), "")
    md = re.sub(r"(\d) %", "\\1\u00a0%", md)                       # keep "25.3 %" on one line
    md = re.sub(r"(\d) (MiB|GiB|KiB|MB|GB|KB|µs|ms|MHz|W)\b", "\\1\u00a0\\2", md)  # ... and "112 MiB"
    md = re.sub(r"^## (\d+)\. ", "## ", md, flags=re.M)          # LaTeX numbers sections itself
    md = re.sub(r"^### (\d+)\.(\d+) ", "### ", md, flags=re.M)  # ... and subsections
    # public version: no unread citations, no internal decision-log references
    md = re.sub(r" Jia et al\. \(Volta/Turing dissection\) is \*cited from its abstract;\s*not yet read\*\.", "", md)
    if "Jia" in md or "D-0" in md:
        sys.exit("internal or unread reference left in the public text")
    body = pandoc(md)
    body = add_section_links(body)
    body = body.replace(r"\section{Artifact}", r"\section*{Artifact}")
    # Figure 1 goes before section 5's table; label that table for the caption reference
    for head, fig in ((r"\section{Background}", FIG_DIRTY), (r"\section{Pitfall 2: queue drain}", FIG_QUEUE),
                      (r"\subsection{Kernel by kernel: does a single-kernel benchmark match the same kernel inside the "
                       r"decode proxy?}", FIG_ERRORS)):
        if head not in body:
            sys.exit(f"heading for figure not found: {head}")
        body = body.replace(head, head + "\n" + fig, 1)
    body = body.replace(r"\section{How big is the bias? Nine GPUs}",
                        r"\section{How big is the bias? Nine GPUs}" + "\n" + FIGURE, 1)
    i = body.find(r"\section{How big is the bias? Nine GPUs}")
    j = body.find(r"\end{longtable}", i)
    if i < 0 or j < 0:
        sys.exit("section 5 table not found")
    body = wrap_tables(body)
    body = space_arch_groups(body)
    body = body.replace(r"\_", r"\_\allowbreak{}")              # let long identifiers break after "_"
    body = add_citations(body)
    body = body.replace(r"\begin{verbatim}", r"\begin{codebox}" + "\n" + r"\begin{Verbatim}[fontsize=\small]")
    body = body.replace(r"\end{verbatim}", r"\end{Verbatim}" + "\n" + r"\end{codebox}")
    body = body.replace(r"\begin{longtable}", r"{\small" + "\n" + r"\begin{longtable}")
    body = body.replace(r"\end{longtable}", r"\end{longtable}" + "\n}")
    out = PREAMBLE % {"title": ascii_tex(pandoc(title).strip()),
                      "abstract": ascii_tex(pandoc(abstract).strip())} + ascii_tex(body) + BIB
    (HERE / "main.tex").write_text(out, encoding="ascii")
    print(f"wrote {HERE / 'main.tex'} ({len(out.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
