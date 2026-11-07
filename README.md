# Dirty L2, Drained Queues: Measurement Pitfalls in GPU Kernel Microbenchmarks

Rungrod Thongjampa, Chanchada Wathanathonnan — Blacktile Research
Contact: rungrod.th@blacktile.dev

Technical report, code and data. Report: [`reports/outreach/arxiv/main.pdf`](reports/outreach/arxiv/main.pdf)
(source: [`reports/TECH_REPORT_DRAFT.md`](reports/TECH_REPORT_DRAFT.md)); arXiv link to be added.

## Summary

Before each timed call, most GPU kernel microbenchmarks evict the L2 cache. In 14 of the 15 tools and benchmark
paths we audited, including `triton.testing.do_bench`, they do it by **writing** a large buffer. The writes leave
dirty L2 lines that the timed kernel must write back. On nine NVIDIA GPUs, a batch-1 fp16 decode GEMV measured
6.6-25.3 % slower with this write flush than with a read flush. The report also covers two related pitfalls:
queue drain (host time inside a CUDA-event interval) and a kernel's own write-back left untimed. It then checks
per-kernel times against the same kernels inside a GEMV-only decode proxy captured in a CUDA graph. The experiments
were pre-registered.

## Layout

| Path | Contents |
|---|---|
| `tools/cleanbench/` | the harness: read flush, GPU busy-wait, steady rotation, CUPTI timing with quality gates (`bench`, `bench_graph`, `compare`, `locked_sm_clock`) |
| `benchmarks/scripts/` | every probe, runner, scorer and unit test; `vast/` holds the rental scripts |
| `benchmarks/src/` | CUDA sources of the early microbenchmarks |
| `benchmarks/results/` | raw JSON/CSV/logs of every run (IP addresses of rented hosts removed) |
| `research/HYPOTHESES.md` | pre-registrations, rules, predictions and results of every experiment (Thai; numbers and code are in English) |
| `research/preregistration/` | sha256 locks, snapshots of every locked file as committed, `verify_locks.py` |
| `research/TOOL_SURVEY.md` | the audit of 15 benchmarking tools, with permalinks |
| `reports/` | the report source, LaTeX, PDF, figures and their build scripts |
| `reports/outreach/repro_inductor_flush_bias.py` | a one-GPU repro of the write-flush bias in PyTorch Inductor's benchmarker |

## Quick start: cleanbench

```bash
pip install torch==2.7.1 triton==3.3.1        # versions used in the report
cd tools/cleanbench && python example.py && python -m unittest test_cleanbench
```

```python
import cleanbench as cb
cold = cb.bench(lambda: f(x, y))                                  # read flush + busy-wait, CUDA events
rot = [lambda i=i: f(xs[i], ys[i]) for i in range(n)]             # copies covering >= 4 x L2
r = cb.bench_graph(rot, footprint_bytes=n * bytes_per_copy)       # CUPTI timing in a CUDA graph
print(r.median_us, r.quality["validated"], r.warnings)
```

`bench_graph` reached 1 % of the in-proxy kernel time only with a locked SM clock, on one GPU. `quality["validated"]`
is False, and the result exploratory, unless the tracing, clock and footprint checks pass. See
[`tools/cleanbench/README.md`](tools/cleanbench/README.md).

## Reproducing the report

- **Scores from the committed data** (no GPU): each experiment's scorer reads its run directory, e.g.
  `python3 benchmarks/scripts/kernel_time_score.py --meta benchmarks/results/kernel_time_20261106-180845/run_r*_meta.json`.
  Unit tests: `cd benchmarks/scripts && python3 -m unittest discover -p 'test_*.py'`.
- **Figures:** `python3 benchmarks/scripts/make_report_figures.py --out reports/outreach/arxiv/fig_bias_vs_l2.pdf` and
  `python3 benchmarks/scripts/make_report_figures_v4.py --outdir reports/outreach/arxiv` (matplotlib).
- **PDF:** `python3 reports/outreach/arxiv/make_tex.py` (needs pandoc), then `tectonic reports/outreach/arxiv/main.tex`.
- **New measurements:** every experiment has a runner `benchmarks/scripts/*_probe.sh` (with `--smoke`). Rented-GPU
  runs used `benchmarks/scripts/vast/vast_run.sh` with your own vast.ai API key and SSH key. Rented hosts differ in
  drivers and power limits, so absolute numbers will differ (the report measured 9-27 % between two hosts of the
  same GPU model).

## Pre-registration

Every experiment's rules and predictions were written and hashed before its measurements were read.
`python3 research/preregistration/verify_locks.py` checks each lock against the snapshot of the locked files:
190 of 191 match. One early lock's protocol file was edited before it was committed; `lock_provenance.json` explains
which. Lock times come from the authors' development history, which is not public, so they are self-attested.

## License

Code: Apache License 2.0 ([`LICENSE`](LICENSE)). Report, figures, pre-registration texts and data: CC BY 4.0
([`LICENSE-CC-BY-4.0.md`](LICENSE-CC-BY-4.0.md)). The read-flush kernel comes from triton-lang/triton PR #12556 (MIT;
[`third_party/TRITON_LICENSE`](third_party/TRITON_LICENSE)). See [`NOTICE`](NOTICE).

## Citation

See [`CITATION.cff`](CITATION.cff). arXiv identifier to be added.
