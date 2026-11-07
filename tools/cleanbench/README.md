# cleanbench

CUDA-event timing of a GPU kernel that avoids three pitfalls of the usual "flush L2, then time" loop
(`triton.testing.do_bench`, PyTorch Inductor's benchmarker and most Python harnesses). One file, PyTorch + Triton.

| Pitfall | What goes wrong | What cleanbench does |
|---|---|---|
| Write-based flush (`cache.zero_()`) | the flush leaves dirty L2 lines; the timed kernel pays their DRAM write-back | flushes by **reading** a buffer of max(256 MiB, 4 × L2) with volatile loads (kernel from triton-lang/triton PR #12556) |
| Queue drain | a flush shorter than the host's dispatch time lets the GPU go idle; the start event fires before the kernel is enqueued, so host time is timed | a **GPU busy-wait** before every start event, sized min(5 ms, max(100 µs, 3 × measured dispatch time)); NVIDIA nvbench's `blocking_kernel` solves the same problem in C++ |
| Who pays for writes | after a read flush, a kernel's own output stays dirty and its write-back happens during the next flush, outside the timed window | `mode="steady"` rotates over copies of the inputs/outputs so each call pays an earlier call's write-back |

```python
import cleanbench as cb

cold = cb.bench(lambda: f(x, y))                       # clean, evicted L2; host latency hidden
steady = cb.bench(rotate=[lambda i=i: f(xs[i], ys[i]) for i in range(n)], mode="steady")
print(cold.median_us, cold.p25_us, cold.p75_us, steady.median_us, cold.warnings)
```

- **cold** times the kernel's work with a clean L2; up to roughly half the L2 of its *own* write-back is not timed.
- **steady** times a stream of independent calls, each paying one output's write-back. Report both for kernels that
  write different amounts (e.g. fused vs unfused).
- For `steady`, pass callables over copies whose total footprint is at least 4 × L2.
- Warnings: median within ~30 timer ticks (≈ 1.02 µs each) with most samples on the tick grid; IQR above 10 % of the
  median; in cold mode, the untimed own write-back.

## Kernel time from GPU timestamps: `bench_graph` / `compare` (validated to 1 % under conditions)

CUDA events time more than the kernel: they add a fixed 2–6 µs per call (launch and event work), which is up to +40 %
for a 10 µs kernel. `bench_graph` reads the kernel's own start and end timestamps from the GPU (CUPTI activity records
through `torch.profiler`; no profiling permission needed). It runs them over a CUDA graph that rotates through copies
of the inputs, so the inputs are cold as in a model step, after at least 1 s of sustained replays so that clocks are in
their loaded state. `compare` times several candidates in interleaved rounds, so drift affects all of them alike.

```python
rot = [lambda i=i: f(xs[i], ys[i]) for i in range(n)]      # copies covering >= 4 x L2
r = cb.bench_graph(rot, footprint_bytes=n * bytes_per_copy) # median kernel time per call, from CUPTI
print(r.median_us, r.quality["validated"], r.warnings)
rs = cb.compare({"fp16": rot16, "int4": rot4}, footprints={"fp16": fp16_bytes, "int4": int4_bytes})
with cb.locked_sm_clock(1000) as locked:                     # needs root / Administrator; resets on exit
    r = cb.bench_graph(rot, footprint_bytes=n * bytes_per_copy)
```

**Quality gates.** `Result.quality` reports the conditions of the 1 % validation, and `quality["validated"]` is True
only if all hold; otherwise the number is exploratory and a warning says why:
- every CUPTI segment held exactly the expected kernels;
- tracing changed the graph's replay time by at most 0.5 % (traced and untraced replays are both timed);
- the driver's clock-event counters were readable and did not grow (these catch throttling, not every small GPU Boost
  step; lock the clock for 1 %);
- the rotated inputs cover at least 4 × L2 (`footprint_bytes`; not checked if omitted).

Candidates run in a seeded random order in every round. Each callable must run its kernels on the current stream
without overlap; its time is the sum of its kernels' durations.

**The clock must be steady for 1 %.** GPU Boost lowers the SM clock when the GPU is hot or at its power limit. Lock it
with `locked_sm_clock` (`nvidia-smi -lgc`, root or Administrator only) at a clock the GPU can hold under load. If you
cannot lock it (cloud containers, WSL), `bench_graph` compares the clock-event counters (`nvidia-smi -q -d PERFORMANCE`)
before and after, and warns if the GPU slowed down.

**Evidence (B23b, RTX 3060 Laptop, SM clock locked at 1,000 MHz; decode proxy with 4 layers of Llama-2-7B shapes,
batch 1, fp16 / int4 / NF4 GEMVs):** the reference is the same kernel's CUPTI time inside a CUDA-graph decode step,
measured in interleaved rounds.

| Harness | Error vs in-model kernel time (12 kernels) | Within 1 % |
|---|---|---|
| CUPTI, rotation in a graph, 1 s warm-up | −0.79 to +0.39 % | 12/12 |
| CUDA events, rotation in a graph | −0.01 to +2.4 % | 6/12 |
| CUPTI, cold (read flush + busy-wait) | −2.0 to +3.4 % | 1/12 |
| CUDA events, cold (`bench`) | +1.3 to +13.3 % | 0/12 |

Without the clock lock (B23), the GPU reached 87 °C and its SM clock varied from 1,005 to 1,747 MHz. The in-model time
of int4 and NF4 kernels then varied by 27–30 % between rounds, and no harness could match it to 1 %.

**Validation** (details in the technical report): on an RTX 3060 Laptop, A100, RTX 5090 and L40S, the cold time did not
depend on the flush size for long kernels, steady minus cold matched the write-back measured with DRAM counters, and the
shipped `do_bench` read 11-25 % above cleanbench's cold time for a 112 MiB decode GEMV.

**Limitations.** `torch.cuda._sleep` is a private PyTorch API. With CUDA events (`bench`), kernels shorter than ~30 µs
are limited by the timer. The 1 % result of `bench_graph` comes from one laptop GPU with a locked clock and kernels of
49–645 µs, and the reference was a GEMV-only decode proxy. Confirmation on data-center GPUs is pending. Only NVIDIA GPUs
were tested.

**Credits.** Read-flush kernel: triton-lang/triton PR #12556 (MIT). Queue-drain remedy: the same idea as NVIDIA nvbench's
`blocking_kernel`. CUPTI kernel timing: also in FlashInfer (`bench_gpu_time_with_cupti`). Clock locking: also in
Triton (`triton.testing.set_gpu_clock`). Throttle detection: also in nvbench. License of this package: to be decided by the author.
