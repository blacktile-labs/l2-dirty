<!-- ร่าง technical report ฉบับที่ 4 (D-108: แก้ตาม review ภายนอก; ฉบับ 3 คือ D-107, ฉบับ 2 คือ D-088)
     ผู้แต่ง (D-109): Rungrod Thongjampa, Chanchada Wathanathonnan · ชื่อใหม่ (PI เลือก 2026-11-07): "Dirty L2, Drained Queues: Measurement Pitfalls in GPU Kernel Microbenchmarks"
     ฉบับ 3 เพิ่ม: §9.1 ตรวจราย kernel (B23 / B23b / B23c), error budget (§12), cleanbench bench_graph, related work ใหม่
     (Speechmatics, Sinha et al., Horace He, CRISP) · ผู้แต่ง: รอ PI ส่งชื่อ (อาจเป็น 3 คน)
     สถานะ: DRAFT ภายใน ห้ามเผยแพร่ PI เป็นผู้เขียนและส่งออกเอง (D-072)
     ฉบับที่ 1 อยู่ใน git history (commit 7e7fb2a) · ตัวเลขอ้างอิง research/HYPOTHESES.md §B6b–§B21,
     benchmarks/results/b17_spread_20261106.json และผลสำรวจ source code (research/TOOL_SURVEY.md)
     เปลี่ยนจากฉบับ 1: ให้เครดิต PR #12556 ชัดขึ้น, เพิ่มวิธีแก้ที่ประเมินด้วย counter, queue drain, ใครจ่ายค่าเขียน,
     เครื่องมือ cleanbench, ช่วงข้าม run ของตัวเลขหลัก, เลิกเรียก read flush ว่า "เวลาจริง" -->

# Dirty L2, Drained Queues: Measurement Pitfalls in GPU Kernel Microbenchmarks

*Draft technical report v4 — Rungrod Thongjampa (corresponding author, rungrod.th@blacktile.dev), Chanchada Wathanathonnan (chanchada.wa@blacktile.dev) — Blacktile Research — 2026-11-07*

## Abstract

GPU kernel microbenchmarks evict cached inputs before each timed call. In 14 of the 15 tools and benchmark paths we
audited, including `triton.testing.do_bench`, the eviction writes a large buffer. The writes leave dirty L2 lines
that the timed kernel must write back, so the flush changes the measurement. With pre-registered experiments on
nine NVIDIA GPUs, a batch-1 fp16 decode GEMV measured 6.6-25.3 % slower with the shipped write flush than with a
read flush, and the bias changed which quantized GEMV appeared faster on some GPUs. Hardware counters on a laptop
GPU confirm the mechanism. Two related pitfalls follow: a flush shorter than the host's dispatch time lets the
command queue drain, so a CUDA-event interval includes host time; and a read flush leaves the measured kernel's own
write-back untimed. Against a four-layer, GEMV-only decode proxy captured in a CUDA graph, summed shipped do_bench
times over-predicted the step by 26-32 % (fp16) on GPUs with large L2 and mis-ranked int4 kernels on an RTX 4060.
Timing kernels from CUPTI activity timestamps over a CUDA graph that rotates through weight copies matched the same
kernels inside the proxy within 0.79 % (12 kernels, ratio of medians over three runs) on one clock-locked laptop
GPU. This agreement was not reproduced on rented RTX 4060 and A100 hosts, which refused clock locking: clock
changes, a kernel-context effect and tracing overhead limited it. We release a harness with explicit quality gates
and list the hazards we measured.

## 1. Summary for practitioners

If you time memory-bound GPU kernels (decode GEMV/GEMM, attention decode, norms, activations, quantized
matmuls) or compare kernels that move different numbers of bytes:

1. **Flush the L2 by reading, not writing.** A write leaves dirty lines that your kernel then writes back (§5).
2. **Read a buffer well above the L2 size.** Our conservative default is max(256 MiB, 4 × L2); an L2-sized read
   left part of the previous call's output dirty, and 2 × L2 was not enough for short kernels with large outputs
   (§6). Check it for your output sizes.
3. **Keep the GPU busy before the start event.** If the flush finishes before the host has enqueued your
   kernel, you time the host (§7). nvbench does this with a blocking kernel; in Python, one line after the flush
   removed it in our measurements: `torch.cuda._sleep(cycles)`.
4. **In a per-call cold loop, do not wait (synchronize, sleep) between flush and timing.** Dirty lines stayed dirty
   while the GPU was idle, and the queue drains (§6, §7).
5. **Decide who pays for writes.** A cold, read-flushed time leaves part of the kernel's own write-back untimed
   (about L2/2 on our laptop GPU, more on an L40S). For kernels that write, report a steady-state time as well (§8).
6. **For a kernel's eventual DRAM write traffic from Nsight Compute, profile one range that ends with a full L2
   flush after the kernel, with a synchronization**; per-kernel counters can miss the write-back (§11).
7. **When a per-kernel number should match the kernel inside a model, time it from CUPTI timestamps over a CUDA
   graph that rotates inputs, after a warm-up, and check the conditions:** tracing must not change the graph's time
   (≤ 0.5 %), the clock must be steady, and the context must resemble the model's. Under these conditions, on one
   clock-locked laptop GPU, this matched 12 kernels of a GEMV-only decode proxy within 0.79 %; where they failed, it
   was off by up to 14 % (§9.1). If the conditions cannot be met, time the model's own graph. CUPTI timing is not
   new (FlashInfer has it, §14).
8. **Make the clock steady, or report how much it moved.** Lock the SM clock where you may (`nvidia-smi -lgc`, root
   only; rented containers refused). For small changes, a clock change Δf/f changes a kernel's time by about
   −s × Δf/f, where s is the share of its time that scales with the SM clock: near 0 for the memory-bound fp16 GEMVs
   and roughly 0.6-0.9 for int4 and NF4 GEMVs on our laptop GPU (§9.1, post hoc and rough).
9. **Tune on the machine you deploy on.** Two hosts with the same H100 NVL ran the same decode step 9-27 % apart (§9).

`cleanbench` (§10) does 1-3, 5 and 7, reports whether the conditions of item 7 held, and warns when a kernel is too
short for the event timer.

## 2. Background

- **Write-back L2.**
  - On NVIDIA GPUs stores allocate in L2 and reach DRAM later. A line written but not yet in DRAM is *dirty*.
  - On four dies we measured with DRAM counters, the L2 starts writing dirty lines back, without evicting them,
    once they exceed between 0.44 and 0.63 of its capacity (§11 has the table). NVIDIA's US patent 9,639,466
    describes a programmable dirty threshold (from its description; we have not analysed the claims).
  - A dirty line that is evicted must be written back first, by whichever access needs its slot.
- **Why tools flush.** A kernel timed repeatedly would find its inputs already in L2. Tools therefore stream a
  large buffer through the L2 before every timed call.
- **The timing loop.** `do_bench` and most Python tools run, per sample:
  `flush(); start.record(); fn(); end.record()`, with no synchronization. A CUDA event is timestamped when the
  GPU reaches it in the stream, not when the host records it.
- **Three things can go wrong**, and they are independent:
  1. the flush leaves dirty lines, which the timed kernel writes back (§5, §6);
  2. the flush ends before the host has enqueued `fn`, so the GPU reaches the start event early (§7);
  3. the timed kernel's own output stays dirty and is written back after the end event (§8).

## 3. Who is affected: a survey of 15 benchmarking tools

We selected tools that LLM-kernel developers commonly use to time or autotune kernels: compiler autotuners, the
benchmark helpers of kernel libraries, serving engines' benchmark scripts and kernel leaderboards. The unit is a tool
or benchmark path, not a random sample, so "14 of 15" describes this selection, not a prevalence. We read the timing
code at each project's `main` branch on 2026-11-06. Seven entries (Triton,
Inductor, nvbench, CUTLASS, FlashInfer, DeepGEMM, KernelBench) were re-checked in source; the other eight were read
once. Permalinks for all entries are in the artifact. Our measurements used Triton 3.3.1, whose `do_bench` flush is
the same as on `main`.

| Tool | Flush by default | Operation | Buffer |
|---|---|---|---|
| Triton `triton.testing.do_bench` (main, v3.9.0) | yes | write (`zero_`) | 256 MiB |
| PyTorch Inductor `InductorBenchmarker` (default autotuning benchmarker) | yes | write (`zero_`) | = L2 size |
| NVIDIA nvbench (cold measurements) | yes | write (`cudaMemsetAsync`) | = L2 size |
| CUTLASS CuTe DSL `tune` (`use_cold_l2`) | yes | write (`cuMemsetD32Async`) | = L2 size |
| FlashInfer `bench_gpu_time` (`cold_l2_cache=True`) | yes | write (`zero_`) | 2 × L2 |
| DeepGEMM `bench_kineto` | yes | write (`zero_`) | 256 MB |
| tritonbench, Liger-Kernel benchmarks | yes | via Triton `do_bench` | 256 MiB |
| Helion autotuner (modes `call` / `replay`) | yes | write | 256 MiB |
| vLLM `benchmarks/` (no shared helper) | per script | write in 19 files | 256 MiB or 2 × L2 |
| SGLang `marker.do_bench` / `bench_kineto` | yes | write | 5 × L2 / 8 GB |
| KernelBench (evaluates LLM-written kernels) | yes | write (`fill_`) | 256 MiB |
| GPU MODE reference-kernels (leaderboard evaluation) | yes | write (`randn`) | 4 GiB / 256 MiB |
| QuACK benchmark scripts | yes | write | 256 MiB |
| TileLang `do_bench` | yes | **read** (fixed 2026-11-04, #3534) | 256 MiB |
| QuACK autotuner, FlashInfer CUDA-graph path, Triton `do_bench_cudagraph` | — | no flush / input rotation | — |

Fourteen of the 15 write the flush buffer in at least one common timing path. Three (Inductor, nvbench,
CUTLASS) use a buffer of exactly L2 size, which matters for §6 and §7.

## 4. Method

- **Pre-registration.** For every experiment, rules, thresholds and predictions were written and locked (sha256
  of code and predictions, plus a git commit) before any GPU was rented or any value was read. Pipeline checks
  before a lock printed only counts, never measurements. Some hypotheses were motivated by earlier results: the
  queue-drain test (§7) and the harness validation (§10) were registered after laptop results suggested them, and
  were then scored on new data. In the nine-GPU study, pilot values were seen before its two amendments, and one
  amendment was written after seeing a single GPU (§13). The main wrong predictions are listed in §13.
- **Hardware.**
  - Timing: rented docker containers on vast.ai (nine GPUs; then an A100-SXM4-40GB, an RTX 5090, an L40S, an RTX 4060
    and an H100 NVL on new hosts for §7-§10), with unlocked clocks.
  - Per-kernel check (§9.1): the laptop GPU, with and without an SM clock lock at 1,000 MHz (`nvidia-smi -lgc`), and
    rented RTX 4060 and A100-SXM4 hosts, which refused clock locking.
  - Counters: an RTX 3060 Laptop GPU (GA106, L2 3 MiB) with Nsight Compute 2024.2.1, `--cache-control none`,
    `--replay-mode app-range`, one range per timed call with a synchronization before and after; plus rented
    VMs for the dirty-cap table. Container hosts did not allow counter access.
- **Timing.** Three harnesses: the real `triton.testing.do_bench` (Triton 3.3.1, median) with `clear_cache`
  patched per mode; our own loop with the same structure, used to add a GPU busy-wait; and `cleanbench` (§10). For
  §9.1, kernel durations from CUPTI activity records, read through `torch.profiler` (no counter permission needed).
  Mode order was shuffled; every kernel was checked against a reference first.
- **Spread.** We report the per-run minimum and maximum for headline numbers.
- **Cost.** About USD 3.05 of rented GPU time for the whole project, of which USD 0.16, 0.46, 0.39 and 0.15 for the
  four follow-ups on fast GPUs (§7-§10).

## 5. How big is the bias? Nine GPUs

B = t(shipped write flush) / t(read flush) − 1, in percent, from the median times over 5 runs. The reference kernel
is a batch-1 fp16 GEMV (`F.linear`, W 14336 × 4096, 112 MiB, larger than every L2 here); its range is the minimum and
maximum of the per-run B. The other columns: the same GEMV under an L2-sized write flush instead of 256 MiB ("L2-sized write"); SDPA
decode ("SDPA") with a 256 MiB KV cache; Liger RMSNorm on 64 MiB; Liger SwiGLU on 2 × 56 MiB. L2 sizes are as reported by the
CUDA device query.

| GPU | Arch | L2 (MiB) | GEMV (%) | range | L2-sized write (%) | SDPA (%) | RMSNorm (%) | SwiGLU (%) |
|------------------|----------|-------:|--------:|------------:|--------:|--------:|--------:|--------:|
| RTX 4060 | Ada | 24 | +6.7 | 6.3–6.7 | +6.4 | +7.1 | +15.6 | +11.4 |
| RTX 4060 Ti | Ada | 32 | +8.6 | 8.5–8.8 | +8.7 | +9.2 | +23.4 | +16.8 |
| RTX 5060 Ti | Blackwell | 32 | +6.6 | 6.5–6.9 | +7.3 | +3.6 | +11.0 | +9.1 |
| A100 PCIe 40GB | Ampere | 40 | **+16.0** | 15.0–16.0 | +16.0 | +7.6 | +15.6 | +12.8 |
| H100 NVL | Hopper | 60 | **+23.1** | 22.7–24.4 | +22.9 | +7.9 | +13.8 | +12.7 |
| RTX 4080 SUPER | Ada | 64 | +16.0 | 15.5–16.7 | +16.1 | +18.4 | +54.6 | +30.9 |
| RTX 4090 | Ada | 72 | +17.4 | 16.9–17.7 | +17.7 | +19.6 | +57.1 | +32.1 |
| L40S | Ada | 96 | **+25.3** | 24.7–25.3 | +25.6 | +28.3 | **+88.0** | +47.8 |
| RTX 5090 | Blackwell | 96 | +10.8 | 9.2–11.0 | +10.1 | +5.5 | +26.5 | +23.9 |

- **Every GPU is affected.** The pre-registered "affected" threshold was ≥ 5 %. The run-to-run range of the
  GEMV number is at most 1.8 points on every GPU.
- **Validity check amended.** The pre-registered check that the read flush evicts failed on the four GPUs with
  ≥ 64 MiB of L2. We replaced it after seeing one GPU (§13); the results of those four GPUs, and the trend below,
  are scored under the amended check, not as originally registered.
- **Within Ada the bias rises with L2 size** (Spearman ρ = 1.0 over five GPUs, 24 to 96 MiB by device query). Five
  points, and bandwidth also differs between them.
- **A smaller RMSNorm is hit harder.** A 16 MiB Liger RMSNorm is timed 16-110 % slow, against 11-88 % at 64 MiB,
  on every one of the nine GPUs. Attention decode shows no consistent trend with size.
- **A one-shot number is not a residue measurement.** The implied leftover dirty data, B × 112 MiB / L2, is
  0.27-0.31 of the L2 on Ada, 0.43-0.45 on A100/H100 and 0.13-0.23 on Blackwell. This is computed from B itself, not measured
  independently; on the laptop GPU, where counters were available, the residue after the shipped flush was
  0.50 of the L2 (§6).

### Effect on decisions

D is the relative change of the int4-over-fp16 speedup under the shipped flush, relative to a read flush (batch-1
decode shapes 4096², 14336 × 4096 and 4096 × 14336; PyTorch tinygemm and bitsandbytes NF4).

| GPU | Worst D | Direction |
|---|---|---|
| RTX 4060 / 4060 Ti | −24 % / −23 % | under-reported |
| RTX 4080 SUPER / 4090 | −17 % / −18 % | under-reported |
| L40S | −17 % | depends on shape |
| A100 | +31 % (4096², bitsandbytes) | depends on shape |
| H100 NVL | +22 % | **over-reported** on every shape |
| RTX 5090 | +37 % (4096², tinygemm; a ≈ 10 µs kernel, ≈ 10 % timer resolution) | depends on shape |
| RTX 5060 Ti | −7 % | depends on shape |

- **H100 NVL, bitsandbytes NF4 at 14336 × 4096:** the shipped flush reports 1.26× over fp16; with a read flush it
  is 1.04×, a gain about six times smaller.
- **RTX 4080 SUPER, tinygemm at 14336 × 4096:** 2.96× under the shipped flush, 3.56× with a read flush.
- **RTX 4060 and 4060 Ti (14336 × 4096):** the shipped flush makes bitsandbytes look faster than tinygemm (2.98×
  vs 2.79× on the 4060); with a read flush tinygemm is faster (3.67× vs 3.29×).
- **Why the direction varies.** The penalty is the write-back of the dirty lines a kernel happens to evict (issue
  #12554 already observed on an A100 that a kernel reading fewer bytes evicts fewer dirty lines); it is not
  proportional to the kernel's time, so a ratio of two kernels can move either way.
- **Autotuning is hurt less.** In an earlier experiment on an RTX 2060 SUPER, RTX 3090 and RTX 4080 SUPER (36 Triton
  GEMV configurations plus cuBLAS), the winner changed on 2 of 3 GPUs but the wrong pick cost ≤ 1.4 %. In a
  separate int8/int4 study, a W4A16 configuration chosen under the write flush on the RTX 4080 SUPER was 22.6 %
  slower than the best. FlashInfer PR #8076 (open) reports that the write flush reverses the order of CUTLASS FP4
  GEMM tactics at decode shapes on an RTX 5090.

## 6. What removes the residue, and what does not

Measured with DRAM counters on the RTX 3060 Laptop (L2 3 MiB): DRAM bytes written during the timed call of a
cuBLAS GEMV over 12 MiB of weights (4 × L2), whose own output is only 3 KB. Anything above zero is foreign
write-back. Median of 15 calls; the read flush is the kernel from PR #12556.

| What runs before the timed call | Foreign write-back in the timed window (× L2) |
|---|---|
| nothing (no flush) | 0.006 |
| `zero_()` 256 MiB, as shipped | **0.50** |
| `zero_()` of an L2-sized buffer (Inductor, nvbench, CUTLASS) | **0.59** |
| `zero_()` 256 MiB, then synchronize and sleep 10 ms | **0.47** |
| `zero_()` 256 MiB, then read 256 MiB | 0.008 |
| read 256 MiB | 0.003 |
| `torch.sum` 256 MiB | 0.003 |
| read of an L2-sized buffer | 0.005 |
| read of a 2 × L2 buffer | 0.002 |

- **Waiting does not help.** 95 % of the residue is still there after 10 ms. In a separate counter experiment on
  the same GPU, dirty lines below the threshold were not written back during up to 10 ms of idle time. Waiting
  also makes the timing far worse (§7).
- **Writing then reading works.** It is a minimal patch for tools whose flush is a memset, but a read alone does
  the same job in one pass instead of two.
- **Buffer size.** An L2-sized read evicted the timed kernel's inputs completely, even for a 24 KiB kernel (we had
  predicted it would not). But it leaves the tail of the *previous call's output* dirty: 0.06 L2 of foreign
  write-back for a kernel with a 0.125 L2 output, and 0.26 L2 more than a 256 MiB read for a kernel with a 2 L2
  output, which costs it 6.7 % in time (§7). With a 2 × L2 read, short kernels whose outputs are 0.4-0.5 L2 were
  still 6-9 % slower than with 256 MiB (§10). Hence our default of max(256 MiB, 4 × L2). The 4 × L2 term is a
  conservative choice: it only exceeds 256 MiB on GPUs with more than 64 MiB of L2, where (on the RTX 5090 and L40S)
  the 384 MiB default reproduced the nine-GPU numbers (§10), but we did not compare it with smaller buffers there.
  Triton's 256 MiB buffer has no stated derivation either (triton-lang/triton issue #1252, 2023: "just to make sure
  flush data size > L2 cache size").

## 7. Pitfall 2: queue drain

This is a known property of CUDA-event timing: NVIDIA's nvbench documents that the start event "may be recorded a
non-trivial amount of time before" the timed kernel is ready, and blocks the stream with a `blocking_kernel` until
all work is queued; a 2023 Speechmatics post recommends a `torch.cuda._sleep` before the start event for the same
reason (§14). What we add is where it bites in practice: Python harnesses that flush before every call, once
the flush is shortened.

In the loop `flush(); start.record(); fn(); end.record()`, the start event is timestamped when the GPU reaches it.
If the GPU has already finished the flush when the host enqueues the start event, the timestamp is taken at once,
and the interval includes the time the host still needs to dispatch `fn` (Python, the Triton launcher or
PyTorch's dispatcher). A 256 MiB flush keeps the GPU busy for about 1 ms on the laptop GPU, which hides this; an
L2-sized flush takes 12-18 µs. To test this we inserted a GPU busy-wait of about 300 µs
(`torch.cuda._sleep`) right before the start event. It runs before the start event and is not timed.

**Laptop GPU (host dispatch on WSL2: 72-139 µs).** Time relative to a 256 MiB read flush:

| Kernel, flush | do_bench-style loop | with the busy-wait |
|---|---|---|
| cuBLAS GEMV 12 MiB, L2-sized read | +66 % | **0 %** |
| Triton y = 2x (3 MiB), 2 × L2 read | +167 % | **0 %** |
| Triton y = 2x, L2-sized `zero_()` vs 256 MiB `zero_()` (the Inductor setting) | +187 % | **0 %** |
| cuBLAS GEMV, `zero_()` + 10 ms wait vs `zero_()` | +348 % | **+2.4 %** |
| elementwise add 12 MiB, L2-sized read | +43 % | +6.7 % (the extra write-back of §6) |

- **The 256 MiB write and read flushes were unaffected by the busy-wait** (within 2.2 %; also on the A100, RTX 5090
  and L40S hosts below): they already keep the queue busy.
- **The real `do_bench`, with `clear_cache` patched:** the Triton kernel took 41-47 µs with an L2-sized read
  flush and 13.3 µs with the same flush followed by the busy-wait.

**Fast GPUs, new hosts.** Liger-Kernel RMSNorm on 16 MiB (the kernel behind the +164 to +537 % numbers of the
nine-GPU study), L2-sized `zero_()` relative to 256 MiB `zero_()`; and a 16 MiB GEMV with no flush relative to a
256 MiB read flush. Host dispatch is the median host time to enqueue the RMSNorm; flush time is estimated as
L2 size over the DRAM rate of the 112 MiB GEMV.

| GPU | host dispatch | L2-sized flush (est.) | RMSNorm, L2-sized `zero_()`: loop / busy-wait | GEMV 16 MiB, no flush ÷ read flush: loop / busy-wait |
|---|---|---|---|---|
| A100-SXM4-40GB | 140 µs | ≈ 37 µs | **+203 % / −3 %** | **1.50 / 0.76** |
| RTX 5090 | 43 µs | ≈ 63 µs | 0 % / −9 % | 0.92 / 0.88 |
| L40S | 102 µs | ≈ 138 µs | 0 % / −1.5 % | 0.93 / 0.93 |

- **What it accounts for.** On the A100 host both anomalies of the nine-GPU study reproduced: the L2-sized write flush
  made the short kernel 3× slower, and no flush was 1.5× *slower* than flushing (the failed check of §13). Both
  disappeared with the busy-wait, and no flush became faster, as it should. They are queue drain, not cache
  effects.
- **They depend on the host, not only on the GPU.** On our RTX 5090 and L40S hosts the host dispatched faster than
  the L2-sized flush ran, and neither anomaly appeared. Across the four machines we measured (with the laptop),
  the inflation appeared exactly where host dispatch took longer than the flush; this rule was formed after seeing
  the data. The RTX 5090 host of the nine-GPU study did show +537 %; its dispatch time was not recorded.
- **Who is exposed.** Python harnesses with a short flush on a fast GPU: Inductor's benchmarker (L2-sized buffer),
  any tool after shrinking its flush, and any loop with no flush. Inductor returns the minimum over samples, which
  our repro shows is less affected than the median. C++ harnesses (nvbench, CUTLASS) likely dispatch faster; we
  did not measure them.
- **Related.** Helion's autotuner (PR #4126, merged 2026-10-23) found that for microsecond kernels its benchmarks
  "mostly timed the flush" and added flush regimes; that is a different mechanism with the same lesson, that the
  flush step can dominate short-kernel timings.

## 8. Pitfall 3: who pays for a kernel's own writes

If a kernel's output fits within the amount of dirty data the L2 holds (about half of it on the laptop GPU), the
output is still dirty in the L2 when the kernel ends. Under a read flush, that
write-back happens during the *next* flush, outside any timed window. Counters on the laptop GPU (fraction of
the kernel's own output written to DRAM inside its timed window, median of 15 calls):

| Kernel (elementwise add) | read flush 256 MiB ("cold") | rotated input/output copies ("steady") | no flush |
|---|---|---|---|
| output 0.125 × L2 | **0.00** | **0.99** | 0.00 |
| output 2 × L2 | 0.75 | 1.00 | 0.99 |

- **On the laptop GPU, under a read flush, a kernel with an output below about L2/2 does not pay for writing it at
  all**; a large output pays for all but about L2/2, which equals the residue a long write stream leaves there
  (0.50 L2).
- **Rotating over copies whose total footprint is ≥ 4 × L2 charges one output's write-back per call**, which is
  what a steady stream of independent calls costs.
- **In time** (laptop GPU, `cleanbench`): steady vs cold was +0.2 % for the GEMV (tiny output), +14 % for the
  2 × L2 elementwise kernel (+6.4 µs, against an implied 5.7 µs to write 0.5 L2 at its DRAM rate) and +38 % for a
  0.4 × L2 elementwise kernel (+3.7 µs, against an implied 4.6 µs).
- **On large-L2 GPUs (new hosts)** steady vs cold was, for an elementwise kernel with a 0.25 × L2 output, +39 %
  (A100), +67 % (RTX 5090) and +109 % (L40S); for the 112 MiB GEMV within ±1.1 %; for a Liger RMSNorm with a
  64 MiB output +16 %, +10 % and +72 %. The long-stream residue implied by §5 predicted the first two RMSNorm
  values within 0.006 but under-predicted L40S 2.5-fold: there, even a 0.67 × L2 output was almost entirely
  untimed in cold mode.
- **Prior art.** We have not found this corollary stated. The nearest is a question on NVIDIA's developer forum
  (2026-10-03) noting that Nsight Compute's `--cache-control all` leaves no flush write-back and asking for a
  "dirty" mode, because "real world scenario usually start with L2 dirty from the previous kernel".
- **Neither number is "the" time of the kernel in a model.** If the next operation reads or overwrites the output
  while it is still in L2, the write-back may never happen. Cold and steady are two measurement conditions, not
  guaranteed bounds: cold leaves part of the kernel's own write-back out, steady charges one earlier call's. Kernels that write different amounts (fused vs unfused, or a kernel
  that writes an intermediate) are compared fairly only if both numbers are reported.

## 9. End-to-end check: which number predicts a decode step?

For workload optimization, a per-kernel time is useful if it predicts what the kernel costs inside the workload. We
built a decode proxy: four layers
of Llama-2-7B shapes (hidden 4096, FFN 11008) at batch 1, each running its four weight GEMVs (fused QKV, O, fused
gate+up, down) with its own weights, in fp16 (cuBLAS), int4 (tinygemm) and NF4 (bitsandbytes). The whole step was
captured in one CUDA graph and replayed back to back, as decode loops do; its median time is the reference. Each
harness's per-kernel times were summed over the step and compared with it (pre-registered; five GPUs).

| GPU | L2 | fp16 step: shipped / cold | int4 step: shipped / cold | int4 speedup error: shipped / cold |
|---|---|---|---|---|
| RTX 3060 Laptop | 3 MiB | +5.2 % / +2.0 % | +8.6 % / +1.8 % | −3.1 % / +0.2 % |
| RTX 4060 | 24 MiB | +8.2 % / +0.6 % | **+43.6 %** / +2.0 % | **−24.6 %** / −1.4 % |
| A100-SXM4-40GB | 40 MiB | **+26.3 %** / +5.2 % | **+46.2 %** / +17.5 % | −13.6 % / −10.4 % |
| H100 NVL | 60 MiB | **+32.4 %** / +8.4 % | +22.6 % / +17.2 % | +8.0 % / −7.5 % |
| L40S | 96 MiB | **+28.1 %** / +2.0 % | **+52.5 %** / +7.1 % | −16.1 % / −4.8 % |

- **The shipped write flush over-predicts the step** on every GPU with ≥ 40 MiB of L2, by 26-32 % for fp16 and
  23-53 % for int4.
- **It also ranks kernels wrongly.** On the RTX 4060, summed shipped times put NF4 ahead of tinygemm; in the step,
  tinygemm is 15 % faster, and the read-flush times rank them correctly. On the H100 NVL, the step's NF4 speedup over
  fp16 is 1.14×; the shipped times predict 1.32×, the read-flush times 1.09×.
- **The read flush with a busy-wait ("cold") is close on the consumer GPUs and the L40S** (within 2 % for fp16 and
  2-7 % for int4), **but not on the A100 and H100**: 5-8 % for fp16 and 17 % for int4, against our pre-registered
  10 %. There an int4 GEMV takes 18-28 µs, and event timing adds a fixed per-call cost that graph replay does not
  (2.2-3.4 µs per call on the rented RTX 4060 and A100 hosts of §9.1; the H100 was not re-measured).
- **Limitations.** The proxy has only GEMVs (no attention, norms or activations). NF4 was mispredicted by every
  harness on the L40S (cold +36 %; cause unknown).

**Can a per-kernel time get within 1 %?** The cold time's excess over the step was 2-5 µs per kernel on every GPU,
which suggests a fixed per-kernel cost of event timing rather than a cache effect. We therefore timed each kernel the
way the step runs it: one CUDA graph calling the kernel once per weight copy (copies covering ≥ 4 × L2, so every call
is cold), replayed back to back, divided by the number of calls (pre-registered; the A100 host was the one used in
B19 and B21, the H100 NVL host was new):

| GPU | step error, fp16 / int4: graph-batched | cold | shipped | int4 speedup error: graph-batched |
|---|---|---|---|---|
| A100-SXM4-40GB | **+1.0 % / +1.6 %** | +6.0 % / +17.5 % | +26.9 % / +47.1 % | **−0.6 %** |
| H100 NVL | −12.2 % / −19.5 % | −4.5 % / −6.4 % | +16.8 % / +0.5 % | +9.1 % |

- **On the A100 host, close:** the step time within 1.0-1.6 % (NF4 0.2 %) and the int4 speedup within 0.6 %.
- **On the H100 NVL, no per-kernel method came within 5 %.** The graph-batched time was 12-20 % *below* the step.
  Where 1 % matters on such GPUs, time the model's own graph.
- **The decode proxy changed across hosts, while isolated-kernel times were stable.** The same H100 NVL step took
  532 µs on one rented host and
  616 µs on another (fp16 +16 %, int4 +27 %, NF4 +9 %). The isolated kernels did not move: their cold times on the
  two hosts agreed within −0.8 to +4.3 % (all 12 kernel rows), and the sum of the second host's graph-batched times
  matched the *first* host's step within 2.4 % (fp16 +1.7 %, int4 +2.4 %, NF4 −2.2 %). So the H100 miss is a
  property of how the second host ran the 16-kernel step, not of per-kernel timing or of the chip (post hoc,
  two hosts, one run set each). The hosts differed in power limit (400 W vs 300 W), driver (580.178 vs 615.71) and
  virtualization (VM pass-through vs unreported); we did not log clocks or throttle reasons during the step, so we
  cannot attribute the difference. The A100 runs of §9 (B21 and B22) used one host, so the A100 1 % result is also
  from one host.

### 9.1 Kernel by kernel: does a single-kernel benchmark match the same kernel inside the decode proxy?

Sums over a step can hide errors that cancel. We therefore compared each kernel's benchmark time with the same
kernel's time inside the decode step (pre-registered):

- **Reference.** The step's CUDA graph, replayed back to back after at least 1 s of sustained replays; each kernel's
  duration from CUPTI activity records (the GPU's own start and end timestamps), 4 layers × 300 replays.
- **Harnesses** on the same kernels:
  - *CUPTI rotation*: CUPTI durations over a CUDA graph that calls the kernel once per weight copy (copies ≥ 4 × L2,
    as in §9), after the same warm-up;
  - the same after 1 s of idle GPU and no warm-up;
  - `cleanbench`'s cold loop (read flush, busy-wait) timed by CUPTI;
  - `cleanbench` cold as in §9 (CUDA events);
  - the graph-batched event time of §9.
- **Interleaving.** The reference and CUPTI rotation were measured in three alternating rounds per run, so slow
  drift affects both; three runs per GPU.
- **Gates.** Tracing must not change the step's CUDA-event time by more than 0.5 %; the reference must be stable
  (IQR ≤ 5 % of the median); every trace segment must hold exactly the expected kernels. After the first laptop run
  failed, an amendment (locked before the next run) added a clock gate: the SM clock within ±1 % of its median in
  ≥ 99 % of busy samples, and no thermal or hardware slowdown in the driver's counters.

| RTX 3060 Laptop, SM clock locked at 1,000 MHz (observed 1,005) | Error vs the same kernel in the step (12 kernels) | Within 1 % |
|---|---|---|
| CUPTI rotation, 1 s warm-up | **−0.79 to +0.39 %** | **12/12** |
| CUPTI rotation, after 1 s idle | −1.6 to +0.9 % | 10/12 |
| Graph-batched CUDA events (§9) | 0.0 to +2.4 % | 6/12 |
| CUPTI, `cleanbench` cold loop | −2.0 to +3.4 % | 1/12 |
| CUDA events, `cleanbench` cold loop | +1.3 to +13.3 % | 0/12 |

Error is the ratio of the harness's median to the reference median, each taken over three runs. All gates passed
and every pre-registered prediction for this run held. The kernels took 49-645 µs. Unaggregated, the per-run errors
were within 0.86 % (36 of 36 kernel-runs), and the errors of the paired rounds (reference and harness measured back
to back, 108 pairs) had a median of 0.30 %, a 95th percentile of 0.95 % and a maximum of 1.10 % (104 of 108 within
1 %). On the RTX 4060 below, the paired-round 95th percentile was 13.8 %.

- **What was needed.**
  - *The GPU's timestamps instead of events.* On the two rented hosts, event-bracketed intervals exceeded the CUPTI
    kernel durations in the same cold loop by 2.2-3.4 µs per call (3-22 µs on the laptop under WSL2). The excess
    includes launch latency and idle gaps inside the event interval, not only the cost of the events themselves.
  - *Rotation inside a graph instead of a flush.* The cold loop was 1-3 % off even when timed by CUPTI: it differs
    from the step in cache state and context, not only in the timer.
  - *Interleaving.* Even with the clock locked, the reference moved 1.2-2.2 % between rounds.
  - *A steady clock.* The unlocked laptop run failed its gates: the GPU reached 87 °C, the SM clock ranged
    1,005-1,747 MHz, and the driver's thermal-slowdown counter rose by 53 s. Between rounds, the in-step time of the
    fp16 kernels moved 2-3 %, that of the int4 and NF4 kernels 27-30 %.
- **Rented hosts refused clock locking** ("The current user does not have permission to change clocks"), as we
  predicted.
  - **RTX 4060:** the pre-registered gates and the thermal check passed, but the clock gate did not: the clock was not steady
    (2,355-2,805 MHz; power-capped in 130 of 202 busy samples). CUPTI rotation was within 0.02 % for fp16 and 1.6 % for int4, but 6.8-14.2 % *below* the step for
    NF4 (bitsandbytes), with similar clocks in both phases (medians 2,535 and 2,573 MHz): this kernel ran faster
    when repeated than among other kernels (cause unknown). Events in the cold loop: +0.4 to +13.5 %.
  - **A100-SXM4:** the tracing gate failed (CUPTI added 1.0-2.2 % to the step, whose kernels took 14-138 µs), so
    this run is descriptive only: CUPTI rotation −4.1 to +3.7 %, events in the cold loop +3.0 to +22.2 %. The clock
    was power-capped (1,260-1,410 MHz).
  - The RTX 4090 and H100 NVL hosts never became reachable and were destroyed unused.
- **Clock sensitivity (post hoc).** A two-term frequency-scaling model, t(f) = a/f + b, splits a kernel's time into a
  part that scales with the SM clock f and a part that does not (DRAM at a fixed memory clock). With s = (a/f₀)/t at
  clock f₀, a change to f₁ changes the time by s × (f₀/f₁ − 1), which for small changes is about −s × Δf/f; since
  0 ≤ s ≤ 1, it is at most f₀/f₁ − 1 (a 20 % clock drop can add up to 25 %). From the laptop's unlocked
  and locked runs (`clock_sensitivity.py`; the unlocked clock summarized from 13-22 one-second `nvidia-smi` samples
  per variant, as a median or a mean): s = 0.00-0.04 for the fp16 GEMVs, 0.79-0.91 for int4 and 0.61-0.66 for NF4.
  Two clock states determine the two parameters exactly, so this is a rough estimate, not a validated model. It is
  consistent, on the laptop, with fp16 timing withstanding clock changes and int4 / NF4 not, and it makes the clock
  an unlikely cause of the RTX 4060's NF4 gap (whose s we did not measure). Richer GPU frequency-scaling models exist
  (e.g. CRISP, §14); we have not tested a correction based on this one.

## 10. cleanbench

`cleanbench.py` (one file, PyTorch and Triton) times a callable with:

- a **read flush** (the PR #12556 kernel) over a buffer of max(256 MiB, 4 × L2), cached per device;
- a **GPU busy-wait** before every start event, sized min(5 ms, max(100 µs, 3 × the measured host dispatch time)), a
  simpler, fixed-length stand-in for nvbench's blocking kernel;
- `mode="cold"` (flush before every call) or `mode="steady"` (no flush; the caller passes callables over copies,
  whose total footprint should be ≥ 4 × L2, and the harness rotates over them);
- median, mean, IQR, p95, and warnings when the median is within 30 timer ticks (≈ 1.02 µs each) and most samples
  fall on the tick grid, when the IQR exceeds 10 % of the median, and, in cold mode, that the kernel's own
  write-back is partly untimed.

```python
import cleanbench as cb
cold = cb.bench(lambda: f(x, y))                                   # read flush + busy-wait
rot = [lambda i=i: f(xs[i], ys[i]) for i in range(n)]          # copies covering >= 4 x L2
steady = cb.bench(rotate=rot, mode="steady")
print(cold.median_us, steady.median_us, cold.warnings)
```

**Validation on the laptop GPU** (per cell, 15 `bench()` calls of 100 timed calls each over 3 runs; value = median
over the calls of each call's mean; the `do_bench` column compares `do_bench`'s median with that value):

| Kernel | cold, 2 × L2 flush vs 256 MiB | steady vs cold | shipped `do_bench` vs cold |
|---|---|---|---|
| cuBLAS GEMV 12 MiB | −0.05 % | +0.2 % | +12 % |
| elementwise add, 2 × L2 | +1.6 % | +14 % | +15 % |
| elementwise add, 0.4 × L2 | +5.6 % | +38 % | +36 % |
| Triton y = 2x, 0.5 × L2 | +9.4 % | +46 % | +41 % |

The first column is why the default buffer is not 2 × L2. On the A100, RTX 5090 and L40S hosts, the shipped
`do_bench` read +16 %, +11 % and +25 % above `cleanbench`'s cold time for the 112 MiB GEMV, matching the nine-GPU
study's +16.0 %, +10.8 % and +25.3 %, measured on different hosts against a different baseline.

**Kernel time from the GPU's timestamps (`bench_graph`, `compare`).** `bench_graph(rotate)` captures callables over
input copies (≥ 4 × L2 in total) in one CUDA graph, keeps the GPU busy for at least 1 s, and reads each call's
kernel durations from CUPTI; `compare` times several candidates in interleaved rounds, in a seeded random order per
round. `locked_sm_clock(mhz)` locks the SM clock where permitted (root or Administrator) and always resets it.
`Result.quality` reports the conditions of §9.1 and marks a result as validated only if all hold: exact CUPTI
records, a tracing effect ≤ 0.5 % (traced and untraced replays are both timed), readable clock-event counters that
did not grow, and inputs covering ≥ 4 × L2 (`footprint_bytes`); otherwise the result is labelled exploratory. The
counters catch throttling, not every small boost step, so a locked clock remains the condition under which 1 % was
reached. Each callable must run on the current stream without overlapping kernels.

**Limitations.** `torch.cuda._sleep` is a private PyTorch API. Steady mode and `bench_graph` need the caller to build
input/output copies. With CUDA events (`bench`), kernels shorter than about 30 µs remain limited by the timer's
≈ 1 µs resolution. `bench_graph` reached 1 % only with a locked clock, on one GPU (§9.1), and CUPTI tracing itself
added 1-2 % on an A100.

## 11. Measurement pitfall: per-kernel DRAM write counters lose bytes

We ran a kernel writing w × L2, followed by a flush. One range enclosing both counts ≈ 100 % of the written
bytes. The sum of per-kernel `dram__bytes_write.sum` does not:

| GPU | lost at w = 0.75 / 1.0 / 1.5 L2 |
|---|---|
| GA106 | 8.9 % / 6.7 % / 0.1 % |
| GA102 | 14.4 % / 9.2 % / 3.5 % |
| TU106 | 14.7 % / 16.9 % / 5.7 % |
| AD103 | 5.8 % / 4.9 % / 3.2 % |

- **Per-kernel attribution misses them.** Two back-to-back ranges also lose bytes: 4-9 % on GA102, 5-13 % on TU106
  and 1-2 % on AD103. All with `--cache-control none`, Nsight Compute 2024.2.1.
- **Still present in the latest release, on GA106.** Re-run on GA106 with Nsight Compute 2026.4.0 (newest in NVIDIA's package
  repository on 2026-11-06), back to back with 2024.2.1: 8.1 % / 5.4 % / 0.2 % lost at w = 0.75 / 1.0 / 1.5 L2, against
  8.6 % / 5.8 % / 0.2 % with 2024.2.1 in the same session (pre-registered: "persists").
- **Not yet measured on A100/H100.** The other three dies were measured with 2024.2.1 only. One attempt with Nsight Compute
  2025.2.1 on an RTX 5060 Ti returned n/a for the DRAM metrics.
- **The dirty threshold behind §2**, write-back fraction at a dirty footprint of 0.4375 / 0.625 L2: GA106 (3 MiB)
  0.014 / 0.991; GA102 (6 MiB) 0.009 / 0.974; TU106 (4 MiB) 0.016 / 0.976; AD103 (64 MiB) 0.071 / 0.933.

## 12. Recommendations for tool authors

- **Triton `do_bench`:** PR #12556 would remove pitfall 1. Its 256 MiB buffer already avoided pitfall 2 on the four
  machines where we tested it with a busy-wait; a busy-wait after the flush would make that independent of GPU and
  host speed.
- **PyTorch Inductor:** switching `zero_()` to a read on the existing L2-sized buffer is not enough: it leaves the
  previous call's output partly dirty (§6) and lets the queue drain on fast GPUs (§7). Read at least
  max(256 MiB, 4 × L2), or add a busy-wait.
- **nvbench, CUTLASS:** a read kernel instead of the memset (or a read pass after it).
- **Leaderboards and LLM-kernel evaluations (KernelBench, GPU MODE):** state the flush method; consider reporting a
  steady-state time, since generated kernels differ in how much they write.
- **All tools, when per-kernel numbers should match a model:** consider CUPTI durations over a CUDA graph that rotates
  inputs, with a warm-up and interleaved candidates, and report whether tracing, clock and footprint conditions held
  (§9.1); where they do not, time the model's graph. Record the SM clock and the driver's clock-event counters with
  every result.

**Observed timing hazards** (from this report; the rows measure different quantities under different conditions,
overlap, and are not additive):

| Hazard | What is distorted | Observed here (setting) | Remedy (status) |
|---|---|---|---|
| Write flush (dirty L2) | isolated cold time | +6.6 to +25.3 % (decode GEMV, nine GPUs) | read flush (validated with counters on one GPU, timing on nine) |
| Queue drain | isolated cold time | up to +203 % (short kernel, A100 host) | busy-wait before the start event (rescued on two hosts) |
| Own write-back untimed | cold vs stream of calls | up to about L2/2 of the output (laptop counters) | also report steady state (§8) |
| Event interval vs kernel duration | per-call time | 2.2-3.4 µs per call (two rented hosts) | CUPTI durations (§9.1) |
| Cold loop vs the proxy's context | per-kernel time in the proxy | 1-3 % (laptop, locked clock) | rotation inside a CUDA graph (one GPU) |
| SM clock changes | any time | 27-30 % between rounds for int4 / NF4 (unlocked laptop) | lock the clock, or report the change (§9.1) |
| CUPTI tracing | traced vs untraced graph | 0.03-2.2 % of the proxy step, up to 3.7 % of a rotation graph (A100) | check tracing on vs off (§9.1) |
| Kernel context (NF4, RTX 4060) | per-kernel time in the proxy | up to 14 % | time the model's graph (cause unknown) |
| Host (same GPU model) | absolute step time | 9-27 % (two H100 NVL hosts) | tune on the deployment host (cause unknown) |

## 13. Negative results, deviations and limitations

**Wrong or partly wrong predictions** (all pre-registered):
- An L2-sized read flush evicted even a 24 KiB kernel's inputs; we predicted it would leave them warm.
- The busy-wait removed the inflation for two of three kernels; for the third, 6.7 % remained, which matches, after
  the fact, the +6.6 % implied by the extra write-back the counters measured; the locked rule scored "partial".
- We predicted that waiting after a write flush slows timing through lower clocks; the busy-wait showed it is
  queue drain.
- We predicted `cleanbench`'s cold time would not depend on the flush buffer between 2 × L2 and 256 MiB; it does
  for short kernels with outputs of 0.4-0.5 L2.
- In the nine-GPU study, we predicted only mild distortions of int4 speedups and marginal effects on GPUs with
  ≤ 32 MiB of L2; the RTX 4060 / 4060 Ti speedups moved by −23 to −24 % and every kernel family was affected there.
- Earlier: cache models (LRU plus dirty cap) did not predict single-kernel write-back on TU106/AD103 or the residue
  after long write streams; a "bandwidth slack absorbs the write-back" hypothesis passed only through a
  weight-format confound; an L2-sized write flush was not more biased than the 256 MiB one for the large GEMV.
- For graph-batched timing (§9) we predicted the step within 2 % on the A100 and H100; the A100 was within 1.6 %, the
  H100 NVL 12-20 % off. Two attempts on the laptop GPU failed the stability gate because another program was using
  the GPU.
- In the end-to-end check (§9) we predicted the read flush with a busy-wait would come within 10 % of the decode
  step on all five GPUs and predict the int4 speedup within 5 % on the three with ≥ 40 MiB of L2; on the A100 and
  H100 it missed the int4 step by 17 % and the speedup by 8-10 %.
- In the per-kernel check (§9.1) we predicted that the gates would pass on rented hosts (the A100 failed the tracing
  gate), that the A100's clock would be steady (it was power-capped) and that every kernel on the RTX 4060 would be
  within 1 % (NF4 was 7-14 % off). For the unlocked laptop run we had flagged the gates as at risk and made no
  accuracy prediction; it failed on thermal slowdown.
- On fast GPUs we predicted that the L2-sized-flush anomaly would reproduce on the RTX 5090 (not on this host);
  that a 16 MiB GEMV would run ≥ 15 % faster with no flush than with a read flush once the queue is busy (A100
  yes; RTX 5090 12 %, L40S 7 %); and that the long-stream residue would predict the cold/steady gap of the 64 MiB
  RMSNorm on at least 3 of 4 GPUs (2 of the 3 measured; L40S off 2.5-fold; H100 not measured).

**Deviation.** In the nine-GPU study, our pre-registered check that the read flush evicts (no flush ≥ 1.2× faster
on 32 MiB of weights) failed on all four GPUs with L2 ≥ 64 MiB; we replaced it after seeing one GPU with "the read
flush leaves the kernel at least as cold as `torch.sum`", validated earlier with counters. On an A100 host we
later reproduced the failed check (no flush 1.5× slower than a read flush) and reversed it with a GPU busy-wait
(0.76×): queue drain, not eviction, had made "no flush" look slow (§7).

**Deviations in §9.1.** The clock-locked laptop run is an amendment written after the unlocked run had failed its
gates; its predictions were locked before it ran. The rented runs added a clock-lock attempt in a wrapper script;
the measurement code and the rules were unchanged, and the clock-gate script was changed to make the locked clock
optional.

**Limitations.**
- No full-model measurement: the reference in §9 is a four-layer, GEMV-only decode proxy, and the report is about
  what harnesses time, not about model throughput.
- The 1 % per-kernel result (§9.1) comes from one laptop GPU with a locked clock and kernels of 49-645 µs, against a
  GEMV-only proxy; on rented data-center GPUs we could not lock the clock and did not reach 1 %.
- The tool survey (§3) is a selection of commonly used tools, read once for eight of them.
- One host per GPU model in the nine-GPU study (two for the RTX 4080 SUPER; the A100, RTX 5090 and L40S were
  re-measured on new hosts), rented containers with unlocked clocks, Triton 3.3.1 and torch 2.7.1.
- Counter results (§6, §8, §11) come from one laptop GPU (§6, §8) or four consumer dies (§11); data-center L2
  policies may differ.
- The host dispatch times on the laptop (WSL2) are longer than on a typical Linux server.

## 14. Related work and credit

- **Write-flush bias and the read flush:** reported in triton-lang/triton issue #12554 (2026-10-30), with Nsight
  Compute measurements on an A100 showing the write-back saturating at about half the L2 and kernels that read
  fewer bytes evicting fewer dirty lines; the read flush we use throughout is the kernel proposed in
  triton-lang/triton PR #12556 (measured there on an RTX 5090; open and awaiting maintainer review on 2026-11-06).
  PyTorch PR #203681 (open) addresses it for ROCm; TileOPs PR #2822 and TileLang #3534 have switched to read
  flushes. Helion PR #4126 reworked its autotuner's flush regimes for microsecond kernels. FlashInfer PR #8076 (open)
  proposes a read flush for its cold-L2 autotuning.
- **Dirty L2 state:** deepseek-ai/DeepEP-Ascend issue #17 reports the same write-flush problem on an NPU; an NVIDIA
  developer-forum thread (2026-10-03) asks Nsight Compute for a dirty-L2 cache-control mode.
- **Cold timing and input rotation:** ROCm hip-ep PR #1238 times kernel tests cold and notes that its tuners rotate
  over KV-cache copies; we use rotation to charge a kernel's own write-back (§8). We do not claim the discovery of the bias or of the read flush; our contribution is
  the cross-GPU and cross-tool measurement, the evaluation of alternative fixes with counters, showing that
  shortened flushes expose a known launch-latency pitfall in Python harnesses, the write-accounting pitfall, and
  `cleanbench`.
- **Dirty threshold:** NVIDIA US 9,639,466 B2; Finn, *What happens when a GPU writes memory* (2026).
- **Event timing and launch latency:** NVIDIA nvbench's `blocking_kernel` holds the stream until all timed work is
  queued, for the same reason as our busy-wait (§7).
- **Kernel-timing practice:** a Speechmatics engineering post (Atkins and MacLeod, 2023) already recommends a
  `torch.cuda._sleep` before the start event so that short kernels and their events are enqueued together, locking
  clocks with `nvidia-smi -lgc`, and CUDA graphs; it flushes by writing an L2-sized buffer. FlashInfer's
  `bench_gpu_time_with_cupti` reads kernel durations from CUPTI; Triton's `testing.set_gpu_clock` locks clocks; nvbench
  detects throttling. We do not claim these techniques; §9.1 measures which of them are needed, and how far each one
  gets, against the kernel's time inside a model.
- **GPU performance variability:** Sinha et al. (SC22; arXiv 2208.11035) report that GPUs of the same model vary by
  up to 22 % across nodes of HPC centers; we saw 9-27 % between two H100 NVL hosts in one decode step. H. He (2024) shows that power
  limits make matmul time depend on the input values.
- **Flush-buffer size:** triton-lang/triton issue #1252 (2023) asked why `do_bench` flushes 256 MB; the answer was
  only to exceed the L2 size.
- **Frequency-scaling performance models:** CRISP (Nath and Tullsen, MICRO 2015) models GPU performance under DVFS;
  §9.1 uses only the simplest two-term form.
- **Counter attribution:** TileOPs issue #1990.
- **GPU microbenchmarking:** Ernst et al. 2021.

## Artifact

The public repository (https://github.com/blacktile-labs/l2-dirty) contains every probe, protocol and scorer with unit tests; the
pre-registration texts (in Thai) and every sha256 lock, with a snapshot of each locked file as committed
(`research/preregistration/verify_locks.py` checks them: 190 of 191 match; one early lock's protocol file was edited
before it was committed); raw JSON/CSV for every run, with the IP addresses of rented hosts removed; the tool-survey
permalinks; `cleanbench`; and the scripts that build this report and its figures. Lock timestamps come from the
authors' development history and are self-attested. Rented-host conditions (drivers, power limits) cannot be
reproduced exactly. Code is under the Apache License 2.0; the report, figures and data under CC BY 4.0.
