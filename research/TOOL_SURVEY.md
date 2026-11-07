# Survey: L2 flush ในเครื่องมือ benchmark ยอดนิยม (D-075, 2026-11-06)

**สถานะ:**
- **วิธีสำรวจ:** subagent อ่าน source แบบอ่านอย่างเดียวที่ `main` HEAD วันที่ 2026-11-06
- **ตรวจซ้ำแล้ว:** ผู้ช่วยวิจัยตรวจซ้ำใน source ที่ดาวน์โหลดมาสำหรับ Triton, Inductor, nvbench, CUTLASS, FlashInfer, DeepGEMM และ KernelBench (D-078/D-079)
- **ยังไม่ได้ตรวจซ้ำ:** แถวอื่น
- **ก่อนอ้างภายนอก:** ให้เปิด permalink ตรวจอีกครั้ง

**การนับ:** มี 15 เครื่องมือ
- มีเครื่องมือเดียว (TileLang) ที่ล้างด้วยการอ่าน
- ที่เหลือ 14 ตัวเขียนใน path ที่ใช้กันทั่วไปอย่างน้อยหนึ่ง path
  - vLLM เขียนใน benchmark script 19 ไฟล์ ไม่ใช่ helper กลาง
  - QuACK เขียนใน script แต่ autotuner ไม่ล้าง

| Tool | Flush by default | Operation | Buffer | Permalink (file:line) | Uses Triton do_bench? | Related issue/PR |
|---|---|---|---|---|---|---|
| Triton `do_bench` (main, v3.9.0) | yes | write `cache.zero_()` | 256 MiB | [driver.py L409-419](https://github.com/triton-lang/triton/blob/27ee20c554e516db15040cf683296b88304a78e9/third_party/nvidia/backend/driver.py#L409-L419), [testing.py L350](https://github.com/triton-lang/triton/blob/27ee20c554e516db15040cf683296b88304a78e9/python/triton/testing.py#L350), [v3.9.0 driver.py L405-415](https://github.com/triton-lang/triton/blob/b7ddcf86c095de79527ba3098503e73158bb94bd/third_party/nvidia/backend/driver.py#L405-L415) | source | #12554, #12556 (open) |
| PyTorch Inductor `InductorBenchmarker` (default, `use_experimental_benchmarker=True`) | yes | write `zero_()` | = L2 size (256 MiB on ROCm) | [benchmarking.py L753-782](https://github.com/pytorch/pytorch/blob/0341691db7d4c2a39967a883e537743f582803ae/torch/_inductor/runtime/benchmarking.py#L753-L782), [config.py L558-562](https://github.com/pytorch/pytorch/blob/0341691db7d4c2a39967a883e537743f582803ae/torch/_inductor/config.py#L558-L562) | fallback `TritonBenchmarker` only | #203681 (open, ROCm) |
| nvbench (cold measurement) | yes | write `cudaMemsetAsync` | = L2 size | [l2flush.cuh L69-76](https://github.com/NVIDIA/nvbench/blob/41fc6a46358c4d2b202b7e714da7333d3b4cc498/nvbench/detail/l2flush.cuh#L69-L76) | no | #270 (related) |
| FlashInfer `bench_gpu_time` | yes (`cold_l2_cache=True`) | write `zero_()` (CUDA-graph path rotates inputs) | 2 × L2 | [utils.py L880-940](https://github.com/flashinfer-ai/flashinfer/blob/5033e71a875d37978da614610070f785ca754945/flashinfer/testing/utils.py#L880-L940) | no | #8076 (autotuner) |
| DeepGEMM `bench_kineto` | yes (`flush_l2=True`) | write `zero_()` on a new tensor | 256e6 B | [bench.py L92-108](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/testing/bench.py#L92-L108) | no | #68 (other) |
| TileLang `do_bench` | yes | **read** `sum` | 256 MiB | [torch_bench.py L78](https://github.com/tile-ai/tilelang/blob/9009dbc5e79fe199ae0c248b9fca947dcde6f02f/tilelang/profiler/torch_bench.py#L78) | no | #3534 (merged 2026-11-04) |
| tritonbench | yes | write | 256 MiB | [parser.py L329-333](https://github.com/meta-pytorch/tritonbench/blob/6394394849c41c23699f757b4cc2712d35ad4d86/tritonbench/utils/parser.py#L329-L333), [run.py L877](https://github.com/meta-pytorch/tritonbench/blob/6394394849c41c23699f757b4cc2712d35ad4d86/tritonbench/components/do_bench/run.py#L877) | yes | – |
| Helion autotuner | yes (`call`) | write (`call`, `replay`); none (`off`) | 256 MiB | [benchmarking.py L74-86](https://github.com/pytorch/helion/blob/46e131e5dad0d45df1cc77c0277a79ee4b69efc4/helion/autotuner/benchmarking.py#L74-L86), [L1407-1411](https://github.com/pytorch/helion/blob/46e131e5dad0d45df1cc77c0277a79ee4b69efc4/helion/autotuner/benchmarking.py#L1407-L1411) | own copy + Triton `clear_cache` | #4126 (frequency) |
| vLLM `benchmarks/` | per script | write in 19 files; none in 20 (`do_bench_cudagraph`) | 256 MiB or 2 × L2 | [example L371-375](https://github.com/vllm-project/vllm/blob/aa4022dc8176db8f769f93445d845f8755f9b4ca/benchmarks/kernels/benchmark_multimem_reduce_scatter.py#L371-L375) | mostly | – |
| SGLang `marker.do_bench` | yes | write before each graph replay | 5 × L2 | [marker.py L101-108](https://github.com/sgl-project/sglang/blob/188cf7da66938ca2e1ea9cbf7ebbf9a2c81edc2a/python/sglang/kernels/jit/benchmark/marker.py#L101-L108), [L472](https://github.com/sgl-project/sglang/blob/188cf7da66938ca2e1ea9cbf7ebbf9a2c81edc2a/python/sglang/kernels/jit/benchmark/marker.py#L472) | 55/87 kernel benchmark files call it | – |
| SGLang `bench_kineto` | yes | write | 8e9 B | [bench_utils.py L58-91](https://github.com/sgl-project/sglang/blob/188cf7da66938ca2e1ea9cbf7ebbf9a2c81edc2a/python/sglang/srt/utils/bench_utils.py#L58-L91) | no | – |
| Liger-Kernel benchmarks | yes | write | 256 MiB | [utils.py L134-170](https://github.com/linkedin/Liger-Kernel/blob/8c7a5078318da1295dc3ca445f9d85e9e9e153f5/benchmark/scripts/utils.py#L134-L170) | yes | – |
| KernelBench (`timing_method="cuda_event"`) | yes | write `fill_(42)` | 256 MiB | [timing.py L139-141](https://github.com/ScalingIntelligence/KernelBench/blob/423217d9fda91e0c2d67e4a43bf62f96f6d104f1/src/kernelbench/timing.py#L139-L141), [L259](https://github.com/ScalingIntelligence/KernelBench/blob/423217d9fda91e0c2d67e4a43bf62f96f6d104f1/src/kernelbench/timing.py#L259) | no (opt-in modes yes) | #171 (other) |
| QuACK | autotuner: rotation; scripts: yes | autotuner none; scripts write | 256 MiB | [autotuner.py L169-172](https://github.com/Dao-AILab/quack/blob/bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8/quack/autotuner.py#L169-L172), [benchmark_rmsnorm.py L52](https://github.com/Dao-AILab/quack/blob/bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8/benchmarks/benchmark_rmsnorm.py#L52) | scripts yes | – |
| GPU MODE reference-kernels (leaderboard eval) | yes | write `torch.randn` | 4 GiB (nvidia problems) / 256 MiB | [utils.py L183-188](https://github.com/gpu-mode/reference-kernels/blob/7b62f5f110a884c005c0da258a68b05b7522411c/problems/nvidia/utils.py#L183-L188), [eval.py L241](https://github.com/gpu-mode/reference-kernels/blob/7b62f5f110a884c005c0da258a68b05b7522411c/problems/nvidia/eval.py#L241) | no | – |
| CUTLASS CuTe DSL `tune` | yes (`use_cold_l2=True`) | write `cuMemsetD32Async` | = L2 size | [testing.py L767-798](https://github.com/NVIDIA/cutlass/blob/954c8d7be60d730cc3f51e8f1ffe9850dce054d5/python/CuTeDSL/cutlass/testing.py#L767-L798) | no | – |

**หมายเหตุ:**
- Triton `do_bench_cudagraph` ไม่ล้าง L2 เลย
- DeepGEMM `bench()` ล้างครั้งเดียวก่อน warmup จึงวัดแบบ L2 อุ่น
- ความเห็นในโค้ดของ KernelBench บอกว่า L40S มี L2 48 MB แต่ device query ของเราได้ 96 MiB
