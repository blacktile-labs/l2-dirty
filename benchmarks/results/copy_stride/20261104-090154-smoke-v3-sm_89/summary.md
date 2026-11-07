# copy_stride summary -- run `20261104-090154-smoke-v3-sm_89`

- Device: NVIDIA GeForce RTX 4060 (cc 8.9), driver 13000, runtime 13000
- n_src = 67108864 uint32 (256 MiB), flush_l2 = True, reps = 10, warmup = 3
- Correctness failures: 0 (per-variant mismatches: {'empty_kernel': 0, 'memcpy_d2d': 0, 'copy_contig_u32': 0, 'copy_contig_u32x4': 0, 'copy_strided_read_u32_s1': 0, 'copy_strided_read_u32_s2': 0, 'copy_strided_read_u32_s8': 0, 'copy_strided_read_u32_s16': 0, 'copy_strided_read_u32_s32': 0})
- L2 fetch granularity reported by driver: 64 B
- MEASURED reference BW (cudaMemcpy D2D, logical bytes / median): 259.4 GB/s
- ESTIMATED theoretical DRAM BW (device attributes, x2 DDR assumed): 272.0 GB/s
- MEASURED empty-kernel event floor (median): 2.56 us

Columns `est_*` and `pred_*` are MODEL ESTIMATES (bench_common.py, assumptions A1-A5);
`pred_us` = estimated bytes / measured memcpy BW. Everything else is measured or derived.

| variant | stride | n_out | reps | median_us | p95_us | min_us | max_us | cv_pct | useful_GBps | est_MiB_g32 | pred_us_g32 | meas_over_pred_g32 | est_MiB_g64 | pred_us_g64 | meas_over_pred_g64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| empty_kernel | 1 | 0 | 10 | 2.560 | 3.635 | 2.048 | 4.096 | 26.893 | 0.000 |  |  |  |  |  |  |
| memcpy_d2d | 1 | 67108864 | 10 | 2069.504 | 2072.218 | 2067.456 | 2073.600 | 0.081 | 259.420 |  |  |  |  |  |  |
| copy_contig_u32 | 1 | 67108864 | 10 | 2072.576 | 2073.514 | 2071.424 | 2073.600 | 0.038 | 259.036 |  |  |  |  |  |  |
| copy_contig_u32x4 | 1 | 67108864 | 10 | 2068.480 | 2070.067 | 2067.456 | 2070.528 | 0.046 | 259.549 |  |  |  |  |  |  |
| copy_strided_read_u32_s1 | 1 | 67108864 | 10 | 2072.576 | 2073.600 | 2071.552 | 2073.600 | 0.036 | 259.036 | 512.000 | 2069.504 | 1.001 | 512.000 | 2069.504 | 1.001 |
| copy_strided_read_u32_s2 | 2 | 33554432 | 10 | 1534.976 | 1535.632 | 1533.952 | 1536.064 | 0.040 | 174.879 | 384.000 | 1552.128 | 0.989 | 384.000 | 1552.128 | 0.989 |
| copy_strided_read_u32_s8 | 8 | 8388608 | 10 | 1179.840 | 1180.672 | 1178.624 | 1180.672 | 0.056 | 56.880 | 288.000 | 1164.096 | 1.014 | 288.000 | 1164.096 | 1.014 |
| copy_strided_read_u32_s16 | 16 | 4194304 | 10 | 590.848 | 591.872 | 590.848 | 591.872 | 0.089 | 56.790 | 144.000 | 582.048 | 1.015 | 272.000 | 1099.424 | 0.537 |
| copy_strided_read_u32_s32 | 32 | 2097152 | 10 | 296.960 | 297.626 | 296.960 | 297.856 | 0.097 | 56.497 | 72.000 | 291.024 | 1.020 | 136.000 | 549.712 | 0.540 |
