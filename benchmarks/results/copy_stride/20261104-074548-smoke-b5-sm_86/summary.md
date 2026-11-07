# copy_stride summary -- run `20261104-074548-smoke-b5-sm_86`

- Device: NVIDIA GeForce RTX 3060 (cc 8.6), driver 12020, runtime 12020
- n_src = 67108864 uint32 (256 MiB), flush_l2 = True, reps = 10, warmup = 3
- Correctness failures: 0 (per-variant mismatches: {'empty_kernel': 0, 'memcpy_d2d': 0, 'copy_contig_u32': 0, 'copy_contig_u32x4': 0, 'copy_strided_read_u32_s1': 0, 'copy_strided_read_u32_s2': 0, 'copy_strided_read_u32_s8': 0, 'copy_strided_read_u32_s16': 0, 'copy_strided_read_u32_s32': 0})
- L2 fetch granularity reported by driver: 64 B
- MEASURED reference BW (cudaMemcpy D2D, logical bytes / median): 333.3 GB/s
- ESTIMATED theoretical DRAM BW (device attributes, x2 DDR assumed): 360.0 GB/s
- MEASURED empty-kernel event floor (median): 2.56 us

Columns `est_*` and `pred_*` are MODEL ESTIMATES (bench_common.py, assumptions A1-A5);
`pred_us` = estimated bytes / measured memcpy BW. Everything else is measured or derived.

| variant | stride | n_out | reps | median_us | p95_us | min_us | max_us | cv_pct | useful_GBps | est_MiB_g32 | pred_us_g32 | meas_over_pred_g32 | est_MiB_g64 | pred_us_g64 | meas_over_pred_g64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| empty_kernel | 1 | 0 | 10 | 2.560 | 3.072 | 2.048 | 3.072 | 21.082 | 0.000 |  |  |  |  |  |  |
| memcpy_d2d | 1 | 67108864 | 10 | 1610.752 | 1611.381 | 1610.752 | 1611.712 | 0.019 | 333.305 |  |  |  |  |  |  |
| copy_contig_u32 | 1 | 67108864 | 10 | 1622.624 | 1623.603 | 1622.016 | 1624.064 | 0.043 | 330.866 |  |  |  |  |  |  |
| copy_contig_u32x4 | 1 | 67108864 | 10 | 1609.728 | 1610.805 | 1609.632 | 1610.848 | 0.028 | 333.517 |  |  |  |  |  |  |
| copy_strided_read_u32_s1 | 1 | 67108864 | 10 | 1624.064 | 1624.627 | 1623.040 | 1625.088 | 0.030 | 330.573 | 512.000 | 1610.752 | 1.008 | 512.000 | 1610.752 | 1.008 |
| copy_strided_read_u32_s2 | 2 | 33554432 | 10 | 1196.224 | 1197.056 | 1195.008 | 1197.056 | 0.065 | 224.402 | 384.000 | 1208.064 | 0.990 | 384.000 | 1208.064 | 0.990 |
| copy_strided_read_u32_s8 | 8 | 8388608 | 10 | 897.024 | 897.645 | 896.960 | 898.048 | 0.036 | 74.813 | 288.000 | 906.048 | 0.990 | 288.000 | 906.048 | 0.990 |
| copy_strided_read_u32_s16 | 16 | 4194304 | 10 | 450.560 | 451.123 | 449.856 | 451.584 | 0.105 | 74.473 | 144.000 | 453.024 | 0.995 | 272.000 | 855.712 | 0.527 |
| copy_strided_read_u32_s32 | 32 | 2097152 | 10 | 227.328 | 227.891 | 226.304 | 228.352 | 0.212 | 73.802 | 72.000 | 226.512 | 1.004 | 136.000 | 427.856 | 0.531 |
