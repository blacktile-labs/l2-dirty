# copy_stride summary -- run `20261104-075048-smoke-b5-sm_89`

- Device: NVIDIA GeForce RTX 4060 (cc 8.9), driver 13000, runtime 13000
- n_src = 67108864 uint32 (256 MiB), flush_l2 = True, reps = 10, warmup = 3
- Correctness failures: 0 (per-variant mismatches: {'empty_kernel': 0, 'memcpy_d2d': 0, 'copy_contig_u32': 0, 'copy_contig_u32x4': 0, 'copy_strided_read_u32_s1': 0, 'copy_strided_read_u32_s2': 0, 'copy_strided_read_u32_s8': 0, 'copy_strided_read_u32_s16': 0, 'copy_strided_read_u32_s32': 0})
- L2 fetch granularity reported by driver: 64 B
- MEASURED reference BW (cudaMemcpy D2D, logical bytes / median): 259.6 GB/s
- ESTIMATED theoretical DRAM BW (device attributes, x2 DDR assumed): 272.0 GB/s
- MEASURED empty-kernel event floor (median): 2.67 us

Columns `est_*` and `pred_*` are MODEL ESTIMATES (bench_common.py, assumptions A1-A5);
`pred_us` = estimated bytes / measured memcpy BW. Everything else is measured or derived.

| variant | stride | n_out | reps | median_us | p95_us | min_us | max_us | cv_pct | useful_GBps | est_MiB_g32 | pred_us_g32 | meas_over_pred_g32 | est_MiB_g64 | pred_us_g64 | meas_over_pred_g64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| empty_kernel | 1 | 0 | 10 | 2.672 | 3.072 | 2.048 | 3.072 | 19.410 | 0.000 |  |  |  |  |  |  |
| memcpy_d2d | 1 | 67108864 | 10 | 2068.416 | 2068.480 | 2067.456 | 2068.480 | 0.022 | 259.557 |  |  |  |  |  |  |
| copy_contig_u32 | 1 | 67108864 | 10 | 2072.576 | 2073.600 | 2072.576 | 2073.600 | 0.021 | 259.036 |  |  |  |  |  |  |
| copy_contig_u32x4 | 1 | 67108864 | 10 | 2067.456 | 2068.422 | 2067.456 | 2068.480 | 0.019 | 259.677 |  |  |  |  |  |  |
| copy_strided_read_u32_s1 | 1 | 67108864 | 10 | 2073.600 | 2074.624 | 2072.576 | 2074.624 | 0.039 | 258.908 | 512.000 | 2068.416 | 1.003 | 512.000 | 2068.416 | 1.003 |
| copy_strided_read_u32_s2 | 2 | 33554432 | 10 | 1536.000 | 1537.587 | 1534.816 | 1538.048 | 0.057 | 174.763 | 384.000 | 1551.312 | 0.990 | 384.000 | 1551.312 | 0.990 |
| copy_strided_read_u32_s8 | 8 | 8388608 | 10 | 1180.672 | 1180.771 | 1178.624 | 1180.800 | 0.061 | 56.840 | 288.000 | 1163.484 | 1.015 | 288.000 | 1163.484 | 1.015 |
| copy_strided_read_u32_s16 | 16 | 4194304 | 10 | 591.872 | 591.872 | 591.744 | 591.872 | 0.007 | 56.692 | 144.000 | 581.742 | 1.017 | 272.000 | 1098.846 | 0.539 |
| copy_strided_read_u32_s32 | 32 | 2097152 | 10 | 297.984 | 298.547 | 296.960 | 299.008 | 0.186 | 56.302 | 72.000 | 290.871 | 1.024 | 136.000 | 549.423 | 0.542 |
