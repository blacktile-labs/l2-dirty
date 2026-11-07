# fusion summary -- run `20261104-085102-v3-sm_86-r3`

- Device: NVIDIA GeForce RTX 3060 (cc 8.6), driver 12020, runtime 12020, L2 2359296 B
- Correctness failures: 0 (24 checks passed)
- reps 30, warmup 5, steady iters 10, flush_l2 True
- ESTIMATED theoretical DRAM BW 360.0 GB/s; ESTIMATED FP32 peak 12845 GFLOP/s (device attributes)

`useful_GBps` = algorithmic bytes of the task / median time (DERIVED). `isolated` = L2 flushed before each rep; `steady` = back-to-back calls.

| variant | params | mode | reps | median_us | p95_us | cv_pct | useful_GBps |
|---|---|---|---|---|---|---|---|
| fused | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | isolated | 30 | 4.096 | 5.120 | 8.616 | 144.000 |
| fused | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | steady | 30 | 2.340 | 2.355 | 2.207 | 252.008 |
| materialized | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | isolated | 30 | 7.168 | 7.282 | 7.003 | 82.286 |
| materialized | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | steady | 30 | 4.282 | 4.310 | 1.052 | 137.745 |
| stage2_cold | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | isolated | 30 | 4.096 | 4.437 | 8.992 | 144.000 |
| stage2_warm | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | isolated | 30 | 4.080 | 4.096 | 11.518 | 144.565 |
| fused | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | isolated | 30 | 8.192 | 8.192 | 0.831 | 324.000 |
| fused | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | steady | 30 | 8.888 | 8.976 | 0.589 | 298.628 |
| materialized | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | isolated | 30 | 14.336 | 15.360 | 3.654 | 185.143 |
| materialized | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | steady | 30 | 16.282 | 16.393 | 0.507 | 163.015 |
| stage2_cold | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | isolated | 30 | 8.192 | 8.192 | 5.978 | 324.000 |
| stage2_warm | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | isolated | 30 | 7.904 | 8.192 | 5.314 | 335.806 |
| fused | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | isolated | 30 | 10.240 | 10.307 | 3.756 | 316.800 |
| fused | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | steady | 30 | 12.018 | 12.088 | 0.516 | 269.942 |
| materialized | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | isolated | 30 | 21.504 | 22.528 | 2.658 | 150.857 |
| materialized | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | steady | 30 | 23.552 | 23.658 | 0.268 | 137.739 |
| stage2_cold | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | isolated | 30 | 9.504 | 10.240 | 4.680 | 341.333 |
| stage2_warm | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | isolated | 30 | 13.056 | 13.312 | 3.346 | 248.471 |
| fused | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | isolated | 30 | 11.264 | 12.288 | 3.809 | 340.364 |
| fused | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | steady | 30 | 13.725 | 13.824 | 0.404 | 279.334 |
| materialized | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | isolated | 30 | 25.600 | 25.600 | 2.001 | 149.760 |
| materialized | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | steady | 30 | 27.238 | 27.338 | 0.178 | 140.754 |
| stage2_cold | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | isolated | 30 | 11.264 | 11.971 | 2.347 | 340.364 |
| stage2_warm | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | isolated | 30 | 14.336 | 15.360 | 3.086 | 267.429 |
| fused | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | isolated | 30 | 13.312 | 13.890 | 2.881 | 332.308 |
| fused | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | steady | 30 | 15.462 | 15.565 | 0.345 | 286.100 |
| materialized | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | isolated | 30 | 28.672 | 29.696 | 1.932 | 154.286 |
| materialized | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | steady | 30 | 30.720 | 30.803 | 0.149 | 144.000 |
| stage2_cold | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | isolated | 30 | 13.312 | 13.507 | 3.007 | 332.308 |
| stage2_warm | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | isolated | 30 | 16.384 | 17.408 | 1.886 | 270.000 |
| fused | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | isolated | 30 | 16.384 | 16.384 | 2.915 | 324.000 |
| fused | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | steady | 30 | 18.227 | 18.332 | 0.336 | 291.239 |
| materialized | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | isolated | 30 | 33.792 | 34.816 | 1.460 | 157.091 |
| materialized | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | steady | 30 | 36.156 | 36.248 | 0.153 | 146.822 |
| stage2_cold | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | isolated | 30 | 16.240 | 16.384 | 2.982 | 326.873 |
| stage2_warm | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | isolated | 30 | 19.456 | 19.456 | 2.151 | 272.842 |
| fused | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | isolated | 30 | 19.456 | 20.480 | 2.124 | 333.474 |
| fused | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | steady | 30 | 21.709 | 21.848 | 0.267 | 298.865 |
| materialized | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | isolated | 30 | 40.960 | 41.984 | 1.241 | 158.400 |
| materialized | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | steady | 30 | 43.220 | 43.314 | 0.114 | 150.119 |
| stage2_cold | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | isolated | 30 | 19.456 | 19.696 | 1.448 | 333.474 |
| stage2_warm | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | isolated | 30 | 22.528 | 23.552 | 1.622 | 288.000 |
| fused | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | isolated | 30 | 25.136 | 25.600 | 1.864 | 328.514 |
| fused | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | steady | 30 | 27.215 | 27.280 | 0.213 | 303.424 |
| materialized | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | isolated | 30 | 52.224 | 52.832 | 1.008 | 158.118 |
| materialized | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | steady | 30 | 54.050 | 54.067 | 0.101 | 152.777 |
| stage2_cold | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | isolated | 30 | 24.576 | 25.600 | 1.555 | 336.000 |
| stage2_warm | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | isolated | 30 | 28.048 | 28.672 | 1.606 | 294.407 |
| fused | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | isolated | 30 | 28.672 | 28.757 | 1.062 | 329.143 |
| fused | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | steady | 30 | 30.720 | 30.822 | 0.163 | 307.200 |
| materialized | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | isolated | 30 | 59.392 | 59.474 | 0.702 | 158.897 |
| materialized | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | steady | 30 | 61.112 | 61.213 | 0.082 | 154.424 |
| stage2_cold | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | isolated | 30 | 28.672 | 28.742 | 1.454 | 329.143 |
| stage2_warm | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | isolated | 30 | 31.744 | 31.974 | 1.033 | 297.290 |
| fused | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | isolated | 30 | 35.840 | 35.946 | 0.577 | 329.143 |
| fused | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | steady | 30 | 37.888 | 37.964 | 0.112 | 311.351 |
| materialized | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | isolated | 30 | 73.728 | 73.728 | 0.691 | 160.000 |
| materialized | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | steady | 30 | 75.379 | 75.469 | 0.078 | 156.496 |
| stage2_cold | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | isolated | 30 | 35.840 | 35.840 | 1.115 | 329.143 |
| stage2_warm | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | isolated | 30 | 38.912 | 39.125 | 0.917 | 303.158 |
| fused | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | isolated | 30 | 43.008 | 43.432 | 0.821 | 329.143 |
| fused | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | steady | 30 | 45.046 | 45.127 | 0.101 | 314.252 |
| materialized | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | isolated | 30 | 88.064 | 88.352 | 0.617 | 160.744 |
| materialized | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | steady | 30 | 89.702 | 89.802 | 0.076 | 157.809 |
| stage2_cold | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | isolated | 30 | 42.960 | 43.008 | 1.081 | 329.511 |
| stage2_warm | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | isolated | 30 | 46.080 | 46.080 | 0.920 | 307.200 |
| fused | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | isolated | 30 | 57.344 | 57.379 | 0.516 | 329.143 |
| fused | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | steady | 30 | 59.392 | 59.485 | 0.150 | 317.793 |
| materialized | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | isolated | 30 | 116.672 | 116.789 | 0.411 | 161.773 |
| materialized | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | steady | 30 | 118.374 | 118.544 | 0.109 | 159.447 |
| stage2_cold | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | isolated | 30 | 57.344 | 57.362 | 0.793 | 329.143 |
| stage2_warm | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | isolated | 30 | 60.416 | 60.547 | 0.671 | 312.407 |

### Fusion payoff by size and arithmetic intensity

`mat/fused` = materialized time / fused time (MEASURED; >1 means fusion helps). `warm/cold` = stage2 time right after stage1 / after an L2 flush (MEASURED; <1 means t was partly served from L2). `est_r_*` = fraction of t predicted to survive in L2 by the two residency models. `AI_fused` = FLOP per algorithmic DRAM byte. Estimated ridge point = 12845 GFLOP/s / 360 GB/s = 35.7 FLOP/B (ESTIMATE).

| l2x | n | K | fused_us(iso) | mat_us(iso) | mat/fused(iso) | mat/fused(steady) | warm/cold | est_r_lru_cliff | est_r_proportional | AI_fused |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.125000 | 73728 | 1 | 4.096 | 7.168 | 1.750 | 1.830 | 0.996 | 1.000 | 1.000 | 0.500 |
| 0.562500 | 331776 | 1 | 8.192 | 14.336 | 1.750 | 1.832 | 0.965 | 0.000 | 0.889 | 0.500 |
| 0.687500 | 405504 | 1 | 10.240 | 21.504 | 2.100 | 1.960 | 1.374 | 0.000 | 0.727 | 0.500 |
| 0.812500 | 479232 | 1 | 11.264 | 25.600 | 2.273 | 1.985 | 1.273 | 0.000 | 0.615 | 0.500 |
| 0.937500 | 552960 | 1 | 13.312 | 28.672 | 2.154 | 1.987 | 1.231 | 0.000 | 0.533 | 0.500 |
| 1.125000 | 663552 | 1 | 16.384 | 33.792 | 2.062 | 1.984 | 1.198 | 0.000 | 0.444 | 0.500 |
| 1.375000 | 811008 | 1 | 19.456 | 40.960 | 2.105 | 1.991 | 1.158 | 0.000 | 0.364 | 0.500 |
| 1.750000 | 1032192 | 1 | 25.136 | 52.224 | 2.078 | 1.986 | 1.141 | 0.000 | 0.286 | 0.500 |
| 2.000000 | 1179648 | 1 | 28.672 | 59.392 | 2.071 | 1.989 | 1.107 | 0.000 | 0.250 | 0.500 |
| 2.500000 | 1474560 | 1 | 35.840 | 73.728 | 2.057 | 1.990 | 1.086 | 0.000 | 0.200 | 0.500 |
| 3.000000 | 1769472 | 1 | 43.008 | 88.064 | 2.048 | 1.991 | 1.073 | 0.000 | 0.167 | 0.500 |
| 4.000000 | 2359296 | 1 | 57.344 | 116.672 | 2.035 | 1.993 | 1.054 | 0.000 | 0.125 | 0.500 |
