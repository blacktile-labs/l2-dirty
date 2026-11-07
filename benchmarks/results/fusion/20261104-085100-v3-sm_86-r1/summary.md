# fusion summary -- run `20261104-085100-v3-sm_86-r1`

- Device: NVIDIA GeForce RTX 3060 (cc 8.6), driver 12020, runtime 12020, L2 2359296 B
- Correctness failures: 0 (24 checks passed)
- reps 30, warmup 5, steady iters 10, flush_l2 True
- ESTIMATED theoretical DRAM BW 360.0 GB/s; ESTIMATED FP32 peak 12845 GFLOP/s (device attributes)

`useful_GBps` = algorithmic bytes of the task / median time (DERIVED). `isolated` = L2 flushed before each rep; `steady` = back-to-back calls.

| variant | params | mode | reps | median_us | p95_us | cv_pct | useful_GBps |
|---|---|---|---|---|---|---|---|
| fused | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | isolated | 30 | 4.096 | 4.498 | 7.261 | 144.000 |
| fused | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | steady | 30 | 2.333 | 2.355 | 1.730 | 252.818 |
| materialized | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | isolated | 30 | 7.168 | 7.256 | 5.861 | 82.286 |
| materialized | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | steady | 30 | 4.301 | 4.373 | 2.271 | 137.136 |
| stage2_cold | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | isolated | 30 | 4.096 | 4.195 | 10.376 | 144.000 |
| stage2_warm | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | isolated | 30 | 4.000 | 4.096 | 11.469 | 147.456 |
| fused | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | isolated | 30 | 8.192 | 8.798 | 4.707 | 324.000 |
| fused | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | steady | 30 | 8.904 | 8.998 | 0.715 | 298.092 |
| materialized | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | isolated | 30 | 14.336 | 15.360 | 3.668 | 185.143 |
| materialized | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | steady | 30 | 16.282 | 16.469 | 0.532 | 163.015 |
| stage2_cold | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | isolated | 30 | 8.192 | 8.192 | 5.387 | 324.000 |
| stage2_warm | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | isolated | 30 | 7.840 | 8.192 | 6.059 | 338.547 |
| fused | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | isolated | 30 | 10.240 | 10.240 | 4.668 | 316.800 |
| fused | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | steady | 30 | 12.059 | 12.173 | 0.601 | 269.002 |
| materialized | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | isolated | 30 | 21.504 | 22.398 | 2.001 | 150.857 |
| materialized | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | steady | 30 | 23.552 | 23.651 | 0.274 | 137.739 |
| stage2_cold | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | isolated | 30 | 9.216 | 10.240 | 4.800 | 352.000 |
| stage2_warm | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | isolated | 30 | 13.088 | 13.312 | 3.216 | 247.863 |
| fused | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | isolated | 30 | 11.344 | 12.288 | 3.995 | 337.963 |
| fused | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | steady | 30 | 13.722 | 13.824 | 0.404 | 279.395 |
| materialized | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | isolated | 30 | 25.600 | 25.618 | 1.901 | 149.760 |
| materialized | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | steady | 30 | 27.238 | 27.337 | 0.205 | 140.754 |
| stage2_cold | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | isolated | 30 | 11.264 | 12.288 | 3.397 | 340.364 |
| stage2_warm | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | isolated | 30 | 14.336 | 15.360 | 2.626 | 267.429 |
| fused | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | isolated | 30 | 13.312 | 13.312 | 1.418 | 332.308 |
| fused | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | steady | 30 | 15.462 | 15.562 | 0.335 | 286.100 |
| materialized | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | isolated | 30 | 28.672 | 29.696 | 1.668 | 154.286 |
| materialized | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | steady | 30 | 30.720 | 30.822 | 0.181 | 144.000 |
| stage2_cold | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | isolated | 30 | 13.312 | 13.522 | 2.907 | 332.308 |
| stage2_warm | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | isolated | 30 | 16.384 | 16.610 | 2.498 | 270.000 |
| fused | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | isolated | 30 | 16.384 | 16.434 | 2.738 | 324.000 |
| fused | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | steady | 30 | 18.227 | 18.330 | 0.285 | 291.239 |
| materialized | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | isolated | 30 | 33.952 | 34.816 | 1.631 | 156.351 |
| materialized | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | steady | 30 | 36.221 | 36.287 | 0.173 | 146.556 |
| stage2_cold | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | isolated | 30 | 15.568 | 16.384 | 3.145 | 340.983 |
| stage2_warm | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | isolated | 30 | 19.456 | 19.456 | 2.019 | 272.842 |
| fused | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | isolated | 30 | 19.456 | 20.480 | 1.606 | 333.474 |
| fused | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | steady | 30 | 21.721 | 21.811 | 0.238 | 298.693 |
| materialized | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | isolated | 30 | 40.960 | 41.984 | 1.181 | 158.400 |
| materialized | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | steady | 30 | 43.227 | 43.315 | 0.121 | 150.093 |
| stage2_cold | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | isolated | 30 | 19.456 | 20.178 | 1.840 | 333.474 |
| stage2_warm | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | isolated | 30 | 22.528 | 23.552 | 1.647 | 288.000 |
| fused | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | isolated | 30 | 24.880 | 25.600 | 1.880 | 331.895 |
| fused | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | steady | 30 | 27.201 | 27.280 | 0.204 | 303.569 |
| materialized | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | isolated | 30 | 52.224 | 52.224 | 0.870 | 158.118 |
| materialized | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | steady | 30 | 54.045 | 54.116 | 0.103 | 152.790 |
| stage2_cold | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | isolated | 30 | 24.576 | 25.600 | 1.674 | 336.000 |
| stage2_warm | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | isolated | 30 | 27.776 | 28.672 | 1.535 | 297.290 |
| fused | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | isolated | 30 | 28.672 | 29.322 | 1.341 | 329.143 |
| fused | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | steady | 30 | 30.720 | 30.821 | 0.190 | 307.200 |
| materialized | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | isolated | 30 | 59.392 | 59.506 | 0.761 | 158.897 |
| materialized | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | steady | 30 | 61.121 | 61.199 | 0.090 | 154.400 |
| stage2_cold | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | isolated | 30 | 28.672 | 28.672 | 1.321 | 329.143 |
| stage2_warm | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | isolated | 30 | 31.744 | 31.878 | 0.866 | 297.290 |
| fused | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | isolated | 30 | 35.840 | 36.518 | 0.923 | 329.143 |
| fused | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | steady | 30 | 37.885 | 37.962 | 0.127 | 311.376 |
| materialized | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | isolated | 30 | 73.728 | 73.728 | 0.683 | 160.000 |
| materialized | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | steady | 30 | 75.385 | 75.511 | 0.100 | 156.482 |
| stage2_cold | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | isolated | 30 | 35.728 | 35.840 | 1.398 | 330.175 |
| stage2_warm | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | isolated | 30 | 38.912 | 38.912 | 0.720 | 303.158 |
| fused | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | isolated | 30 | 43.008 | 43.090 | 0.878 | 329.143 |
| fused | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | steady | 30 | 45.053 | 45.113 | 0.132 | 314.203 |
| materialized | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | isolated | 30 | 88.000 | 88.064 | 0.559 | 160.861 |
| materialized | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | steady | 30 | 89.706 | 89.795 | 0.078 | 157.802 |
| stage2_cold | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | isolated | 30 | 42.752 | 43.008 | 1.127 | 331.114 |
| stage2_warm | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | isolated | 30 | 46.048 | 46.080 | 0.946 | 307.413 |
| fused | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | isolated | 30 | 57.344 | 57.539 | 0.522 | 329.143 |
| fused | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | steady | 30 | 59.390 | 59.536 | 0.153 | 317.801 |
| materialized | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | isolated | 30 | 116.272 | 116.806 | 0.475 | 162.329 |
| materialized | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | steady | 30 | 118.374 | 118.477 | 0.071 | 159.447 |
| stage2_cold | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | isolated | 30 | 57.344 | 57.344 | 0.808 | 329.143 |
| stage2_warm | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | isolated | 30 | 60.416 | 60.501 | 0.575 | 312.407 |

### Fusion payoff by size and arithmetic intensity

`mat/fused` = materialized time / fused time (MEASURED; >1 means fusion helps). `warm/cold` = stage2 time right after stage1 / after an L2 flush (MEASURED; <1 means t was partly served from L2). `est_r_*` = fraction of t predicted to survive in L2 by the two residency models. `AI_fused` = FLOP per algorithmic DRAM byte. Estimated ridge point = 12845 GFLOP/s / 360 GB/s = 35.7 FLOP/B (ESTIMATE).

| l2x | n | K | fused_us(iso) | mat_us(iso) | mat/fused(iso) | mat/fused(steady) | warm/cold | est_r_lru_cliff | est_r_proportional | AI_fused |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.125000 | 73728 | 1 | 4.096 | 7.168 | 1.750 | 1.844 | 0.977 | 1.000 | 1.000 | 0.500 |
| 0.562500 | 331776 | 1 | 8.192 | 14.336 | 1.750 | 1.829 | 0.957 | 0.000 | 0.889 | 0.500 |
| 0.687500 | 405504 | 1 | 10.240 | 21.504 | 2.100 | 1.953 | 1.420 | 0.000 | 0.727 | 0.500 |
| 0.812500 | 479232 | 1 | 11.344 | 25.600 | 2.257 | 1.985 | 1.273 | 0.000 | 0.615 | 0.500 |
| 0.937500 | 552960 | 1 | 13.312 | 28.672 | 2.154 | 1.987 | 1.231 | 0.000 | 0.533 | 0.500 |
| 1.125000 | 663552 | 1 | 16.384 | 33.952 | 2.072 | 1.987 | 1.250 | 0.000 | 0.444 | 0.500 |
| 1.375000 | 811008 | 1 | 19.456 | 40.960 | 2.105 | 1.990 | 1.158 | 0.000 | 0.364 | 0.500 |
| 1.750000 | 1032192 | 1 | 24.880 | 52.224 | 2.099 | 1.987 | 1.130 | 0.000 | 0.286 | 0.500 |
| 2.000000 | 1179648 | 1 | 28.672 | 59.392 | 2.071 | 1.990 | 1.107 | 0.000 | 0.250 | 0.500 |
| 2.500000 | 1474560 | 1 | 35.840 | 73.728 | 2.057 | 1.990 | 1.089 | 0.000 | 0.200 | 0.500 |
| 3.000000 | 1769472 | 1 | 43.008 | 88.000 | 2.046 | 1.991 | 1.077 | 0.000 | 0.167 | 0.500 |
| 4.000000 | 2359296 | 1 | 57.344 | 116.272 | 2.028 | 1.993 | 1.054 | 0.000 | 0.125 | 0.500 |
