# fusion summary -- run `20261104-085101-v3-sm_86-r2`

- Device: NVIDIA GeForce RTX 3060 (cc 8.6), driver 12020, runtime 12020, L2 2359296 B
- Correctness failures: 0 (24 checks passed)
- reps 30, warmup 5, steady iters 10, flush_l2 True
- ESTIMATED theoretical DRAM BW 360.0 GB/s; ESTIMATED FP32 peak 12845 GFLOP/s (device attributes)

`useful_GBps` = algorithmic bytes of the task / median time (DERIVED). `isolated` = L2 flushed before each rep; `steady` = back-to-back calls.

| variant | params | mode | reps | median_us | p95_us | cv_pct | useful_GBps |
|---|---|---|---|---|---|---|---|
| fused | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | isolated | 30 | 4.096 | 4.731 | 6.235 | 144.000 |
| fused | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | steady | 30 | 2.336 | 2.355 | 1.827 | 252.493 |
| materialized | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | isolated | 30 | 6.864 | 7.238 | 6.815 | 85.930 |
| materialized | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | steady | 30 | 4.291 | 4.301 | 0.670 | 137.456 |
| stage2_cold | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | isolated | 30 | 4.096 | 4.202 | 9.121 | 144.000 |
| stage2_warm | n=73728;l2x=0.125000;K=1;l2_bytes=2359296 | isolated | 30 | 3.776 | 4.096 | 10.431 | 156.203 |
| fused | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | isolated | 30 | 8.192 | 8.192 | 3.274 | 324.000 |
| fused | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | steady | 30 | 8.909 | 8.994 | 0.802 | 297.924 |
| materialized | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | isolated | 30 | 14.336 | 15.360 | 3.302 | 185.143 |
| materialized | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | steady | 30 | 16.282 | 16.384 | 0.447 | 163.015 |
| stage2_cold | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | isolated | 30 | 8.192 | 8.192 | 4.881 | 324.000 |
| stage2_warm | n=331776;l2x=0.562500;K=1;l2_bytes=2359296 | isolated | 30 | 7.472 | 8.192 | 6.333 | 355.221 |
| fused | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | isolated | 30 | 10.144 | 10.240 | 4.821 | 319.798 |
| fused | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | steady | 30 | 12.077 | 12.159 | 0.489 | 268.624 |
| materialized | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | isolated | 30 | 21.504 | 22.442 | 2.076 | 150.857 |
| materialized | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | steady | 30 | 23.552 | 23.654 | 0.224 | 137.739 |
| stage2_cold | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | isolated | 30 | 9.328 | 10.240 | 4.969 | 347.774 |
| stage2_warm | n=405504;l2x=0.687500;K=1;l2_bytes=2359296 | isolated | 30 | 13.184 | 13.312 | 3.425 | 246.058 |
| fused | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | isolated | 30 | 11.280 | 12.288 | 3.822 | 339.881 |
| fused | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | steady | 30 | 13.722 | 13.807 | 0.339 | 279.395 |
| materialized | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | isolated | 30 | 25.600 | 25.600 | 1.857 | 149.760 |
| materialized | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | steady | 30 | 27.241 | 27.337 | 0.205 | 140.736 |
| stage2_cold | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | isolated | 30 | 11.264 | 11.957 | 2.316 | 340.364 |
| stage2_warm | n=479232;l2x=0.812500;K=1;l2_bytes=2359296 | isolated | 30 | 14.336 | 15.360 | 2.961 | 267.429 |
| fused | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | isolated | 30 | 13.312 | 13.888 | 2.846 | 332.308 |
| fused | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | steady | 30 | 15.462 | 15.565 | 0.352 | 286.100 |
| materialized | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | isolated | 30 | 28.672 | 29.696 | 1.595 | 154.286 |
| materialized | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | steady | 30 | 30.720 | 30.822 | 0.194 | 144.000 |
| stage2_cold | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | isolated | 30 | 13.312 | 13.490 | 1.890 | 332.308 |
| stage2_warm | n=552960;l2x=0.937500;K=1;l2_bytes=2359296 | isolated | 30 | 16.384 | 17.106 | 1.616 | 270.000 |
| fused | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | isolated | 30 | 16.384 | 16.384 | 2.889 | 324.000 |
| fused | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | steady | 30 | 18.306 | 18.330 | 0.269 | 289.990 |
| materialized | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | isolated | 30 | 33.792 | 34.816 | 1.361 | 157.091 |
| materialized | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | steady | 30 | 36.147 | 36.250 | 0.153 | 146.856 |
| stage2_cold | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | isolated | 30 | 15.584 | 16.384 | 3.211 | 340.632 |
| stage2_warm | n=663552;l2x=1.125000;K=1;l2_bytes=2359296 | isolated | 30 | 19.456 | 19.456 | 1.770 | 272.842 |
| fused | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | isolated | 30 | 19.456 | 20.480 | 1.965 | 333.474 |
| fused | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | steady | 30 | 21.709 | 21.811 | 0.262 | 298.865 |
| materialized | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | isolated | 30 | 40.960 | 41.984 | 1.094 | 158.400 |
| materialized | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | steady | 30 | 43.215 | 43.315 | 0.172 | 150.136 |
| stage2_cold | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | isolated | 30 | 19.456 | 19.626 | 1.852 | 333.474 |
| stage2_warm | n=811008;l2x=1.375000;K=1;l2_bytes=2359296 | isolated | 30 | 22.528 | 23.077 | 1.161 | 288.000 |
| fused | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | isolated | 30 | 25.200 | 25.600 | 1.720 | 327.680 |
| fused | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | steady | 30 | 27.217 | 27.327 | 0.268 | 303.391 |
| materialized | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | isolated | 30 | 52.224 | 52.483 | 0.919 | 158.118 |
| materialized | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | steady | 30 | 54.035 | 54.076 | 0.104 | 152.818 |
| stage2_cold | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | isolated | 30 | 24.576 | 25.600 | 1.639 | 336.000 |
| stage2_warm | n=1032192;l2x=1.750000;K=1;l2_bytes=2359296 | isolated | 30 | 27.648 | 28.672 | 1.436 | 298.667 |
| fused | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | isolated | 30 | 28.672 | 29.197 | 1.604 | 329.143 |
| fused | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | steady | 30 | 30.720 | 30.822 | 0.182 | 307.200 |
| materialized | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | isolated | 30 | 59.392 | 59.392 | 0.843 | 158.897 |
| materialized | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | steady | 30 | 61.101 | 61.133 | 0.068 | 154.452 |
| stage2_cold | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | isolated | 30 | 28.672 | 28.707 | 1.682 | 329.143 |
| stage2_warm | n=1179648;l2x=2.000000;K=1;l2_bytes=2359296 | isolated | 30 | 31.744 | 31.814 | 1.009 | 297.290 |
| fused | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | isolated | 30 | 35.840 | 36.053 | 0.781 | 329.143 |
| fused | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | steady | 30 | 37.888 | 37.968 | 0.131 | 311.351 |
| materialized | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | isolated | 30 | 73.488 | 73.746 | 0.711 | 160.523 |
| materialized | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | steady | 30 | 75.366 | 75.469 | 0.065 | 156.523 |
| stage2_cold | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | isolated | 30 | 35.840 | 35.840 | 1.308 | 329.143 |
| stage2_warm | n=1474560;l2x=2.500000;K=1;l2_bytes=2359296 | isolated | 30 | 38.912 | 38.930 | 0.737 | 303.158 |
| fused | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | isolated | 30 | 43.008 | 43.168 | 0.639 | 329.143 |
| fused | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | steady | 30 | 45.056 | 45.133 | 0.116 | 314.182 |
| materialized | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | isolated | 30 | 88.064 | 88.555 | 0.663 | 160.744 |
| materialized | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | steady | 30 | 89.701 | 89.805 | 0.080 | 157.812 |
| stage2_cold | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | isolated | 30 | 42.240 | 43.008 | 1.148 | 335.127 |
| stage2_warm | n=1769472;l2x=3.000000;K=1;l2_bytes=2359296 | isolated | 30 | 45.872 | 46.080 | 1.070 | 308.593 |
| fused | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | isolated | 30 | 57.344 | 57.866 | 0.778 | 329.143 |
| fused | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | steady | 30 | 59.394 | 59.493 | 0.131 | 317.785 |
| materialized | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | isolated | 30 | 115.952 | 116.736 | 0.445 | 162.777 |
| materialized | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | steady | 30 | 118.374 | 118.477 | 0.076 | 159.447 |
| stage2_cold | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | isolated | 30 | 57.264 | 57.344 | 0.829 | 329.603 |
| stage2_warm | n=2359296;l2x=4.000000;K=1;l2_bytes=2359296 | isolated | 30 | 60.416 | 60.699 | 0.627 | 312.407 |

### Fusion payoff by size and arithmetic intensity

`mat/fused` = materialized time / fused time (MEASURED; >1 means fusion helps). `warm/cold` = stage2 time right after stage1 / after an L2 flush (MEASURED; <1 means t was partly served from L2). `est_r_*` = fraction of t predicted to survive in L2 by the two residency models. `AI_fused` = FLOP per algorithmic DRAM byte. Estimated ridge point = 12845 GFLOP/s / 360 GB/s = 35.7 FLOP/B (ESTIMATE).

| l2x | n | K | fused_us(iso) | mat_us(iso) | mat/fused(iso) | mat/fused(steady) | warm/cold | est_r_lru_cliff | est_r_proportional | AI_fused |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.125000 | 73728 | 1 | 4.096 | 6.864 | 1.676 | 1.837 | 0.922 | 1.000 | 1.000 | 0.500 |
| 0.562500 | 331776 | 1 | 8.192 | 14.336 | 1.750 | 1.828 | 0.912 | 0.000 | 0.889 | 0.500 |
| 0.687500 | 405504 | 1 | 10.144 | 21.504 | 2.120 | 1.950 | 1.413 | 0.000 | 0.727 | 0.500 |
| 0.812500 | 479232 | 1 | 11.280 | 25.600 | 2.270 | 1.985 | 1.273 | 0.000 | 0.615 | 0.500 |
| 0.937500 | 552960 | 1 | 13.312 | 28.672 | 2.154 | 1.987 | 1.231 | 0.000 | 0.533 | 0.500 |
| 1.125000 | 663552 | 1 | 16.384 | 33.792 | 2.062 | 1.975 | 1.248 | 0.000 | 0.444 | 0.500 |
| 1.375000 | 811008 | 1 | 19.456 | 40.960 | 2.105 | 1.991 | 1.158 | 0.000 | 0.364 | 0.500 |
| 1.750000 | 1032192 | 1 | 25.200 | 52.224 | 2.072 | 1.985 | 1.125 | 0.000 | 0.286 | 0.500 |
| 2.000000 | 1179648 | 1 | 28.672 | 59.392 | 2.071 | 1.989 | 1.107 | 0.000 | 0.250 | 0.500 |
| 2.500000 | 1474560 | 1 | 35.840 | 73.488 | 2.050 | 1.989 | 1.086 | 0.000 | 0.200 | 0.500 |
| 3.000000 | 1769472 | 1 | 43.008 | 88.064 | 2.048 | 1.991 | 1.086 | 0.000 | 0.167 | 0.500 |
| 4.000000 | 2359296 | 1 | 57.344 | 115.952 | 2.022 | 1.993 | 1.055 | 0.000 | 0.125 | 0.500 |
