# gather summary -- run `20261104-071046-rtx4060-probe-default`

- Device: NVIDIA GeForce RTX 4060 (cc 8.9), driver 13000, runtime 13000, L2 25165824 B
- Correctness failures: 0 (6 checks passed)
- reps 20, warmup 3, steady iters 5, flush_l2 True
- ESTIMATED theoretical DRAM BW 272.0 GB/s; ESTIMATED FP32 peak 15391 GFLOP/s (device attributes)

`useful_GBps` = algorithmic bytes of the task / median time (DERIVED). `isolated` = L2 flushed before each rep; `steady` = back-to-back calls.

| variant | params | mode | reps | median_us | p95_us | cv_pct | useful_GBps |
|---|---|---|---|---|---|---|---|
| gather | T=67108864;M=4194304;dist=uniform | isolated | 20 | 1656.832 | 1657.907 | 0.060 | 30.378 |
| gather | T=67108864;M=4194304;dist=uniform | steady | 20 | 1662.579 | 1664.051 | 0.088 | 30.273 |
| gather_pair64 | T=67108864;M=4194304;dist=uniform | isolated | 20 | 2798.592 | 2809.958 | 0.366 | 23.980 |
| gather_pair64 | T=67108864;M=4194304;dist=uniform | steady | 20 | 2797.568 | 2799.849 | 0.175 | 23.988 |
| gather_pair128 | T=67108864;M=4194304;dist=uniform | isolated | 20 | 2798.528 | 2806.886 | 0.372 | 23.980 |
| gather_pair128 | T=67108864;M=4194304;dist=uniform | steady | 20 | 2798.899 | 2805.393 | 0.328 | 23.977 |
| gather | T=67108864;M=4194304;dist=zipf1.0 | isolated | 20 | 731.136 | 732.160 | 0.062 | 68.840 |
| gather | T=67108864;M=4194304;dist=zipf1.0 | steady | 20 | 730.806 | 731.181 | 0.046 | 68.871 |
| gather_pair64 | T=67108864;M=4194304;dist=zipf1.0 | isolated | 20 | 817.664 | 822.091 | 0.454 | 82.074 |
| gather_pair64 | T=67108864;M=4194304;dist=zipf1.0 | steady | 20 | 809.280 | 813.984 | 0.290 | 82.924 |
| gather_pair128 | T=67108864;M=4194304;dist=zipf1.0 | isolated | 20 | 818.176 | 824.422 | 0.512 | 82.023 |
| gather_pair128 | T=67108864;M=4194304;dist=zipf1.0 | steady | 20 | 811.417 | 813.926 | 0.214 | 82.706 |

### Fetch-granularity probe (isolated mode)

`meas_ratio` = pair-variant time / gather time (MEASURED). `pred_g*` = ratio expected if DRAM moves g-byte chunks per miss with no reuse (ESTIMATE; meaningful for random dists). l2_fetch_granularity reported by driver: 64 B (requested: -1).

| variant | dist | pair_us | gather_us | meas_ratio | pred_g32 | pred_g64 | pred_g128 |
|---|---|---|---|---|---|---|---|
| gather_pair64 | uniform | 2798.592 | 1656.832 | 1.689 | 1.800 | 1.000 | 1.000 |
| gather_pair128 | uniform | 2798.528 | 1656.832 | 1.689 | 1.800 | 1.889 | 1.000 |
| gather_pair64 | zipf1.0 | 817.664 | 731.136 | 1.118 | 1.800 | 1.000 | 1.000 |
| gather_pair128 | zipf1.0 | 818.176 | 731.136 | 1.119 | 1.800 | 1.889 | 1.000 |

### Traffic bounds (isolated mode)

`est_*_MiB` are model bytes; `impl_GBps_*` = estimate / measured median. If the *compulsory* (lower-bound) bytes imply more than the estimated DRAM peak (272 GB/s), the model's 32-B-sector assumption is contradicted.

| variant | dist | T | median_us | hot_sector_share | max_elem_count | est_MiB_compulsory | est_MiB_hot_set | est_MiB_no_reuse | impl_GBps_compulsory | impl_GBps_no_reuse | compulsory_exceeds_peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gather | uniform | 67108864 | 1656.832 | 0.400 | 5 | 132.757 | 132.756 | 160.000 | 84.019 | 101.261 | no |
| gather | zipf1.0 | 67108864 | 731.136 | 0.881 | 226070 | 71.190 | 71.190 | 160.000 | 102.098 | 229.468 | no |
