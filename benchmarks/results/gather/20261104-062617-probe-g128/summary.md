# gather summary -- run `20261104-062617-probe-g128`

- Device: NVIDIA GeForce RTX 3060 (cc 8.6), driver 12020, runtime 12020, L2 2359296 B
- Correctness failures: 0 (6 checks passed)
- reps 20, warmup 3, steady iters 5, flush_l2 True
- ESTIMATED theoretical DRAM BW 360.0 GB/s; ESTIMATED FP32 peak 12845 GFLOP/s (device attributes)

`useful_GBps` = algorithmic bytes of the task / median time (DERIVED). `isolated` = L2 flushed before each rep; `steady` = back-to-back calls.

| variant | params | mode | reps | median_us | p95_us | cv_pct | useful_GBps |
|---|---|---|---|---|---|---|---|
| gather | T=16777216;M=4194304;dist=uniform | isolated | 20 | 1657.856 | 1658.242 | 0.035 | 30.359 |
| gather | T=16777216;M=4194304;dist=uniform | steady | 20 | 1657.242 | 1657.831 | 0.015 | 30.371 |
| gather_pair64 | T=16777216;M=4194304;dist=uniform | isolated | 20 | 1659.904 | 1660.979 | 0.043 | 40.429 |
| gather_pair64 | T=16777216;M=4194304;dist=uniform | steady | 20 | 1659.379 | 1660.406 | 0.030 | 40.442 |
| gather_pair128 | T=16777216;M=4194304;dist=uniform | isolated | 20 | 1659.904 | 1660.928 | 0.035 | 40.429 |
| gather_pair128 | T=16777216;M=4194304;dist=uniform | steady | 20 | 1659.494 | 1659.908 | 0.016 | 40.439 |
| gather | T=16777216;M=4194304;dist=zipf1.0 | isolated | 20 | 906.240 | 907.264 | 0.058 | 55.539 |
| gather | T=16777216;M=4194304;dist=zipf1.0 | steady | 20 | 905.421 | 905.660 | 0.022 | 55.589 |
| gather_pair64 | T=16777216;M=4194304;dist=zipf1.0 | isolated | 20 | 904.192 | 905.216 | 0.067 | 74.220 |
| gather_pair64 | T=16777216;M=4194304;dist=zipf1.0 | steady | 20 | 903.639 | 904.008 | 0.025 | 74.265 |
| gather_pair128 | T=16777216;M=4194304;dist=zipf1.0 | isolated | 20 | 904.192 | 905.226 | 0.054 | 74.220 |
| gather_pair128 | T=16777216;M=4194304;dist=zipf1.0 | steady | 20 | 903.578 | 903.987 | 0.026 | 74.270 |

### Fetch-granularity probe (isolated mode)

`meas_ratio` = pair-variant time / gather time (MEASURED). `pred_g*` = ratio expected if DRAM moves g-byte chunks per miss with no reuse (ESTIMATE; meaningful for random dists). l2_fetch_granularity reported by driver: 128 B (requested: 128).

| variant | dist | pair_us | gather_us | meas_ratio | pred_g32 | pred_g64 | pred_g128 |
|---|---|---|---|---|---|---|---|
| gather_pair64 | uniform | 1659.904 | 1657.856 | 1.001 | 1.800 | 1.000 | 1.000 |
| gather_pair128 | uniform | 1659.904 | 1657.856 | 1.001 | 1.800 | 1.889 | 1.000 |
| gather_pair64 | zipf1.0 | 904.192 | 906.240 | 0.998 | 1.800 | 1.000 | 1.000 |
| gather_pair128 | zipf1.0 | 904.192 | 906.240 | 0.998 | 1.800 | 1.889 | 1.000 |

### Traffic bounds (isolated mode)

`est_*_MiB` are model bytes; `impl_GBps_*` = estimate / measured median. If the *compulsory* (lower-bound) bytes imply more than the estimated DRAM peak (360 GB/s), the model's 32-B-sector assumption is contradicted.

| variant | dist | T | median_us | hot_sector_share | max_elem_count | est_MiB_compulsory | est_MiB_hot_set | est_MiB_no_reuse | impl_GBps_compulsory | impl_GBps_no_reuse | compulsory_exceeds_peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gather | uniform | 16777216 | 1657.856 | 0.099 | 6 | 87.358 | 149.560 | 160.000 | 55.253 | 101.198 | no |
| gather | zipf1.0 | 16777216 | 906.240 | 0.706 | 244059 | 59.477 | 71.851 | 160.000 | 68.819 | 185.130 | no |
