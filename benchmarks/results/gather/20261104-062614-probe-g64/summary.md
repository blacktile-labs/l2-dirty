# gather summary -- run `20261104-062614-probe-g64`

- Device: NVIDIA GeForce RTX 3060 (cc 8.6), driver 12020, runtime 12020, L2 2359296 B
- Correctness failures: 0 (6 checks passed)
- reps 20, warmup 3, steady iters 5, flush_l2 True
- ESTIMATED theoretical DRAM BW 360.0 GB/s; ESTIMATED FP32 peak 12845 GFLOP/s (device attributes)

`useful_GBps` = algorithmic bytes of the task / median time (DERIVED). `isolated` = L2 flushed before each rep; `steady` = back-to-back calls.

| variant | params | mode | reps | median_us | p95_us | cv_pct | useful_GBps |
|---|---|---|---|---|---|---|---|
| gather | T=16777216;M=4194304;dist=uniform | isolated | 20 | 1257.472 | 1257.523 | 0.048 | 40.026 |
| gather | T=16777216;M=4194304;dist=uniform | steady | 20 | 1257.091 | 1257.677 | 0.025 | 40.038 |
| gather_pair64 | T=16777216;M=4194304;dist=uniform | isolated | 20 | 1259.520 | 1260.544 | 0.068 | 53.281 |
| gather_pair64 | T=16777216;M=4194304;dist=uniform | steady | 20 | 1258.906 | 1259.524 | 0.022 | 53.307 |
| gather_pair128 | T=16777216;M=4194304;dist=uniform | isolated | 20 | 2008.064 | 2016.410 | 0.283 | 33.420 |
| gather_pair128 | T=16777216;M=4194304;dist=uniform | steady | 20 | 2008.640 | 2013.444 | 0.156 | 33.410 |
| gather | T=16777216;M=4194304;dist=zipf1.0 | isolated | 20 | 686.976 | 687.104 | 0.075 | 73.266 |
| gather | T=16777216;M=4194304;dist=zipf1.0 | steady | 20 | 686.154 | 686.615 | 0.040 | 73.353 |
| gather_pair64 | T=16777216;M=4194304;dist=zipf1.0 | isolated | 20 | 686.080 | 686.374 | 0.083 | 97.815 |
| gather_pair64 | T=16777216;M=4194304;dist=zipf1.0 | steady | 20 | 685.034 | 685.261 | 0.028 | 97.964 |
| gather_pair128 | T=16777216;M=4194304;dist=zipf1.0 | isolated | 20 | 908.288 | 908.582 | 0.069 | 73.885 |
| gather_pair128 | T=16777216;M=4194304;dist=zipf1.0 | steady | 20 | 906.992 | 907.069 | 0.019 | 73.991 |

### Fetch-granularity probe (isolated mode)

`meas_ratio` = pair-variant time / gather time (MEASURED). `pred_g*` = ratio expected if DRAM moves g-byte chunks per miss with no reuse (ESTIMATE; meaningful for random dists). l2_fetch_granularity reported by driver: 64 B (requested: 64).

| variant | dist | pair_us | gather_us | meas_ratio | pred_g32 | pred_g64 | pred_g128 |
|---|---|---|---|---|---|---|---|
| gather_pair64 | uniform | 1259.520 | 1257.472 | 1.002 | 1.800 | 1.000 | 1.000 |
| gather_pair128 | uniform | 2008.064 | 1257.472 | 1.597 | 1.800 | 1.889 | 1.000 |
| gather_pair64 | zipf1.0 | 686.080 | 686.976 | 0.999 | 1.800 | 1.000 | 1.000 |
| gather_pair128 | zipf1.0 | 908.288 | 686.976 | 1.322 | 1.800 | 1.889 | 1.000 |

### Traffic bounds (isolated mode)

`est_*_MiB` are model bytes; `impl_GBps_*` = estimate / measured median. If the *compulsory* (lower-bound) bytes imply more than the estimated DRAM peak (360 GB/s), the model's 32-B-sector assumption is contradicted.

| variant | dist | T | median_us | hot_sector_share | max_elem_count | est_MiB_compulsory | est_MiB_hot_set | est_MiB_no_reuse | impl_GBps_compulsory | impl_GBps_no_reuse | compulsory_exceeds_peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gather | uniform | 16777216 | 1257.472 | 0.099 | 6 | 87.358 | 149.560 | 160.000 | 72.845 | 133.420 | no |
| gather | zipf1.0 | 16777216 | 686.976 | 0.706 | 244059 | 59.477 | 71.851 | 160.000 | 90.784 | 244.218 | no |
