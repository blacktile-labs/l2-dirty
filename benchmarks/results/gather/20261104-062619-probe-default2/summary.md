# gather summary -- run `20261104-062619-probe-default2`

- Device: NVIDIA GeForce RTX 3060 (cc 8.6), driver 12020, runtime 12020, L2 2359296 B
- Correctness failures: 0 (6 checks passed)
- reps 20, warmup 3, steady iters 5, flush_l2 True
- ESTIMATED theoretical DRAM BW 360.0 GB/s; ESTIMATED FP32 peak 12845 GFLOP/s (device attributes)

`useful_GBps` = algorithmic bytes of the task / median time (DERIVED). `isolated` = L2 flushed before each rep; `steady` = back-to-back calls.

| variant | params | mode | reps | median_us | p95_us | cv_pct | useful_GBps |
|---|---|---|---|---|---|---|---|
| gather | T=16777216;M=4194304;dist=uniform | isolated | 20 | 1239.040 | 1240.064 | 0.046 | 40.621 |
| gather | T=16777216;M=4194304;dist=uniform | steady | 20 | 1238.640 | 1239.043 | 0.024 | 40.635 |
| gather_pair64 | T=16777216;M=4194304;dist=uniform | isolated | 20 | 1975.296 | 1985.741 | 0.342 | 33.974 |
| gather_pair64 | T=16777216;M=4194304;dist=uniform | steady | 20 | 1975.769 | 1981.057 | 0.134 | 33.966 |
| gather_pair128 | T=16777216;M=4194304;dist=uniform | isolated | 20 | 1977.856 | 1985.741 | 0.481 | 33.930 |
| gather_pair128 | T=16777216;M=4194304;dist=uniform | steady | 20 | 1976.116 | 1980.333 | 0.193 | 33.960 |
| gather | T=16777216;M=4194304;dist=zipf1.0 | isolated | 20 | 675.840 | 676.134 | 0.087 | 74.473 |
| gather | T=16777216;M=4194304;dist=zipf1.0 | steady | 20 | 674.983 | 675.430 | 0.038 | 74.567 |
| gather_pair64 | T=16777216;M=4194304;dist=zipf1.0 | isolated | 20 | 759.296 | 762.880 | 0.286 | 88.383 |
| gather_pair64 | T=16777216;M=4194304;dist=zipf1.0 | steady | 20 | 759.056 | 760.209 | 0.091 | 88.411 |
| gather_pair128 | T=16777216;M=4194304;dist=zipf1.0 | isolated | 20 | 758.720 | 762.880 | 0.344 | 88.450 |
| gather_pair128 | T=16777216;M=4194304;dist=zipf1.0 | steady | 20 | 759.034 | 760.419 | 0.114 | 88.414 |

### Fetch-granularity probe (isolated mode)

`meas_ratio` = pair-variant time / gather time (MEASURED). `pred_g*` = ratio expected if DRAM moves g-byte chunks per miss with no reuse (ESTIMATE; meaningful for random dists). l2_fetch_granularity reported by driver: 64 B (requested: -1).

| variant | dist | pair_us | gather_us | meas_ratio | pred_g32 | pred_g64 | pred_g128 |
|---|---|---|---|---|---|---|---|
| gather_pair64 | uniform | 1975.296 | 1239.040 | 1.594 | 1.800 | 1.000 | 1.000 |
| gather_pair128 | uniform | 1977.856 | 1239.040 | 1.596 | 1.800 | 1.889 | 1.000 |
| gather_pair64 | zipf1.0 | 759.296 | 675.840 | 1.123 | 1.800 | 1.000 | 1.000 |
| gather_pair128 | zipf1.0 | 758.720 | 675.840 | 1.123 | 1.800 | 1.889 | 1.000 |

### Traffic bounds (isolated mode)

`est_*_MiB` are model bytes; `impl_GBps_*` = estimate / measured median. If the *compulsory* (lower-bound) bytes imply more than the estimated DRAM peak (360 GB/s), the model's 32-B-sector assumption is contradicted.

| variant | dist | T | median_us | hot_sector_share | max_elem_count | est_MiB_compulsory | est_MiB_hot_set | est_MiB_no_reuse | impl_GBps_compulsory | impl_GBps_no_reuse | compulsory_exceeds_peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gather | uniform | 16777216 | 1239.040 | 0.099 | 6 | 87.358 | 149.560 | 160.000 | 73.929 | 135.405 | no |
| gather | zipf1.0 | 16777216 | 675.840 | 0.706 | 244059 | 59.477 | 71.851 | 160.000 | 92.279 | 248.242 | no |
