# gather summary -- run `20261104-071057-rtx4060-probe-g64`

- Device: NVIDIA GeForce RTX 4060 (cc 8.9), driver 13000, runtime 13000, L2 25165824 B
- Correctness failures: 0 (6 checks passed)
- reps 20, warmup 3, steady iters 5, flush_l2 True
- ESTIMATED theoretical DRAM BW 272.0 GB/s; ESTIMATED FP32 peak 15391 GFLOP/s (device attributes)

`useful_GBps` = algorithmic bytes of the task / median time (DERIVED). `isolated` = L2 flushed before each rep; `steady` = back-to-back calls.

| variant | params | mode | reps | median_us | p95_us | cv_pct | useful_GBps |
|---|---|---|---|---|---|---|---|
| gather | T=67108864;M=4194304;dist=uniform | isolated | 20 | 1626.208 | 1627.242 | 0.033 | 30.950 |
| gather | T=67108864;M=4194304;dist=uniform | steady | 20 | 1631.821 | 1632.266 | 0.018 | 30.844 |
| gather_pair64 | T=67108864;M=4194304;dist=uniform | isolated | 20 | 1628.304 | 1630.144 | 0.050 | 41.214 |
| gather_pair64 | T=67108864;M=4194304;dist=uniform | steady | 20 | 1633.584 | 1634.541 | 0.043 | 41.081 |
| gather_pair128 | T=67108864;M=4194304;dist=uniform | isolated | 20 | 2663.936 | 2675.571 | 0.318 | 25.192 |
| gather_pair128 | T=67108864;M=4194304;dist=uniform | steady | 20 | 2663.322 | 2671.267 | 0.200 | 25.197 |
| gather | T=67108864;M=4194304;dist=zipf1.0 | isolated | 20 | 723.968 | 724.992 | 0.080 | 69.522 |
| gather | T=67108864;M=4194304;dist=zipf1.0 | steady | 20 | 723.760 | 724.412 | 0.056 | 69.542 |
| gather_pair64 | T=67108864;M=4194304;dist=zipf1.0 | isolated | 20 | 742.400 | 743.424 | 0.050 | 90.394 |
| gather_pair64 | T=67108864;M=4194304;dist=zipf1.0 | steady | 20 | 727.655 | 743.073 | 1.184 | 92.226 |
| gather_pair128 | T=67108864;M=4194304;dist=zipf1.0 | isolated | 20 | 949.248 | 949.299 | 0.053 | 70.697 |
| gather_pair128 | T=67108864;M=4194304;dist=zipf1.0 | steady | 20 | 942.083 | 942.929 | 0.135 | 71.235 |

### Fetch-granularity probe (isolated mode)

`meas_ratio` = pair-variant time / gather time (MEASURED). `pred_g*` = ratio expected if DRAM moves g-byte chunks per miss with no reuse (ESTIMATE; meaningful for random dists). l2_fetch_granularity reported by driver: 64 B (requested: 64).

| variant | dist | pair_us | gather_us | meas_ratio | pred_g32 | pred_g64 | pred_g128 |
|---|---|---|---|---|---|---|---|
| gather_pair64 | uniform | 1628.304 | 1626.208 | 1.001 | 1.800 | 1.000 | 1.000 |
| gather_pair128 | uniform | 2663.936 | 1626.208 | 1.638 | 1.800 | 1.889 | 1.000 |
| gather_pair64 | zipf1.0 | 742.400 | 723.968 | 1.025 | 1.800 | 1.000 | 1.000 |
| gather_pair128 | zipf1.0 | 949.248 | 723.968 | 1.311 | 1.800 | 1.889 | 1.000 |

### Traffic bounds (isolated mode)

`est_*_MiB` are model bytes; `impl_GBps_*` = estimate / measured median. If the *compulsory* (lower-bound) bytes imply more than the estimated DRAM peak (272 GB/s), the model's 32-B-sector assumption is contradicted.

| variant | dist | T | median_us | hot_sector_share | max_elem_count | est_MiB_compulsory | est_MiB_hot_set | est_MiB_no_reuse | impl_GBps_compulsory | impl_GBps_no_reuse | compulsory_exceeds_peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gather | uniform | 67108864 | 1626.208 | 0.400 | 5 | 132.757 | 132.756 | 160.000 | 85.601 | 103.168 | no |
| gather | zipf1.0 | 67108864 | 723.968 | 0.881 | 226070 | 71.190 | 71.190 | 160.000 | 103.109 | 231.740 | no |
