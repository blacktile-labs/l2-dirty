# B17 local rescore (scorer at commit with B17b amendment); per-GPU JSON/TXT + cross-GPU all.json
python3 -I benchmarks/scripts/flush_bias_score.py --meta <gpu>/benchmarks/results/flush_bias_*/run_r*_meta.json --json-out <gpu>.json
python3 -I benchmarks/scripts/flush_bias_score.py --all b17_*.json --json-out all.json
