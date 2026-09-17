#!/usr/bin/env bash
# One command to reproduce every number in reports/ and the whitepaper.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 0/5 run tests =="
pytest tests/ -q

echo; echo "== 1/5 build dataset (three-way disjoint template split) =="
python3 scripts/make_dataset.py --n-train 12000 --n-dev 1800 --n-test 3000

echo; echo "== 2/5 hyperparameter sweep (selection on shifted dev only) =="
python3 scripts/sweep.py

echo; echo "== 3/5 train + ablations + minimal pairs + robustness =="
python3 scripts/train_eval.py

echo; echo "== 4/5 footprint / latency / throughput =="
python3 scripts/benchmark.py

echo; echo "== 5/5 error analysis =="
python3 scripts/error_analysis.py

echo; echo "done. see reports/"
