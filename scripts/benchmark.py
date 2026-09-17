"""Footprint / latency / throughput evidence.

Every number is reported WITH its measurement boundary, hardware, input length
distribution and batch size, because 'single-digit millisecond' is not a
claim you can check otherwise.
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pg import bench
from pg.baseline import BaselineAdapter
from pg.data import read_tsv
from pg.hashfast import HashFast

BUDGET = 500_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="reports/benchmark.json")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()

    texts = [t for t, _, _ in read_tsv(f"{a.data}/test.tsv")]
    hw = bench.hardware()
    stats = bench.input_stats(texts)
    print("hardware:", json.dumps(hw, indent=2))
    print("input:   ", json.dumps(stats))
    print()

    models = {}
    with open("artifacts/baseline.pkl", "rb") as f:
        models["baseline"] = BaselineAdapter(pickle.load(f))
    models["hashfast"] = HashFast.load("artifacts/hashfast.npz")

    out = {"hardware": hw, "input_stats": stats,
           "parameter_budget": BUDGET, "models": {}}

    for name, m in models.items():
        p = m.n_params()
        rec = {
            "n_params": int(p),
            "pct_of_500M_budget": round(p / BUDGET * 100, 4),
            "fp32_size_mb": round(p * 4 / 1e6, 2),
            "int8_size_mb": round(p * 1 / 1e6, 2),
            "latency_end_to_end_bs1": bench.latency_single(m, texts, n=a.n),
            "latency_model_only_bs1": bench.latency_model_only(m, texts, n=a.n),
            "throughput": {},
        }
        for bs in (1, 32, 256, 1024):
            rec["throughput"][str(bs)] = bench.throughput(m, texts, batch_size=bs,
                                                          rounds=10)
        out["models"][name] = rec

        e2e = rec["latency_end_to_end_bs1"]
        print(f"=== {name} ===")
        print(f"  params            {p:>12,}  ({rec['pct_of_500M_budget']}% of 500M budget)")
        print(f"  size fp32 / int8  {rec['fp32_size_mb']:>8.2f} MB / {rec['int8_size_mb']:.2f} MB")
        print(f"  latency bs=1 end-to-end   p50 {e2e['p50_ms']:.3f} ms  "
              f"p90 {e2e['p90_ms']:.3f}  p99 {e2e['p99_ms']:.3f}  max {e2e['max_ms']:.3f}")
        mo = rec["latency_model_only_bs1"]
        print(f"  latency bs=1 model-only   p50 {mo['p50_ms']:.3f} ms  "
              f"p99 {mo['p99_ms']:.3f}")
        for bs, t in rec["throughput"].items():
            print(f"  throughput bs={bs:<5} {t['msgs_per_sec']:>12,.0f} msg/s   "
                  f"{t['ms_per_msg']:.4f} ms/msg amortised")
        print(f"  VERDICT single-digit-ms @ bs=1 p99: "
              f"{'PASS' if e2e['p99_ms'] < 10 else 'FAIL'}")
        print(f"  VERDICT <=500M params: {'PASS' if p <= BUDGET else 'FAIL'}")
        print()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
