"""Hyperparameter sweep. Selection happens on the SHIFTED dev split only;
the test split is never touched here.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pg import baseline, metrics
from pg.data import L2I, read_tsv
from pg.hashfast import HashFast


def xy(rows):
    return [t for t, _, _ in rows], [L2I[l] for _, l, _ in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="reports/sweep.json")
    a = ap.parse_args()

    Xtr, ytr = xy(read_tsv(f"{a.data}/train.tsv"))
    Xdv, ydv = xy(read_tsv(f"{a.data}/dev.tsv"))
    res = {"baseline": [], "hashfast": []}

    print("--- baseline: regularisation strength C x char n-gram range ---")
    for C in (0.5, 1.0, 4.0, 16.0):
        for lo, hi in ((2, 4), (2, 5), (3, 6)):
            p = baseline.build(C=C)
            p.set_params(feats__char__ngram_range=(lo, hi))
            p.fit(Xtr, ytr)
            f1 = metrics.report(ydv, p.predict(Xdv))["macro_f1"]
            res["baseline"].append({"C": C, "char_ngram": [lo, hi], "dev_macro_f1": f1})
            print(f"  C={C:<5} char={lo}-{hi}  dev macro-F1 {f1:.4f}")

    print("\n--- hashfast: n-gram range x pooling x dim (early-stopped on dev) ---")
    grid = [(lo, hi, pool, dim, buckets)
            for lo, hi in ((2, 4), (2, 5), (3, 5))
            for pool in ("mean", "mean+max")
            for dim in (32, 64)
            for buckets in (100_000,)]
    for lo, hi, pool, dim, buckets in grid:
                lr = 0.3
                m = HashFast(buckets=buckets, dim=dim, seed=0,
                             cmin=lo, cmax=hi, pooling=pool)
                enc_tr, enc_dv = m.encode_all(Xtr), m.encode_all(Xdv)
                h = m.fit(enc_tr, ytr, epochs=12, lr=lr, batch_size=64,
                          val=(enc_dv, ydv), verbose=False,
                          class_weight="balanced")
                f1 = max(e.get("val_macro_f1", 0) for e in h)
                res["hashfast"].append({
                    "buckets": buckets, "dim": dim, "lr": lr,
                    "char_ngram": [lo, hi], "pooling": pool,
                    "best_epoch": getattr(m, "best_epoch", None),
                    "n_params": m.n_params(), "dev_macro_f1": f1})
                print(f"  char={lo}-{hi} {pool:<9} d={dim:<3} "
                      f"ep*={getattr(m,'best_epoch',None):<3} "
                      f"params={m.n_params():>9,}  dev macro-F1 {f1:.4f}")

    for k in res:
        res[k].sort(key=lambda d: -d["dev_macro_f1"])
        print(f"\nbest {k}: {json.dumps(res[k][0])}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
