"""Build train/dev/test TSVs.

Default: synthetic Hinglish support corpus with HELD-OUT templates in test.
With --sentimix <conll> it will instead/also pull real SentiMix data.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pg import data, datagen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--sentimix", default=None,
                    help="path to SemEval-2020 Task 9 CoNLL file")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    # THREE-WAY TEMPLATE SPLIT.
    # train / dev / test use disjoint template pools, so dev measures
    # generalisation to unseen phrasings and is usable for model selection.
    # An in-distribution dev split saturates at 100% and selects nothing.
    tr = datagen.generate_balanced(max(1, a.n_train // 3), "train", seed=a.seed)
    dv = datagen.generate_balanced(max(1, a.n_dev // 3), "dev", seed=a.seed + 1)
    te = datagen.generate_balanced(max(1, a.n_test // 3), "test", seed=a.seed + 2)

    # hard guarantee: no surface string shared across splits
    tr_set = {t for t, _, _ in tr}
    dv = [r for r in dv if r[0] not in tr_set]
    te = [r for r in te if r[0] not in tr_set]

    if a.sentimix:
        extra = data.load_sentimix(a.sentimix)
        cut = int(len(extra) * 0.8)
        tr += [(t, l, s) for t, l, s in extra[:cut]]
        te += [(t, l, s) for t, l, s in extra[cut:]]
        print(f"merged {len(extra)} SentiMix rows")

    for name, rows in (("train", tr), ("dev", dv), ("test", te)):
        p = os.path.join(a.out, f"{name}.tsv")
        data.write_tsv(rows, p)
        counts = {}
        for _, l, _ in rows:
            counts[l] = counts.get(l, 0) + 1
        print(f"{p:<20} {len(rows):>6} rows  {counts}")

    print("\ntemplate pools -> train: rest | dev:",
          sorted(datagen.HELD_DEV), "| test:", sorted(datagen.HELD_OUT))


if __name__ == "__main__":
    main()
