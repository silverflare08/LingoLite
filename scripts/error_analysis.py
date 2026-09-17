"""Error analysis for the whitepaper.

Three views:
  1. error rate per held-out template id -> which PHRASINGS fail, not just
     which classes, which is what tells you whether the failure is lexical
     coverage or genuine ambiguity.
  2. the actual misclassified strings, grouped by (true -> predicted).
  3. the highest-weighted features per class in the linear model, which shows
     directly whether emoji tokens and sub-word fragments carry the signal.
"""
import argparse
import collections
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pg.baseline import BaselineAdapter
from pg.data import L2I, LABELS, read_tsv
from pg.hashfast import HashFast

LINES = []


def log(s=""):
    print(s)
    LINES.append(str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--examples", type=int, default=6)
    a = ap.parse_args()

    rows = read_tsv(f"{a.data}/test.tsv")
    X = [t for t, _, _ in rows]
    y = np.array([L2I[l] for _, l, _ in rows])
    tids = [s for _, _, s in rows]

    with open("artifacts/baseline.pkl", "rb") as f:
        pipe = pickle.load(f)
    hf = HashFast.load("artifacts/hashfast.npz")

    preds = {"baseline": np.asarray(pipe.predict(X)),
             "hashfast": np.asarray(hf.predict(X))}
    summary = {}

    for name, p in preds.items():
        log(f"\n================ {name} ================")
        # --- per-template error rate ---
        per = collections.defaultdict(lambda: [0, 0])
        for i, tid in enumerate(tids):
            per[tid][1] += 1
            if p[i] != y[i]:
                per[tid][0] += 1
        log("  error rate by held-out template:")
        tbl = sorted(per.items(), key=lambda kv: -kv[1][0] / kv[1][1])
        for tid, (err, tot) in tbl:
            bar = "#" * int(30 * err / tot)
            log(f"    {tid:<6} {err:>4}/{tot:<5} {err/tot:6.1%}  {bar}")
        summary[name] = {"per_template": {k: {"errors": v[0], "n": v[1],
                                              "rate": v[0] / v[1]}
                                          for k, v in per.items()}}

        # --- example errors by confusion cell ---
        log("  example errors:")
        cells = collections.defaultdict(list)
        for i in range(len(X)):
            if p[i] != y[i]:
                cells[(LABELS[y[i]], LABELS[p[i]])].append((X[i], tids[i]))
        for (t, pr), ex in sorted(cells.items(), key=lambda kv: -len(kv[1])):
            log(f"    true={t} -> pred={pr}  ({len(ex)} cases)")
            for s, tid in ex[:a.examples]:
                log(f"        [{tid}] {s}")

    # --- linear model introspection ---
    log("\n================ top features (linear model) ================")
    feats = np.array(pipe.named_steps["feats"].get_feature_names_out())
    coef = pipe.named_steps["clf"].coef_
    top = {}
    for ci, lab in enumerate(LABELS):
        idx = np.argsort(coef[ci])[::-1][:20]
        top[lab] = [(feats[j], float(coef[ci][j])) for j in idx]
        log(f"  {lab}:")
        log("    " + "  ".join(f"{feats[j]!r}:{coef[ci][j]:.2f}" for j in idx[:12]))
    summary["top_features"] = top

    os.makedirs(a.out, exist_ok=True)
    json.dump(summary, open(f"{a.out}/error_analysis.json", "w"), indent=2)
    open(f"{a.out}/error_analysis.log", "w").write("\n".join(LINES))
    log(f"\nwrote {a.out}/error_analysis.json and .log")


if __name__ == "__main__":
    main()
