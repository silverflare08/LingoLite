"""Train both models, run the emoji/punctuation ablation and the robustness
suite, and dump everything to reports/results.json + a readable log.
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pg import baseline, metrics, minimal_pairs, perturb
from pg.data import L2I, LABELS, read_tsv
from pg.hashfast import HashFast

LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


def xy(rows):
    return [t for t, _, _ in rows], [L2I[l] for _, l, _ in rows]


def eval_baseline(pipe, rows):
    X, y = xy(rows)
    return metrics.report(y, pipe.predict(X))


def eval_hashfast(m, rows):
    X, y = xy(rows)
    return metrics.report(y, m.predict(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="reports")
    # defaults below are the configurations selected by scripts/sweep.py on
    # the SHIFTED dev split. Test data was not used for selection.
    ap.add_argument("--buckets", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--cmin", type=int, default=2)
    ap.add_argument("--cmax", type=int, default=4)
    ap.add_argument("--pooling", default="mean+max")
    ap.add_argument("--C", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    tr = read_tsv(os.path.join(a.data, "train.tsv"))
    dv = read_tsv(os.path.join(a.data, "dev.tsv"))
    te = read_tsv(os.path.join(a.data, "test.tsv"))
    log(f"train {len(tr)}  dev {len(dv)}  test {len(te)}")
    log("NOTE: test uses templates and spelling variants never seen in train.")

    results = {"data": {"train": len(tr), "dev": len(dv), "test": len(te)}}

    # ---------------- Experiment 1: baseline ----------------
    log("\n=== E1  baseline: char_wb 2-5 + word 1-2 TF-IDF -> logistic regression ===")
    Xtr, ytr = xy(tr)
    t0 = time.perf_counter()
    pipe = baseline.build(C=a.C)
    pipe.set_params(feats__char__ngram_range=(a.cmin, a.cmax))
    pipe.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    nfeat = len(pipe.named_steps["feats"].get_feature_names_out())
    log(f"  fit {fit_s:.1f}s   features {nfeat}   learned params {baseline.n_params(pipe)}")
    rep_dev = eval_baseline(pipe, dv)
    rep_te = eval_baseline(pipe, te)
    log(metrics.fmt(rep_dev, " dev:"))
    log(metrics.fmt(rep_te, " test (held-out templates):"))
    results["baseline"] = {"dev": rep_dev, "test": rep_te, "fit_seconds": fit_s,
                           "n_features": int(nfeat),
                           "n_params": baseline.n_params(pipe)}
    with open("artifacts/baseline.pkl", "wb") as f:
        pickle.dump(pipe, f)

    # ---------------- Experiment 2: HashFast ----------------
    log(f"\n=== E2  HashFast (buckets={a.buckets}, dim={a.dim}, "
        f"char {a.cmin}-{a.cmax}, pooling={a.pooling}) ===")
    m = HashFast(n_classes=3, buckets=a.buckets, dim=a.dim, seed=0,
                 cmin=a.cmin, cmax=a.cmax, pooling=a.pooling)
    enc_tr = m.encode_all(Xtr)
    Xdv, ydv = xy(dv)
    enc_dv = m.encode_all(Xdv)
    t0 = time.perf_counter()
    hist = m.fit(enc_tr, ytr, epochs=a.epochs, lr=0.3, batch_size=64,
                 val=(enc_dv, ydv), class_weight="balanced")
    fit_s = time.perf_counter() - t0
    log(f"  fit {fit_s:.1f}s   params {m.n_params():,} "
        f"({m.size_mb():.1f} MB fp32)  budget 500,000,000 -> "
        f"{m.n_params()/500e6*100:.2f}% used")
    rep_dev = eval_hashfast(m, dv)
    rep_te = eval_hashfast(m, te)
    log(metrics.fmt(rep_dev, " dev:"))
    log(metrics.fmt(rep_te, " test (held-out templates):"))
    m.save("artifacts/hashfast.npz")
    results["hashfast"] = {"dev": rep_dev, "test": rep_te, "fit_seconds": fit_s,
                           "n_params": m.n_params(), "size_mb": m.size_mb(),
                           "history": hist, "buckets": a.buckets, "dim": a.dim}

    # ---------------- Experiment 3: emoji / punctuation ablation -------------
    log("\n=== E3  ablation: what are emoji and punctuation worth? ===")
    abl = {}
    for name, ke, kp in (("full", True, True),
                         ("no_emoji", False, True),
                         ("no_punct", True, False),
                         ("no_emoji_no_punct", False, False)):
        p = baseline.build(keep_emoji=ke, keep_punct=kp, C=a.C)
        p.set_params(feats__char__ngram_range=(a.cmin, a.cmax))
        p.fit(Xtr, ytr)
        r = eval_baseline(p, te)
        h = HashFast(buckets=a.buckets, dim=a.dim, seed=0, cmin=a.cmin,
                     cmax=a.cmax, pooling=a.pooling,
                     keep_emoji=ke, keep_punct=kp)
        h.fit(h.encode_all(Xtr), ytr, epochs=a.epochs, lr=0.3, batch_size=64,
              verbose=False, class_weight="balanced",
              val=(h.encode_all(Xdv), ydv))
        rh = eval_hashfast(h, te)
        abl[name] = {"baseline_macro_f1": r["macro_f1"],
                     "hashfast_macro_f1": rh["macro_f1"]}
        log(f"  {name:<20} baseline {r['macro_f1']:.4f}   hashfast {rh['macro_f1']:.4f}")
    results["ablation_emoji_punct"] = abl

    # sarcasm slice: positive lexicon + sarcastic emoji, label = negative
    sarc = [r for r in te if r[2].startswith("s")]
    if sarc:
        log(f"\n  sarcasm slice (n={len(sarc)}): positive words + 🙄/😒 -> negative")
        rb, rh = eval_baseline(pipe, sarc), eval_hashfast(m, sarc)
        log(f"    baseline recall-on-slice {rb['accuracy']:.4f} | "
            f"hashfast {rh['accuracy']:.4f}")
        sarc_noemo = perturb.apply("emoji_removed", sarc)
        rb2, rh2 = eval_baseline(pipe, sarc_noemo), eval_hashfast(m, sarc_noemo)
        log(f"    same slice, emojis stripped: baseline {rb2['accuracy']:.4f} | "
            f"hashfast {rh2['accuracy']:.4f}  <- the drop IS the emoji's contribution")
        results["sarcasm_slice"] = {
            "n": len(sarc),
            "with_emoji": {"baseline": rb["accuracy"], "hashfast": rh["accuracy"]},
            "without_emoji": {"baseline": rb2["accuracy"], "hashfast": rh2["accuracy"]}}

    # ---------------- Experiment 3b: minimal-pair probe ----------------
    log("\n=== E3b  minimal-pair probe: does an emoji actually flip the label? ===")
    log("  identical carrier text, positive emoji vs sarcastic emoji.")
    log("  flip_accuracy = fraction of pairs where BOTH members are correct.")
    mp = {}
    for name, fn in (("baseline", lambda t: pipe.predict(t)),
                     ("hashfast", lambda t: m.predict(t))):
        r = minimal_pairs.evaluate(fn, L2I)
        mp[name] = {k: v for k, v in r.items() if k != "detail"}
        log(f"  {name:<10} flip_acc {r['flip_accuracy']:.3f}   "
            f"pos-member {r['positive_member_acc']:.3f}   "
            f"sarc-member {r['sarcastic_member_acc']:.3f}  (n={r['n_pairs']})")
        for d in r["detail"][:4]:
            log(f"      {d['carrier']:<26} +emoji {'OK ' if d['pos_correct'] else 'MISS'}"
                f"   sarc {'OK ' if d['sarc_correct'] else 'MISS'}")
    results["minimal_pairs"] = mp

    # ---------------- Experiment 4: robustness ----------------
    log("\n=== E4  robustness: meaning-preserving perturbations of the test set ===")
    base_b, base_h = results["baseline"]["test"]["macro_f1"], results["hashfast"]["test"]["macro_f1"]
    rob = {"clean": {"baseline": base_b, "hashfast": base_h}}
    log(f"  {'perturbation':<18}{'baseline':>10}{'drop':>8}{'hashfast':>11}{'drop':>8}")
    log(f"  {'clean':<18}{base_b:>10.4f}{'--':>8}{base_h:>11.4f}{'--':>8}")
    for name in perturb.SUITE:
        pr = perturb.apply(name, te)
        fb = eval_baseline(pipe, pr)["macro_f1"]
        fh = eval_hashfast(m, pr)["macro_f1"]
        rob[name] = {"baseline": fb, "hashfast": fh,
                     "baseline_drop": base_b - fb, "hashfast_drop": base_h - fh}
        log(f"  {name:<18}{fb:>10.4f}{base_b-fb:>8.4f}{fh:>11.4f}{base_h-fh:>8.4f}")
    results["robustness"] = rob

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(a.out, "train_eval.log"), "w") as f:
        f.write("\n".join(LOG))
    log(f"\nwrote {a.out}/results.json and {a.out}/train_eval.log")


if __name__ == "__main__":
    main()
