"""Proves the encoder is task-agnostic: same HashFast architecture, same
normalize()/featurize() pipeline, trained on INTENT labels instead of
sentiment labels. Only n_classes and the training labels change.

This is item 5 from README's "next steps": adding a second downstream task
without touching the model architecture.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pg import datagen, metrics
from pg.data import LABELS as SENTIMENT_LABELS  # noqa: F401 (for contrast)
from pg.hashfast import HashFast


def main():
    tr = datagen.generate_balanced_intent(600, "train", seed=13)
    dv = datagen.generate_balanced_intent(80, "dev", seed=14)
    te = datagen.generate_balanced_intent(150, "test", seed=15)

    labels = sorted({l for _, l, _ in tr + dv + te})
    l2i = {l: i for i, l in enumerate(labels)}
    print(f"intents: {labels}\n")

    xy = lambda rows: ([t for t, _, _ in rows], [l2i[l] for _, l, _ in rows])
    Xtr, ytr = xy(tr)
    Xdv, ydv = xy(dv)
    Xte, yte = xy(te)

    m = HashFast(n_classes=len(labels), buckets=100_000, dim=64,
                cmin=2, cmax=4, pooling="mean+max", seed=0)
    m.fit(m.encode_all(Xtr), ytr, epochs=12, lr=0.3, batch_size=64,
          val=(m.encode_all(Xdv), ydv), verbose=False, class_weight="balanced")

    pred = m.predict(Xte)
    rep = metrics.report(yte, pred)
    print(f"intent classification -- macro-F1 {rep['macro_f1']:.4f}  "
          f"accuracy {rep['accuracy']:.4f}")
    print(f"same architecture as sentiment (pg.hashfast.HashFast), "
          f"same normalize()/featurize() pipeline, {m.n_params():,} params")
    print("\nsample predictions:")
    for t, p, y in list(zip(Xte, pred, yte))[:6]:
        mark = "OK " if p == y else "MISS"
        print(f"  [{mark}] pred={labels[p]:<16} true={labels[y]:<16} {t}")


if __name__ == "__main__":
    main()
