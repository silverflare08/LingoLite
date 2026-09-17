"""Minimal-pair probe for pragmatic flips.

The problem statement's sharpest claim is that an emoji can invert meaning:
"bohot badhiya service" is praise, "bohot badhiya service 🙄" is a complaint.

Aggregate accuracy cannot demonstrate that a model handles this, because a
model can score well by memorising the surrounding words. A minimal pair
isolates it: identical carrier text, one emoji changed, expected label flipped.
The metric is FLIP ACCURACY — the fraction of pairs where the model gets BOTH
members right. A model that ignores emojis scores ~0 here no matter how good
its headline F1 is.
"""
CARRIERS = [
    "bohot badhiya service",
    "wah kya service hai",
    "bahut accha kaam",
    "superb delivery",
    "excellent support team",
    "great experience",
    "acha product mila",
    "shandar service bhai",
    "kya baat hai bhai",
    "mast delivery thi",
]

POS_EMO = ["😊", "👍", "❤️", "🙏"]
SARC_EMO = ["🙄", "😒", "🤡", "😑"]


def build():
    """Returns list of (text_pos, 'positive', text_sarc, 'negative', carrier)."""
    pairs = []
    for i, c in enumerate(CARRIERS):
        pairs.append((f"{c} {POS_EMO[i % len(POS_EMO)]}", "positive",
                      f"{c} {SARC_EMO[i % len(SARC_EMO)]}", "negative", c))
    return pairs


def evaluate(predict_fn, l2i):
    """predict_fn(list_of_texts) -> array of class indices."""
    pairs = build()
    texts = [p[0] for p in pairs] + [p[2] for p in pairs]
    pred = list(predict_fn(texts))
    n = len(pairs)
    rows, both, pos_ok, neg_ok = [], 0, 0, 0
    for i, (tp, lp, ts, ls, carrier) in enumerate(pairs):
        ok_p = pred[i] == l2i[lp]
        ok_s = pred[n + i] == l2i[ls]
        pos_ok += ok_p
        neg_ok += ok_s
        both += ok_p and ok_s
        rows.append({"carrier": carrier, "pos_text": tp, "sarc_text": ts,
                     "pos_correct": bool(ok_p), "sarc_correct": bool(ok_s)})
    return {"n_pairs": n,
            "flip_accuracy": both / n,
            "positive_member_acc": pos_ok / n,
            "sarcastic_member_acc": neg_ok / n,
            "detail": rows}
