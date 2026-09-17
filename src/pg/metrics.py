import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from .data import LABELS


def report(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(LABELS)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                        "f1": float(f[i]), "support": int(sup[i])}
            for i in range(len(LABELS))},
        "confusion": confusion_matrix(
            y_true, y_pred, labels=range(len(LABELS))).tolist(),
    }


def fmt(rep, title=""):
    lines = []
    if title:
        lines.append(title)
    lines.append(f"  accuracy {rep['accuracy']:.4f}   macro-F1 {rep['macro_f1']:.4f}")
    for lab, d in rep["per_class"].items():
        lines.append(f"    {lab:<9} P {d['precision']:.3f}  R {d['recall']:.3f} "
                     f" F1 {d['f1']:.3f}  n={d['support']}")
    lines.append("  confusion (rows=true " + ",".join(LABELS) + "):")
    for lab, row in zip(LABELS, rep["confusion"]):
        lines.append(f"    {lab:<9} " + " ".join(f"{v:5d}" for v in row))
    return "\n".join(lines)
