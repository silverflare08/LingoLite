import csv
import os
from . import datagen

LABELS = datagen.LABELS
L2I = {l: i for i, l in enumerate(LABELS)}


def write_tsv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["text", "label", "source"])
        for r in rows:
            w.writerow(r)


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        return [(row["text"], row["label"], row.get("source", "")) for row in r]


def load_sentimix(conll_path):
    """Parser for the SemEval-2020 Task 9 (SentiMix Hi-En) CoNLL-style files.

    Format:  meta <id> <label> / <token> <lang_tag> / blank line between posts.
    Returns the same (text, label, source) triple as everything else, so you
    can swap it into any script with --train/--test paths.
    """
    rows, toks, label = [], [], None
    with open(conll_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if toks and label:
                    rows.append((" ".join(toks), label, "sentimix"))
                toks, label = [], None
                continue
            parts = line.split("\t")
            if parts[0] == "meta":
                label = parts[2].strip().lower()
                continue
            toks.append(parts[0])
    if toks and label:
        rows.append((" ".join(toks), label, "sentimix"))
    return [(t, l, s) for t, l, s in rows if l in L2I]
