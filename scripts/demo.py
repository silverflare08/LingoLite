"""Interactive demo. Type a message, see the predicted label and how long it
took. This is the single most convincing thing in the repo -- watching it
answer in a fraction of a millisecond is a better argument than any number in
the whitepaper.

Usage:
    python3 scripts/demo.py
    python3 scripts/demo.py --model baseline
    python3 scripts/demo.py "order abhi tk nhi aya 😡"     # one-shot, no prompt
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pg.baseline import BaselineAdapter
from pg.data import LABELS
from pg.hashfast import HashFast

EMOJI = {"negative": "😠", "neutral": "😐", "positive": "😊"}


def load(name):
    if name == "baseline":
        with open("artifacts/baseline.pkl", "rb") as f:
            return BaselineAdapter(pickle.load(f))
    return HashFast.load("artifacts/hashfast.npz")


def classify(model, text):
    t0 = time.perf_counter()
    idx = int(model.predict([text])[0])
    ms = (time.perf_counter() - t0) * 1e3
    return LABELS[idx], ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default=None,
                    help="classify this one message and exit")
    ap.add_argument("--model", choices=["hashfast", "baseline"], default="hashfast")
    a = ap.parse_args()

    if not os.path.exists("artifacts/hashfast.npz"):
        print("No trained model found. Run ./run_all.sh first.")
        return

    model = load(a.model)
    print(f"loaded '{a.model}' model\n")

    if a.text:
        label, ms = classify(model, a.text)
        print(f"{EMOJI[label]} {label:<9} ({ms:.3f} ms)   {a.text}")
        return

    print("Type a Hinglish message and press Enter. Ctrl+C or 'quit' to exit.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in ("quit", "exit"):
            break
        label, ms = classify(model, text)
        print(f"  {EMOJI[label]} {label:<9} ({ms:.3f} ms)\n")


if __name__ == "__main__":
    main()
