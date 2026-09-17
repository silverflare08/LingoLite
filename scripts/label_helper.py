"""Hand-labeling helper for real messages.

Workflow:
  1. Paste raw messages, one per line, into data/raw_unlabeled.txt
     (one real support message / review per line; no labels yet).
  2. Run this script. It shows you each message and asks for a label.
  3. Your answers are saved to data/hand_labeled.tsv in the same
     (text, label, source) format everything else in this repo uses.

Re-running is safe: already-labeled lines are skipped.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

RAW = "data/raw_unlabeled.txt"
OUT = "data/hand_labeled.tsv"
KEYS = {"1": "negative", "2": "neutral", "3": "positive", "s": "SKIP"}


def already_done():
    if not os.path.exists(OUT):
        return set()
    with open(OUT, encoding="utf-8") as f:
        return {row["text"] for row in csv.DictReader(f, delimiter="\t")}


def main():
    if not os.path.exists(RAW):
        os.makedirs("data", exist_ok=True)
        with open(RAW, "w", encoding="utf-8") as f:
            f.write("# paste one real message per line below, then re-run this script\n")
        print(f"Created {RAW} -- paste real messages into it (one per line), then re-run.")
        return

    with open(RAW, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    done = already_done()
    todo = [l for l in lines if l not in done]
    if not todo:
        print(f"Nothing new to label. {len(done)} messages already in {OUT}.")
        return

    new_file = not os.path.exists(OUT)
    out = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(out, delimiter="\t")
    if new_file:
        w.writerow(["text", "label", "source"])

    print(f"{len(todo)} messages to label. 1=negative 2=neutral 3=positive s=skip q=quit\n")
    for i, text in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {text}")
        while True:
            ans = input("  label> ").strip().lower()
            if ans == "q":
                out.close()
                print(f"\nstopped early. saved so far in {OUT}")
                return
            if ans in KEYS:
                break
            print("  type 1, 2, 3, s (skip) or q (quit)")
        if KEYS[ans] != "SKIP":
            w.writerow([text, KEYS[ans], "hand_labeled"])
            out.flush()
        print()

    out.close()
    print(f"done. saved to {OUT}")


if __name__ == "__main__":
    main()
