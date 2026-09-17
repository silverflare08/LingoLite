# LingoLite — Hinglish code-mixed sentiment, small and fast

Reference implementation for the Inter IIT Bootcamp 2026 problem
*The Polyglot's Shorthand* (450 pts). Declared scope: **Hindi–English Romanized
code-mixed sentiment analysis**, 3-way.

```bash
pip install -r requirements.txt
./run_all.sh
```

Runs in a few minutes on one CPU core, no GPU, no downloads.

## Results at a glance

| model | params | % of 500M budget | test macro-F1 | e2e latency p50 (bs=1) | throughput |
|---|---|---|---|---|---|
| TF-IDF + LogReg | 7,980 | 0.0016% | **0.956** | 1.105 ms | 13,253 msg/s (bs 1024) |
| HashFast (hashed n-gram bag) | 6,400,387 | 1.28% | 0.903 | **0.114 ms** | 7,669 msg/s (bs 1) |

Both **PASS** ≤500M parameters and single-digit-millisecond latency, on a single
core with no accelerator. Full measurement conditions in `whitepaper.md` §7.

Three findings worth your attention before you read the code:

1. **char 2–4 grams, not 3–5.** Hinglish tokens are short; wider n-grams
   memorise spellings instead of generalising across them (dev macro-F1
   0.967 vs 0.599).
2. **Mean pooling destroys the emoji signal.** An emoji is ~2 of ~100 n-grams.
   Adding a max-pooled channel moved dev macro-F1 0.798 → 0.967.
3. **Aggregate F1 cannot tell you whether your model reads emoji.** A
   minimal-pair probe showed both models at 0.00–0.10 flip accuracy while
   reporting 0.83–0.95 macro-F1. The cause was a corpus defect. See
   `whitepaper.md` §6.2 — this is the most useful thing in the repo.

## Project structure

```
polyglot-shorthand/
├── README.md                  overview + how to run
├── whitepaper.md               full write-up and results
├── LICENSE
├── .gitignore
├── requirements.txt
├── pytest.ini
├── run_all.sh                  runs everything, start to finish
│
├── src/pg/                     the library code
├── scripts/                    scripts you run from the terminal
│   ├── demo.py                     try a message, see label + timing
│   ├── label_helper.py             hand-label your own real messages
│   ├── intent_demo.py              same model, trained on intent instead of sentiment
│   └── ...                         (dataset build, sweep, train, benchmark, errors)
├── tests/                      automated checks
├── configs/                    settings / rules, kept out of the code
├── data/                       generated dataset (not committed to git)
├── artifacts/                  trained models (not committed to git)
└── reports/                    results and logs
```

```bash
pip install -r requirements.txt
./run_all.sh
```

### Try it yourself

```bash
python3 scripts/demo.py                                    # interactive
python3 scripts/demo.py "order abhi tk nhi aya 😡"          # one message
python3 scripts/label_helper.py                             # hand-label real messages
python3 scripts/intent_demo.py                              # same model, different task
```

## Using real data

The generated corpus exists so the pipeline runs end to end today. It does
**not** measure real-world accuracy, and the whitepaper says so in §2.

```bash
python3 scripts/make_dataset.py --sentimix path/to/sentimix_train_conll.txt
./run_all.sh
```

`pg.data.load_sentimix` parses the official CoNLL format. For the submission you
want SentiMix for training plus a hand-labelled support-message test set,
since SentiMix is Twitter-domain.

## Extending to other tasks

The problem statement allows one or several downstream tasks. The encoder is
task-agnostic: intent classification is the same `HashFast` with a different
number of output classes and an intent-labelled corpus. Summarisation and QA
are *generative* and do not fit a bag-of-n-grams encoder — if you take those on,
they need a separate seq2seq model, and the latency budget will be the binding
constraint, not the parameter budget.
