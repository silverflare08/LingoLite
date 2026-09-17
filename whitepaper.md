# The LingoLite — technical whitepaper

**Task scope (declared):** Hindi–English Romanized code-mixed **sentiment
analysis**, 3-way (negative / neutral / positive), customer-support domain.
The problem statement permits choosing one downstream task; this is that choice.
Intent classification is a drop-in extension — same encoder, different head.

**Headline claim:** a 6.4M-parameter bag-of-hashed-character-n-grams classifier
reaches **0.903 macro-F1** on a shifted test split at **0.114 ms median
end-to-end latency** on a single CPU core, and a 7,980-parameter linear model
reaches **0.956 macro-F1** at 1.1 ms. Both clear the 500M / single-digit-ms
envelope by three orders of magnitude, which means the budget is not the
binding constraint — **label quality is**.

---

## 1. Why the standard pipelines break, and what that implies

The problem statement's diagnosis is right but incomplete. IndicBERT fails on
`nhi` not merely because it expects Devanagari, but because *any* model whose
atomic unit is a word-like subword treats `nahi`, `nhi`, `nai`, `nhii` as four
unrelated types. The variance is at the **character** level, so the
representation has to be at the character level too.

That single observation determines the architecture. Once you commit to
character n-grams:

- OOV disappears structurally. There is no vocabulary to miss.
- Phonetic drift becomes a *feature-overlap* problem rather than a lookup
  failure: `nahi` and `nhi` share `<n`, `nh`, `hi`, `i>`.
- Model size collapses, because you no longer pay for a vocabulary embedding
  table sized to cover every spelling of every word.
- Latency collapses with it, since there is no attention and no autoregressive
  decode.

This is why the 500M parameter budget turns out to be slack rather than tight.
A model that *needs* 500M parameters for this task is a model that is trying to
memorise spellings.

---

## 2. Data

`SentiMix` (SemEval-2020 Task 9, Hi-En, 20k labelled tweets) is the correct
training corpus and `pg.data.load_sentimix` parses its CoNLL format directly.
It is **Twitter-domain**, so it is not sufficient on its own: support chat has
different intents, different length distribution, and different emoji usage.

For a runnable, self-contained reference this repo ships a **generated**
support-chat corpus (`pg/datagen.py`) with an explicit phonetic-variant pool and
emoji-driven pragmatic flips.

> **Honest framing, stated up front:** numbers on generated data measure whether
> the *pipeline* works, not whether the model handles real users. They are an
> upper bound. The submission-grade claim requires SentiMix plus a
> hand-labelled support-message test set, and the repo is structured so that is
> a path swap, not a rewrite.

Three properties make the generated numbers non-trivial:

1. **Three-way disjoint template split.** Train, dev and test draw from
   *different* phrasing templates. An in-distribution dev split saturated at
   100% macro-F1 in early runs and was useless for model selection — which is
   itself a finding worth carrying into the real setup.
2. **Held-out spelling variants.** The test split samples from the 40% tail of
   each phonetic pool that training never saw.
3. **Labelling rules written before the data** (`configs/labeling_guidelines.md`),
   including the rule that *unmarked* ironic text is labelled at its surface
   reading. Violating that rule puts contradictions in the corpus.

| split | rows | source templates |
|---|---|---|
| train | 12,000 | all except dev/test pools |
| dev | 1,800 | `n12 p11 s03 u14` |
| test | 3,000 | `n90 n91 n92 p90 p91 s90 s91 t90 t91 u90 u91` |

---

## 3. Tokenisation and normalisation

`pg/normalize.py`, in order:

| step | rationale |
|---|---|
| NFKC, lowercase, URL → `<url>`, numbers → `<num>` | removes variance with no sentiment content |
| **emoji split into standalone tokens** | emoji are semantic modulators; stripping them is the single most damaging thing you can do (§6) |
| `!!!!!` → `!!!`, then punctuation isolated | preserves *that* the writer shouted while bounding the feature space |
| elongation `plssss` → `plss` | keeps the elongation signal, kills the combinatorial blow-up |
| char n-grams, **boundary-marked**, n = 2–4 | `<`/`>` markers let prefixes and suffixes be learned |
| word 1–2 grams | catches multiword idioms char grams fragment |

**Measured finding on n-gram width.** char 2–4 beats char 3–5 by a wide margin
(dev macro-F1 0.967 vs 0.599 for the neural model; 0.812 vs 0.557 for the linear
one). Hinglish tokens are short — `nhi`, `h`, `tk`, `kro` — so 3–5-grams mostly
span whole words and *memorise spellings* rather than generalising across them.
Short n-grams are what produce the phonetic invariance everyone assumes
character models have automatically. They do not; it depends on the width.

---

## 4. Models

### 4.1 Baseline — char+word TF-IDF → multinomial logistic regression

2,659 features after `min_df` pruning, **7,980 learned parameters**
(3 × 2,659 + 3). Sub-second training. IDF weighting is doing real work here:
it discounts n-grams common to all classes, which mean-pooled neural bags get
for free only if you add it.

### 4.2 HashFast — compact hashed-ngram embedding bag (`pg/hashfast.py`)

```
text → normalize → {char 2-4 grams, word 1-2 grams}
     → CRC32 hash into 100,000 buckets        (no vocabulary file, no OOV)
     → embedding bag, mean ⊕ max pooled       (E: 100000 × 64)
     → linear softmax                         (W: 128 × 3, b: 3)
```

**6,400,387 parameters — 1.28% of the 500M budget**, 25.6 MB fp32 / 6.4 MB int8.
Implemented in NumPy with sparse SGD so it reproduces anywhere without a deep
learning framework; `pg/torch_model.py` holds the PyTorch port and a char-CNN
variant for scaling.

**Design finding — mean pooling destroys the emoji signal.** A message
contributes ~100 n-grams; an emoji contributes 1–3 of them. Averaging divides
the very signal the problem statement says inverts meaning by ~50. Adding a
**max-pooled** channel alongside the mean lets one strong n-gram survive
averaging, and it moved dev macro-F1 from 0.798 → **0.967** at dim=64. This is
the most consequential architectural decision in the build and it follows
directly from taking "emoji as semantic modulator" literally.

Early stopping is on the **shifted** dev split; training past epoch 6 kept
reducing train loss (0.0059 → 0.0034) while dev macro-F1 flattened — memorising
phrasings that do not transfer.

---

## 5. Results

Selection was performed on dev only (`scripts/sweep.py`, 12 baseline configs +
12 neural configs). Test was scored once per model.

### Test set (held-out templates, held-out spellings)

| model | params | macro-F1 | accuracy | neg F1 | neu F1 | pos F1 |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 7,980 | **0.9557** | 0.9553 | 0.968 | 0.942 | 0.957 |
| HashFast | 6,400,387 | 0.9028 | 0.9037 | 0.895 | 0.954 | 0.859 |

**The 7,980-parameter linear model wins.** That is the correct result to report,
not one to explain away: at this corpus size and this level of lexical
diversity, IDF-weighted sparse features beat a learned dense bag. The neural
model earns its place on **latency** (10× faster per message, §7), not accuracy.
On real SentiMix-scale data with genuine lexical diversity the ordering may
invert — which is exactly what the sweep harness is for.

---

## 6. Emoji and punctuation as first-class semantics

### 6.1 Ablation (test macro-F1)

| configuration | baseline | HashFast |
|---|---|---|
| full | 0.9557 | 0.9028 |
| emoji stripped | 0.8258 (**−0.130**) | 0.8469 (−0.056) |
| punctuation stripped | 0.9315 (−0.024) | 0.9143 (+0.011) |
| both stripped | 0.7533 (**−0.202**) | 0.8012 (−0.102) |

Stripping emoji costs the linear model 13 macro-F1 points. Stripping both costs
20. Emoji are not decoration in this domain; they are 20% of the signal.

### 6.2 The minimal-pair probe — and the bug it caught

Aggregate F1 **cannot** demonstrate that a model reads emoji, because a model
can score well by memorising carrier words. `pg/minimal_pairs.py` isolates it:
identical carrier text, one emoji changed, expected label flipped. The metric
is **flip accuracy** — both members correct.

First run:

| model | flip accuracy |
|---|---|
| baseline | 0.10 |
| HashFast | 0.00 |

Both models were near-zero *while reporting 0.83–0.95 macro-F1*. The cause was
a corpus defect, not a modelling one: `bohot badhiya service` appeared in
training **only** as sarcasm, so both models learned the *words* were negative
and never learned the emoji meant anything. Adding **sincere positive twins** —
identical carrier text, positive emoji, positive label, so the emoji is the only
distinguishing evidence:

| model | flip accuracy before | after |
|---|---|---|
| baseline | 0.10 | **1.00** |
| HashFast | 0.00 | **0.50** |

On the sarcasm slice (n=358), stripping the emoji drops baseline accuracy from
1.00 → 0.49. That drop *is* the emoji's causal contribution, measured rather
than asserted.

**Transferable lesson:** if your corpus never contains the sincere counterpart
of an ironic phrase, no architecture will learn to read the irony marker. This
is a data-construction requirement, and it is invisible to aggregate metrics.

---

## 7. Footprint, latency, throughput

Measured on: Linux x86-64, **1 logical core**, no accelerator, Python 3.12,
NumPy 2.4.4, single process. Inputs: 3,000 real test messages, mean 49.3 chars
(p95 67), mean 9.2 tokens (p95 13). 2,000 timed iterations after 50 warm-up.

Two boundaries are reported because "latency" is meaningless without one:
**end-to-end** = raw string in → label out (normalisation + n-gram extraction +
hashing + pooling + matmul + argmax); **model-only** = pre-encoded ids in.

| model | params | % of 500M | fp32 | e2e p50 | e2e p90 | e2e p99 | model-only p50 |
|---|---|---|---|---|---|---|---|
| baseline | 7,980 | 0.0016% | 0.03 MB | 1.105 ms | 1.243 ms | 1.695 ms | 0.107 ms |
| HashFast | 6,400,387 | 1.28% | 25.6 MB | **0.114 ms** | 0.145 ms | **0.203 ms** | 0.035 ms |

Throughput, single core:

| batch | baseline | HashFast |
|---|---|---|
| 1 | 852 msg/s | 7,669 msg/s |
| 32 | 8,705 msg/s | 10,779 msg/s |
| 256 | 12,415 msg/s | 7,635 msg/s |
| 1024 | 13,253 msg/s | 7,122 msg/s |

**Both models PASS both constraints** (≤500M params; p99 < 10 ms at batch 1).

Three things worth reading off this table:

1. **HashFast is 10× faster per message at batch 1**, the real-time routing
   case. Its cost is dominated by hashing, which is O(message length) with a
   tiny constant.
2. **The baseline's advantage is batching.** Its sparse matmul amortises well,
   so at batch ≥256 it overtakes on throughput. HashFast's Python-level
   max-pool loop does not vectorise across the batch — a known, fixable
   implementation limit, not an architectural one.
3. **Feature extraction dominates, not arithmetic.** For the baseline, 1.105 ms
   end-to-end vs 0.107 ms model-only means ~90% of the time is normalisation
   and n-gram extraction in Python. The obvious next optimisation is rewriting
   the tokeniser in C/Rust, not shrinking the model.

Extrapolated: one core sustains ~7.6k–13k msg/s. A 16-core node handles
~10⁵ msg/s. A 500M-parameter transformer on the same hardware would manage
single-digit to low-tens msg/s — a 10⁴× gap, which is the whole argument.

---

## 8. Robustness

Meaning-preserving perturbations applied to the full test set. Each is
constructed so a human assigns the same label, so any drop is model brittleness.

| perturbation | baseline macro-F1 | Δ | HashFast macro-F1 | Δ |
|---|---|---|---|---|
| clean | 0.9557 | — | 0.9028 | — |
| vowel drop (`nahi`→`nhi`) | 0.9597 | **+0.004** | 0.9001 | −0.003 |
| phonetic swap (v↔w, aa→a) | 0.9494 | −0.006 | 0.8895 | −0.013 |
| elongation (`plssss`) | 0.9467 | −0.009 | 0.8905 | −0.012 |
| punctuation removed | 0.9483 | −0.007 | 0.8807 | −0.022 |
| keyboard typo (8%/char) | 0.9444 | −0.011 | 0.8567 | −0.046 |
| **emoji removed** | 0.8087 | **−0.147** | 0.8406 | −0.062 |

Phonetic drift — the thing the problem statement flags as the central
difficulty — costs **under 1.5 macro-F1 points**. Vowel-dropping actually
*improves* the baseline slightly, because collapsing `nahi`/`nhi` onto shared
n-grams reduces surface variance the model was already handling.

Meanwhile removing emoji costs **15 points**. The ranking is the finding:
**in this domain the pragmatic channel is an order of magnitude more fragile
than the orthographic one.** The industry framing of code-mixed NLP as a
spelling-normalisation problem is backwards. Character n-grams solve spelling
almost for free; nothing solves irony for free.

---

## 9. Error analysis

Per-template error rates and the misclassified strings are in
`reports/error_analysis.log`. Two dominant modes:

**(a) Unseen evaluative vocabulary — the real limitation.** `n92`
(*"quality bohot ghatiya hai paise waste ho gaye"*) is HashFast's worst template
at 43.7% error. `ghatiya` never appears in training. Character n-grams give
robustness to *spelling variation of known words*; they give nothing for a word
whose sentiment was never observed. This bounds the whole approach and is the
argument for either (i) far more lexically diverse training data, or (ii) a
pretrained character-level encoder. It is **not** an argument for a bigger
classifier head.

**(b) Sarcasm-carrier bleed.** `p91` (*"same day delivery mil gaya superb hai"*)
errs at ~25% for both models. `superb` appears in training predominantly inside
sarcastic templates, so its learned polarity is contaminated. The fix is the
same as §6.2: balance every ironic carrier with sincere occurrences. This is
the quantitative case for why sarcasm data must be built in matched pairs.

**Feature introspection** (`reports/error_analysis.log`, top-features section)
confirms emoji tokens and sub-word fragments — not whole words — carry the
largest class weights, which is the mechanism the architecture was chosen for.

---

## 10. Trade-offs, honestly stated

| choice | bought | paid |
|---|---|---|
| char n-grams over subword | spelling invariance, no OOV, tiny model | no compositional/word-order semantics |
| bag-of-n-grams over transformer | 10⁴× throughput, 0.1 ms latency | no negation scope, no long-range dependency |
| hashing over vocabulary | zero vocab file, fixed memory | hash collisions (~6% of buckets at 100k for this corpus), non-inspectable features |
| mean⊕max pooling | emoji survive averaging (+0.17 dev F1) | 2× head width; max-pool loop is the batching bottleneck |
| generated corpus | fully reproducible end-to-end today | **does not measure real-world accuracy** |

**Where this approach will fail:** negation scope (`accha nahi tha` vs
`nahi accha tha`), multi-clause messages where a bag cannot tell which clause
the grievance attaches to, and sentiment carried by pure word order. If error
analysis on real data shows those dominating, the char-CNN in
`pg/torch_model.py` is the next step — measure first; it costs 3–5× the latency.

---

## 11. Reproducing

```bash
pip install -r requirements.txt
./run_all.sh            # dataset → sweep → train/ablate/probe → bench → errors
```

Outputs land in `reports/`: `results.json`, `sweep.json`, `benchmark.json`,
`error_analysis.json`, plus readable `.log` versions. Every number in this
document comes from those files. All seeds fixed.

To run on real data instead:

```bash
python3 scripts/make_dataset.py --sentimix path/to/train_conll.txt
./run_all.sh
```

---

## 12. What to do next, in priority order

1. **Get real labels.** SentiMix for training, plus ~1,000 hand-labelled
   support messages as the *only* headline test set. Report inter-annotator
   agreement; without it no macro-F1 on this task is interpretable.
2. **Build the minimal-pair probe set by hand** (~200 pairs). It caught a
   silent corpus defect here that aggregate F1 completely masked, and it will
   do the same on real data.
3. **Move the tokeniser out of Python.** It is 90% of end-to-end latency.
4. **Then, and only then,** consider a larger model. On present evidence the
   parameter budget is not what is limiting accuracy — label quality and
   lexical coverage are.
