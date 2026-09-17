"""Synthetic Hinglish customer-support corpus.

WHY THIS EXISTS
---------------
SentiMix (Hi-En) is the right training corpus, but it is (a) Twitter-domain and
(b) not redistributable here. This generator gives you a runnable, reproducible
stand-in so the whole pipeline executes end to end today, and a clean slot to
drop real data into tomorrow (see pg.data.load_sentimix).

HONESTY CONSTRAINT
------------------
Scores on generated data are a *sanity check on the pipeline*, not evidence of
real-world accuracy. To make them mean anything at all, the test split uses
TEMPLATES THE MODEL NEVER SAW (held-out template ids) and a disjoint slice of
the spelling-variant pool. Do not quote these numbers as your final result.
"""
import random

LABELS = ["negative", "neutral", "positive"]

# --- phonetic variant pool: the core of the "spelling drift" problem ---------
VAR = {
    "nahi":   ["nahi", "nhi", "nai", "nahii", "nhii"],
    "hai":    ["hai", "h", "hy", "hai"],
    "kya":    ["kya", "kia", "kyaa", "kya"],
    "karo":   ["karo", "kro", "kar do", "krdo", "kardo"],
    "aaya":   ["aaya", "aya", "ayaa", "aaya"],
    "abhi":   ["abhi", "abi", "abhii"],
    "tak":    ["tak", "tk", "takk"],
    "please": ["please", "plz", "pls", "plss", "plzz"],
    "bhai":   ["bhai", "bhaii", "bhayi", "bro"],
    "order":  ["order", "ordr", "oder"],
    "mujhe":  ["mujhe", "mjhe", "muje", "mujhee"],
    "bohot":  ["bohot", "bahut", "bht", "bohut", "bhut"],
    "accha":  ["accha", "acha", "achha", "acchaa"],
    "paisa":  ["paisa", "paise", "pese", "paisay"],
    "wapas":  ["wapas", "vapas", "wapis"],
    "milega": ["milega", "milga", "mlega", "milegaa"],
    "jaldi":  ["jaldi", "jldi", "jaldii"],
    "kitne":  ["kitne", "ktne", "kitnee"],
    "gaya":   ["gaya", "gya", "gaya", "gyaa"],
    "raha":   ["raha", "rha", "rhaa", "rah"],
    "kaise":  ["kaise", "kese", "kaisee"],
}

ITEM = ["order", "parcel", "package", "item", "product", "delivery", "refund"]

POS_EMO = ["😊", "🙏", "❤️", "👍", "😍"]
NEG_EMO = ["😡", "😭", "😠", "👎", "💔"]
SARC_EMO = ["🙄", "😒", "🤡", "😑"]

# --- templates. {x} slots resolve through VAR; %I% -> ITEM -------------------
# Each entry: (template_id, label, text)
TEMPLATES = [
    # negative: complaint
    ("n01", "negative", "{bhai} %I% {abhi} {tak} {nahi} {aaya} {hai}"),
    ("n02", "negative", "{paisa} kat {gaya} par %I% {nahi} {aaya}"),
    ("n03", "negative", "delivered bol {raha} {hai} lekin mila hi {nahi}"),
    ("n04", "negative", "{bohot} kharab service {hai} {mujhe} refund chahiye"),
    ("n05", "negative", "3 din se wait kar {raha} hu koi response {nahi}"),
    ("n06", "negative", "wrong item bhej diya {hai} {wapas} {karo}"),
    ("n07", "negative", "worst experience ever %I% damaged {aaya}"),
    ("n08", "negative", "{paisa} {wapas} {karo} {please} {bohot} late ho {gaya}"),
    ("n09", "negative", "customer care call hi {nahi} uthata koi help {karo}"),
    ("n10", "negative", "{mujhe} cancel karna {hai} par option hi {nahi} {hai}"),

    # extra in-domain templates (train pool) for surface diversity
    ("u09", "neutral", "{kya} is %I% pe warranty {hai}"),
    ("u10", "neutral", "slot change karna {hai} {kaise} hoga"),
    ("u11", "neutral", "{mujhe} bill pe gst number add karna {hai}"),
    ("u12", "neutral", "%I% {kitne} baje {tak} deliver hota {hai}"),
    ("u13", "neutral", "payment mode change ho sakta {hai} {kya}"),
    ("u14", "neutral", "{mujhe} pickup schedule karna {hai} {abhi}"),
    ("p08", "positive", "{bohot} {jaldi} resolve ho {gaya} thanks team"),
    ("p09", "positive", "%I% expected se {accha} nikla {bohot} khush"),
    ("p10", "positive", "smooth process tha koi dikkat {nahi} {aaya}"),
    ("p11", "positive", "{bhai} app {bohot} easy {hai} use karna"),
    ("n11", "negative", "{kitne} baar bola par koi sunta hi {nahi}"),
    ("n12", "negative", "%I% {abhi} {tak} pickup {nahi} hua refund kab {milega}"),
    # HELD OUT for test (never trained on):
    ("n90", "negative", "app crash ho {raha} {hai} payment fail {hai} {bohot} irritating"),
    ("n91", "negative", "%I% ka status update hi {nahi} ho {raha} {kitne} din lagenge"),
    ("n92", "negative", "quality {bohot} ghatiya {hai} paise waste ho gaye"),

    # neutral: query / factual
    ("u01", "neutral", "%I% {kitne} din me {milega}"),
    ("u02", "neutral", "{kya} %I% kal {aayega}"),
    ("u03", "neutral", "tracking id share {karo} {please}"),
    ("u04", "neutral", "{mujhe} address change karna {hai} {kaise} karu"),
    ("u05", "neutral", "cod available {hai} {kya} is %I% {hai}"),
    ("u06", "neutral", "delivery time {kya} {hai} {bhai}"),
    ("u07", "neutral", "{mujhe} invoice chahiye download {kaise} karu"),
    ("u08", "neutral", "return window {kitne} din ka {hai}"),
    # HELD OUT:
    ("u90", "neutral", "{kya} exchange ka bhi option {hai} is %I% pe"),
    ("u91", "neutral", "pin code {kitne} baje {tak} serviceable {hai}"),

    # positive: praise / thanks
    ("p01", "positive", "delivery {jaldi} {aaya} thank you"),
    ("p02", "positive", "{bohot} {accha} service {hai} {bhai}"),
    ("p03", "positive", "product quality {accha} {hai} {bohot} khush hu"),
    ("p04", "positive", "refund {jaldi} mil {gaya} thanks a lot"),
    ("p05", "positive", "packaging {bohot} {accha} thi {mujhe} pasand {aaya}"),
    ("p06", "positive", "support team {bohot} helpful {hai} issue solve ho {gaya}"),
    ("p07", "positive", "great experience %I% time pe {aaya}"),
    # HELD OUT:
    ("p90", "positive", "delivery partner {bohot} polite tha {accha} laga"),
    ("p91", "positive", "same day delivery mil {gaya} superb {hai} {bhai}"),

    # SARCASM: positive lexicon + sarcastic emoji -> negative.
    # This is the pragmatic-flip case the problem statement calls out.
    ("s01", "negative", "{bohot} badhiya service"),
    ("s02", "negative", "wah {kya} service {hai}"),
    ("s03", "negative", "superb {bhai} fir se wrong item"),
    ("s04", "negative", "{bohot} {accha} kaam"),
    ("s05", "negative", "mast service thi"),
    ("s90", "negative", "excellent {bohot} {accha} experience"),
    ("s91", "negative", "shandar service {bhai}"),

    # POSITIVE TWINS of the sarcastic carriers. Identical text, positive emoji,
    # positive label. Without these the corpus never shows the model a case
    # where "bohot badhiya service" is sincere, so it learns the WORDS are
    # negative instead of learning to read the emoji. See reports/ E3b.
    ("t01", "positive", "{bohot} badhiya service"),
    ("t02", "positive", "wah {kya} service {hai}"),
    ("t04", "positive", "{bohot} {accha} kaam"),
    ("t05", "positive", "mast service thi"),
    ("t90", "positive", "excellent {bohot} {accha} experience"),
    ("t91", "positive", "shandar service {bhai}"),
]

HELD_OUT = {t[0] for t in TEMPLATES if t[0][1:].startswith("9")}
# Dev must also be a shifted split, else it saturates at 100% and cannot be
# used to select hyperparameters. These ids are excluded from training.
HELD_DEV = {"n12", "u14", "p11", "s03"}
SARCASTIC = {t[0] for t in TEMPLATES if t[0].startswith("s")}
# Sincere twins: same carrier text, positive emoji mandatory.
TWIN = {t[0] for t in TEMPLATES if t[0].startswith("t")}


def _fill(tpl: str, rng: random.Random, variant_slice) -> str:
    out = tpl
    while "%I%" in out:
        out = out.replace("%I%", rng.choice(ITEM), 1)
    for key, pool in VAR.items():
        token = "{" + key + "}"
        while token in out:
            sub = variant_slice(pool, rng)
            out = out.replace(token, sub, 1)
    # unresolved slots like {aayega} -> strip braces
    out = out.replace("{", "").replace("}", "")
    return out


def _train_variant(pool, rng):
    # first 60% of each spelling pool
    k = max(1, int(len(pool) * 0.6))
    return rng.choice(pool[:k])


def _test_variant(pool, rng):
    # remaining 40% -> unseen spellings at eval time
    k = max(1, int(len(pool) * 0.6))
    tail = pool[k:] or pool[:k]
    return rng.choice(tail)



PREFIX = ["", "", "", "hello ", "hi ", "sir ", "{bhai} ", "arey "]
SUFFIX_OK = ["", "", "", " {please}", " {bhai}", " thanks", " kindly help"]
SUFFIX_NEG = ["", "", "", " {please}", " {bhai}", " urgent", " kindly dekho"]


def _decorate(text, label, rng, vfn):
    pre = rng.choice(PREFIX)
    suf = rng.choice(SUFFIX_NEG if label == "negative" else SUFFIX_OK)
    out = pre + text + suf
    for key, pool in VAR.items():
        token = "{" + key + "}"
        while token in out:
            out = out.replace(token, vfn(pool, rng), 1)
    return out.strip()


def _emoji_for(label, tpl_id, rng):
    """Attach an emoji ~60% of the time, consistent with the labeling rules.

    Exception: sarcastic templates ALWAYS carry a sarcastic marker. Without it,
    'bohot badhiya service' is genuinely positive, and labelling it negative
    would put a contradiction in the training data. The emoji is the only thing
    that licenses the negative label, which is exactly the property the
    minimal-pair probe measures.
    """
    if tpl_id in SARCASTIC:
        return " " + rng.choice(SARC_EMO)
    if tpl_id in TWIN:
        return " " + rng.choice(POS_EMO)
    if rng.random() > 0.6:
        return ""
    if label == "positive":
        return " " + rng.choice(POS_EMO)
    if label == "negative":
        return " " + rng.choice(NEG_EMO)
    return ""


def _punct(label, tpl_id, rng):
    if tpl_id.startswith("u") and rng.random() < 0.7:
        return "?"
    if label == "negative" and rng.random() < 0.3:
        return rng.choice(["!", "!!", "!!!"])
    return ""


def generate(n, split="train", seed=13):
    """Yield (text, label, template_id). split='test' uses held-out templates
    and held-out spelling variants only."""
    rng = random.Random(seed)
    if split == "train":
        pool = [t for t in TEMPLATES if t[0] not in HELD_OUT | HELD_DEV]
        vfn = _train_variant
    elif split == "dev":
        pool = [t for t in TEMPLATES if t[0] in HELD_DEV]
        vfn = _test_variant
    else:
        pool = [t for t in TEMPLATES if t[0] in HELD_OUT]
        vfn = _test_variant
    rows = []
    for _ in range(n):
        tid, label, tpl = rng.choice(pool)
        text = _fill(tpl, rng, vfn)
        text += _punct(label, tid, rng)
        text += _emoji_for(label, tid, rng)
        rows.append((text, label, tid))
    return rows


def generate_balanced(n_per_class, split="train", seed=13, max_tries=40):
    """Class-balanced sampling.

    Negative templates have more fillable slots than neutral ones, so naive
    dedup-then-split skews hard toward negative. Here we sample per class and
    prefer unseen surface strings, falling back to repeats only once a class's
    unique pool is genuinely exhausted.
    """
    rng = random.Random(seed)
    if split == "train":
        pool = [t for t in TEMPLATES if t[0] not in HELD_OUT | HELD_DEV]
        vfn = _train_variant
    elif split == "dev":
        pool = [t for t in TEMPLATES if t[0] in HELD_DEV]
        vfn = _test_variant
    else:
        pool = [t for t in TEMPLATES if t[0] in HELD_OUT]
        vfn = _test_variant

    by_label = {}
    for tid, lab, tpl in pool:
        by_label.setdefault(lab, []).append((tid, lab, tpl))

    rows, seen = [], set()
    for lab, tpls in by_label.items():
        got = 0
        while got < n_per_class:
            for _ in range(max_tries):
                tid, _l, tpl = rng.choice(tpls)
                text = _fill(tpl, rng, vfn)
                text += _punct(lab, tid, rng)
                text += _emoji_for(lab, tid, rng)
                text = _decorate(text, lab, rng, vfn)
                if text not in seen:
                    break
            seen.add(text)
            rows.append((text, lab, tid))
            got += 1
    rng.shuffle(rows)
    return rows


# --- Intent labels (extension: same encoder, different head) ------------
# Maps each template id to an intent. This is deliberately simple: it proves
# the encoder is task-agnostic without building a second annotation effort.
INTENT_MAP = {
    "n01": "delivery_delay", "n02": "delivery_delay", "n03": "delivery_delay",
    "n04": "complaint", "n05": "delivery_delay", "n06": "wrong_item",
    "n07": "wrong_item", "n08": "refund_request", "n09": "complaint",
    "n10": "cancellation", "n11": "complaint", "n12": "refund_request",
    "n90": "complaint", "n91": "delivery_delay", "n92": "complaint",
    "u01": "delivery_query", "u02": "delivery_query", "u03": "tracking_query",
    "u04": "account_query", "u05": "payment_query", "u06": "delivery_query",
    "u07": "account_query", "u08": "policy_query", "u09": "policy_query",
    "u10": "account_query", "u11": "account_query", "u12": "delivery_query",
    "u13": "payment_query", "u14": "account_query",
    "u90": "policy_query", "u91": "delivery_query",
    "p01": "praise", "p02": "praise", "p03": "praise", "p04": "praise",
    "p05": "praise", "p06": "praise", "p07": "praise", "p08": "praise",
    "p09": "praise", "p10": "praise", "p11": "praise",
    "p90": "praise", "p91": "praise",
    "s01": "complaint", "s02": "complaint", "s03": "complaint",
    "s04": "complaint", "s05": "complaint", "s90": "complaint", "s91": "complaint",
    "t01": "praise", "t02": "praise", "t04": "praise", "t05": "praise",
    "t90": "praise", "t91": "praise",
}


def generate_balanced_intent(n_per_class, split="train", seed=13):
    """Same generator, same templates, different label: intent instead of
    sentiment. Demonstrates the encoder in hashfast.py is task-agnostic --
    only n_classes and the label column change."""
    rows = generate_balanced(n_per_class, split, seed)
    return [(text, INTENT_MAP.get(tid, "other"), tid) for text, _sent, tid in rows]
