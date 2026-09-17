"""Normalization for Latinized, code-mixed South Asian text.

Design rule: emojis and punctuation are SEMANTIC MODULATORS, never noise.
They are preserved and promoted to standalone tokens so downstream
feature extractors can see them explicitly.
"""
import re
import unicodedata

# Unicode blocks that cover the emoji we actually see in support chat.
EMOJI_RANGES = (
    "\U0001F300-\U0001FAFF"   # pictographs, emoticons, supplemental
    "\u2600-\u27BF"           # misc symbols + dingbats
    "\uFE0F\u200D"            # variation selector, ZWJ
    "\U0001F1E6-\U0001F1FF"   # regional indicators
)
EMOJI_RE = re.compile(f"[{EMOJI_RANGES}]+")
# Punctuation we treat as a pragmatic signal (intensity / question / ellipsis).
PUNCT_RE = re.compile(r"([!?.]{1,}|,|;|:)")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NUM_RE = re.compile(r"\b\d[\d,]*\b")
SPACE_RE = re.compile(r"\s+")

# Elongation: "plsssss" -> "plss". Two repeats kept so that the *fact* of
# elongation survives as a feature while the combinatorial blow-up dies.
ELONG_RE = re.compile(r"(.)\1{2,}")


def split_emoji(text: str) -> str:
    """Put whitespace around every emoji run so it becomes its own token."""
    return EMOJI_RE.sub(lambda m: " " + " ".join(m.group(0)) + " ", text)


def normalize(text: str, keep_emoji: bool = True, keep_punct: bool = True) -> str:
    """Canonical surface form. Cheap: pure regex, no model, no lookup table.

    keep_emoji / keep_punct exist so the ablation study can switch them off
    and measure what those signals are actually worth.
    """
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = URL_RE.sub(" <url> ", t)
    t = t.lower()

    if keep_emoji:
        t = split_emoji(t)
    else:
        t = EMOJI_RE.sub(" ", t)

    t = NUM_RE.sub(" <num> ", t)

    if keep_punct:
        # collapse "!!!!!" -> "!!!" then isolate it
        t = re.sub(r"([!?])\1{2,}", r"\1\1\1", t)
        t = PUNCT_RE.sub(r" \1 ", t)
    else:
        t = PUNCT_RE.sub(" ", t)

    t = re.sub(r"[^\w\s!?.,;:<>" + EMOJI_RANGES + r"]", " ", t)
    t = ELONG_RE.sub(r"\1\1", t)
    t = SPACE_RE.sub(" ", t).strip()
    return t


def char_ngrams(text: str, nmin: int = 3, nmax: int = 5):
    """Boundary-marked character n-grams over the whole normalized string.

    Boundary markers ('<' ... '>') let the model learn prefixes/suffixes,
    which is what makes 'nhi' / 'nahi' / 'nai' land near each other.
    """
    grams = []
    for tok in text.split():
        s = "<" + tok + ">"
        L = len(s)
        for n in range(nmin, nmax + 1):
            if L < n:
                continue
            for i in range(L - n + 1):
                grams.append(s[i:i + n])
    return grams


def word_ngrams(text: str, nmax: int = 2):
    toks = text.split()
    out = list(toks)
    for n in range(2, nmax + 1):
        out += [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return out


def featurize(text: str, cmin=3, cmax=5, wmax=2):
    """Full symbolic feature list used by the hashed neural model."""
    return char_ngrams(text, cmin, cmax) + ["W:" + w for w in word_ngrams(text, wmax)]
