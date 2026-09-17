"""Perturbations that emulate real spelling drift, typos and emoji loss.

Each perturbation is meaning-preserving by construction (a human reading the
perturbed text assigns the same label), so any accuracy drop is pure model
brittleness. That is what makes 'robustness drop' a defensible metric.
"""
import random
import re
from .normalize import EMOJI_RE

# adjacent keys on a QWERTY phone keyboard
NEIGH = {
    "a": "qsw", "b": "vgn", "c": "xdv", "d": "sfe", "e": "wrd", "f": "dgr",
    "g": "fht", "h": "gjy", "i": "uok", "j": "hku", "k": "jli", "l": "ko",
    "m": "nj", "n": "bm", "o": "ipl", "p": "ol", "q": "wa", "r": "etf",
    "s": "adw", "t": "ryg", "u": "yih", "v": "cb", "w": "qes", "x": "zsc",
    "y": "tuh", "z": "xa",
}

PHONETIC = [
    ("aa", "a"), ("ee", "i"), ("oo", "u"), ("ai", "ei"),
    ("v", "w"), ("w", "v"), ("j", "z"), ("s", "sh"), ("kh", "k"),
]


def drop_vowels(text, rng, p=0.5):
    """'nahi' -> 'nhi'. Never touches the first character of a token."""
    out = []
    for tok in text.split():
        if EMOJI_RE.search(tok) or len(tok) < 4:
            out.append(tok); continue
        chars = [tok[0]] + [
            c for c in tok[1:]
            if not (c in "aeiou" and rng.random() < p)
        ]
        out.append("".join(chars) or tok)
    return " ".join(out)


def keyboard_typo(text, rng, p=0.08):
    out = []
    for ch in text:
        if ch in NEIGH and rng.random() < p:
            out.append(rng.choice(NEIGH[ch]))
        else:
            out.append(ch)
    return "".join(out)


def phonetic_swap(text, rng, p=0.4):
    for a, b in PHONETIC:
        if rng.random() < p:
            text = text.replace(a, b)
    return text


def elongate(text, rng, p=0.3):
    out = []
    for tok in text.split():
        if tok and tok[-1].isalpha() and rng.random() < p:
            tok = tok + tok[-1] * rng.randint(2, 4)
        out.append(tok)
    return " ".join(out)


def strip_emoji(text, rng=None):
    return re.sub(r"\s+", " ", EMOJI_RE.sub(" ", text)).strip()


def strip_punct(text, rng=None):
    return re.sub(r"\s+", " ", re.sub(r"[!?.,;:]", " ", text)).strip()


SUITE = {
    "vowel_drop": drop_vowels,
    "keyboard_typo": keyboard_typo,
    "phonetic_swap": phonetic_swap,
    "elongation": elongate,
    "emoji_removed": lambda t, r: strip_emoji(t),
    "punct_removed": lambda t, r: strip_punct(t),
}


def apply(name, rows, seed=7):
    rng = random.Random(seed)
    fn = SUITE[name]
    return [(fn(t, rng), l, s) for t, l, s in rows]
