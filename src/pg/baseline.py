"""Level 1: character n-gram TF-IDF + logistic regression.

Rationale (goes in the whitepaper): a word-level vocabulary is exactly what
breaks on Romanized code-mixed text, because every spelling of a word is a new
type. Character n-grams with word-boundary markers make 'nahi' / 'nhi' / 'nai'
share most of their active features, so the classifier generalises across
spellings without any normalisation lexicon.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from .normalize import normalize


class _Normalize:
    """Picklable module-level callable (a closure would break joblib/pickle)."""

    def __init__(self, keep_emoji=True, keep_punct=True):
        self.keep_emoji, self.keep_punct = keep_emoji, keep_punct

    def __call__(self, texts):
        return [normalize(t, self.keep_emoji, self.keep_punct) for t in texts]


def make_normalizer(keep_emoji=True, keep_punct=True):
    return FunctionTransformer(_Normalize(keep_emoji, keep_punct), validate=False)


def build(keep_emoji=True, keep_punct=True, C=4.0, max_features=200_000, seed=0):
    feats = FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5),
            min_df=2, sublinear_tf=True, max_features=max_features)),
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1,
            sublinear_tf=True, token_pattern=r"[^\s]+")),
    ])
    clf = LogisticRegression(
        C=C, max_iter=2000, class_weight="balanced",
        solver="lbfgs", random_state=seed)
    return Pipeline([
        ("norm", make_normalizer(keep_emoji, keep_punct)),
        ("feats", feats),
        ("clf", clf),
    ])


def n_params(pipe):
    """Effective learned parameters = coefficient matrix + intercepts."""
    clf = pipe.named_steps["clf"]
    return int(clf.coef_.size + clf.intercept_.size)


class BaselineAdapter:
    """Gives the sklearn pipeline the same predict/encode surface as HashFast
    so the benchmark harness can time both without special-casing."""

    def __init__(self, pipe):
        self.pipe = pipe

    def predict(self, texts):
        return np.asarray(self.pipe.predict(texts))

    def encode_all(self, texts):
        return self.pipe.named_steps["feats"].transform(
            self.pipe.named_steps["norm"].transform(texts))

    def predict_encoded(self, X):
        return np.asarray(self.pipe.named_steps["clf"].predict(X))

    def n_params(self):
        return n_params(self.pipe)
