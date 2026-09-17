"""Tests for pg.hashfast.

These check the model's plumbing (shapes, saving/loading, parameter budget),
not its accuracy — accuracy is measured by scripts/train_eval.py and belongs
in reports/, not in a unit test that runs in under a second.
"""
import numpy as np

from pg.hashfast import HashFast


def _toy_model():
    return HashFast(n_classes=3, buckets=2000, dim=8, seed=0)


def test_predict_returns_one_label_per_input():
    m = _toy_model()
    out = m.predict(["order nhi aya", "kya haal hai", "thanks bhai"])
    assert len(out) == 3
    assert set(out.tolist()) <= {0, 1, 2}


def test_empty_string_does_not_crash():
    # An empty or all-punctuation message should still get a prediction,
    # not throw — this is what a real ingestion queue will send eventually.
    m = _toy_model()
    out = m.predict([""])
    assert len(out) == 1


def test_training_reduces_loss():
    # Not a claim about accuracy — just that the gradient step is actually
    # doing something, i.e. the hand-written backprop isn't a no-op.
    m = _toy_model()
    texts = ["order nhi aya", "bohot accha service", "kya haal hai"] * 20
    labels = [0, 2, 1] * 20
    enc = m.encode_all(texts)
    hist = m.fit(enc, labels, epochs=5, lr=0.5, batch_size=8, verbose=False)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]


def test_param_count_matches_declared_formula():
    # buckets * dim (embedding table) + dim*2 * n_classes (mean+max head)
    # + n_classes (bias). If this test breaks, the whitepaper's parameter
    # table is wrong and needs updating alongside the code.
    m = HashFast(n_classes=3, buckets=1000, dim=16, pooling="mean+max")
    expected = 1000 * 16 + (16 * 2) * 3 + 3
    assert m.n_params() == expected


def test_save_and_load_roundtrip(tmp_path):
    m = _toy_model()
    texts = ["order nhi aya", "bohot accha", "kya haal"] * 10
    labels = [0, 2, 1] * 10
    m.fit(m.encode_all(texts), labels, epochs=3, verbose=False)
    path = tmp_path / "model.npz"
    m.save(str(path))

    loaded = HashFast.load(str(path))
    before = m.predict(texts)
    after = loaded.predict(texts)
    assert np.array_equal(before, after)


def test_stays_within_parameter_budget():
    # The actual configuration used in scripts/train_eval.py, checked
    # against the 500M-parameter ceiling from the problem statement.
    m = HashFast(n_classes=3, buckets=100_000, dim=64, pooling="mean+max")
    assert m.n_params() <= 500_000_000
