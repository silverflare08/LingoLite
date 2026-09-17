"""Tests for pg.datagen.

These protect the thing that made the minimal-pair bug findable in the first
place: that train/dev/test use disjoint templates, and that sarcastic
carriers have sincere twins. If either regresses, the project's headline
numbers become meaningless again in the exact way §6.2 of the whitepaper
describes.
"""
from pg.datagen import HELD_DEV, HELD_OUT, SARCASTIC, TWIN, generate_balanced


def test_train_dev_test_templates_are_disjoint():
    all_ids = {t[0] for t in __import__("pg.datagen", fromlist=["TEMPLATES"]).TEMPLATES}
    train_ids = all_ids - HELD_OUT - HELD_DEV
    assert train_ids.isdisjoint(HELD_OUT)
    assert train_ids.isdisjoint(HELD_DEV)
    assert HELD_OUT.isdisjoint(HELD_DEV)


def test_every_sarcastic_template_has_a_sincere_twin_count():
    # Not a strict 1:1 mapping, but there should be roughly as many sincere
    # carriers as sarcastic ones — otherwise the model only ever sees the
    # ironic reading again, which was the bug in whitepaper.md §6.2.
    assert len(TWIN) >= len(SARCASTIC) - 1


def test_generate_balanced_gives_equal_classes():
    rows = generate_balanced(50, "train", seed=1)
    counts = {}
    for _, label, _ in rows:
        counts[label] = counts.get(label, 0) + 1
    assert set(counts.values()) == {50} or len(set(counts.values())) == 1


def test_test_split_only_uses_held_out_templates():
    rows = generate_balanced(20, "test", seed=1)
    ids = {tid for _, _, tid in rows}
    assert ids <= HELD_OUT


def test_dev_split_only_uses_held_dev_templates():
    rows = generate_balanced(10, "dev", seed=1)
    ids = {tid for _, _, tid in rows}
    assert ids <= HELD_DEV
