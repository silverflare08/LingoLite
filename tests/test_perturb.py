"""Tests for pg.perturb and pg.minimal_pairs.

Guards the two things the robustness story depends on: perturbations must
actually change the text (otherwise the robustness numbers are measuring
nothing), and the minimal-pair probe must produce genuinely matched pairs.
"""
from pg.minimal_pairs import build
from pg.perturb import apply


def test_vowel_drop_actually_changes_text():
    rows = [("nahi bhai order abhi tak nahi aya", "negative", "n01")]
    out = apply("vowel_drop", rows)
    assert out[0][0] != rows[0][0]


def test_emoji_removed_strips_all_emoji():
    rows = [("bohot accha 😊 thanks 🙏", "positive", "p01")]
    out = apply("emoji_removed", rows)
    assert "😊" not in out[0][0] and "🙏" not in out[0][0]


def test_perturbation_never_changes_the_label():
    # A perturbation is only valid for the robustness suite if it preserves
    # meaning. We can't check meaning automatically, but we CAN check the
    # one thing that would silently invalidate every robustness number:
    # the (text, label, source) triple's label must pass through untouched.
    rows = [("order nhi aya bhai", "negative", "n01"),
            ("delivery jaldi aya thanks", "positive", "p01")]
    for name in ("vowel_drop", "keyboard_typo", "phonetic_swap",
                "elongation", "emoji_removed", "punct_removed"):
        out = apply(name, rows)
        assert [l for _, l, _ in out] == [l for _, l, _ in rows], name


def test_minimal_pairs_are_true_minimal_pairs():
    # Same carrier text, only the emoji differs, opposite expected labels.
    for pos_text, pos_label, sarc_text, sarc_label, carrier in build():
        assert pos_label == "positive" and sarc_label == "negative"
        assert pos_text.startswith(carrier) and sarc_text.startswith(carrier)
        assert pos_text != sarc_text
