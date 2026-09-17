"""Tests for pg.normalize.

Each test checks ONE claim the whitepaper makes about the design. If a test
here fails, a specific sentence in whitepaper.md is now false — that's the
point of testing: it keeps the write-up honest as the code changes.
"""
from pg.normalize import char_ngrams, featurize, normalize


def test_emoji_is_kept_not_stripped():
    # Claim: emoji are preserved as semantic signal, not deleted as noise.
    out = normalize("bohot badhiya service 🙄")
    assert "🙄" in out


def test_emoji_removed_when_flag_is_off():
    # The ablation study needs this switch to actually turn emoji off.
    out = normalize("bohot badhiya service 🙄", keep_emoji=False)
    assert "🙄" not in out


def test_lowercases_text():
    assert normalize("ORDER NHI AYA") == normalize("order nhi aya")


def test_excessive_punctuation_is_capped_not_deleted():
    # "!!!!!" should shrink to a bounded number of "!", not vanish and not
    # explode the feature space with every possible length.
    out = normalize("urgent!!!!!!!!!!")
    assert "!" in out
    assert "!!!!!!!!!!" not in out


def test_elongation_is_shortened_but_kept():
    # "plsssss" should not survive as-is (infinite variants), but the FACT
    # that the word was stretched should still be visible in the output.
    out = normalize("plsssss")
    assert out != "plsssss"
    assert "pl" in out


def test_spelling_variants_share_character_ngrams():
    # This is the core claim of the whole project: different spellings of the
    # same word should overlap heavily in character n-gram space, which is
    # what lets the model generalise across "nahi" / "nhi" / "nai" without a
    # spelling-correction dictionary.
    #
    # cmin/cmax=2,4 on purpose: that's the window the actual model uses
    # (whitepaper.md §3). The wider default (3-5) window was tried first and
    # scored worse for exactly this reason -- on short Hinglish tokens a
    # 5-letter window spans the whole word, so two spellings of the same
    # short word barely overlap. Re-run this test with (3, 5) and watch it
    # fail; that failure IS the finding written up in the whitepaper.
    a = set(char_ngrams(normalize("nahi"), 2, 4))
    b = set(char_ngrams(normalize("nhi"), 2, 4))
    overlap = len(a & b) / min(len(a), len(b))
    assert overlap > 0.3, f"expected meaningful overlap, got {overlap:.2f}"


def test_unrelated_words_do_not_share_many_ngrams():
    # Sanity check on the above: overlap should mean something, i.e. two
    # unrelated words should NOT overlap nearly as much.
    a = set(char_ngrams(normalize("nahi"), 2, 4))
    c = set(char_ngrams(normalize("delivery"), 2, 4))
    overlap = len(a & c) / min(len(a), len(c))
    assert overlap < 0.3


def test_featurize_returns_both_char_and_word_features():
    feats = featurize(normalize("order nhi aya"))
    has_char = any(not f.startswith("W:") for f in feats)
    has_word = any(f.startswith("W:") for f in feats)
    assert has_char and has_word
