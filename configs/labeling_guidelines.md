# Labelling guidelines — Hinglish support-chat sentiment

Annotator disagreement on code-mixed text is high, and most of it comes from
emoji. These rules exist so that "the emoji flips the meaning" is a *decidable*
claim rather than a vibe. Any model evaluation is only as good as this file.

## Classes

| Label | Definition | Test |
|---|---|---|
| `negative` | The writer expresses dissatisfaction, complaint, frustration, or a grievance about the service/product. | Would a support agent need to apologise or remediate? |
| `neutral` | The writer requests information or performs a transaction with no evaluative stance. | Is this answerable with a fact, with nobody at fault? |
| `positive` | The writer expresses satisfaction, praise, or thanks. | Would a support agent reasonably say "glad to hear it"? |

## Rule 1 — Sentiment is about the writer's stance, not the event

"order kal ayega kya?" is `neutral` even though the order has not arrived.
"order abhi tk nhi aya" is `negative`, because the phrasing carries grievance.

## Rule 2 — Emoji override lexicon, but only in specific classes

An emoji changes the label when the carrier text is evaluative and the emoji
contradicts it:

| Text | Label | Why |
|---|---|---|
| `bohot badhiya service 😊` | positive | emoji agrees with lexicon |
| `bohot badhiya service 🙄` | negative | eye-roll marks the praise as ironic |
| `order kal ayega kya? 😊` | neutral | emoji is politeness, not evaluation; carrier is not evaluative |

Sarcasm-marking emoji (closed set): 🙄 😒 🤡 😑
Sincerity-marking emoji (closed set): 😊 👍 ❤️ 🙏 😍
Distress emoji: 😡 😭 😠 👎 💔

## Rule 3 — Bare ironic text without a marker is NOT labelled negative

`bohot badhiya service` with no emoji and no context is `positive`. Labelling
it negative puts a contradiction into the corpus: the annotator is using
knowledge the model cannot see. If the sarcasm is not marked by an emoji,
punctuation, or contradicting content ("superb, fir se wrong item"), label the
surface reading.

This rule is enforced in the generator: sarcastic items always carry a marker.

## Rule 4 — Mixed messages take the dominant grievance

"delivery fast thi but item galat hai" → `negative`. A complaint anywhere in
the message outranks praise, because routing consequences follow the complaint.

## Rule 5 — Politeness particles are not sentiment

"pls", "kindly", "bhai", "sir" do not make a message positive.

## Ambiguity handling

Items that two annotators label differently after applying the rules above are
recorded but excluded from the scored test set, and reported as an
`ambiguous_rate`. Hiding them inside the accuracy number is how sentiment
benchmarks end up with an unreachable ceiling nobody accounts for.
