"""Fixed probe texts for the setup-time checks of the local model apps (setup.sh ends with one).

A probe check classifies a few unambiguous AG News texts and fails setup (non-zero exit) when a label is
wrong; a parity check compares the local runtime with a reference on the same texts. Part of the shared
benchmark library (shared/); used only at setup, never in a timed run.
"""

from __future__ import annotations

import sys

# Three unambiguous texts (AG News label ids), for probe checks.
AG_NEWS_PROBES: dict[str, str] = {
    "The striker scored twice as the club won the championship final on Sunday.": "sports",
    "Shares fell 5% after the bank reported weaker quarterly earnings and cut its outlook.": "business",
    "Researchers released a new open-source compiler for quantum processors and a faster chip design.": "sci_tech",
}

# Twenty texts (four per topic plus four mixed or ambiguous ones), for parity checks: ambiguous texts
# make a runtime difference show up as an argmax flip or a probability gap.
PARITY_TEXTS: list[str] = [
    *AG_NEWS_PROBES,
    "The prime minister met European leaders in Brussels to discuss the refugee crisis.",
    "Rebels attacked a convoy near the capital, killing six soldiers, officials said.",
    "The United Nations called for a ceasefire after a week of fighting along the border.",
    "Voters in Brazil head to the polls on Sunday in a closely fought presidential election.",
    "The quarterback threw for 300 yards in a comeback win over the defending champions.",
    "Tennis star advances to the semifinals after a five-set battle at Wimbledon.",
    "The Olympic committee announced the host city for the 2032 summer games.",
    "Oil prices climbed to a record high as supply concerns rattled traders.",
    "The retailer will cut 2,000 jobs and close 40 stores to reduce costs.",
    "The central bank raised interest rates by a quarter point to curb inflation.",
    "NASA's rover found new evidence of ancient water on Mars, scientists said.",
    "A security flaw in the popular web browser lets attackers run code remotely.",
    "Microsoft sued a rival over patents for its search engine technology.",
    "The government fined the chipmaker for anticompetitive pricing in Europe.",
    "A football club was sold to an investment fund for 2 billion dollars.",
    "Stocks of video game makers rose after the console launch sold out.",
    "Probe text without a clear topic: the weather was mild on Tuesday.",
]


def check_probes(predict, label_of=lambda p: p.label, name: str = "probe check") -> None:
    """Classify AG_NEWS_PROBES with predict(texts) -> predictions; exit 1 if any label is wrong."""
    preds = predict(list(AG_NEWS_PROBES))
    failed = 0
    for (text, want), pred in zip(AG_NEWS_PROBES.items(), preds):
        got = label_of(pred)
        ok = got == want
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {got:<9} (want {want:<9}) p={pred.confidence:.3f}  {text[:60]}")
    if failed:
        sys.exit(f"{name} failed: {failed} of {len(AG_NEWS_PROBES)} probes misclassified")
    print(f"{name} passed")
