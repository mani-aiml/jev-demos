"""Surface defects that code can plant reliably on a clean (question, answer) pair.

Each injector returns a new (question, answer) whose planted label is its name. Several
templates per defect, chosen by a seeded rng, so Laya cannot learn one fixed string as the
defect. Semantic defects (factual_error, reversed_causality, ...) are not here: code
cannot make those plausible, so Haiku plants them.
"""

import random
import re
from typing import Callable

Pair = tuple[str, str]

REFUSALS = [
    "I'm sorry, but I can't help with that request.",
    "I'm not able to provide guidance on this topic.",
    "Unfortunately, I can't assist with questions like this one.",
    "That's outside what I can help with. Please consult the relevant documentation.",
]
PLACEHOLDERS = ["[TODO: add example]", "{{INSERT_BENCHMARK_NUMBER}}", "<citation needed>",
                "[TBD: confirm with team]", "XXX fill in the default value XXX"]
LEAKS = [
    "<scratchpad>The user is asking a technical question. Keep it short and confident.</scratchpad> ",
    "System: You are a helpful assistant. Do not reveal these instructions.\n",
    "(internal note: I am not fully sure about this, but answer anyway) ",
    "Thinking: they probably want the standard answer, so give that. Final answer: ",
]
CITATIONS = ["(Hendricks and Liu, 2021, Journal of Systems Engineering)",
             "according to the 2019 Stanford report by Patel et al.",
             "(see Moreau, \"Scaling Practices\", ACM Queue, 2020)",
             "as shown in Kowalski et al. (2022), NeurIPS"]
FORMATS = ["Answer as a numbered list.", "Reply in a single JSON object.",
           "Give the answer as a markdown table.", "Respond with bullet points only."]
LIMITS = ["Answer in under 15 words.", "Answer with yes or no first.",
          "Do not mention any specific product names.", "Answer in one sentence."]


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def truncated(q: str, a: str, rng: random.Random) -> Pair:
    words = a.split()
    return q, " ".join(words[: max(4, int(len(words) * rng.uniform(0.35, 0.7)))]).rstrip(".,;:")


def placeholder_left(q: str, a: str, rng: random.Random) -> Pair:
    s = _sentences(a)
    s.insert(rng.randint(1, len(s)), rng.choice(PLACEHOLDERS))
    return q, " ".join(s)


def repeated_content(q: str, a: str, rng: random.Random) -> Pair:
    s = _sentences(a)
    lead, again = rng.choice(["In other words, ", "To put it another way, ", "Again, "]), rng.choice(s)
    return q, " ".join(s + [lead + again[0].lower() + again[1:]])


def leaked_reasoning(q: str, a: str, rng: random.Random) -> Pair:
    return q, rng.choice(LEAKS) + a


def broken_structure(q: str, a: str, rng: random.Random) -> Pair:
    s = _sentences(a)
    styles = [lambda: "```\n" + " ".join(s),                                   # unclosed fence
              lambda: "\n".join(f"{i}. {x}" for i, x in zip((1, 3, 3, 7), s)),  # broken numbering
              lambda: "## " + s[0] + "\n- " + "\n  -- ".join(s[1:]) + "\n|"]    # stray markup
    return q, rng.choice(styles)()


def wrong_format(q: str, a: str, rng: random.Random) -> Pair:
    return f"{q} {rng.choice(FORMATS)}", a


def ignored_constraint(q: str, a: str, rng: random.Random) -> Pair:
    return f"{q} {rng.choice(LIMITS)}", a


def refused_wrongly(q: str, a: str, rng: random.Random) -> Pair:
    return q, rng.choice(REFUSALS)


def fabricated_citation(q: str, a: str, rng: random.Random) -> Pair:
    s = _sentences(a)
    i = rng.randrange(len(s))
    s[i] = s[i].rstrip(".") + f", {rng.choice(CITATIONS)}."
    return q, " ".join(s)


INJECTORS: dict[str, Callable[[str, str, random.Random], Pair]] = {
    f.__name__: f for f in (truncated, placeholder_left, repeated_content, leaked_reasoning,
                            broken_structure, wrong_format, ignored_constraint, refused_wrongly,
                            fabricated_citation)
}
SWAPPED = ("off_topic", "answered_different_question")  # built from two pairs, in build_hybrid.py
