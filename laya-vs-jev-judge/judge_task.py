"""The judging task both engines answer, defined once so the comparison is like-for-like.

Two questions, chosen to sit on opposite ends of the cardinality range that separates
these models: a 2-way verdict (the high-volume call) and a 40-way criterion lookup
(where per-option token budget starts to matter).
"""

from typing import Any

VERDICT_OPTIONS: dict[str, str] = {
    "pass": "the answer is factually correct, responsive to the question, and free of invented detail",
    "rework": "the answer is wrong, evasive, incomplete, or states something it cannot support",
}

# 40 failure modes: the high-cardinality question. Kept flat and mutually exclusive.
CRITERIA: dict[str, str] = {
    "none": "no defect worth flagging",
    "factual_error": "states something verifiably untrue",
    "fabricated_citation": "cites a source, paper, or author that does not exist",
    "fabricated_number": "gives a statistic or figure with no basis",
    "outdated_fact": "was true once but is no longer",
    "wrong_entity": "answers about the wrong person, product, or company",
    "unit_error": "right magnitude, wrong unit or scale",
    "arithmetic_error": "the arithmetic does not check out",
    "reversed_causality": "gets cause and effect the wrong way round",
    "overgeneralised": "states a narrow finding as a universal rule",
    "unsupported_claim": "asserts something it gives no grounds for",
    "hedged_to_uselessness": "so qualified it conveys nothing",
    "evasive": "avoids answering the question asked",
    "answered_different_question": "responds to a question that was not asked",
    "incomplete": "addresses only part of a multi-part question",
    "ignored_constraint": "breaks an explicit constraint in the prompt",
    "wrong_format": "does not follow the requested output format",
    "too_verbose": "buries the answer in padding",
    "too_terse": "omits detail the question required",
    "contradicts_itself": "makes two claims that cannot both hold",
    "contradicts_source": "disagrees with the provided context",
    "ignored_context": "does not use context it was given",
    "misread_context": "uses the context but misreads it",
    "stale_assumption": "assumes a fact not in evidence",
    "circular_reasoning": "restates the claim as its own justification",
    "false_precision": "gives more significant figures than the method supports",
    "confused_correlation": "treats correlation as causation",
    "wrong_scope": "answers at the wrong level of abstraction",
    "missing_caveat": "omits a limitation that materially changes the answer",
    "unsafe_advice": "recommends something harmful or negligent",
    "outside_expertise": "gives regulated advice it should decline",
    "leaked_reasoning": "exposes scratchpad or system text",
    "refused_wrongly": "declines a benign request",
    "tone_mismatch": "wrong register for the audience",
    "repeated_content": "repeats the same point in different words",
    "broken_structure": "headings, lists, or code blocks are malformed",
    "wrong_language": "responds in the wrong language",
    "truncated": "stops mid-thought",
    "placeholder_left": "leaves a TODO or template marker in the output",
    "off_topic": "not about the subject at all",
}


def laya_questions() -> dict[str, dict[str, Any]]:
    """The same two questions in Laya's schema."""
    return {
        "verdict": {
            "type": "choice",
            "instructions": "Judge the ANSWER against the QUESTION. Is it acceptable as it stands?",
            "criteria": VERDICT_OPTIONS,
        },
        "criterion": {
            "type": "choice",
            "instructions": "Which single defect best describes what is wrong with the ANSWER?",
            "criteria": CRITERIA,
        },
    }


def jev_questions() -> dict[str, Any]:
    """The same two questions in Jev's schema."""
    from typesafe_sdk import Choice

    return {
        "verdict": Choice(
            instructions="Judge the ANSWER against the QUESTION. Is it acceptable as it stands?",
            criteria=VERDICT_OPTIONS,
        ),
        "criterion": Choice(
            instructions="Which single defect best describes what is wrong with the ANSWER?",
            criteria=CRITERIA,
        ),
    }


def state_for(question: str, answer: str) -> dict[str, str]:
    """The item under judgement, identical for both engines."""
    return {"QUESTION": question, "ANSWER": answer}
