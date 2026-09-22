"""Helpers for asking Jev which lookups to run: the candidates, the question, the bookkeeping."""

import json
import re
from dataclasses import dataclass
from typing import Any

from typesafe_sdk import Noul

from tools import ID_PATTERNS, REGIONS, TOOLS

JEV_USD_PER_TOKEN = 0.042 / 1_000_000

Call = tuple[str, str]


@dataclass
class JevStats:
    """What Jev cost: calls, input tokens and the dollars they add up to."""

    jev_calls: int = 0
    jev_tokens: int = 0

    def add_usage(self, input_tokens: int | None) -> None:
        self.jev_calls += 1
        self.jev_tokens += input_tokens or 0

    @property
    def jev_usd(self) -> float:
        return self.jev_tokens * JEV_USD_PER_TOKEN


def candidate_calls(state: dict[str, Any], called: set[Call]) -> list[Call]:
    """Every (tool, argument) pair that is possible given the ids seen so far."""
    text = json.dumps(state)
    values = {arg: sorted(set(re.findall(pattern, text))) for arg, pattern in ID_PATTERNS.items()}
    values["region"] = list(REGIONS)
    pairs = [(name, value) for name, spec in TOOLS.items() for value in values[spec.arg]]
    return [pair for pair in pairs if pair not in called]


def question(call: Call) -> Noul:
    """One yes/no question per candidate. The wording matters: asking whether a lookup is "the
    very next one" split the probability across parallel lookups and hit about half the time;
    "one of the lookups to run now, possibly alongside others" hits 97 percent."""
    name, value = call
    spec = TOOLS[name]
    return Noul(
        instructions=f"{name}({spec.arg}={value}) is one of the lookups the assistant should run now, "
        f"possibly alongside others, to complete the task. {name} returns: {spec.description}"
    )
