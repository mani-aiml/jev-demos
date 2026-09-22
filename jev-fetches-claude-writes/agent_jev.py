"""Same agent, with Jev as a branch predictor: likely lookups start while Claude is still deciding."""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from typesafe_sdk import AsyncTypeSafeClient, Noul

from agent_claude import MODEL, RunResult, run_agent
from tools import ID_PATTERNS, REGIONS, TOOLS, call_tool

SPECULATE_AT = 0.7
MAX_SPECULATIONS = 6
JEV_USD_PER_TOKEN = 0.042 / 1_000_000

Call = tuple[str, str]


@dataclass
class SpecStats:
    hits: int = 0
    misses: int = 0
    wasted: int = 0
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
    name, value = call
    spec = TOOLS[name]
    return Noul(
        instructions=f"{name}({spec.arg}={value}) is one of the lookups the assistant should run now, "
        f"possibly alongside others, to complete the task. {name} returns: {spec.description}"
    )


class JevExecutor:
    """Asks Jev one yes/no question per possible lookup, all in one call, and starts the likely ones early."""

    def __init__(self, client: AsyncTypeSafeClient | None = None) -> None:
        self.client = client or AsyncTypeSafeClient()
        self.stats = SpecStats()
        self.history: list[str] = []
        self.called: set[Call] = set()
        self.pending: dict[Call, asyncio.Task[str]] = {}
        self.predictor: asyncio.Task[None] | None = None

    def begin_step(self, task: str) -> None:
        self.finish()
        state = {"task": task, "lookups_so_far": self.history}
        self.predictor = asyncio.create_task(self._speculate(state))

    async def _speculate(self, state: dict[str, Any]) -> None:
        calls = candidate_calls(state, self.called)
        if not calls:
            return
        questions = {f"q{i}": question(call) for i, call in enumerate(calls)}
        response = await self.client.system_one(state=state, questions=questions)
        self.stats.add_usage(response.usage.input_tokens)
        scored = sorted(((response.answers[f"q{i}"].noul, call) for i, call in enumerate(calls)), reverse=True)
        for probability, call in scored[:MAX_SPECULATIONS]:
            if probability >= SPECULATE_AT:
                self.pending[call] = asyncio.create_task(call_tool(*call))

    async def run(self, name: str, value: str) -> str:
        early = self.pending.pop((name, value), None)
        if early is None:
            self.stats.misses += 1
            result = await call_tool(name, value)
        else:
            self.stats.hits += 1
            result = await early
        self.called.add((name, value))
        self.history.append(f"{name}({value}) -> {result}")
        return result

    def finish(self) -> None:
        """Drop any speculation Claude did not ask for."""
        if self.predictor is not None:
            self.predictor.cancel()
        for task in self.pending.values():
            task.cancel()
        self.stats.wasted += len(self.pending)
        self.pending.clear()


async def run_agent_with_jev(task: str, model: str = MODEL) -> tuple[RunResult, SpecStats]:
    executor = JevExecutor()
    result = await run_agent(task, executor, model)
    return result, executor.stats
