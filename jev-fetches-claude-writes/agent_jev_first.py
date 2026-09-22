"""Jev fetches, Claude writes: Jev picks the lookups, the harness runs them, Claude answers in one call with no tools.

On a MISSING reply the run falls back to agent_claude.py, the plain tool loop, so the answer is never written from missing facts."""

import asyncio
from typing import Any

from anthropic import AsyncAnthropic
from typesafe_sdk import AsyncTypeSafeClient

from agent_claude import MAX_TOKENS, MODEL, RunResult, run_agent
from agent_jev_helper import JevStats, candidate_calls, question
from tools import call_tool

FETCH_AT = 0.7
MAX_ROUNDS = 4
MISSING = "MISSING:"
SYSTEM = (
    "You are a support operations agent. Answer the task from the facts given, in three sentences or fewer. "
    f"If a fact you need is not in the list, reply with only '{MISSING} <what you need>'."
)


async def gather_facts(task: str, jev: AsyncTypeSafeClient, stats: JevStats) -> list[str]:
    """Round by round, run every lookup Jev rates likely, until nothing is."""
    history: list[str] = []
    called: set[tuple[str, str]] = set()
    for _ in range(MAX_ROUNDS):
        state: dict[str, Any] = {"task": task, "lookups_so_far": history}
        calls = candidate_calls(state, called)
        if not calls:
            break
        response = await jev.system_one(state=state, questions={f"q{i}": question(c) for i, c in enumerate(calls)})
        stats.add_usage(response.usage.input_tokens)
        chosen = [call for i, call in enumerate(calls) if response.answers[f"q{i}"].noul >= FETCH_AT]
        if not chosen:
            break
        results = await asyncio.gather(*(call_tool(*call) for call in chosen))
        called.update(chosen)
        history += [f"{name}({value}) -> {out}" for (name, value), out in zip(chosen, results)]
    return history


async def run_agent_jev_first(task: str, model: str = MODEL) -> tuple[RunResult, JevStats, bool]:
    """Returns the run, Jev's stats, and whether Claude had to fall back to the tool-using agent."""
    stats = JevStats()
    result = RunResult(model)
    facts = await gather_facts(task, AsyncTypeSafeClient(), stats)
    result.tool_calls = len(facts)
    prompt = f"Task: {task}\n\nFacts:\n" + "\n".join(f"- {fact}" for fact in facts)
    response = await AsyncAnthropic().messages.create(
        model=model, max_tokens=MAX_TOKENS, system=SYSTEM, messages=[{"role": "user", "content": prompt}]
    )
    result.add_usage(response.usage)
    result.answer = "".join(block.text for block in response.content if block.type == "text")
    fell_back = result.answer.startswith(MISSING)
    if fell_back:
        fallback = await run_agent(task, model=model)
        result.answer, result.tool_calls = fallback.answer, result.tool_calls + fallback.tool_calls
        result.claude_calls += fallback.claude_calls
        result.input_tokens += fallback.input_tokens
        result.output_tokens += fallback.output_tokens
    result.stop()
    return result, stats, fell_back
