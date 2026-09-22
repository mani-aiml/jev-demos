"""Baseline agent: Claude picks a tool, then waits for it. Nothing runs ahead."""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from anthropic import AsyncAnthropic
from anthropic.types import Message, Usage
from dotenv import load_dotenv

from tools import TOOL_SCHEMAS, TOOLS, call_tool

load_dotenv(Path(__file__).parent / ".env")

MODEL = "claude-sonnet-5"
USD_PER_MILLION = {"claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0)}
MAX_TOKENS = 4096
MAX_STEPS = 8
SYSTEM = (
    "You are a support operations agent. Gather facts with the lookup tools, "
    "calling independent lookups in parallel in one turn. "
    "Then answer in three sentences or fewer."
)


class ToolExecutor(Protocol):
    def begin_step(self, task: str) -> None: ...
    async def run(self, name: str, value: str) -> str: ...
    def finish(self) -> None: ...


class DirectExecutor:
    """Runs each tool only when Claude asks for it."""

    def begin_step(self, task: str) -> None:
        pass

    async def run(self, name: str, value: str) -> str:
        return await call_tool(name, value)

    def finish(self) -> None:
        pass


@dataclass
class RunResult:
    model: str
    seconds: float = 0.0
    claude_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    answer: str = ""
    started: float = field(default_factory=time.perf_counter, repr=False)

    def add_usage(self, usage: Usage) -> None:
        self.claude_calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

    def stop(self) -> None:
        self.seconds = time.perf_counter() - self.started

    @property
    def claude_usd(self) -> float:
        usd_in, usd_out = USD_PER_MILLION[self.model]
        return (self.input_tokens * usd_in + self.output_tokens * usd_out) / 1_000_000


async def _tool_results(response: Message, executor: ToolExecutor) -> list[dict[str, Any]]:
    uses = [block for block in response.content if block.type == "tool_use"]
    outputs = await asyncio.gather(*(executor.run(use.name, use.input[TOOLS[use.name].arg]) for use in uses))
    return [{"type": "tool_result", "tool_use_id": use.id, "content": out} for use, out in zip(uses, outputs)]


async def run_agent(task: str, executor: ToolExecutor | None = None, model: str = MODEL) -> RunResult:
    executor = executor or DirectExecutor()
    client = AsyncAnthropic()
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    result = RunResult(model)
    for _ in range(MAX_STEPS):
        executor.begin_step(task)
        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, system=SYSTEM, tools=TOOL_SCHEMAS, messages=messages
        )
        result.add_usage(response.usage)
        if response.stop_reason != "tool_use":
            break
        results = await _tool_results(response, executor)
        result.tool_calls += len(results)
        messages += [{"role": "assistant", "content": response.content}, {"role": "user", "content": results}]
    executor.finish()
    result.answer = "".join(block.text for block in response.content if block.type == "text")
    result.stop()
    return result
