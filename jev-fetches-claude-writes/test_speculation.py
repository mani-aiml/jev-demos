"""Offline tests for the speculation logic. No API calls."""

import asyncio
from types import SimpleNamespace

import agent_jev
import tools
from agent_jev import JevExecutor, candidate_calls

TASK = "Where is order O-1042?"


class FakeJev:
    """Answers every question with a fixed probability per tool name."""

    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities

    async def system_one(self, state: dict, questions: dict) -> SimpleNamespace:
        names = {key: q.instructions.split("(")[0] for key, q in questions.items()}
        answers = {key: SimpleNamespace(noul=self.probabilities.get(name, 0.0)) for key, name in names.items()}
        return SimpleNamespace(answers=answers, usage=SimpleNamespace(input_tokens=100))


def _no_latency(monkeypatch) -> None:
    monkeypatch.setattr(tools, "TOOL_LATENCY_S", 0)


def test_candidates_come_from_ids_in_state_and_skip_called() -> None:
    state = {"task": TASK, "lookups_so_far": []}
    calls = candidate_calls(state, {("get_order", "O-1042")})
    assert ("get_shipment", "O-1042") in calls
    assert ("get_order", "O-1042") not in calls
    assert not any(name == "get_customer" for name, _ in calls)
    assert ("get_refund_policy", "EU") in calls


def test_hit_miss_and_wasted_are_counted(monkeypatch) -> None:
    _no_latency(monkeypatch)

    async def scenario() -> JevExecutor:
        executor = JevExecutor(client=FakeJev({"get_shipment": 0.9, "get_invoice": 0.8}))
        executor.begin_step(TASK)
        await executor.predictor
        assert "held at customs" in await executor.run("get_shipment", "O-1042")
        assert "O-1042" not in await executor.run("get_inventory", "SKU-301")
        executor.finish()
        return executor

    executor = asyncio.run(scenario())
    assert (executor.stats.hits, executor.stats.misses, executor.stats.wasted) == (1, 1, 1)
    assert executor.stats.jev_tokens == 100


def test_nothing_is_speculated_below_threshold(monkeypatch) -> None:
    _no_latency(monkeypatch)

    async def scenario() -> JevExecutor:
        executor = JevExecutor(client=FakeJev({"get_shipment": agent_jev.SPECULATE_AT - 0.01}))
        executor.begin_step(TASK)
        await executor.predictor
        return executor

    assert asyncio.run(scenario()).pending == {}


def test_jev_first_stops_when_nothing_is_likely(monkeypatch) -> None:
    from agent_jev_first import gather_facts
    from agent_jev import SpecStats

    _no_latency(monkeypatch)
    stats = SpecStats()
    facts = asyncio.run(gather_facts(TASK, FakeJev({"get_shipment": 0.9}), stats))
    assert len(facts) == 1 and facts[0].startswith("get_shipment(O-1042) -> ") and "held at customs" in facts[0]
    assert stats.jev_calls == 2
