"""Offline tests for the planning logic. No API calls."""

import asyncio
from types import SimpleNamespace

import tools
from agent_jev_first import FETCH_AT, gather_facts
from agent_jev_helper import JevStats, candidate_calls, question

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


def test_question_names_the_tool_and_asks_for_one_of_now() -> None:
    q = question(("get_shipment", "O-1042"))
    assert q.instructions.startswith("get_shipment(order_id=O-1042) is one of the lookups")
    assert "possibly alongside others" in q.instructions


def test_gather_facts_runs_the_likely_lookup_and_stops(monkeypatch) -> None:
    _no_latency(monkeypatch)
    stats = JevStats()
    facts = asyncio.run(gather_facts(TASK, FakeJev({"get_shipment": 0.9}), stats))
    assert len(facts) == 1 and facts[0].startswith("get_shipment(O-1042) -> ") and "held at customs" in facts[0]
    assert stats.jev_calls == 2          # one round that fetched, one that found nothing likely
    assert stats.jev_tokens == 200


def test_nothing_is_fetched_below_threshold(monkeypatch) -> None:
    _no_latency(monkeypatch)
    stats = JevStats()
    facts = asyncio.run(gather_facts(TASK, FakeJev({"get_shipment": FETCH_AT - 0.01}), stats))
    assert facts == [] and stats.jev_calls == 1
