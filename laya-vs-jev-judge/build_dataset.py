"""Build the judged corpus in two resumable stages.

    stage 1  generate (question, answer) with a planted defect  -> data/generated.jsonl
    stage 2  gold-label each one blind                          -> data/judged.jsonl

Both stages skip ids they already hold, so a rate limit costs time and never money twice.

Two independent references come out of this:
  planted  - the defect the generator was told to introduce (a controlled label)
  gold     - what a frontier judge says, reading only question and answer

The video quotes agreement with `gold`. `planted` is the honesty check: if the frontier
judge cannot recover a defect it was never told about, no small model will either.
"""

import asyncio
import json
import random
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic, APIStatusError

from judge_task import CRITERIA

GENERATOR_MODEL = "claude-haiku-4-5-20251001"
GOLD_MODEL = "claude-sonnet-5"
DATA = Path(__file__).parent / "data"
GENERATED, JUDGED = DATA / "generated.jsonl", DATA / "judged.jsonl"

TOPICS = [
    "LLM inference and serving", "retrieval and vector search", "AI agents and tool use",
    "Python performance", "distributed systems", "SQL and data modelling",
    "Kubernetes and deployment", "security and authentication", "statistics for ML",
    "GPU hardware", "evaluation and benchmarking", "networking",
]

GEN_PROMPT = """Write one realistic technical QUESTION about {topic}, then an ANSWER to it.

The answer must contain exactly this defect: {defect} ({description})

Make the defect realistic and not obvious, the kind a competent model actually produces.
The answer should be 2-4 sentences and otherwise fluent and confident.

Return strict JSON: {{"question": "...", "answer": "..."}}"""

CLEAN_PROMPT = """Write one realistic technical QUESTION about {topic}, then a genuinely
correct, complete ANSWER to it, 2-4 sentences.

Return strict JSON: {{"question": "...", "answer": "..."}}"""

GOLD_PROMPT = """You are grading an AI-generated answer.

QUESTION: {question}
ANSWER: {answer}

Decide two things:
1. verdict: "pass" if the answer is correct, responsive and free of invented detail; "rework" otherwise.
2. criterion: the single best-fitting label from this list for what is wrong ("none" if nothing is):
{criteria}

Return strict JSON: {{"verdict": "...", "criterion": "..."}}"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def _text_of(msg: Any) -> str:
    """The text blocks only. Sonnet 5 may put a ThinkingBlock first, which has no .text."""
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()


async def json_call(
    client: AsyncAnthropic, model: str, prompt: str, tries: int = 6
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """One call that must return a JSON object, plus what it cost in tokens."""
    for attempt in range(tries):
        try:
            msg = await client.messages.create(
                model=model, max_tokens=2000, messages=[{"role": "user", "content": prompt}]
            )
            usage = {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens}
            text = _text_of(msg)
            start, end = text.find("{"), text.rfind("}")
            parsed = json.loads(text[start : end + 1]) if start >= 0 else None
            return parsed, usage
        except APIStatusError as err:
            if err.status_code not in (429, 500, 502, 503, 529) or attempt == tries - 1:
                return None, {}
            await asyncio.sleep(min(2**attempt + random.random(), 30))
        except Exception:
            return None, {}
    return None, {}


async def generate(client: AsyncAnthropic, target: int, concurrency: int = 10) -> None:
    """Top the generated corpus up to `target` items."""
    have = {row["id"] for row in read_jsonl(GENERATED)}
    todo = [i for i in range(target) if i not in have]
    print(f"stage 1: have {len(have)}, generating {len(todo)}")
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    defects = [k for k in CRITERIA if k != "none"]

    async def one(index: int) -> None:
        topic = random.choice(TOPICS)
        planted = "none" if index % 3 == 0 else random.choice(defects)
        prompt = (
            CLEAN_PROMPT.format(topic=topic) if planted == "none"
            else GEN_PROMPT.format(topic=topic, defect=planted, description=CRITERIA[planted])
        )
        async with sem:
            pair, _ = await json_call(client, GENERATOR_MODEL, prompt)
        if pair and "question" in pair and "answer" in pair:
            async with lock:
                append_jsonl(GENERATED, {"id": index, "topic": topic, "planted": planted, **pair})

    await asyncio.gather(*(one(i) for i in todo))
    print(f"stage 1 complete: {len(read_jsonl(GENERATED))} generated")


async def gold_label(client: AsyncAnthropic, concurrency: int = 4) -> None:
    """Grade every generated item that has no gold label yet."""
    done = {row["id"] for row in read_jsonl(JUDGED)}
    todo = [row for row in read_jsonl(GENERATED) if row["id"] not in done]
    print(f"stage 2: have {len(done)}, labelling {len(todo)}")
    listing = "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    failures = 0

    async def one(item: dict[str, Any]) -> None:
        nonlocal failures
        prompt = GOLD_PROMPT.format(question=item["question"], answer=item["answer"], criteria=listing)
        async with sem:
            got, usage = await json_call(client, GOLD_MODEL, prompt)
        verdict, criterion = (got or {}).get("verdict"), (got or {}).get("criterion")
        if verdict not in {"pass", "rework"} or criterion not in CRITERIA:
            failures += 1
            return
        async with lock:
            append_jsonl(JUDGED, {
                **item, "gold_verdict": verdict, "gold_criterion": criterion,
                "gold_tokens_in": usage.get("in", 0), "gold_tokens_out": usage.get("out", 0),
            })

    await asyncio.gather(*(one(item) for item in todo))
    print(f"stage 2 complete: {len(read_jsonl(JUDGED))} judged, {failures} failed")


def report() -> None:
    rows = read_jsonl(JUDGED)
    recovered = sum(1 for r in rows if r["planted"] == r["gold_criterion"])
    agreed = sum(1 for r in rows if (r["planted"] == "none") == (r["gold_verdict"] == "pass"))
    rework = sum(1 for r in rows if r["gold_verdict"] == "rework")
    print(f"\nitems: {len(rows)}   gold says rework: {rework} ({rework/len(rows):.1%})")
    print(f"gold recovered the planted criterion: {recovered}/{len(rows)} ({recovered/len(rows):.1%})")
    print(f"gold verdict matched planted:         {agreed}/{len(rows)} ({agreed/len(rows):.1%})")
    print(f"distinct gold criteria present: {len({r['gold_criterion'] for r in rows})} of {len(CRITERIA)}")
    tin = sum(r.get("gold_tokens_in", 0) for r in rows) / len(rows)
    tout = sum(r.get("gold_tokens_out", 0) for r in rows) / len(rows)
    usd = (tin * 3.0 + tout * 15.0) / 1_000_000
    print(f"measured gold judge: {tin:.0f} in / {tout:.0f} out per item = ${usd*1000:.2f} per 1k items")


async def main(target: int = 500) -> None:
    client = AsyncAnthropic()
    await generate(client, target)
    await gold_label(client)
    report()


if __name__ == "__main__":
    asyncio.run(main())
