"""Run both engines over the judged corpus and record answer, certainty and latency per item.

Both get the identical state and the identical two questions in one call, so neither is
paying for round trips the other avoids. The latencies are not like-for-like and the
video must say so: Laya is local compute on this Mac, Jev is a hosted API including
network. They are reported separately, never as a single speedup number.
"""

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

DEMO = Path(__file__).parent
DATA = DEMO / "data" / "judged.jsonl"


def load_env() -> None:
    """Read demo/.env into the environment without needing a dependency for it."""
    import os

    env = DEMO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def load_items() -> list[dict[str, Any]]:
    return [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]


def run_laya(items: list[dict[str, Any]], subfolder: str, device: str = "mps") -> dict[str, Any]:
    """Every item through one Laya checkpoint, timed per call after a warm-up."""
    import laya

    from judge_task import laya_questions, state_for

    questions = laya_questions()
    t0 = time.time()
    agent = laya.load("convaiinnovations/laya", subfolder=subfolder, device=device)
    load_s = time.time() - t0

    agent.predict(state_for(items[0]["question"], items[0]["answer"]), questions)  # warm

    rows, latencies = [], []
    for item in items:
        state = state_for(item["question"], item["answer"])
        start = time.perf_counter()
        result = agent.predict(state, questions)
        latencies.append((time.perf_counter() - start) * 1000)
        answers = result["answers"]
        rows.append({
            "id": item["id"],
            "verdict": answers["verdict"]["choice"],
            "verdict_conf": float(answers["verdict"]["confidence"]),
            "criterion": answers["criterion"]["choice"],
            "criterion_conf": float(answers["criterion"]["confidence"]),
        })
    return {
        "engine": f"laya:{subfolder or 'english'}",
        "device": device,
        "load_s": round(load_s, 1),
        "latency_ms": _latency_summary(latencies),
        "usd": 0.0,
        "rows": rows,
    }


async def run_jev(items: list[dict[str, Any]], concurrency: int = 8) -> dict[str, Any]:
    """Every item through Jev's hosted API, timed end to end."""
    from typesafe_sdk import AsyncTypeSafeClient

    from judge_task import jev_questions, state_for

    questions = jev_questions()
    client = AsyncTypeSafeClient()
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    tokens = 0
    rows: list[dict[str, Any]] = []

    async def one(item: dict[str, Any]) -> None:
        nonlocal tokens
        async with sem:
            start = time.perf_counter()
            response = await client.system_one(
                state=state_for(item["question"], item["answer"]), questions=questions
            )
            latencies.append((time.perf_counter() - start) * 1000)
        tokens += response.usage.input_tokens or 0
        rows.append({
            "id": item["id"],
            "verdict": response.answers["verdict"].choice,
            "verdict_conf": float(response.answers["verdict"].confidence),
            "criterion": response.answers["criterion"].choice,
            "criterion_conf": float(response.answers["criterion"].confidence),
        })

    await asyncio.gather(*(one(item) for item in items))
    rows.sort(key=lambda r: r["id"])
    return {
        "engine": "jev",
        "device": "hosted api",
        "load_s": 0.0,
        "latency_ms": _latency_summary(latencies),
        "usd": round(tokens * 0.042 / 1_000_000, 6),
        "input_tokens": tokens,
        "rows": rows,
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50": round(statistics.median(ordered), 1),
        "p90": round(ordered[int(len(ordered) * 0.9)], 1),
        "mean": round(statistics.fmean(ordered), 1),
        "n": len(ordered),
    }


OUT = DEMO / "data" / "engine_runs.json"


def _save(result: dict[str, Any]) -> None:
    """Persist as each engine finishes, so one failure never discards the others."""
    existing = json.loads(OUT.read_text()) if OUT.exists() else []
    kept = [r for r in existing if r["engine"] != result["engine"]]
    OUT.write_text(json.dumps(kept + [result], indent=2))
    print(f"  saved {result['engine']}")


async def main() -> None:
    load_env()
    items = load_items()
    print(f"items: {len(items)}")

    for subfolder in ("typed-decisions", None):
        print(f"--- laya:{subfolder or 'english'} ---")
        result = run_laya(items, subfolder=subfolder)
        print(json.dumps(result["latency_ms"]))
        _save(result)

    print("--- jev ---")
    result = await run_jev(items)
    print(json.dumps(result["latency_ms"]), f"${result['usd']}")
    _save(result)


if __name__ == "__main__":
    asyncio.run(main())
