"""Time Jev on every held-out item, one call at a time, and keep each pick, clock and token count.

The notebook keeps only Jev's scores and a median of ten calls. The video's grading showdown needs
each answer's own pick and time, and the tokens that price it, so this runs the pass/rework question
sequentially, the way Laya is timed, and saves every row.

    python jev_timing.py            # writes runs/jev_verdict_timed.json
"""
import asyncio
import json
import statistics
import time
from pathlib import Path

from evaluate import score
from judge_task import state_for
from nb_helpers import _jev_questions, _load_env, load_test

OUT = Path(__file__).parent / "runs" / "jev_verdict_timed.json"
PRICE_PER_INPUT_TOKEN = 0.042 / 1e6  # TypeSafe blog, 15 Sep 2026; output tokens are free


async def main() -> None:
    from typesafe_sdk import AsyncTypeSafeClient

    _load_env()
    client, question, items = AsyncTypeSafeClient(), {"verdict": _jev_questions()["verdict"]}, load_test()
    rows = []
    for it in items:
        start = time.perf_counter()
        r = await client.system_one(state=state_for(it["question"], it["answer"]), questions=question)
        ms = (time.perf_counter() - start) * 1000
        a = r.answers["verdict"]
        rows.append({"id": it["id"], "pick": a.choice, "conf": float(a.confidence), "ms": round(ms, 2),
                     "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens})
    result = score({"rows": rows, "p50_ms": round(statistics.median(x["ms"] for x in rows), 1)}, items, "verdict")
    tokens = sum(x["input_tokens"] for x in rows)
    result.update(measured=time.strftime("%Y-%m-%d %H:%M"), total_ms=round(sum(x["ms"] for x in rows), 1),
                  input_tokens=tokens, cost_usd=round(tokens * PRICE_PER_INPUT_TOKEN, 5))
    OUT.write_text(json.dumps(result, indent=1))
    print(f"Jev, pass/rework vs key {result['agreement']:.3f}, p50 {result['p50_ms']} ms, "
          f"500 answers in {result['total_ms'] / 1000:.1f} s, {tokens} input tokens, ${result['cost_usd']}")


if __name__ == "__main__":
    asyncio.run(main())
