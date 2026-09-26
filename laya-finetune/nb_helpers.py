"""Small, readable steps for the notebook. Data preparation, training and scoring are imported
from the modules that produced the numbers, so the notebook shows that code, not a copy of it."""

import asyncio
import collections
import os
import platform
import statistics
import time
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

# tqdm warns that ipywidgets is missing, and the warning prints the venv path (the username)
warnings.filterwarnings("ignore", message="IProgress not found")

from evaluate import score
from finetune_data import SHORT_CRITERIA, TEST_SET, read_jsonl
from judge_task import VERDICT_OPTIONS, state_for
from prepare_data import closest_test_question

DEMO = Path(__file__).parent


def versions() -> None:
    for pkg in ("laya", "laya-mlx", "torch", "typesafe-sdk"):
        print(f"{pkg:14s} {version(pkg)}")
    print(f"{'machine':14s} {platform.machine()}, macOS {platform.mac_ver()[0]}")


def show_item(item: dict[str, Any]) -> None:
    """One graded item: what was asked, what was answered, and the defect that was planted."""
    print("QUESTION:", item["question"])
    print("ANSWER:  ", item["answer"][:400] + ("..." if len(item["answer"]) > 400 else ""))
    print("planted defect (the answer key):", item["planted"])


def test_summary(items: list[dict[str, Any]]) -> None:
    clean = sum(it["planted"] == "none" for it in items)
    print(f"{len(items)} held-out items, {clean} planted clean ({clean / len(items):.0%}), "
          f"{len({it['planted'] for it in items}) - 1} defect types present")


def sources_summary(paths: list[str]) -> None:
    for path in paths:
        rows = read_jsonl(DEMO / path)
        made = collections.Counter(r.get("source", "haiku") for r in rows)
        print(f"{path:38s} {len(rows):5d} rows   {dict(made)}")


def show_near_copy(dropped: list[dict[str, Any]]) -> None:
    """The first dropped question that is reworded rather than identical, beside its test twin."""
    for row in dropped:
        similarity, twin = closest_test_question(row["question"])
        if similarity < 0.95:
            print(f"similarity {similarity:.2f}\n  TRAIN: {row['question']}\n  TEST:  {twin}")
            return


def _load_env() -> None:
    env = DEMO / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _jev_questions() -> dict[str, Any]:
    from typesafe_sdk import Choice

    return {"verdict": Choice(instructions="Judge the ANSWER against the QUESTION. Is it acceptable as it stands?",
                              criteria=VERDICT_OPTIONS),
            "criterion": Choice(instructions="Which single defect best describes what is wrong with the ANSWER?",
                                criteria=SHORT_CRITERIA)}


async def run_jev(items: list[dict[str, Any]], concurrency: int = 8) -> dict[str, Any]:
    """Jev on the same items and the same short labels Laya is scored with."""
    from typesafe_sdk import AsyncTypeSafeClient

    _load_env()
    client, qs, sem = AsyncTypeSafeClient(), _jev_questions(), asyncio.Semaphore(concurrency)
    single = []
    for it in items[:10]:  # latency: ten sequential one-question calls, as Laya is timed
        start = time.perf_counter()
        await client.system_one(state=state_for(it["question"], it["answer"]), questions={"verdict": qs["verdict"]})
        single.append((time.perf_counter() - start) * 1000)

    async def one(it: dict[str, Any]):
        async with sem:
            return it["id"], (await client.system_one(state=state_for(it["question"], it["answer"]), questions=qs)).answers

    answers = dict(await asyncio.gather(*(one(it) for it in items)))
    out = {}
    for name in qs:
        rows = [{"id": i, "pick": a[name].choice, "conf": float(a[name].confidence)} for i, a in answers.items()]
        out[name] = score({"rows": rows, "p50_ms": round(statistics.median(single), 1)}, items, name)
    print(f"Jev one-question call, median of 10 sequential: {out['verdict']['p50_ms']} ms (hosted API, incl. network)")
    return out


def scoreboard(results: dict[str, dict[str, Any]]) -> None:
    print(f"{'':24s} {'pass/rework':>12} {'40 options':>11} {'p50 ms':>8}   (vs the answer key; vs Sonnet in brackets)")
    for name, r in results.items():
        v, c = r["verdict"], r["criterion"]
        print(f"{name:24s} {v['agreement']:>6.3f} ({v['agreement_sonnet']:.3f}) "
              f"{c['agreement']:>5.3f} ({c['agreement_sonnet']:.3f}) {v['p50_ms']:>7}")


def load_test() -> list[dict[str, Any]]:
    return read_jsonl(TEST_SET)
