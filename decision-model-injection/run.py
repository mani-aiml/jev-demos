"""Run every message, clean and attacked, through Jev, Laya and Claude. Appends to runs/raw.jsonl.

    python run.py --engine laya --repeats 1
    python run.py --engine jev --repeats 3
    python run.py --engine claude --repeats 3
    python run.py --engine jev --detector        (the injection-check question on its own)
    python run.py --engine jev --followups       (the fake-answer follow-up attacks only)
    python run.py --engine jev --source llmail --repeats 3             (LLMail-Inject sample)
    python run.py --engine claude --source llmail --repeats 3 --detector
"""

import argparse
import asyncio
import functools
import json
import time

from engines import CLAUDE_MODEL, DEMO, ask_claude, ask_jev, ask_laya, load_env
from task import DETECTOR, FOLLOWUPS, QUESTIONS, QUESTIONS_EXPLICIT, QUESTIONS_NARROW, variants

RAW = DEMO / "runs" / "raw.jsonl"
LLMAIL = DEMO / "data" / "llmail_sample.jsonl"
LLMAIL_RAW = DEMO / "runs" / "llmail_raw.jsonl"
HELDOUT = DEMO / "data" / "llmail_heldout.jsonl"
HELDOUT_RAW = DEMO / "runs" / "llmail_heldout_raw.jsonl"
LAYA_ID = "convaiinnovations/laya"


def save(rows: list[dict]) -> None:
    """LLMail results go to their own file so report.py keeps reading only our twelve emails."""
    path = {"llmail": LLMAIL_RAW, "heldout": HELDOUT_RAW}.get(rows[0]["source"], RAW)
    with path.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def record(engine: str, rep: int, v: dict, answers: dict, ms: float, qset: str) -> dict:
    return {"engine": engine, "rep": rep, "qset": qset, "mid": v["mid"], "attack": v["attack"],
            "gold_human": v["gold_human"], "gold_route": v["gold_route"], "ms": round(ms, 1),
            "answers": answers, "source": v.get("source", "handwritten"), "meta": v.get("meta", {})}


def run_laya(repeats: int, questions: dict, qset: str, subfolder: str | None, items: list[dict]) -> None:
    import laya

    agent = laya.load(LAYA_ID, subfolder=subfolder, device="mps")
    engine = f"laya:{subfolder or 'english'}"
    ask_laya(agent, items[0]["text"], questions)  # warm-up
    for rep in range(repeats):
        rows = []
        for v in items:
            start = time.perf_counter()
            answers = ask_laya(agent, v["text"], questions)
            rows.append(record(engine, rep, v, answers, (time.perf_counter() - start) * 1000, qset))
        save(rows)
        print(f"{engine} rep {rep}: {len(rows)} rows")


async def run_api(engine: str, repeats: int, questions: dict, qset: str, concurrency: int, items: list[dict],
                  claude_model: str = CLAUDE_MODEL) -> None:
    if engine == "jev":
        from typesafe_sdk import AsyncTypeSafeClient
        client, ask = AsyncTypeSafeClient(), ask_jev
    else:
        from anthropic import AsyncAnthropic
        client, ask = AsyncAnthropic(), functools.partial(ask_claude, model=claude_model)
        engine = engine if claude_model == CLAUDE_MODEL else claude_model
    sem = asyncio.Semaphore(concurrency)

    async def one(rep: int, v: dict) -> dict | None:
        async with sem:
            start = time.perf_counter()
            try:
                answers = await ask(client, v["text"], questions)
            except Exception as error:  # one bad item must not throw away a paid batch
                print(f"  {engine} rep {rep} {v['mid']}: {type(error).__name__}: {str(error)[:120]}")
                return None
            return record(engine, rep, v, answers, (time.perf_counter() - start) * 1000, qset)

    for rep in range(repeats):
        rows = [r for r in await asyncio.gather(*(one(rep, v) for v in items)) if r]
        if rows:
            save(rows)
        print(f"{engine} rep {rep}: {len(rows)} of {len(items)} rows")


def load_items(source: str, followups: bool) -> list[dict]:
    """Our twelve emails with their attacks, or the LLMail-Inject sample (llmail_sample.py)."""
    if source in ("llmail", "heldout"):
        path = LLMAIL if source == "llmail" else HELDOUT
        if not path.exists():
            raise SystemExit("run llmail_sample.py first (with --heldout for the fresh set)")
        return [json.loads(line) | {"source": source} for line in path.read_text().splitlines()]
    return variants(FOLLOWUPS) if followups else variants()


def unanswered(items: list[dict], engine: str, qset: str) -> list[dict]:
    """Emails this engine has not answered for this question set yet (a run cut off by a limit)."""
    done = set()
    if LLMAIL_RAW.exists():
        for line in LLMAIL_RAW.read_text().splitlines():
            r = json.loads(line)
            if r["engine"] == engine and r["qset"] == qset:
                done.add(r["mid"])
    return [v for v in items if v["mid"] not in done]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["jev", "laya", "claude"], required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--detector", action="store_true")
    p.add_argument("--laya-subfolder", default=None)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--followups", action="store_true", help="only the FOLLOWUPS attacks")
    p.add_argument("--source", choices=["handwritten", "llmail", "heldout"], default="handwritten")
    p.add_argument("--resume", action="store_true", help="llmail only: skip emails already answered")
    p.add_argument("--explicit", action="store_true", help="needs_human criteria that name send/forward/share")
    p.add_argument("--narrow", action="store_true", help="explicit, limited to someone other than the customer")
    p.add_argument("--claude-model", default=CLAUDE_MODEL, help="e.g. claude-haiku-4-5-20251001")
    args = p.parse_args()
    load_env()
    questions, qset = (DETECTOR, "detector") if args.detector else (QUESTIONS, "main")
    if args.explicit:
        questions, qset = QUESTIONS_EXPLICIT, "main_explicit"
    if args.narrow:
        questions, qset = QUESTIONS_NARROW, "main_narrow"
    items = load_items(args.source, args.followups)
    if args.resume and args.source == "llmail":  # resume covers the main sample only
        items = unanswered(items, args.engine if args.claude_model == CLAUDE_MODEL or args.engine != "claude"
                           else args.claude_model, qset)
        print(f"resume: {len(items)} emails left")
    if args.engine == "laya":
        run_laya(args.repeats, questions, qset, args.laya_subfolder, items)
    else:
        asyncio.run(run_api(args.engine, args.repeats, questions, qset, args.concurrency, items, args.claude_model))


if __name__ == "__main__":
    main()
