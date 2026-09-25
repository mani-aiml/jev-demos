"""Small, readable steps for the recorded notebook. Scoring, questions and the Jev run are
reused from the harness modules, so the notebook shows the same code that produced the
numbers rather than a second implementation of it."""

import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import collections
import platform
import statistics
import time
from importlib.metadata import version
from typing import Any

import compare
from judge_task import CRITERIA, VERDICT_OPTIONS, laya_questions, state_for
from run_engines import _latency_summary, load_env, load_items, run_jev

__all__ = [
    "CRITERIA", "VERDICT_OPTIONS", "QUESTIONS", "versions", "load_items", "corpus_summary", "load_laya",
    "time_laya", "run_laya", "score", "four_framings", "budget_check", "criterion_three_ways",
    "jev_single_call", "run_jev", "run_jev_short", "show_confidence_source", "cascade_table",
]

QUESTIONS = laya_questions()
SHORT_CRITERIA = {k: k.replace("_", " ") for k in CRITERIA}


class _Tee:
    """Mirror everything printed to a log, so the recorded run's numbers can be read back."""

    def __init__(self, stream, path: str):
        self.stream, self.log = stream, open(path, "a")

    def write(self, text: str) -> int:
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def __getattr__(self, name: str):
        return getattr(self.stream, name)


def versions() -> None:
    import os
    import sys

    os.makedirs("recording", exist_ok=True)
    sys.stdout = _Tee(sys.stdout, "recording/printed.log")
    load_env()
    for pkg in ("laya", "laya-mlx", "typesafe-sdk"):
        print(f"{pkg:14s} {version(pkg)}")
    print(f"{'machine':14s} {platform.machine()}, {platform.mac_ver()[0] or platform.system()}")


def corpus_summary(items: list[dict[str, Any]]) -> None:
    n = len(items)
    rework = sum(r["gold_verdict"] == "rework" for r in items)
    none = sum(r["gold_criterion"] == "none" for r in items)
    planted_v = sum((r["planted"] == "none") == (r["gold_verdict"] == "pass") for r in items)
    planted_c = sum(r["planted"] == r["gold_criterion"] for r in items)
    usd = sum(r["gold_tokens_in"] * 3 + r["gold_tokens_out"] * 15 for r in items) / n / 1e6
    print(f"items: {n}")
    print(f"reference judge says rework: {rework/n:.1%}   says none: {none/n:.1%}")
    print(f"reference judge vs what I planted: verdict {planted_v/n:.1%}, exact defect {planted_c/n:.1%}")
    print(f"reference judge cost: ${usd*1000:.2f} per 1,000 items (measured tokens)")


def load_laya():
    """Load quietly, then re-show only Laya's own calibration warning, which the video cites.

    The kernel prints library warnings with full file paths and the loader draws a download
    bar; neither says anything, so both are captured and dropped here.
    """
    import contextlib
    import io
    import sys

    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("always")
        import laya_mlx

        start = time.perf_counter()
        agent = laya_mlx.load("convaiinnovations/laya", subfolder="typed-decisions")
        elapsed = time.perf_counter() - start
    for w in caught:
        if "temperatures" in str(w.message):
            print(f"{w.category.__name__}: {w.message}", file=sys.stderr)
    print(f"typed-decisions checkpoint loaded in {elapsed:.1f} s")
    return agent


def time_laya(agent, items: list[dict[str, Any]], n: int = 50) -> None:
    q = {"verdict": QUESTIONS["verdict"]}
    agent.predict(state_for(items[0]["question"], items[0]["answer"]), q)
    lat = []
    for it in items[:n]:
        start = time.perf_counter()
        agent.predict(state_for(it["question"], it["answer"]), q)
        lat.append((time.perf_counter() - start) * 1000)
    print(f"one 2-option call, median of {n}: {statistics.median(lat):.1f} ms (local, no network)")


def run_laya(agent, items, questions: dict[str, Any]) -> dict[str, Any]:
    rows, lat = [], []
    for it in items:
        start = time.perf_counter()
        answers = agent.predict(state_for(it["question"], it["answer"]), questions)["answers"]
        lat.append((time.perf_counter() - start) * 1000)
        row = {"id": it["id"]}
        for key in questions:
            row[key] = answers[key]["choice"].replace(" ", "_")
            row[f"{key}_conf"] = float(answers[key]["confidence"])
        rows.append(row)
    return {"rows": rows, "latency_ms": _latency_summary(lat)}


def score(run: dict[str, Any], items, field: str) -> None:
    by_id = {r["id"]: r for r in items}
    value = compare.agreement(run["rows"], by_id, field)
    picks = collections.Counter(r[field] for r in run["rows"])
    p50 = run.get("latency_ms", {}).get("p50")
    timing = f"   p50 {p50} ms" if isinstance(p50, (int, float)) else ""
    print(f"{field}: agreement with reference {value:.3f}{timing}   top picks {picks.most_common(2)}")


def four_framings(agent, items) -> None:
    ins = "Judge the ANSWER against the QUESTION. Is it acceptable as it stands?"
    variants = {
        "original, pass listed first": (ins, {"pass": VERDICT_OPTIONS["pass"], "rework": VERDICT_OPTIONS["rework"]}),
        "flipped, rework listed first": (ins, {"rework": VERDICT_OPTIONS["rework"], "pass": VERDICT_OPTIONS["pass"]}),
        "short labels": ("Is this answer correct and complete?", {"pass": "correct", "rework": "has an error"}),
        "negated framing": ("Does the ANSWER contain a factual error, an invented detail, or an evasion?",
                            {"rework": "yes, it has a defect", "pass": "no, it is clean"}),
    }
    for name, (text, opts) in variants.items():
        run = run_laya(agent, items, {"verdict": {"type": "choice", "instructions": text, "criteria": opts}})
        picks = collections.Counter(r["verdict"] for r in run["rows"])
        print(f"{name:30s} pass {picks['pass']:3d} / rework {picks['rework']:3d}")


def budget_check(agent) -> None:
    """Measure with Laya's own prefix builder, so the numbers are what the model receives."""
    from laya_mlx.common import build_prefix

    q = QUESTIONS["criterion"]
    internal = {"t": "choice", "ins": q["instructions"], "crit": q["criteria"]}
    tok, budget = agent.tok, agent.cfg.get("head_max_len", 192)
    wanted = sum(1 + len(tok(" %s: %s" % (k, v), add_special_tokens=False)["input_ids"][:48])
                 for k, v in CRITERIA.items())
    ins_tokens = len(tok("choice question: " + q["instructions"], add_special_tokens=False)["input_ids"])
    ids, markers = build_prefix(tok, internal, budget)
    slots = [b - a for a, b in zip(markers, markers[1:] + [len(ids) - 1])]
    kept_ins = markers[0] - 2
    print(f"shared budget for question + options (head_max_len): {budget} tokens")
    print(f"my 40 options with descriptions want:                {wanted} tokens")
    print(f"each option slot after trimming:                     {min(slots)} to {max(slots)} tokens (1 is the marker)")
    print(f"instruction tokens kept:                             {kept_ins} of {ins_tokens}")


def criterion_three_ways(agent, items) -> dict[str, Any]:
    q = QUESTIONS["criterion"]
    configs = [("long descriptions, budget 256", q, 256),
               ("short labels,      budget 256", {**q, "criteria": SHORT_CRITERIA}, 256),
               ("long descriptions, budget 1024", q, 1024)]
    runs = {}
    for name, question, budget in configs:
        agent.cfg["head_max_len"] = budget
        runs[name] = run_laya(agent, items, {"criterion": question})
        print(name, end="   ")
        score(runs[name], items, "criterion")
    agent.cfg["head_max_len"] = 256
    return runs


async def jev_single_call(items, n: int = 10) -> None:
    from typesafe_sdk import AsyncTypeSafeClient

    from judge_task import jev_questions

    client, lat = AsyncTypeSafeClient(), []
    for it in items[:n]:
        start = time.perf_counter()
        await client.system_one(state=state_for(it["question"], it["answer"]), questions=jev_questions())
        lat.append((time.perf_counter() - start) * 1000)
    print(f"one call, median of {n}, sequential: {statistics.median(lat):.0f} ms (hosted API, includes network)")


async def run_jev_short(items) -> dict[str, Any]:
    from typesafe_sdk import AsyncTypeSafeClient, Choice
    import asyncio

    q = {"criterion": Choice(instructions=QUESTIONS["criterion"]["instructions"], criteria=SHORT_CRITERIA)}
    client, sem = AsyncTypeSafeClient(), asyncio.Semaphore(8)

    async def one(it):
        async with sem:
            r = await client.system_one(state=state_for(it["question"], it["answer"]), questions=q)
        a = r.answers["criterion"]
        return {"id": it["id"], "criterion": a.choice.replace(" ", "_"), "criterion_conf": float(a.confidence)}

    rows = await asyncio.gather(*(one(it) for it in items))
    return {"rows": list(rows)}


def show_confidence_source() -> None:
    import inspect

    import laya

    print(inspect.getsource(laya.confidence_from_probs))
    src = inspect.getsource(laya.agent).splitlines()
    hit = next(i for i, line in enumerate(src) if "confidence_from_probs(p, k)" in line)
    print(f"laya/agent.py line {hit + 1}:  {src[hit].strip()}")


def cascade_table(runs: dict[str, dict[str, Any]], items) -> None:
    compare.load()
    by_id = {r["id"]: r for r in items}
    print(f"all items to the reference judge: ${compare.GOLD_USD_PER_ITEM*1000:.2f} per 1,000\n")
    for name, (run, usd_per_item) in runs.items():
        for field in ("verdict", "criterion"):
            if field not in run["rows"][0]:
                continue
            c = compare.cascade(run["rows"], by_id, field, usd_per_item)
            if c["coverage"] == 0:
                print(f"{name:28s} {field:9s} auto-accept 0.0%   no confidence cut reaches 95%, so the whole bill stays")
                continue
            print(f"{name:28s} {field:9s} auto-accept {c['coverage']:.1%}   blended agreement "
                  f"{c['blended_agreement']:.1%}   ${c['usd_per_1k']:.2f} per 1,000   "
                  f"bill deleted {c.get('saving_vs_all_gold', 0):.1%}")
