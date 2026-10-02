"""Score a Strands Decider checkpoint on the 500 held-out items Laya and Jev were scored on.

Same two questions, same short labels and the same planted answer key as
laya-finetune/demo/evaluate.py, so the three engines sit in one table. `judge_task.py` is that
video's file, copied unchanged. Each question is asked on its own, timed per call after a warm-up.

    python decider_eval.py StrandsAgents/strands-decider-2B-hobson-v19 data/test_judged.jsonl
    python decider_eval.py runs/<fine-tuned checkpoint> data/test_judged.jsonl
"""

import collections
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from strands_decider.infer import load_engine

from judge_task import CRITERIA, laya_questions, state_for

COVERAGE = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
RUNS = Path(__file__).parent / "runs"
SHORT_CRITERIA = {k: k.replace("_", " ") for k in CRITERIA}


def questions() -> dict[str, dict[str, Any]]:
    """Both questions, the 40-way one with short labels, as Laya and Jev were asked."""
    qs = laya_questions()
    qs["criterion"] = {**qs["criterion"], "criteria": SHORT_CRITERIA}
    return qs


def answer_key(row: dict[str, Any], reference: str) -> dict[str, str]:
    """What was planted, or what the Sonnet judge said."""
    if reference == "sonnet":
        return {"verdict": row["gold_verdict"], "criterion": row["gold_criterion"]}
    return {"verdict": "pass" if row["planted"] == "none" else "rework", "criterion": row["planted"]}


def ask(engine, items: list[dict[str, Any]], name: str, question: dict[str, Any]) -> dict[str, Any]:
    """Every item through one question, timed per call after a warm-up call."""
    engine.ask(state_for(items[0]["question"], items[0]["answer"]), {name: question})
    rows, latency = [], []
    for it in items:
        start = time.perf_counter()
        answer = engine.ask(state_for(it["question"], it["answer"]), {name: question}).answers[name]
        latency.append((time.perf_counter() - start) * 1000)
        rows.append({"id": it["id"], "pick": answer.choice.replace(" ", "_"),
                     "conf": float(answer.confidence), "ms": round(latency[-1], 2)})
    return {"rows": rows, "p50_ms": round(statistics.median(latency), 1)}


def agreement(rows: list[dict[str, Any]], key: dict[int, str]) -> float:
    return sum(r["pick"] == key[r["id"]] for r in rows) / len(rows)


def risk_coverage(rows: list[dict[str, Any]], key: dict[int, str]) -> list[tuple[float, float]]:
    """Agreement on the most confident share of items, for each share in COVERAGE."""
    ranked = sorted(rows, key=lambda r: r["conf"], reverse=True)
    return [(share, round(agreement(ranked[: max(1, int(len(ranked) * share))], key), 4)) for share in COVERAGE]


def score(run: dict[str, Any], items: list[dict[str, Any]], name: str) -> dict[str, Any]:
    planted = {it["id"]: answer_key(it, "planted")[name] for it in items}
    sonnet = {it["id"]: answer_key(it, "sonnet")[name] for it in items}
    return {"agreement": round(agreement(run["rows"], planted), 3),
            "agreement_sonnet": round(agreement(run["rows"], sonnet), 3),
            "p50_ms": run["p50_ms"],
            "top_picks": collections.Counter(r["pick"] for r in run["rows"]).most_common(3),
            "risk_coverage": risk_coverage(run["rows"], planted),
            "rows": run["rows"]}


def evaluate(checkpoint: str, test_set: str) -> dict[str, Any]:
    """Both questions for one checkpoint, printed and saved to runs/eval_<name>.json."""
    items = [json.loads(line) for line in Path(test_set).read_text().splitlines() if line.strip()]
    engine, out = load_engine(checkpoint), {}
    for name, question in questions().items():
        out[name] = score(ask(engine, items, name, question), items, name)
        s = out[name]
        print(f"{name:9s} vs key {s['agreement']:.3f}  vs Sonnet {s['agreement_sonnet']:.3f}  "
              f"p50 {s['p50_ms']} ms  top {s['top_picks']}", flush=True)
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"eval_{Path(checkpoint).name}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])
