"""Score a Laya checkpoint on the 500 held-out test items, on the MLX runtime.

Each question is asked on its own, so agreement and latency compare like-for-like across
checkpoints. The answer key is the planted label; agreement with the Sonnet reference judge is
reported alongside it for continuity with earlier zero-shot numbers.

    python evaluate.py base                 # the zero-shot typed-decisions checkpoint
    python evaluate.py runs/all2082         # a fine-tuned checkpoint
"""

import collections
import json
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any

from finetune_data import MODEL_ID, SUBFOLDER, TEST_SET, labels, questions, read_jsonl
from judge_task import state_for

COVERAGE = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
RUNS = Path(__file__).parent / "runs"


def load(model: str):
    """Load on MLX. Warnings are re-shown as their message only: the default form prints the
    library's full path, which carries the local username into a recording."""
    import laya_mlx

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = (laya_mlx.load(MODEL_ID, subfolder=SUBFOLDER) if model == "base"
                 else laya_mlx.load(str(Path(model).resolve())))
    for w in caught:
        print(f"{w.category.__name__}: {w.message}", file=sys.stderr)
    return agent


def ask(agent, items: list[dict[str, Any]], name: str, question: dict[str, Any]) -> dict[str, Any]:
    """Every item through one question, timed per call after a warm-up call."""
    agent.predict(state_for(items[0]["question"], items[0]["answer"]), {name: question})
    rows, latency = [], []
    for it in items:
        start = time.perf_counter()
        answer = agent.predict(state_for(it["question"], it["answer"]), {name: question})["answers"][name]
        latency.append((time.perf_counter() - start) * 1000)
        rows.append({"id": it["id"], "pick": answer["choice"].replace(" ", "_"),
                     "conf": float(answer["confidence"]), "ms": round(latency[-1], 2)})
    return {"rows": rows, "p50_ms": round(statistics.median(latency), 1)}


def agreement(rows: list[dict[str, Any]], key: dict[int, str]) -> float:
    return sum(r["pick"] == key[r["id"]] for r in rows) / len(rows)


def risk_coverage(rows: list[dict[str, Any]], key: dict[int, str]) -> list[tuple[float, float]]:
    """Agreement on the most confident share of items, for each share in COVERAGE."""
    ranked = sorted(rows, key=lambda r: r["conf"], reverse=True)
    return [(share, round(agreement(ranked[: max(1, int(len(ranked) * share))], key), 4)) for share in COVERAGE]


def score(run: dict[str, Any], items: list[dict[str, Any]], name: str) -> dict[str, Any]:
    planted = {it["id"]: labels(it, "planted")[name] for it in items}
    sonnet = {it["id"]: labels(it, "sonnet")[name] for it in items}
    return {"agreement": round(agreement(run["rows"], planted), 3),
            "agreement_sonnet": round(agreement(run["rows"], sonnet), 3),
            "p50_ms": run["p50_ms"],
            "top_picks": collections.Counter(r["pick"] for r in run["rows"]).most_common(3),
            "risk_coverage": risk_coverage(run["rows"], planted),
            "rows": run["rows"]}


def evaluate(model: str) -> dict[str, Any]:
    """Both questions for one checkpoint, printed and saved to runs/eval_<name>.json."""
    items, agent, out = read_jsonl(TEST_SET), load(model), {}
    for name, question in questions("short").items():
        out[name] = score(ask(agent, items, name, question), items, name)
        s = out[name]
        print(f"{name:9s} vs key {s['agreement']:.3f}  vs Sonnet {s['agreement_sonnet']:.3f}  "
              f"p50 {s['p50_ms']} ms  top {s['top_picks']}")
    tag = "base" if model == "base" else Path(model).name
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"eval_{tag}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    evaluate(sys.argv[1])
