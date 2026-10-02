"""The verdict asked in the exact form Decider's adequacy training used (data/adequacy.py, v19):
a noul over "Request:/Response:" text. Reports accuracy at 0.5 and AUC, which needs no threshold.

    python diag_native.py data/test_judged.jsonl
"""

import json
import sys
from pathlib import Path
from typing import Any

from strands_decider.infer import load_engine

from decider_eval import answer_key, questions
from judge_task import state_for

CKPT = "StrandsAgents/strands-decider-2B-hobson-v19"
ADEQUATE = {"false": "the response is not adequate", "true": "the response is adequate"}
VARIANTS = [  # data/adequacy.py QUESTION and VARIANTS, verbatim
    "Does the response adequately answer the request: correct, complete and on point?",
    "Is this response good enough to send as the answer to the request?",
    "Would a careful reviewer accept this response as an adequate answer to the request?",
    "Does the response do what the request asks, without significant errors or omissions?",
]


def native_state(it: dict[str, Any]) -> str:
    return f"Request:\n{it['question'].strip()}\n\nResponse:\n{it['answer'].strip()}"


def auc(p_ok: dict[int, float], key: dict[int, str]) -> float:
    """Chance that a random pass item scores above a random rework item."""
    good = [p_ok[i] for i in p_ok if key[i] == "pass"]
    bad = [p_ok[i] for i in p_ok if key[i] == "rework"]
    return sum((g > b) + 0.5 * (g == b) for g in good for b in bad) / (len(good) * len(bad))


def report(name: str, p_ok: dict[int, float], key: dict[int, str]) -> None:
    picks = {i: "pass" if p >= 0.5 else "rework" for i, p in p_ok.items()}
    acc = sum(picks[i] == key[i] for i in picks) / len(picks)
    caught = sum(picks[i] == "rework" for i in picks if key[i] == "rework")
    print(f"{name:62s} acc {acc:.3f}  caught {caught:3d}/333  AUC {auc(p_ok, key):.3f}", flush=True)


def main(test_set: str) -> None:
    items = [json.loads(line) for line in Path(test_set).read_text().splitlines() if line.strip()]
    key = {it["id"]: answer_key(it, "planted")["verdict"] for it in items}
    eng = load_engine(CKPT)
    verdict = questions()["verdict"]
    p_pass = {it["id"]: eng.ask(state_for(it["question"], it["answer"]), {"q": verdict}).answers["q"].probabilities["pass"]
              for it in items}
    report("as scored: choice pass/rework, JSON state", p_pass, key)
    for text in VARIANTS:
        q = {"type": "noul", "instructions": text, "criteria": ADEQUATE}
        p_ok = {it["id"]: eng.ask(native_state(it), {"q": q}).answers["q"].noul for it in items}
        report(f"native noul: {text[:48]}", p_ok, key)


if __name__ == "__main__":
    main(sys.argv[1])
