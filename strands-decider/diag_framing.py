"""Is the low out-of-the-box verdict score the harness or the model? Same 500 items, equivalent framings.

    python diag_framing.py data/test_judged.jsonl
"""

import collections
import json
import sys
from pathlib import Path
from typing import Any, Callable

from strands_decider.infer import load_engine

from decider_eval import answer_key, questions
from judge_task import VERDICT_OPTIONS, state_for

CKPT = "StrandsAgents/strands-decider-2B-hobson-v19"
DEFECT_NOUL = ("Does the ANSWER have a problem: is it wrong, evasive, incomplete, off-topic, "
               "or does it state something it cannot support?")
ACCEPT_NOUL = "Is the ANSWER factually correct, responsive to the QUESTION, and free of invented detail?"


def json_state(it: dict[str, Any]) -> Any:
    return state_for(it["question"], it["answer"])


def text_state(it: dict[str, Any]) -> str:
    return f"QUESTION: {it['question']}\n\nANSWER: {it['answer']}"


def run(eng, items, q: dict[str, Any], state: Callable = json_state, noul_rework_if: bool | None = None) -> dict[int, str]:
    """Verdict per item. For a noul question, `noul_rework_if` is the answer that means rework."""
    picks = {}
    for it in items:
        a = eng.ask(state(it), {"q": q}).answers["q"]
        if noul_rework_if is None:
            picks[it["id"]] = a.choice
        else:
            picks[it["id"]] = "rework" if (a.noul >= 0.5) == noul_rework_if else "pass"
    return picks


def main(test_set: str) -> None:
    items = [json.loads(line) for line in Path(test_set).read_text().splitlines() if line.strip()]
    key = {it["id"]: answer_key(it, "planted")["verdict"] for it in items}
    verdict = questions()["verdict"]
    eng = load_engine(CKPT)
    base = run(eng, items, verdict)
    uncached = run(load_engine(CKPT, use_prefix_cache=False), items, verdict)
    print(f"prefix cache on vs off: {sum(base[i] == uncached[i] for i in base)}/500 identical picks")
    variants = {
        "as scored (JSON state, pass first)": base,
        "options reversed (rework first)": run(eng, items, {**verdict, "criteria": dict(reversed(VERDICT_OPTIONS.items()))}),
        "plain-text state": run(eng, items, verdict, state=text_state),
        "noul: does the ANSWER have a problem?": run(eng, items, {"type": "noul", "instructions": DEFECT_NOUL}, noul_rework_if=True),
        "noul: is the ANSWER acceptable?": run(eng, items, {"type": "noul", "instructions": ACCEPT_NOUL}, noul_rework_if=False),
    }
    for name, p in variants.items():
        acc = sum(p[i] == key[i] for i in p) / len(p)
        caught = sum(p[i] == "rework" for i in p if key[i] == "rework")
        print(f"{name:40s} acc {acc:.3f}  caught {caught:3d}/333  said rework {sum(v == 'rework' for v in p.values())}")
    by = collections.defaultdict(lambda: [0, 0])
    for it in items:
        if it["planted"] != "none":
            by[it["planted"]][0] += base[it["id"]] == "rework"
            by[it["planted"]][1] += 1
    print("\nas scored, defects caught per planted type:")
    for t, (c, n) in sorted(by.items(), key=lambda x: -x[1][0] / x[1][1]):
        print(f"  {t:28s} {c}/{n}")


if __name__ == "__main__":
    main(sys.argv[1])
