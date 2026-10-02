"""Laya's 2,082 judging rows as Strands Decider training examples, split and weighted exactly as Laya's.

Reuses laya-finetune/demo's own pieces, so both models see the same rows, labels and weights:
the shuffle seed and 10% calibration share of its train.py, `weight_to_design` (one clean answer
in three, the generation design), the planted labels and the short 40-way labels.

The output follows Decider's own on-disk format (strands_decider/data/format.py `Example`, as in
its committed data/synthetic/*.jsonl): one row per (state, question), `options` as [name,
description] pairs in canonical order, `label` an index into them, `weight` read by the loss.
The state is the same {"QUESTION", "ANSWER"} dict the evaluation sends.

    python build_decider_data.py [seed]   # in laya-finetune's environment: it needs laya, for finetune_data

The seed sets the row order and so the calibration split, as in Laya's train.py; a seed other than
the default writes judge_<part>_s<seed>.jsonl.
"""

import collections
import json
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
# laya-finetune's code: its sibling folder in jev-demos, or its demo/ folder in the video workspace.
LAYA = next(p for p in (HERE.parent / "laya-finetune", HERE.parents[1] / "laya-finetune" / "demo")
            if (p / "finetune_data.py").exists())
sys.path.insert(0, str(LAYA))
from finetune_data import labels, questions, read_jsonl  # noqa: E402
from judge_task import state_for  # noqa: E402
from prepare_data import weight_to_design  # noqa: E402

CALIB_SHARE, SEED = 0.1, 20260924  # laya-finetune/demo/train.py
OUT = HERE / "data"


def examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Two examples per row, one per question, each carrying the row's design weight."""
    out = []
    for row in rows:
        key = labels(row, "planted")
        for name, q in questions("short").items():
            options = [[k, v] for k, v in q["criteria"].items()]
            out.append({"kind": "choice", "state": state_for(row["question"], row["answer"]),
                        "instructions": q["instructions"], "options": options,
                        "label": [k for k, _ in options].index(key[name]),
                        "task": f"judge:{name}", "weight": row["weight"], "instruction_variants": []})
    return out


def report(name: str, ex: list[dict[str, Any]]) -> None:
    """Raw and weighted clean share per question, so the mix is checked, not assumed."""
    for q in ("verdict", "criterion"):
        rows = [e for e in ex if e["task"] == f"judge:{q}"]
        clean = [e for e in rows if e["options"][e["label"]][0] in ("pass", "none")]
        raw, wtd = len(clean) / len(rows), sum(e["weight"] for e in clean) / sum(e["weight"] for e in rows)
        print(f"{name:5s} {q:9s} {len(rows):5d} examples  clean raw {raw:.1%}  weighted {wtd:.1%}")


def main(seed: int = SEED) -> None:
    rows = read_jsonl(LAYA / "data" / "train_all.jsonl")
    random.Random(seed).shuffle(rows)
    weight_to_design(rows)
    n_calib = int(len(rows) * CALIB_SHARE)
    OUT.mkdir(exist_ok=True)
    suffix = "" if seed == SEED else f"_s{seed}"
    for name, part in (("calib", rows[:n_calib]), ("train", rows[n_calib:])):
        ex = examples(part)
        (OUT / f"judge_{name}{suffix}.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ex))
        report(name, ex)
    types = collections.Counter(r["planted"] for r in rows[n_calib:] if r["planted"] != "none")
    print(f"train defect types {len(types)}, rows per type {min(types.values())} to {max(types.values())}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else SEED)
