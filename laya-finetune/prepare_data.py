"""Data transformation, run before any training: combine sources, drop test copies, fix the mix.

    python prepare_data.py --out data/train_all.jsonl --plot plots/mix_train_all.png \\
        data/train_judged.jsonl data/train_dropped_near_test.jsonl data/hybrid_haiku.jsonl data/hybrid.jsonl

Three stages, each shown in the table and the plot so a skew is caught before training:
1. combine every generated source (the first file wins on a duplicate id);
2. drop any row whose question is a near-copy of a test question;
3. weight rows back to the generation design (`weight_to_design`, applied again at train time
   by `train.py --mix`). Weights are not written to the output; the design is.
"""

import argparse
import collections
import copy
import difflib
import json
import re
from pathlib import Path
from typing import Any

from finetune_data import DESIGN_CLEAN_SHARE, TEST_SET, read_jsonl

NEAR_TEST = 0.8  # question similarity at which a training row counts as a test copy


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _test_questions() -> list[str]:
    return [normalise(r["question"]) for r in read_jsonl(TEST_SET)]


def is_near(question: str, tests: list[str], cutoff: float = NEAR_TEST) -> bool:
    q = normalise(question)
    for t in tests:  # the quick ratios are upper bounds on ratio(), so they only skip misses
        sm = difflib.SequenceMatcher(None, q, t)
        if sm.real_quick_ratio() >= cutoff and sm.quick_ratio() >= cutoff and sm.ratio() >= cutoff:
            return True
    return False


def closest_test_question(question: str) -> tuple[float, str]:
    """The most similar test question and its similarity, to show what a near-copy looks like."""
    q, raw = normalise(question), [r["question"] for r in read_jsonl(TEST_SET)]
    return max((difflib.SequenceMatcher(None, q, normalise(t)).ratio(), t) for t in raw)


def drop_test_copies(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(kept, dropped). Haiku repeats itself: 284 of its first 998 questions were near-copies."""
    tests = _test_questions()
    near = [is_near(r["question"], tests) for r in rows]
    return [r for r, n in zip(rows, near) if not n], [r for r, n in zip(rows, near) if n]


def weight_to_design(rows: list[dict[str, Any]], clean_share: float = DESIGN_CLEAN_SHARE,
                     balance_types: bool = False) -> None:
    """Weight rows back to the generation design: `clean_share` clean, the rest defective.

    With `balance_types`, every defect type also gets an equal share, as it had at generation.
    The prior comes from the generation design, never from the test set's labels.
    """
    n, counts = len(rows), collections.Counter(r["planted"] for r in rows)
    defect_types = [t for t in counts if t != "none"]
    for r in rows:
        if r["planted"] == "none":
            r["weight"] = clean_share / (counts["none"] / n)
        elif balance_types:
            r["weight"] = (1 - clean_share) / len(defect_types) / (counts[r["planted"]] / n)
        else:
            r["weight"] = (1 - clean_share) / ((n - counts["none"]) / n)


def weighted(rows: list[dict[str, Any]], mix: str) -> list[dict[str, Any]]:
    """A weighted copy, leaving the input untouched."""
    out = copy.deepcopy(rows)
    weight_to_design(out, balance_types=mix == "balanced")
    return out


def combine(paths: list[str]) -> list[dict[str, Any]]:
    seen: dict[int, dict[str, Any]] = {}
    for path in paths:
        for r in read_jsonl(Path(path)):
            seen.setdefault(r["id"], r)
    return list(seen.values())


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless: the plot goes to a file
    from mix_report import plot, print_table, problems, shares

    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", required=True)
    args = ap.parse_args()
    raw = combine(args.sources)
    kept, dropped = drop_test_copies(raw)
    stages = [("as generated", raw, False), ("after dropping test copies", kept, False),
              ("weighted: clean 1 in 3", weighted(kept, "design"), True),
              ("weighted: types balanced too", weighted(kept, "balanced"), True)]
    print_table(stages)
    for name, rows, is_weighted in stages[1:]:
        print(f"{name:30s} {'; '.join(problems(shares(rows, is_weighted))) or 'on design'}")
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    plot(stages, args.plot, compare=(1, 3))
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"dropped {len(dropped)} test copies; wrote {len(kept)} rows to {args.out}; plot {args.plot}")


if __name__ == "__main__":
    main()
