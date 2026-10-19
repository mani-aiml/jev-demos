"""Score every arm in runs/ against photos/labels.csv.

Usage: score.py [--detail ARM]
Per arm and marketplace: how many photos were published, sent back or given to a person, and how many automatic
decisions were wrong. Each repeat is scored on its own and the counts are averaged.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import rules as R

HERE = Path(__file__).resolve().parent.parent
GOOD = {"pass": "publish", "fail": "send back"}


def decide(row: dict, a: dict, market: str) -> tuple[str, str]:
    if "skipped" in a:
        return "send back", "size"
    return R.verdict(R.compliant(a, market, int(row["width"]), int(row["height"])))


def score(arm: Path, labels: dict, detail: bool) -> None:
    runs = [json.loads(line) for line in open(arm)]
    reps = max(r["repeat"] for r in runs) + 1
    for market in R.MARKETS:
        n = defaultdict(int)
        for a in runs:
            row = labels[a["photo"]]
            got, why = decide(row, a, market)
            n["to a person" if got == "ask a person" else "right" if got == GOOD[row[market]] else "WRONG"] += 1
            if detail and a["repeat"] == 0 and got != GOOD[row[market]]:
                print(f"    {market:8} {a['photo']:16} label {row[market]:4} got {got} ({why})")
        print(f"  {arm.stem:12} {market:8}", "  ".join(f"{k} {v / reps:4.1f}" for k, v in sorted(n.items())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail")
    args = ap.parse_args()
    labels = {r["photo"]: r for r in csv.DictReader(open(HERE / "photos" / "labels.csv"))}
    for arm in sorted((HERE / "runs").glob("*.jsonl")):
        score(arm, labels, args.detail == arm.stem)


if __name__ == "__main__":
    main()
