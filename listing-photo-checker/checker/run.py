"""Run one arm over the 20-photo set.

Usage: run.py ARM [--repeat N]     ARM: jev-1.14.0 | jev-1.13.0 | siglip | qwen
Appends one JSON line per photo per repeat to runs/<ARM>.jsonl.
"""
import argparse
import csv
import json
from pathlib import Path

import photos
import rules as R

HERE = Path(__file__).resolve().parent.parent
SET, RUNS = HERE / "photos" / "set", HERE / "runs"


def load_arm(name: str):
    if name.startswith("jev"):
        import arm_jev
        return arm_jev.make(name)
    if name == "siglip":
        import arm_siglip
        return arm_siglip.make()
    import arm_qwen
    return arm_qwen.make()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    answer = load_arm(args.arm)
    RUNS.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(HERE / "photos" / "labels.csv")))
    with open(RUNS / f"{args.arm}.jsonl", "a") as out:
        for rep in range(args.repeat):
            for r in rows:
                w, h = int(r["width"]), int(r["height"])
                if not any(R.code_checks(m, w, h)["size"] for m in R.MARKETS):
                    out.write(json.dumps({"photo": r["photo"], "repeat": rep, "skipped": "too small"}) + "\n")
                    continue                 # a line of code already said no: the photo never goes to a model
                photo, tiles = photos.prepare(SET / f"{r['photo']}.jpg")
                a = answer(photo, tiles, R.listing(r["photo"]))
                out.write(json.dumps({"photo": r["photo"], "repeat": rep, **a}) + "\n")
                out.flush()
                print(rep, r["photo"], a["seconds"], "s", {k: round(v, 2) for k, v in a["rules"].items()}, flush=True)


if __name__ == "__main__":
    main()
