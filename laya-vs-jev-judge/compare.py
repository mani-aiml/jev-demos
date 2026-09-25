"""Score both engines against gold, and price the cascade.

Two deliberate choices, both of which the script prints so the video cannot quietly
misstate them:

1. No ECE. Expected calibration error assumes the confidence is a probability of being
   correct. Laya's is normalised Shannon entropy, 1 - H(p)/log(k), which is a certainty
   score on a different scale. Comparing the two engines by ECE would be comparing
   different quantities. Risk-coverage is valid for any score that merely ranks, and it
   is also the thing a cascade actually needs.

2. Gold is the escalation target. An escalated item is therefore right by construction,
   so the headline number is not "accuracy" but "share of the frontier judge's bill you
   can delete while still reproducing N% of its verdicts".
"""

import json
from pathlib import Path
from typing import Any

DEMO = Path(__file__).parent
GOLD_USD_IN, GOLD_USD_OUT = 3.0 / 1_000_000, 15.0 / 1_000_000  # Sonnet list price
GOLD_USD_PER_ITEM = 0.0  # measured from the corpus in load()


def load() -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    global GOLD_USD_PER_ITEM
    items = {
        json.loads(line)["id"]: json.loads(line)
        for line in (DEMO / "data" / "judged.jsonl").read_text().splitlines()
        if line.strip()
    }
    rows = list(items.values())
    tin = sum(r.get("gold_tokens_in", 0) for r in rows) / len(rows)
    tout = sum(r.get("gold_tokens_out", 0) for r in rows) / len(rows)
    GOLD_USD_PER_ITEM = tin * GOLD_USD_IN + tout * GOLD_USD_OUT
    runs = json.loads((DEMO / "data" / "engine_runs.json").read_text())
    return items, runs


def agreement(rows: list[dict[str, Any]], items: dict[int, dict[str, Any]], field: str) -> float:
    hits = sum(1 for r in rows if r[field] == items[r["id"]][f"gold_{field}"])
    return hits / len(rows)


def risk_coverage(
    rows: list[dict[str, Any]], items: dict[int, dict[str, Any]], field: str
) -> list[dict[str, float]]:
    """Accuracy on the most-confident share, swept over coverage."""
    ranked = sorted(rows, key=lambda r: r[f"{field}_conf"], reverse=True)
    out = []
    for fraction in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        take = ranked[: max(1, int(len(ranked) * fraction))]
        hits = sum(1 for r in take if r[field] == items[r["id"]][f"gold_{field}"])
        out.append({
            "coverage": fraction,
            "accuracy_on_covered": round(hits / len(take), 4),
            "threshold": round(take[-1][f"{field}_conf"], 4),
        })
    return out


def cascade(
    rows: list[dict[str, Any]], items: dict[int, dict[str, Any]], field: str,
    engine_usd_per_item: float, target: float = 0.95,
) -> dict[str, Any]:
    """Largest share we can auto-accept while still reproducing `target` of gold's verdicts."""
    ranked = sorted(rows, key=lambda r: r[f"{field}_conf"], reverse=True)
    best = {"coverage": 0.0, "blended_agreement": 1.0, "usd_per_1k": GOLD_USD_PER_ITEM * 1000}
    for cut in range(1, len(ranked) + 1):
        take = ranked[:cut]
        hits = sum(1 for r in take if r[field] == items[r["id"]][f"gold_{field}"])
        if hits / cut < target:
            continue
        coverage = cut / len(ranked)
        usd = (engine_usd_per_item * len(ranked) + GOLD_USD_PER_ITEM * (len(ranked) - cut))
        best = {
            "coverage": round(coverage, 4),
            "accuracy_on_covered": round(hits / cut, 4),
            "blended_agreement": round(coverage * (hits / cut) + (1 - coverage), 4),
            "usd_per_1k": round(usd / len(ranked) * 1000, 4),
            "saving_vs_all_gold": round(1 - (usd / len(ranked)) / GOLD_USD_PER_ITEM, 4),
        }
    return best


def main() -> None:
    items, runs = load()
    print(f"items: {len(items)}   gold judge: ${GOLD_USD_PER_ITEM*1000:.2f} per 1k items\n")

    for run in runs:
        per_item = run["usd"] / max(1, len(run["rows"]))
        print(f"=== {run['engine']}  ({run['device']}) ===")
        print(f"  latency ms: {run['latency_ms']}   cost/1k: ${per_item*1000:.4f}")
        for field in ("verdict", "criterion"):
            k = 2 if field == "verdict" else 40
            print(f"  -- {field} (k={k}) --")
            print(f"     agreement with gold: {agreement(run['rows'], items, field):.3f}")
            print(f"     risk-coverage: {json.dumps(risk_coverage(run['rows'], items, field))}")
            print(f"     cascade@95%:   {json.dumps(cascade(run['rows'], items, field, per_item))}")
        print()


if __name__ == "__main__":
    main()
