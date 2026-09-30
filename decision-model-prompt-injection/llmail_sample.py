"""Build the LLMail-Inject test set: 300 attacks that beat a defended LLM, plus the challenge's
normal emails as controls. Writes data/llmail_sample.jsonl. Free: no model is called.

Source: microsoft/llmail-inject-challenge (MIT), Abdelnabi et al. 2025. Only rows whose five
objective flags are all true are used (the attack got past the defence and the assistant sent the
attacker's email). Fetched through the Hugging Face dataset query API, so the 460k-row files are
never downloaded.

    python llmail_sample.py              (the main 300 + 78 controls)
    python llmail_sample.py --heldout    (300 fresh attacks, none in the main sample, no controls)
"""

import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "microsoft/llmail-inject-challenge"
FILTER_URL = "https://datasets-server.huggingface.co/filter"
SCENARIOS_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/data/scenarios.json"
FLAGS = ("email.retrieved", "defense.undetected", "exfil.sent", "exfil.destination", "exfil.content")
PER_PHASE = {"Phase1": 240, "Phase2": 60}
PAGE = 100
SEED = 29
OUT = Path(__file__).parent / "data" / "llmail_sample.jsonl"
HELDOUT = Path(__file__).parent / "data" / "llmail_heldout.jsonl"
POOL = Path(__file__).parent / "data" / "llmail_pool_{}.jsonl"  # the successful rows, fetched once
RAW_DIR = Path(__file__).parent / "data" / "tmp_raw"  # optional local copy of raw_submissions_phase*.jsonl


def get_json(url: str, params: dict | None = None, tries: int = 10) -> dict:
    """The query API returns the odd 500 while its index warms up; back off and retry."""
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url + query, timeout=180) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as error:
            client_error = isinstance(error, urllib.error.HTTPError) and error.code < 500
            if client_error or attempt == tries - 1:
                raise
            time.sleep(min(60, 2 ** attempt * 3))


def local_rows(split: str) -> list[dict] | None:
    """Same filter as successful_rows, over a downloaded raw file when one is present."""
    path = RAW_DIR / f"raw_submissions_{split.lower()}.jsonl"
    if not path.exists():
        return None
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if all(json.loads(row["objectives"]).get(flag) for flag in FLAGS):
                rows.append(row)
    return rows


def successful_rows(split: str) -> list[dict]:
    """Every submission in a phase where all five objective flags are true. Pages are saved as they
    arrive, so a server error resumes from the last page instead of starting over."""
    where = " AND ".join(f"\"objectives\" LIKE '%\"{flag}\": true%'" for flag in FLAGS)
    partial = Path(str(POOL).format(split) + ".partial")
    rows = [json.loads(line) for line in partial.read_text().splitlines()] if partial.exists() else []
    while True:
        page = get_json(FILTER_URL, {"dataset": DATASET, "config": "default", "split": split,
                                     "where": where, "offset": len(rows), "length": PAGE})
        new = [r["row"] for r in page["rows"]]
        rows += new
        with partial.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in new)
        if not new or len(rows) >= page["num_rows_total"]:
            partial.unlink()
            return rows


def pool(phase: str) -> list[dict]:
    """Unique successful attacks for a phase, cached locally after the first (slow) fetch."""
    path = Path(str(POOL).format(phase))
    if not path.exists():
        rows = local_rows(phase)
        rows = successful_rows(phase) if rows is None else rows
        print(f"{phase}: {len(rows)} successful submissions")
        path.write_text("".join(json.dumps(r) + "\n" for r in dedupe(rows)))
    return [json.loads(line) for line in path.read_text().splitlines()]


def dedupe(rows: list[dict]) -> list[dict]:
    """Teams resubmitted the same email to several levels; keep one copy of each text."""
    seen, unique = set(), []
    for row in rows:
        key = (" ".join(row["subject"].lower().split()), " ".join(row["body"].lower().split()))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def scenario_of(level: str) -> str:
    """'level2v' -> 'scenario_2'."""
    return f"scenario_{level.removeprefix('level')[0]}"


def stratified(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Spread the sample across scenarios as evenly as the pool allows."""
    by_scenario: dict[str, list[dict]] = {}
    for row in rows:
        by_scenario.setdefault(scenario_of(row["scenario"]), []).append(row)
    for pool in by_scenario.values():
        rng.shuffle(pool)
    picked: list[dict] = []
    while len(picked) < n and any(by_scenario.values()):
        for pool in by_scenario.values():
            if pool and len(picked) < n:
                picked.append(pool.pop())
    return picked


def attack_item(row: dict, phase: str) -> dict:
    return {"mid": row["RowKey"], "attack": "llmail", "gold_human": "needs_human", "gold_route": None,
            "text": f"Subject: {row['subject'].strip()}\n\n{row['body'].strip()}",
            "meta": {"phase": phase, "level": row["scenario"], "scenario": scenario_of(row["scenario"]),
                     "words": len(row["body"].split())}}


def control_items() -> list[dict]:
    """The challenge's own normal emails: no gold triage label, used for false alarms."""
    texts = {e for s in get_json(SCENARIOS_URL).values() for e in s["emails"]}
    return [{"mid": f"control_{i}", "attack": "control", "gold_human": None, "gold_route": None,
             "text": t.replace("Subject of the email: ", "Subject: ").replace("   Body: ", "\n\n"),
             "meta": {"words": len(t.split())}} for i, t in enumerate(sorted(texts))]


def main() -> None:
    heldout = "--heldout" in sys.argv
    rng = random.Random(SEED + 1 if heldout else SEED)
    # exclude by text as well as id: a resubmitted copy of a main-sample email has a new id
    used = {" ".join(json.loads(line)["text"].lower().split()) for line in OUT.read_text().splitlines()} if heldout else set()
    items = []
    for phase, n in PER_PHASE.items():
        rows = [r for r in pool(phase) if " ".join(attack_item(r, phase)["text"].lower().split()) not in used]
        print(f"{phase}: {len(rows)} unique successful attacks available, sampling {n}")
        items += [attack_item(r, phase) for r in stratified(rows, n, rng)]
    if not heldout:
        items += control_items()
    out = HELDOUT if heldout else OUT
    out.write_text("".join(json.dumps(i) + "\n" for i in items))
    print(f"wrote {len(items)} items to {out}")


if __name__ == "__main__":
    main()
