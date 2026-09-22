# Jev fetches, Claude writes

The code behind the video "What is Jev? TypeSafe's System One model, paired with Claude".

A support agent answers questions from eight read-only lookups. Run the usual way, three of
every four calls it makes to Claude Sonnet 5 are not writing anything. They are deciding
which record to look up next. This demo hands that decision to Jev, a model that returns
typed decisions with probabilities and never writes a word, and leaves Claude with one call:
the facts, and the answer. Same tasks, same answers, 1.7x faster and 5.8x cheaper on this
catalog.

## Contents

| file | what it is |
|---|---|
| `tools.py` | eight read-only lookups over `data.json`, each with a simulated 1.5 s round trip |
| `agent_claude.py` | arm 1, the baseline: Claude picks a tool, waits for it, repeats |
| `agent_jev_first.py` | arm 2, the pattern in the video: Jev picks the lookups, the harness runs them, Claude writes once |
| `agent_jev_helper.py` | the helpers both arms of the planner share: the candidate lookups from the ids in the state, the question Jev is asked, and Jev's cost bookkeeping |
| `speed_gain.ipynb` | both arms on one task, then eight tasks three times each, with medians and charts |
| `test_planning.py` | offline tests of the planning logic, no API calls |
| `tasks.json` | the eight support tasks |
| `runs.csv` | my 48 runs from 21 Sep 2026, the numbers the video quotes |

## Architecture

### Arm 1: Claude alone

```
task -> Claude, with all eight tool schemas and the whole history
          |  picks one tool
          v
        the tool runs (1.5 s)
          |  result appended to the history
          v
        Claude again ... three or four turns, then it writes the answer
```

Every turn carries the schemas and a history that grows with each result, about 4,000 input
tokens by the last call, and only the last turn produces text.

### Arm 2: Jev fetches, Claude writes

```
task
  |
  v
ids in the task (C-17, O-1042, EU) -> candidate lookups: every tool that takes one of them
  |
  v
Jev, one call: one Noul per candidate, "is this one of the lookups to run now?"
  |
  v
the harness runs every candidate scoring 0.7 or above, in parallel, and keeps the facts
  |
  |  a result can expose new ids (an order id inside a customer record)
  |  so the new candidates go back to Jev: round two, round three, until nothing scores 0.7
  v
Claude, one call: the task and the facts, no tools, no schemas, no history
  |
  |  if the reply starts with "MISSING: <fact>"
  v
fallback to agent_claude.py: the plain tool loop, so nothing is written from missing facts
```

Jev decides which records to read. Claude decides what to say. The harness owns the memory,
the facts list, and the safety rule: only read-only lookups are ever fetched on a
prediction. A predictor must never run a write.

### One task, step by step

| step | arm 1: Claude alone | arm 2: Jev fetches, Claude writes |
|---|---|---|
| 1 | Claude reads the task and the schemas, asks for the customer record | Jev scores six candidate lookups in one call, in about 200 ms |
| 2 | the lookup runs, 1.5 s | the harness runs the five that scored 0.7 or above, in parallel, 1.5 s |
| 3 | Claude reads the history again, asks for the order | the order record exposes a shipment id; Jev scores the new candidates |
| 4 | the lookup runs, 1.5 s | the harness runs the shipment lookup, 1.5 s |
| 5 | Claude asks for the shipment, waits again | Claude gets seven facts in one call and writes the answer |
| 6 | Claude writes the answer | done |

## Results

Eight tasks, both arms, three repeats each: 48 runs on claude-sonnet-5, 21 Sep 2026.

| per task, medians | Claude alone | Jev fetches, Claude writes |
|---|---|---|
| seconds | 8.4 | 5.4 |
| Claude calls | 3 | 1 |
| input tokens | 4,214 | 350 |
| lookups | 4 | 4.5 |
| cost, USD | 0.0128 | 0.0022 |

| totals for the eight tasks, mean of three repeats | Claude alone | Jev fetches, Claude writes |
|---|---|---|
| seconds | 71.0 | 42.3 |
| cost, USD | 0.096 | 0.016 |
| fallbacks to the tool loop | | 0 of 24 |

| task | Claude alone, s | Jev first, s | speedup |
|---|---|---|---|
| account-review | 8.3 | 5.6 | 1.5x |
| backorder | 7.2 | 5.2 | 1.4x |
| dead-switch | 8.3 | 6.2 | 1.3x |
| double-charge | 8.3 | 5.3 | 1.6x |
| in-transit | 12.3 | 3.4 | 3.6x |
| late-gateway | 11.7 | 5.9 | 2.0x |
| policy-only | 4.3 | 3.2 | 1.4x |
| unpaid | 8.9 | 7.4 | 1.2x |

Jev's own bill for all 24 runs was about a quarter of a cent.

## The data is synthetic

The data in this demo is synthetic. The customers, orders, shipments, invoices, tickets and policies in `data.json` and `tasks.json` were made up for this demo and for nothing else. Any resemblance to real people, companies, orders or records is purely coincidental.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest test_planning.py
```

Put your TypeSafe and Anthropic keys in `.env`. The tests are free; they make no API calls.
Then open `speed_gain.ipynb`. The single-task cells cost a few cents. The full 48-run cell
costs about 35 cents on Claude and a fraction of a cent on Jev.

## Two design choices worth knowing

| choice | where | why |
|---|---|---|
| `FETCH_AT = 0.7` | `agent_jev_first.py` | TypeSafe suggests acting above 0.9 and not below 0.5. It is lower here on purpose: a wrong yes costs one wasted read-only lookup, a wrong no costs a missing fact and a fallback. Raise it if your lookups are not free to waste. |
| the question wording | `agent_jev_helper.py`, `question()` | "Is this lookup the very next one?" hit about half the time, because parallel lookups split the probability. "Is this one of the lookups to run now, possibly alongside others?" hits 97 percent. Jev reads the question literally. |

## Read this before trusting the numbers

- Medians over three runs. Claude takes different paths from run to run, and one slow response can flip a task.
- Tool latency is simulated at 1.5 s. Faster tools shrink the time gain and leave the cost gain.
- The MISSING fallback catches under-fetching. It cannot catch a wrong answer written from incomplete facts. You need evals on your own tasks for that.
- Jev's speed and price figures are TypeSafe's own, from their blog and docs as retrieved on 21 Sep 2026. The model was a week old and in early access when this was recorded.
- The eight tasks and the catalog are mine and small. Your numbers will differ.

MIT licence, see the repository root.
