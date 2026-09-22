# Jev fetches, Claude writes

**What is Jev? TypeSafe's System One model, paired with Claude.** The code behind the video.

Three of every four calls my support agent made to Claude Sonnet 5 were not writing anything.
They were deciding which record to look up next. So the lookups were handed to Jev, a model
that returns typed decisions with probabilities and never writes a word, and Claude was left
with one call: the facts, and the answer. Same tasks, same answers, 1.7x faster and 5.8x
cheaper on this catalog.

## The pattern

```
task ──► Jev: one Noul per possible lookup, in ONE call ──► the harness runs every lookup
         scoring 0.7 or above, in parallel, in rounds   ──► Claude writes once, no tools,
         (a result can expose new ids, so ask Jev again)     no schemas, no history
                                                             │
                                             "MISSING: <fact>" ──► fall back to the
                                                                   tool-using agent
```

Only read-only lookups are fetched this way. A predictor must never run a write.

| file | what it is |
|---|---|
| `tools.py` | eight read-only lookups over `data.json`, each with a simulated 1.5 s round trip |
| `agent_claude.py` | the baseline: Claude picks a tool, waits for it, repeats |
| `agent_jev.py` | Jev as a branch predictor: likely lookups start while Claude is still deciding (the fallback arm) |
| `agent_jev_first.py` | Jev fetches, Claude writes: the pattern the video is about |
| `speed_gain.ipynb` | both arms on one task, then eight tasks three times each, medians and charts |
| `test_speculation.py` | offline tests of the planning logic, no API calls |
| `tasks.json`, `runs.csv` | the eight tasks, and my 48 runs from 21 Sep 2026 |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add your TypeSafe and Anthropic keys
pytest test_speculation.py   # free: no API calls
```

Then open `speed_gain.ipynb`. The single-task cells cost a few cents; the full 48-run cell
costs about $0.35 on Claude and a fraction of a cent on Jev.

**Two things in the code are worth knowing before you change them.**

- `FETCH_AT = 0.7` in `agent_jev_first.py`. TypeSafe's guidance is to act above 0.9 and not
  below 0.5. It is pulled down here on purpose: a wrong yes costs one wasted read-only lookup,
  a wrong no costs a missing fact and a fallback. Raise it if your lookups are not free to waste.
- The question wording. "Is this lookup the very next one?" hit about half the time, because
  parallel lookups split the probability. "Is this one of the lookups to run now, possibly
  alongside others?" hits 97 percent. Jev reads the question literally.

## Read this before trusting the numbers

- Medians over 3 runs. Claude takes different paths from run to run.
- Tool latency is simulated at 1.5 s; faster tools shrink the time gain and leave the cost gain.
- The MISSING fallback catches under-fetching. It cannot catch a wrong answer written from
  incomplete facts. You need evals on your own tasks for that.
- Jev's speed and price figures are TypeSafe's own (blog and docs, retrieved 21 Sep 2026); the
  model was a week old and in early access when this was recorded. The notebook outputs and
  `runs.csv` are my numbers on my toy catalog. Yours will differ.

MIT licence (see the repository root). Video: The Agentic Enterprise on YouTube.
