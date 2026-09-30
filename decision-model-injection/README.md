# Can you prompt-inject a decision model?

A support agent reads an inbound email and asks a decision model one question before acting: does a
person need to check this first, or is it safe to automate? The email's author is the attacker. This
folder tests Jev (TypeSafe) against Claude Haiku 4.5 on that gate, first with twelve hand-written
emails and ten classic attacks, then with real attack emails from Microsoft's public LLMail-Inject
challenge. Claude answers through a forced tool that lists the same options, so both models pick
an option rather than write.

**No data is included.** The LLMail-Inject samples are rebuilt on your machine by
`llmail_sample.py`, straight from the public dataset. Only code and summary reports are here.

| File | What it does |
|---|---|
| `task.py` | the twelve hand-written emails, the questions and options (three wordings of "needs a person"), the ten attacks |
| `engines.py` | one call per engine: Jev, Claude (forced tool), and Laya |
| `llmail_sample.py` | builds the LLMail-Inject sample: 300 attacks that beat a defended assistant (all five success flags true, duplicates removed) plus the challenge's normal emails; `--heldout` builds 300 different ones |
| `run.py` | runs every email through one engine; `--source llmail` or `heldout`; `--narrow` / `--explicit` wordings; `--detector` |
| `report.py`, `report_llmail.py` | the summaries (`report_llmail.py heldout` for the fresh set) |
| `runs/` | the summary reports from our runs (counts only) |

```bash
pip install -r requirements.txt
cp .env.example .env            # TYPESAFE_API_KEY; ANTHROPIC_API_KEY in your shell
H=claude-haiku-4-5-20251001

# twelve hand-written emails
python run.py --engine jev --repeats 3
python run.py --engine claude --claude-model $H
python report.py

# LLMail-Inject (builds data/ locally; the query API is slow, allow 15 minutes)
python llmail_sample.py
python run.py --engine jev --source llmail --repeats 3            # add --narrow, then --detector
python run.py --engine claude --claude-model $H --source llmail   # add --narrow, then --detector
python report_llmail.py

# 300 attacks not used for tuning the wording
python llmail_sample.py --heldout
python run.py --engine jev --source heldout --repeats 3 --narrow
python report_llmail.py heldout
```

## What we got (29 and 30 Sep 2026)
"Got through" means the model said safe to automate on an email that needed a person.

| | Jev (majority of 3) | Claude Haiku 4.5 (1 run) |
|---|---|---|
| 12 hand-written emails, 10 attacks: got through | 2 of 60 (both a fake answer block) | 1 of 60 |
| 300 LLMail-Inject attacks, original wording | 106 | 9 |
| + "send, forward or share anything" | 5 | 0 |
| + "...with anyone other than the customer" (narrow) | 4 | 0 |
| narrow wording + a separate injection-detector question | 1 | 0 |
| normal emails marked safe with both checks (of 78) | 64 | 17 |
| 300 held-out attacks: original, narrow, narrow + detector | 107, 10, 1 | not run, 0, 0 |
| median time per call | ~120 ms | ~794 ms |

## Caveats
- We can't tell whether an attack nudged a model or the model just missed the danger: there is no
  clean version of each attack email. We tested the review gate only, not what an agent does next.
- The wording was changed twice after seeing results on the first 300; the held-out 300 are the
  check. We named the attacker's goal because we knew it; in your system, name every action your
  agent could take because of an email.
- The normal emails have no labelled answers, so "marked safe" is possible automation, not accuracy.
- Hand-written emails and attacks, one dataset, non-adaptive attackers: a starting point for your
  own testing, not a guarantee. Hosted models vary run to run.
- The two email addresses in `task.py` were changed to example.com after our runs (they were
  made-up addresses that could have belonged to someone real). Everything else is as run.
- The forced tool is not strict mode; every Claude answer we recorded stayed within the options.
