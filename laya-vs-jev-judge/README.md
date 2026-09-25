# Laya vs Jev, judged on the same task

The code behind the video "Laya vs Jev, Judged on the Same Task" on The Agentic Enterprise.
Two System One models, Jev (TypeSafe, hosted) and Laya (Convai Innovations, open weights), get
the same job: grade 500 AI-written answers. Both answer two questions per item, zero-shot, out
of the box.

| question | Jev | Laya |
|---|---|---|
| 2 options: pass or rework | 0.876 | 0.382 (said "pass" on 492 of 500) |
| 40 options: which failure mode | 0.790 | 0.234 (0.120 in my first, unfair setup) |
| median latency per call | 159 ms (hosted API, includes network) | 39.0 ms (local, `laya-mlx`, 2 options) |

Every score is **agreement with a reference judge** (`claude-sonnet-5`, grading blind), not
correctness. Against the defects I planted, the reference judge itself matched 92.0% of verdicts
and 67.6% of exact defects. The two latencies are measured differently and are never combined
into one speedup figure. Jev's scores move by up to a point between runs; Laya's repeat exactly.

## Files

| file | what it does |
|---|---|
| `judge_task.py` | the two questions and the 40 failure modes, defined once for both engines |
| `build_dataset.py` | generates the 500 items with planted defects and has the reference judge grade them (needs `ANTHROPIC_API_KEY`; the result is already in `data/`) |
| `run_engines.py` | runs Laya (PyTorch) and Jev over every item, saves `data/engine_runs.json` |
| `compare.py` | agreement, risk-coverage and the cascade maths (how much of the reference judge's bill Jev's confidence can remove) |
| `nb_helpers.py`, `laya_vs_jev.ipynb` | the notebook recorded for the video, with its outputs |
| `data/generated.jsonl`, `data/judged.jsonl` | the 500 items and the reference judge's labels |

## Run it

Laya runs locally; `laya-mlx` (an independent port, not an official Convai release) needs Apple
silicon. Jev needs a TypeSafe key.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # TYPESAFE_API_KEY; ANTHROPIC_API_KEY only to rebuild data/
jupyter lab laya_vs_jev.ipynb
```

## The mistake worth knowing about

Laya puts all the options of a question into one shared token budget (`head_max_len`, 256 for
this checkpoint). My 40 options with a sentence of description each came to 492 tokens, so every
option was cut to 6 tokens, one of them a marker. Count your option tokens first
(`budget_check` in `nb_helpers.py`), or keep the labels short.

## Caveats

- The corpus is synthetic. Defects were assigned at random from 40 labels, so their frequencies
  say nothing about real error rates.
- Claims about either model's design come from the makers' own documentation as it stood on
  22 Sep 2026. Both models were days old, and either may have changed since.
- Zero-shot only. Convai's documentation says Laya's base model is meant to be fine-tuned; whether
  a fine-tuned Laya would close the gap is not tested here.
