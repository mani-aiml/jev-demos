# jev-demos

Demos with Jev, TypeSafe's System One model, and other System One models next to it. One folder per demo, each self-contained: its own
README with the architecture, its own requirements and key names. The videos are on
The Agentic Enterprise on YouTube.

| folder | contents |
|---|---|
| [`jev-fetches-claude-writes/`](jev-fetches-claude-writes/) | two agents (Claude alone; Jev fetches, Claude writes), the lookup tools and data, offline tests, and the notebook that runs eight tasks both ways |
| [`laya-vs-jev-judge/`](laya-vs-jev-judge/) | Jev and Laya grading the same 500 AI-written answers, zero-shot: the corpus, the reference judge's labels, the harness, the cascade maths and the recorded notebook |
| [`decision-model-prompt-injection/`](decision-model-prompt-injection/) | can you prompt-inject a decision model? Jev and Claude Haiku 4.5 as a review gate, against twelve hand-written emails with ten classic attacks and against real attacks from Microsoft's LLMail-Inject: the three wordings, the detector question, the harness and the LLMail sampler. Code only: no data, public or synthetic; `llmail_sample.py` fetches the dataset from the source |
| [`laya-finetune/`](laya-finetune/) | Laya fine-tuned on the same grading job and scored against Jev on the same 500 items: the data pipeline and its three checks (test copies, class mix, planted labels), training, scoring, the Jev timing run and the notebook. The fine-tuned checkpoint is not included |
| [`strands-decider/`](strands-decider/) | Strands Decider 2B on the same grading job and the same 500 items, out of the box and fine-tuned on Laya's rows with the same split, mix and seeds: scoring, the harness checks, its native adequacy format, the data conversion and the fine-tune. Code only: it reads `laya-finetune/data/`; no checkpoints |

```bash
cd <folder>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add the keys that folder's README names
```

All data in these demos is synthetic, made up for the demo it sits in and for nothing else. Any
resemblance to real people, companies or records is purely coincidental. The one exception is
`decision-model-prompt-injection/`, which holds no data at all: it downloads public attack emails from
Microsoft's LLMail-Inject (MIT) at run time, and you bring your own hand-written test emails.
`strands-decider/` holds no data either: it reads `laya-finetune/data/`.

MIT licence for everything here.
