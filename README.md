# jev-demos

Demos with Jev, TypeSafe's System One model, and other System One models next to it. One folder per demo, each self-contained: its own
README with the architecture, its own requirements and key names. The videos are on
The Agentic Enterprise on YouTube.

| folder | contents |
|---|---|
| [`jev-fetches-claude-writes/`](jev-fetches-claude-writes/) | two agents (Claude alone; Jev fetches, Claude writes), the lookup tools and data, offline tests, and the notebook that runs eight tasks both ways |
| [`laya-vs-jev-judge/`](laya-vs-jev-judge/) | Jev and Laya grading the same 500 AI-written answers, zero-shot: the corpus, the reference judge's labels, the harness, the cascade maths and the recorded notebook |
| [`laya-finetune/`](laya-finetune/) | Laya fine-tuned on the same grading job and scored against Jev on the same 500 items: the data pipeline and its three checks (test copies, class mix, planted labels), training, scoring, the Jev timing run and the notebook. The fine-tuned checkpoint is not included |

```bash
cd <folder>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add the keys that folder's README names
```

All data in these demos is synthetic, made up for the demo it sits in and for nothing else. Any
resemblance to real people, companies or records is purely coincidental.

MIT licence for everything here.
