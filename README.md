# jev-demos

Demos with **Jev**, TypeSafe's System One model: a model that returns typed decisions with
probabilities and never writes a word. One folder per demo, each self-contained with its own
README, requirements and keys. The videos are on **The Agentic Enterprise** on YouTube.

| folder | what it shows | video |
|---|---|---|
| [`jev-fetches-claude-writes/`](jev-fetches-claude-writes/) | Jev picks the lookups, the harness runs them, Claude writes once: 1.7x faster, 5.8x cheaper on a toy support agent | What is Jev? TypeSafe's System One model, paired with Claude |

Every folder runs on its own:

```bash
cd <folder>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add the keys that folder's README names
```

Jev's own speed and price figures are TypeSafe's; the numbers in each folder are mine, on my
tasks, and yours will differ. MIT licence for everything here.
