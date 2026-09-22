# agentic-enterprise-demos

The code behind the videos on **The Agentic Enterprise**. One folder per video, each
self-contained: its own README, requirements and keys.

| folder | video |
|---|---|
| [`jev-fetches-claude-writes/`](jev-fetches-claude-writes/) | What is Jev? TypeSafe's System One model, paired with Claude |

Every folder runs on its own:

```bash
cd <folder>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add the keys that folder's README names
```

MIT licence for everything here.
