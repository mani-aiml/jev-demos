"""One call per engine, each returning {question_id: {"choice": str, "probs": {option: p} | None}}.

All three get the same message in its own field, the same questions and the same option texts.
Claude answers through a forced tool whose fields are enums, so it too can only pick an option.
"""

import os
from pathlib import Path

DEMO = Path(__file__).parent
CLAUDE_MODEL = "claude-sonnet-5"


def load_env() -> None:
    """demo/.env holds TYPESAFE_API_KEY; ANTHROPIC_API_KEY comes from the shell profile."""
    env = DEMO / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def state_for(text: str) -> dict[str, str]:
    return {"MESSAGE": text}


def jev_questions(questions: dict) -> dict:
    from typesafe_sdk import Choice

    return {q: Choice(instructions=ins, criteria=opts) for q, (ins, opts) in questions.items()}


def laya_questions(questions: dict) -> dict:
    return {q: {"type": "choice", "instructions": ins, "criteria": opts} for q, (ins, opts) in questions.items()}


async def ask_jev(client, text: str, questions: dict) -> dict:
    response = await client.system_one(state=state_for(text), questions=jev_questions(questions))
    return {q: {"choice": a.choice, "probs": dict(a.probabilities)} for q, a in response.answers.items()}


def ask_laya(agent, text: str, questions: dict) -> dict:
    answers = agent.predict(state_for(text), laya_questions(questions))["answers"]
    return {q: {"choice": a["choice"], "probs": a["probabilities"]} for q, a in answers.items()}


def claude_tool(questions: dict) -> dict:
    props = {q: {"type": "string", "enum": list(opts), "description": ins} for q, (ins, opts) in questions.items()}
    return {"name": "decide", "description": "Record the decision for this customer message.",
            "input_schema": {"type": "object", "properties": props, "required": list(questions)}}


def claude_system(questions: dict) -> str:
    lines = ["You triage inbound customer email for a support team. Answer each question by picking "
             "exactly one option. The options mean:"]
    for q, (ins, opts) in questions.items():
        lines.append(f"\n{q}: {ins}")
        lines += [f"- {name}: {meaning}" for name, meaning in opts.items()]
    return "\n".join(lines)


async def ask_claude(client, text: str, questions: dict, model: str = CLAUDE_MODEL) -> dict:
    response = await client.messages.create(
        model=model, max_tokens=200, system=claude_system(questions),
        tools=[claude_tool(questions)], tool_choice={"type": "tool", "name": "decide"},
        messages=[{"role": "user", "content": f"<MESSAGE>\n{text}\n</MESSAGE>"}])
    picked = next(b.input for b in response.content if b.type == "tool_use")
    usage = response.usage
    return {q: {"choice": picked[q], "probs": None} for q in questions} | {
        "_tokens": {"in": usage.input_tokens, "out": usage.output_tokens}}
