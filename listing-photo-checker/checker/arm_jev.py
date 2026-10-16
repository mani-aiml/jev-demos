"""Jev: the rules go in as written, the photo goes in as an image asset. Two calls per photo: the whole photo
(five rules + the box), then the four corner tiles. jev-1.13.0 has no Box question, so it skips the box."""
import os
import time
from pathlib import Path

from typesafe_sdk import Box, Image, Noul, TypeSafeClient

import rules as R

ENV = Path(os.environ.get("JEV_ENV", Path(__file__).resolve().parents[1] / ".env"))   # TYPESAFE_API_KEY=...
HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}
BOX_MODELS = {"jev-1.14.0", "jev-preview"}


def load_key() -> None:
    for line in ENV.read_text().splitlines():
        if line.startswith("TYPESAFE_API_KEY="):
            os.environ.setdefault("TYPESAFE_API_KEY", line.partition("=")[2].strip())


def make(model: str):
    load_key()
    client = TypeSafeClient()
    questions = {k: Noul(instructions=q) for k, (q, _) in R.RULES.items()}
    if model in BOX_MODELS:
        questions["box"] = Box(instructions=R.BOX)
    tile_qs = {f"t{i}": Noul(instructions=R.TILE_RULE.replace("`photo`", f"`tile{i}`")) for i in range(4)}

    def answer(photo: Path, tiles: list[Path], item: str) -> dict:
        t = time.time()
        whole = client.system_one(state={"photo": Image(photo), "listing": item}, questions=questions, model=model,
                                  extra_headers=HEADERS)
        part = client.system_one(state={f"tile{i}": Image(p) for i, p in enumerate(tiles)}, questions=tile_qs,
                                 model=model, extra_headers=HEADERS)
        a = whole.answers
        return {"rules": {k: a[k].noul for k in R.RULES}, "tiles": [part.answers[f"t{i}"].noul for i in range(4)],
                "box": a["box"].box.model_dump() if "box" in a else None,
                "box_confidence": a["box"].confidence if "box" in a else None,
                "tokens": whole.usage.input_tokens + part.usage.input_tokens, "seconds": round(time.time() - t, 2)}

    return answer
