"""The small vision-language arm, run locally: Qwen3-VL-4B reads the same rule text Jev gets, one rule per pass.
P(yes) is the softmax over the next-token logits for "Yes" and "No" (the VQAScore method, Lin et al. 2024). The box
comes from Qwen's own grounding output: bbox_2d in 0-1000 coordinates, parsed from its reply."""
import re
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import rules as R

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def make():
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    yes, no = (proc.tokenizer.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No"))

    def inputs(photo: Path, text: str):
        msg = [{"role": "user", "content": [{"type": "image", "image": str(photo)}, {"type": "text", "text": text}]}]
        return proc.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=True,
                                        return_tensors="pt").to(DEVICE)

    @torch.no_grad()
    def p_yes(photo: Path, question: str) -> float:
        logits = model(**inputs(photo, question + " Answer Yes or No.")).logits[0, -1, [yes, no]].float()
        return logits.softmax(-1)[0].item()

    @torch.no_grad()
    def box(photo: Path, item: str) -> dict | None:
        x = inputs(photo, f"Locate {item} in this photo. Reply with its bbox_2d as JSON.")
        reply = proc.decode(model.generate(**x, max_new_tokens=60, do_sample=False)[0, x["input_ids"].shape[1]:])
        m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", reply)
        if not m:
            return None
        x0, y0, x1, y1 = (min(1000, int(v)) / 1000 for v in m.groups())
        return {"top": y0, "bottom": y1, "left": x0, "right": x1}

    def answer(photo: Path, tiles: list[Path], item: str) -> dict:
        t = time.time()
        rules = {k: p_yes(photo, R.fill_in(q, item)) for k, (q, _) in R.RULES.items()}
        tile_p = [p_yes(p, R.fill_in(R.TILE_RULE, item)) for p in tiles]
        return {"rules": rules, "tiles": tile_p, "box": box(photo, item), "seconds": round(time.time() - t, 2)}

    return answer
