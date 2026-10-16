"""The classic vision arm, run locally: SigLIP 2 answers each rule by comparing the photo with two label prompts (rule met /
rule broken), and OWLv2 finds the item's box. A classifier can't read a rule, so each rule becomes a label pair, written
once before the run, with one wording round on the 3 dev photos like Jev got (15 Oct): the "whole" pair changed (the first
scored a clean mug 0.07). No "added" pair both caught FREE SHIPPING and stayed quiet on clean tiles, so the one that
catches it stayed. P(yes) is the softmax over the pair."""
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, Owlv2ForObjectDetection, Owlv2Processor

import rules as R

CLS, DET = "google/siglip2-so400m-patch14-384", "google/owlv2-base-patch16-ensemble"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MIN_BOX_SCORE = 0.1
PAIRS = {                                    # rule: (the photo if the rule is TRUE, the photo if it is FALSE)
    "white_bg": ("a product photo on a seamless white studio background",
                 "a product photo on a table, colored paper or a textured background"),
    "added": ("a photo with text, a logo, a watermark or a border added on top", "a clean photo with nothing added on top"),
    "extras": ("a photo of {item} alone", "a photo of {item} together with other objects"),
    "sharp": ("a sharp, in-focus photo", "a blurry, out-of-focus photo"),
    "whole": ("a photo of a whole {item}", "a cropped photo showing only part of a {item}"),
}


def make():
    cls_proc, cls = AutoProcessor.from_pretrained(CLS), AutoModel.from_pretrained(CLS).to(DEVICE).eval()
    det_proc, det = Owlv2Processor.from_pretrained(DET), Owlv2ForObjectDetection.from_pretrained(DET).to(DEVICE).eval()

    @torch.no_grad()
    def p_yes(img: Image.Image, pair: tuple[str, str]) -> float:
        x = cls_proc(text=list(pair), images=img, padding="max_length", max_length=64, return_tensors="pt").to(DEVICE)
        return cls(**x).logits_per_image.softmax(-1)[0, 0].item()

    @torch.no_grad()
    def box(img: Image.Image, item: str) -> dict | None:
        x = det_proc(text=[[f"a photo of {item}"]], images=img, return_tensors="pt").to(DEVICE)
        side = max(img.size)                 # OWLv2 pads to a square, so boxes come back in the padded frame
        res = det_proc.post_process_grounded_object_detection(det(**x), threshold=MIN_BOX_SCORE,
                                                     target_sizes=torch.tensor([[side, side]]))[0]
        if len(res["scores"]) == 0:
            return None
        x0, y0, x1, y1 = res["boxes"][res["scores"].argmax()].tolist()
        w, h = img.size
        return {"top": max(0, y0) / h, "bottom": min(h, y1) / h, "left": max(0, x0) / w, "right": min(w, x1) / w}

    def answer(photo: Path, tiles: list[Path], item: str) -> dict:
        t = time.time()
        img = Image.open(photo).convert("RGB")
        item = item.removeprefix("one ")
        rules = {k: p_yes(img, tuple(s.format(item=item) for s in pair)) for k, pair in PAIRS.items()}
        tile_p = [p_yes(Image.open(p).convert("RGB"), PAIRS["added"]) for p in tiles]
        return {"rules": rules, "tiles": tile_p, "box": box(img, item), "seconds": round(time.time() - t, 2)}

    return answer
