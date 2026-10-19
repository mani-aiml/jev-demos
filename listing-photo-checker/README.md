# listing-photo-checker

Can Jev, TypeSafe's decision model, check a product photo against a marketplace's picture rules? Each rule is a
plain-English question; Jev answers with the probability that the answer is yes, and the code turns those
probabilities into one of three actions. The same photos and rules also go to two models that run locally, for
comparison: a classifier (SigLIP 2 for the rules, OWLv2 for the box) and a small vision-language model (Qwen3-VL-4B).

## How it works

- **Rules** (`checker/rules.py`): five yes/no questions about the whole photo (white studio background, anything
  added on top, only the listed item, sharp, whole item in frame), the "added on top?" question again on four corner
  tiles, and a Box question for where the item is. Rules a line of code can answer (size, square) never go to a model.
- **Photos** (`checker/photos.py`): a copy shrunk to Jev's 1568 px limit, plus four corner tiles; the original is
  kept for the size check.
- **Jev** (`checker/arm_jev.py`): two calls per photo, the whole photo with the rules and the box, then the tiles.
  Images need the `TypeSafe-Danger-Zone-Images: true` header (a preview feature; TypeSafe says image support is not
  ready for production yet).
- **Actions**: publish if every rule is at or above 0.9, send back if any rule is at or below 0.2, otherwise a person
  checks it. For Walmart, a box that covers less than 20% of the photo sends it back (item not cropped close).
- **Local arms** (`checker/arm_siglip.py`, `checker/arm_qwen.py`) get the same rule text.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # TYPESAFE_API_KEY=...
cd checker
python run.py jev-1.14.0 --repeat 3   # or jev-1.13.0 | siglip | qwen
python score.py
```

## Bring your own photos

No photos or labels are included. Put your photos in `photos/set/<name>.jpg` and write `photos/labels.csv`:

```
photo,split,ebay,walmart,reason,width,height
mug_white,dev,pass,pass,clean,2200,2200
```

`photo` is the file name without `.jpg`; the part before the first `_` picks the listing text in `rules.py`
(`LISTINGS`). `ebay` and `walmart` are your expected verdicts (`pass` or `fail`), written before you run anything.
`photos/make_set.py` and `photos/generate.sh` (with the prompts in `photos/prompts/`) are how the video's 20 test
photos were made: base photos generated with Codex, then overlays, crops and size copies added in code.

Runs append one JSON line per photo to `runs/<arm>.jsonl`. Tested on a MacBook Pro (M4 Pro, 48 GB); the local arms
use PyTorch MPS.
