"""The listing-photo rules: what every model is asked, and how the answers become eBay and Walmart verdicts.

Every arm answers the same five yes/no rules about the whole photo, the "added on top?" rule again on four corner tiles
(the look-closer step), and where the item is (a box). Rules a line of code can check (size, square) never go to a model.
Thresholds were fixed before any run (outline.md): publish if every rule is >= 0.9 compliant, send back if any is <= 0.2.
"""

# Plain-English rules. `photo` and `listing` are the names in Jev's state; local arms get the same text with the names
# filled in (fill_in). compliant: the answer a good photo gives. Wording was chosen on the 3 dev photos only
# (15 Oct): the first white-background rule, a list of "no texture, creases, color tint or scene", scored a clean white
# mug 0.13 on jev-1.14.0; the plain sentence scores it 0.93. The first run is kept in runs/v1_wording/.
RULES = {
    "white_bg": ("Is `photo` a product shot on a seamless white studio background?", True),
    "added": ("Was any text, logo, watermark or border added on top of `photo` after it was taken? "
              "Text printed on the product itself does not count.", False),
    "extras": ("Is the item in `listing` the only object in `photo`?", True),
    "sharp": ("Is `photo` sharp and in focus?", True),
    "whole": ("Is the whole item in `photo` inside the frame, with no part cut off at the edges?", True),
}
TILE_RULE = ("Was any text, logo or watermark added on top of `photo` after it was taken? "
             "Text printed on the product itself does not count.")
BOX = "the item from `listing` in `photo`"

LISTINGS = {"mug": "one ceramic coffee mug", "whitemug": "one ceramic coffee mug", "spoon": "one wooden cooking spoon",
            "tin": "one tea tin"}
MARKETS = {"ebay": ["added"], "walmart": ["white_bg", "added", "extras", "sharp", "whole"]}
PUBLISH, SEND_BACK = 0.9, 0.2
MIN_FILL = 0.2                              # Walmart: "crop the item as close as possible"; box area / photo area


def listing(photo: str) -> str:
    return LISTINGS[photo.split("_")[0]]


def fill_in(text: str, item: str, photo: str = "this photo") -> str:
    return text.replace("`photo`", photo).replace("`listing`", f"the listing ({item})")


def code_checks(market: str, w: int, h: int) -> dict[str, float]:
    """The rules that are just math, answered with certainty."""
    if market == "ebay":
        return {"size": float(max(w, h) >= 500)}
    return {"size": float(min(w, h) >= 1500), "square": float(w == h)}


def compliant(ans: dict, market: str, w: int, h: int) -> dict[str, float]:
    """Probability that each rule is met. 'added' is the worst of the whole photo and its four tiles."""
    p = {}
    for rule in MARKETS[market]:
        yes = max([ans["rules"][rule]] + ans["tiles"]) if rule == "added" else ans["rules"][rule]
        p[rule] = yes if RULES[rule][1] else 1 - yes
    p.update(code_checks(market, w, h))
    if market == "walmart" and ans.get("box"):
        b = ans["box"]
        p["fill"] = float((b["right"] - b["left"]) * (b["bottom"] - b["top"]) >= MIN_FILL)
    return p


def verdict(p: dict[str, float]) -> tuple[str, str]:
    """('publish' | 'send back' | 'ask a person', the rule that decided it)."""
    worst = min(p, key=p.get)
    if p[worst] <= SEND_BACK:
        return "send back", worst
    if p[worst] >= PUBLISH:
        return "publish", ""
    return "ask a person", worst
