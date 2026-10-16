"""Prepare a test photo the same way for every arm: a copy that fits Jev's 1568 px limit, and four corner tiles."""
from pathlib import Path

from PIL import Image

MAX_SIDE = 1568
WORK = Path(__file__).resolve().parent.parent / "work"


def prepare(src: Path) -> tuple[Path, list[Path]]:
    """Returns (the resized photo, its four quarter tiles). Cached in work/."""
    WORK.mkdir(exist_ok=True)
    photo = WORK / f"{src.stem}.jpg"
    tiles = [WORK / f"{src.stem}_tile{i}.jpg" for i in range(4)]
    if photo.exists() and all(t.exists() for t in tiles):
        return photo, tiles
    with Image.open(src) as im:
        full = im.convert("RGB")
    small = full.copy()
    small.thumbnail((MAX_SIDE, MAX_SIDE))
    small.save(photo, quality=92)
    w, h = full.size
    for i, t in enumerate(tiles):
        x, y = (i % 2) * w // 2, (i // 2) * h // 2
        tile = full.crop((x, y, x + w // 2, y + h // 2))
        tile.thumbnail((MAX_SIDE, MAX_SIDE))
        tile.save(t, quality=92)
    return photo, tiles
