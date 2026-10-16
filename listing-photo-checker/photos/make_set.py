"""Build the 20-photo test set from the Codex base photos in raw/.

Bases are enlarged to 2200 x 2200 (Walmart's size; the Codex output is 1254). The overlays, crops and size copies are
made here in code so every one is exact and repeatable. Writes set/<name>.jpg and a labels.csv template:
fill in the expected eBay and Walmart verdicts yourself before any model runs (the video's answer key is not
published).
"""
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).parent
RAW, OUT = HERE / "raw", HERE / "set"
SIZE = 2200
WHITE = (255, 255, 255)
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
SPLIT = {"mug_white": "dev", "mug_wood": "dev", "mug_freeship": "dev",
         "tin_white": "holdout", "tin_wood": "holdout", "tin_watermark": "holdout"}

def base(name: str) -> Image.Image:
    return Image.open(RAW / f"{name}.png").convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)


def on_white(img: Image.Image, scale: float, size: tuple[int, int] = (SIZE, SIZE), dx: int = 0) -> Image.Image:
    """img shrunk by scale and centred (moved right by dx) on a pure white canvas."""
    small = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", size, WHITE)
    canvas.paste(small, ((size[0] - small.width) // 2 + dx, (size[1] - small.height) // 2))
    return canvas


def text_on(img: Image.Image, text: str, xy, size: int, fill, anchor: str = "mm") -> Image.Image:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(xy, text, font=ImageFont.truetype(FONT, size), fill=fill, anchor=anchor)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def watermark(img: Image.Image) -> Image.Image:
    """A faint store mark in the bottom-right corner: about 15% opacity, easy to miss at full-photo size."""
    return text_on(img, "© SAMPLE STORE", (SIZE - 90, SIZE - 80), 64, (90, 90, 90, 40), "rb")


def variants() -> dict[str, Image.Image]:
    mug = base("mug_white")
    v = {n: base(n) for n in ("mug_white", "spoon_white", "tin_white", "whitemug_white", "mug_slogan", "mug_wood",
                              "spoon_wood", "tin_wood", "mug_offwhite", "mug_creased", "mug_extra")}
    v["mug_tiny"] = on_white(mug, 0.3)
    v["mug_cutoff"] = on_white(mug, 1.0, dx=round(SIZE * 0.42))
    v["mug_wide"] = on_white(mug, 1400 / SIZE, (SIZE, 1400))
    v["mug_blurry"] = mug.filter(ImageFilter.GaussianBlur(18))
    v["mug_400"] = mug.resize((400, 400), Image.LANCZOS)
    v["mug_freeship"] = text_on(mug, "FREE SHIPPING", (SIZE // 2, 170), 170, (220, 30, 40, 255))
    v["mug_border"] = Image.new("RGB", (SIZE, SIZE), (22, 40, 90))
    v["mug_border"].paste(mug.resize((SIZE - 160, SIZE - 160), Image.LANCZOS), (80, 80))
    v["mug_watermark"] = watermark(mug)
    v["tin_watermark"] = watermark(v["tin_white"])
    return v


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with open(HERE / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["photo", "split", "ebay", "walmart", "reason", "width", "height"])
        for name, img in variants().items():
            img.save(OUT / f"{name}.jpg", quality=92)
            w.writerow([name, SPLIT.get(name, "test"), "", "", "", img.width, img.height])     # your answers here


if __name__ == "__main__":
    main()
