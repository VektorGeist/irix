"""Visual memory demo: enroll three screenshots, recall one by text.

Run:  python demo_memory.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from PIL import Image, ImageDraw, ImageFont
from irix.memory import MemoryStore

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)


def shot(text: str, accent: tuple, path: Path) -> None:
    img = Image.new("RGB", (800, 500), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 800, 64], fill=accent)
    d.text((20, 16), text, font=FONT, fill=(255, 255, 255))
    d.text((40, 120), f"details about {text}", font=FONT, fill=(20, 20, 20))
    d.rectangle([40, 200, 400, 320], outline=accent, width=4)
    img.save(path)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="irix-mem-demo-"))
    mem = MemoryStore(tmp)

    scenes = [("payment checkout", (30, 120, 60)),
              ("login dialog", (30, 80, 160)),
              ("error traceback", (160, 40, 40))]
    for s, accent in scenes:
        p = tmp / f"{s.replace(' ', '-')}.png"
        shot(s, accent, p)
        r = mem.remember(str(p), source="demo", caption=s)
        print(f"remembered '{s}' -> id {r['id']}")

    print("\nrecall 'payment':")
    for hit in mem.recall("payment"):
        print(f"  id={hit['id']} caption={hit['caption']!r}")
        print(f"  image={hit['image_path']}")

    print("\nrecall 'error':")
    for hit in mem.recall("error"):
        print(f"  id={hit['id']} caption={hit['caption']!r}")

    print(f"\nmemory lives at {tmp}")


if __name__ == "__main__":
    main()
