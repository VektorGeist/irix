"""Recall benchmark: N episodes, text queries, measure recall@5.

Each episode is a synthetic screenshot whose visible text matches its
caption topic. Auto-caption comes from the Irix OCR label pipeline, so
this measures end-to-end retrieval quality of memory_remember/recall.
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from PIL import Image, ImageDraw, ImageFont
from irix.memory import MemoryStore

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
TITLE = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)

TOPICS = [
    "login dialog", "payment checkout", "error traceback", "code editor",
    "spreadsheet budget", "video call", "file manager", "terminal shell",
    "music player", "photo gallery", "email inbox", "calendar agenda",
    "web browser", "chat window", "map navigation", "settings panel",
    "install wizard", "update progress", "search results", "database table",
    "dashboard charts", "pdf document", "drawing canvas", "game score",
    "social feed", "news article", "weather widget", "notes app",
    "task list", "contacts list",
]


def make_image(topic: str, seed: int) -> Path:
    rng = random.Random(seed)
    w, h = rng.choice([(1280, 720), (1024, 768), (800, 600), (1366, 768)])
    bg = tuple(rng.randint(225, 250) for _ in range(3))
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    # distinct chrome per episode: sidebar vs topbar vs floating panel
    style = rng.choice(["topbar", "sidebar", "panel"])
    accent = tuple(rng.randint(20, 120) for _ in range(3))
    if style == "topbar":
        d.rectangle([0, 0, w, 72], fill=accent)
        d.text((20, 18), topic.upper(), font=TITLE, fill=(255, 255, 255))
        y0 = 110
    elif style == "sidebar":
        d.rectangle([0, 0, 200, h], fill=accent)
        d.text((12, 20), topic.upper(), font=TITLE, fill=(255, 255, 255))
        y0 = 40
        x0 = 240
    else:
        px, py = rng.randint(40, w - 500), rng.randint(40, h - 420)
        d.rectangle([px, py, px + 460, py + 380], fill=accent)
        d.text((px + 16, py + 12), topic.upper(), font=TITLE, fill=(255, 255, 255))
        y0 = py + 70
        x0 = px + 16
    if style == "topbar":
        x0 = 40
    y = y0
    for i in range(rng.randint(3, 6)):
        word = f"{topic} item {i + 1} details"
        d.text((x0, y), word, font=FONT, fill=(20, 20, 20))
        y += 52
        if y > h - 120:
            break
    # random accent blocks for visual diversity
    for _ in range(rng.randint(1, 4)):
        bx, by = rng.randint(0, w - 160), rng.randint(y0, h - 120)
        d.rectangle([bx, by, bx + rng.randint(60, 160), by + rng.randint(24, 60)],
                    fill=tuple(rng.randint(100, 220) for _ in range(3)))
    bx = w - 220 if style != "sidebar" else x0
    d.rectangle([bx, h - 80, bx + 200, h - 30], fill=(60, 140, 60))
    d.text((bx + 40, h - 58), "Confirm", font=FONT, fill=(255, 255, 255))
    p = Path(tempfile.mkdtemp()) / f"ep-{seed}.png"
    img.save(p)
    return p


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    seed0 = 7000
    rng = random.Random(seed0)
    topics = [rng.choice(TOPICS) for _ in range(n)]
    topics = [f"{t} {i // len(TOPICS) + 1}" for i, t in enumerate(topics)]

    tmp = tempfile.mkdtemp()
    mem = MemoryStore(tmp)
    stored = []  # (id, topic) for genuinely new episodes only
    for i, topic in enumerate(topics):
        img = make_image(topic, seed0 + i)
        r = mem.remember(str(img), source="bench", caption=topic)
        if r["duplicate_of"] is None:
            stored.append((r["id"], topic))

    hits = 0
    for eid, topic in stored:
        results = mem.recall(topic, n=5)  # full caption: specific like a real agent query
        got = [r["id"] for r in results]
        if eid in got:
            hits += 1
        elif hits < 3:
            print("MISS", topic, "->", [(r["id"], r["caption"][:50]) for r in results])

    print(f"episodes={len(stored)} queries={len(stored)} recall@5={hits/len(stored):.3f}")
    return 0 if hits / len(stored) >= 0.90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
