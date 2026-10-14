"""bench — the reproducible GUI-grounding benchmark. Model-free, self-contained.

Measures what a screen-vision layer is FOR: given an instruction ("click
Settings"), does the predicted click point land inside the right widget?
That is ScreenSpot's metric (click-in-bbox accuracy), so numbers here are
directly comparable in kind to the public GUI-grounding leaderboards — while
staying runnable by anyone in seconds, on CPU, with `irix bench`.

Two sources:
  * SYNTHETIC screens (`make_screen`) — rendered UIs with exact ground truth:
    light/dark themes, font sizes 12-20px, buttons + distractor panels, an
    optional anti-aliased blur to approximate real rendering. Deterministic
    per seed. Honest scope: crisp synthetic rendering is EASIER than real
    ClearType screens; treat synthetic numbers as the upper bound.
  * A ScreenSpot-style JSONL loader (`load_jsonl`) — one record per line:
    {"image": path, "label": str, "bbox": [x, y, w, h]} for real,
    hand-labelled or downloaded benchmark data.

Reported: grounding accuracy (click-in-bbox), box recall, label accuracy,
and parse latency. No model anywhere in the pipeline being measured.
"""
from __future__ import annotations

import json
import time

_WORDS = ["Open", "Save", "Cancel", "Settings", "Export", "Delete", "Search",
          "Login", "Submit", "Browse", "Next", "Back", "Apply", "Close",
          "Help", "New File", "Download", "Options", "Print", "Rename"]


def make_screen(seed: int = 0, w: int = 1280, h: int = 800,
                dark: bool = False, font_px: int = 15, blur: bool = False):
    """One synthetic UI screen + ground truth [{label, bbox}]. Deterministic."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from .text import system_fonts
    rng = np.random.default_rng(seed)
    bg, panel, fg = ((28, 30, 34), (45, 48, 54), (235, 235, 235)) if dark else \
                    ((242, 243, 245), (255, 255, 255), (25, 25, 25))
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    fonts = system_fonts()
    font = ImageFont.truetype(fonts[int(rng.integers(0, len(fonts)))], font_px)
    d.rectangle([0, 0, w, 34], fill=panel)                      # title bar
    for _ in range(3):                                          # distractor art
        x, y = int(rng.integers(0, w - 220)), int(rng.integers(60, h - 160))
        art = np.array(rng.integers(0, 255, (140, 200, 3)), np.uint8)
        im.paste(Image.fromarray(art), (x, y))
    labels = list(rng.permutation(_WORDS)[:8])
    gt, cells = [], []
    cols, rows = 4, 4
    for i in rng.permutation(cols * rows)[:len(labels)]:        # non-overlapping grid
        cells.append((int(i) % cols, int(i) // cols))
    for label, (cx, cy) in zip(labels, cells):
        tw = int(d.textlength(label, font=font))
        bw, bh = tw + 34, font_px + 18
        x = 40 + cx * (w - 80) // cols + int(rng.integers(0, 30))
        y = 70 + cy * (h - 120) // rows + int(rng.integers(0, 30))
        d.rounded_rectangle([x, y, x + bw, y + bh], radius=5, fill=panel,
                            outline=(120, 122, 128), width=1)
        d.text((x + 17, y + 8), label, fill=fg, font=font)
        gt.append({"label": label, "bbox": [x, y, bw, bh]})
    frame = np.array(im)[:, :, ::-1].copy()                     # RGB -> BGR
    if blur:                                                    # approximate AA
        frame = cv2.GaussianBlur(frame, (3, 3), 0.6)
    return frame, gt


def _inside(pt, bbox) -> bool:
    x, y, bw, bh = bbox
    return x <= pt[0] <= x + bw and y <= pt[1] <= y + bh


def _label_sim(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def evaluate(samples) -> dict:
    """Score grounding on [(frame_or_path, gt_list)] pairs. For each ground-truth
    label: elements.find -> click point -> hit iff inside the true bbox."""
    from . import elements
    n = hits = boxes_found = labels_right = 0
    lat = []
    for frame, gt in samples:
        t0 = time.time()
        els = elements.parse(frame)
        lat.append(time.time() - t0)
        for item in gt:
            n += 1
            if any(_inside((e["cx"], e["cy"]), item["bbox"]) for e in els):
                boxes_found += 1
            best, score = None, 0.0
            for e in els:
                if not e["label"]:
                    continue
                s = _label_sim(e["label"], item["label"])
                if s > score:
                    best, score = e, s
            if best and score >= 0.6:
                if _label_sim(best["label"], item["label"]) >= 0.99:
                    labels_right += 1
                if _inside((best["cx"], best["cy"]), item["bbox"]):
                    hits += 1
    return {"ok": True, "n": n,
            "grounding_acc": round(hits / n, 3) if n else 0.0,
            "box_recall": round(boxes_found / n, 3) if n else 0.0,
            "label_exact": round(labels_right / n, 3) if n else 0.0,
            "parse_s_mean": round(sum(lat) / len(lat), 3) if lat else None,
            "screens": len(lat)}


def run(n_screens: int = 12, seed: int = 0, blur: bool = True) -> dict:
    """The standard synthetic run: half light / half dark, font sizes 12-20px,
    anti-aliased. Deterministic per seed — the number anyone can reproduce."""
    samples = []
    for i in range(n_screens):
        samples.append(make_screen(seed=seed + i, dark=(i % 2 == 1),
                                   font_px=12 + (i % 5) * 2, blur=blur))
    out = evaluate(samples)
    out["source"] = f"synthetic(n={n_screens}, seed={seed}, blur={blur})"
    return out


def load_jsonl(path: str):
    """ScreenSpot-style records -> evaluate() samples. One JSON per line:
    {"image": path, "label": str, "bbox": [x,y,w,h]} — records for the same
    image are grouped into one screen."""
    by_img: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_img.setdefault(r["image"], []).append(
                {"label": r["label"], "bbox": r["bbox"]})
    return list(by_img.items())
