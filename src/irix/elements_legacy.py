"""elements — parse a screen into clickable UI ELEMENTS, deterministically, no model.

Two cheap steps: (1) detect rectangular widgets (Canny edges + contours, with an
interior-texture filter so real UI ≠ game art / photos), (2) OCR just inside each
box (cropped + upscaled, so small button text reads cleanly). Yields a precise
element map [{label, box, center}], so "find the OK button" is a text lookup
against detected boxes — pixel-exact and fast. No vision model anywhere.
"""
from __future__ import annotations

from pathlib import Path

from . import ocr


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _dedup(boxes, iou_thresh=0.55):
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        if all(_iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


def _imread(image_or_path):
    import cv2
    return cv2.imread(image_or_path) if isinstance(image_or_path, str) else image_or_path


def detect_boxes(image_or_path) -> list[tuple]:
    """Rectangular widget-like regions, via Canny edges + contours (OpenCV).
    Accepts a path or a BGR ndarray."""
    try:
        import cv2
    except Exception:
        return []
    img = _imread(image_or_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 22 or bh < 12:
            continue
        if bw > w * 0.7 or bh > h * 0.45:
            continue
        ar = bw / max(1, bh)
        if ar < 1.1 or ar > 14:        # widget-ish aspect ratio
            continue
        # SEGMENTATION: real UI element vs decoration/art. A button is FLAT inside
        # (uniform fill + a little text -> few interior edges); background art is
        # BUSY inside (texture everywhere). Keep flat.
        m = 3
        ix0, iy0, ix1, iy1 = x + m, y + m, x + bw - m, y + bh - m
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        interior_edge_density = float(edges[iy0:iy1, ix0:ix1].mean()) / 255.0
        if interior_edge_density > 0.12:    # too textured -> background/art, skip
            continue
        boxes.append((x, y, bw, bh))
    return _dedup(boxes)


def _ocr_crop(gray, box) -> str:
    """OCR a single cropped widget. ONE call: the Irix engine's binarize is
    auto-polarity (dark- and light-mode both read), so the old 4-call
    polarity×psm sweep is redundant. The crop is INSET past the widget's own
    border — a box's outline rectangle reads as ink and wrecks the line
    segmentation (measured: 'Apply' -> 'L%' with the border, exact without).
    Small crops are upscaled 3x so button text stays legible."""
    import cv2
    from PIL import Image
    x, y, bw, bh = box
    m = max(3, bh // 8)                                # step inside the border
    crop = gray[y + m:y + bh - m, x + m:x + bw - m]
    if crop.size == 0:
        return ""
    if crop.shape[0] < 48:                             # small text: upscale, then
        crop = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                          interpolation=cv2.INTER_CUBIC)
        # unsharp mask: anti-aliasing turns thin strokes mid-gray and Otsu
        # then breaks them — restore stroke contrast before binarization
        # (measured: 'R;ame'/'\"xt' -> 'Rename'/'Next' on 14px AA text)
        crop = cv2.addWeighted(crop, 1.8, cv2.GaussianBlur(crop, (0, 0), 2.0), -0.8, 0)
    txt = ocr.run_image(Image.fromarray(crop), timeout=8)
    return " ".join(txt.split()).strip()


def parse(image_or_path) -> list[dict]:
    """Every detected box with its OCR'd label + click centre. Accepts a path
    or a BGR ndarray; the image is decoded ONCE for the whole parse."""
    import cv2
    img = _imread(image_or_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    out = []
    for box in detect_boxes(img):
        x, y, bw, bh = box
        out.append({"label": _ocr_crop(gray, box),
                    "x": x, "y": y, "w": bw, "h": bh,
                    "cx": x + bw // 2, "cy": y + bh // 2})
    return out


def find(image_path: str, text: str) -> dict:
    """The element whose label best matches `text`. {ok,x,y,label,box} or {error}.
    Deterministic scoring: exact > substring > token-overlap."""
    t = text.lower().strip()
    toks = [w for w in t.split() if len(w) > 1]
    best, best_score = None, 0
    for el in parse(image_path):
        lab = el["label"].lower().strip()
        if not lab:
            continue
        if t == lab:
            score = 4
        elif t in lab or lab in t:
            score = 3
        elif toks and any(tok in lab for tok in toks):
            score = 2
        else:
            score = 0
        if score > best_score:
            best, best_score = el, score
    if best is None:
        return {"error": f"no element matching {text!r}"}
    return {"ok": True, "x": best["cx"], "y": best["cy"],
            "label": best["label"], "box": (best["x"], best["y"], best["w"], best["h"])}


def mark(image_path: str, out_path: str | None = None) -> dict:
    """Set-of-Mark: number every detected box and draw it, returning the annotated
    image path + the boxes with their click centres. The pick is DETERMINISTIC
    (use find() for label lookup) — the drawing is just for inspection/debugging.
    No model picks the number (that was the old VLM path, removed)."""
    import tempfile

    boxes = detect_boxes(image_path)
    if not boxes:
        return {"error": "no boxes detected"}
    from PIL import Image, ImageDraw
    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    marks = []
    for i, (x, y, w, h) in enumerate(boxes):
        d.rectangle([x, y, x + w, y + h], outline=(255, 0, 255), width=2)
        d.rectangle([x, max(0, y - 15), x + 20, y], fill=(255, 0, 255))
        d.text((x + 3, max(0, y - 14)), str(i), fill=(255, 255, 255))
        marks.append({"n": i, "box": (x, y, w, h), "cx": x + w // 2, "cy": y + h // 2})
    out = out_path or str(Path(tempfile.gettempdir()) / "irix_marked.png")
    im.save(out)
    return {"ok": True, "image": out, "marks": marks, "n": len(marks)}
