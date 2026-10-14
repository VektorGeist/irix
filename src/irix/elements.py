"""elements — clickable UI element detection across ALL interfaces, deterministically, no model.

Five complementary stages (all OpenCV + NumPy, CPU-only):

  stage A: original edge-contour pipeline, Canny(50,150)      [desktop-crisp]
  stage B: low-threshold Canny(15,45) contour pipeline        [soft/AA borders,
                                                              phone dark mode]
  stage C: MSER fill-stable regions                          [rounded widgets,
                                                              low-contrast fills,
                                                              icon-only targets]
  stage D: link-styled text (blue hue + underline)            [web links]
  stage E: text-context suppression (label + geometry fusion) [kills word-
           fragment and text-block false positives: a small box embedded
           mid-text-line that OCRs as text-like is a word piece, not a widget;
           a box holding 3+ text lines is a text block, not a widget]

Platform profiles (desktop / phone / web / auto) tune the parameters; 'auto'
selects by aspect ratio, deterministically. The pre-extension implementation
is preserved byte-for-byte as irix.elements_legacy for fallback/comparison.

Public API (unchanged): detect_boxes(image_or_path, profile='auto'), parse,
find, mark. The profile kwarg is optional — all existing positional callers
(mcp_server, cli, fuse, scene, screen, memory, bench) work untouched.
"""
from __future__ import annotations

from pathlib import Path

from irix import ocr

# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------

PROFILES = {
    "desktop": dict(
        canny_pairs=[(50, 150), (15, 45)],
        mser=True, mser_delta=5, mser_min_area=2000, mser_max_area_frac=0.35,
        min_bw=22, min_bh=12, ar=(1.0, 14.0), density_max=0.12,
        density_max_small=0.12,
        fill_std_max=25.0, surround_min=3.0, fill_ratio_min=0.45,
        links=True,
    ),
    "phone": dict(
        canny_pairs=[(50, 150), (15, 45)],
        mser=True, mser_delta=5, mser_min_area=1200, mser_max_area_frac=0.40,
        min_bw=16, min_bh=10, ar=(1.0, 14.0), density_max=0.12,
        density_max_small=0.30,
        fill_std_max=30.0, surround_min=2.0, fill_ratio_min=0.45,
        links=True,
    ),
    "web": dict(
        canny_pairs=[(50, 150), (15, 45)],
        mser=True, mser_delta=5, mser_min_area=1500, mser_max_area_frac=0.35,
        min_bw=18, min_bh=10, ar=(0.5, 14.0), density_max=0.13,
        density_max_small=0.30,
        fill_std_max=28.0, surround_min=2.5, fill_ratio_min=0.45,
        links=True,
    ),
}


def _auto_profile(img) -> str:
    h, w = img.shape[:2]
    return "phone" if (h / max(1, w)) >= 1.4 else "desktop"


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


# ---------------------------------------------------------------------------
# stage A/B: edge-contour widget detection (parameterized original)
# ---------------------------------------------------------------------------

def _texture_ok(x, y, bw, bh, edges, P) -> bool:
    """Interior-texture gate shared by all stages. Small near-square widgets
    (icons) may have interior detail; wide boxes are words; large busy
    regions are art."""
    m = 3
    ix0, iy0, ix1, iy1 = x + m, y + m, x + bw - m, y + bh - m
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    interior = float(edges[iy0:iy1, ix0:ix1].mean()) / 255.0
    ar = bw / max(1, bh)
    dmax = P["density_max"]
    if bw * bh <= 15000 and 0.6 <= ar <= 1.6:
        dmax = P["density_max_small"]
    return interior <= dmax


def _surround_ok(x, y, bw, bh, gray, P) -> bool:
    """The box must differ from its immediate surroundings. Kills fragments
    of smooth gradients/art (surrounded by near-identical pixels) while
    keeping real widgets, even low-contrast ones (a few gray levels)."""
    import cv2
    import numpy as np
    h, w = gray.shape
    m = 10
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(w, x + bw + m), min(h, y + bh + m)
    crop = gray[y0:y1, x0:x1]
    mh, mw = crop.shape
    if mh == 0 or mw == 0:
        return False
    mask = np.zeros((mh, mw), np.uint8)
    mask[y - y0:y - y0 + bh, x - x0:x - x0 + bw] = 1
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * m + 1, 2 * m + 1))
    ring = cv2.dilate(mask, k)
    ring[mask > 0] = 0
    if ring.sum() == 0:
        return False
    return abs(float(crop[mask > 0].mean()) - float(crop[ring > 0].mean())) >= P["surround_min"]


def _contour_boxes(gray, edges, w, h, P) -> list:
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < P["min_bw"] or bh < P["min_bh"]:
            continue
        if bw > w * 0.7 or bh > h * 0.45:
            continue
        ar = bw / max(1, bh)
        if ar < P["ar"][0] or ar > P["ar"][1]:
            continue
        if not _texture_ok(x, y, bw, bh, edges, P):
            continue
        if not _surround_ok(x, y, bw, bh, gray, P):
            continue
        boxes.append((x, y, bw, bh))
    return boxes


def _edge_stage(gray, P) -> list:
    import cv2
    boxes = []
    for lo, hi in P["canny_pairs"]:
        edges = cv2.Canny(gray, lo, hi)
        boxes += _contour_boxes(gray, edges, gray.shape[1], gray.shape[0], P)
    return boxes


# ---------------------------------------------------------------------------
# stage C: MSER fill-stable regions (rounded + low-contrast + icons)
# ---------------------------------------------------------------------------

def _mser_boxes(gray, P) -> list:
    import cv2
    import numpy as np
    h, w = gray.shape
    mser = cv2.MSER_create()
    mser.setDelta(P["mser_delta"])
    mser.setMinArea(P["mser_min_area"])
    mser.setMaxArea(int(w * h * P["mser_max_area_frac"]))
    mser.setMaxVariation(0.25)
    mser.setMinDiversity(0.2)
    regions, bboxes = mser.detectRegions(gray)
    edges = cv2.Canny(gray, 50, 150)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    out = []
    for pts, (x, y, bw, bh) in zip(regions, bboxes):
        x, y, bw, bh = int(x), int(y), int(bw), int(bh)
        ar = bw / max(1, bh)
        if ar < P["ar"][0] or ar > P["ar"][1]:
            continue
        pts = np.asarray(pts).reshape(-1, 2)
        if len(pts) < P["mser_min_area"]:
            continue
        fill_ratio = len(pts) / max(1, bw * bh)
        if fill_ratio < P["fill_ratio_min"]:
            continue
        vals = gray[pts[:, 1].clip(0, h - 1), pts[:, 0].clip(0, w - 1)].astype(float)
        if vals.std() > P["fill_std_max"]:
            continue
        if not _texture_ok(x, y, bw, bh, edges, P):
            continue
        # surround contrast: region mean vs ring just outside the region
        mask = np.zeros((h, w), np.uint8)
        mask[pts[:, 1].clip(0, h - 1), pts[:, 0].clip(0, w - 1)] = 1
        ring = cv2.dilate(mask, k9) - cv2.dilate(mask, k3)
        if ring.sum() == 0:
            continue
        if abs(float(vals.mean()) - float(gray[ring > 0].mean())) < P["surround_min"]:
            continue
        out.append((x, y, bw, bh))
    return out


# ---------------------------------------------------------------------------
# stage D: link-styled text (blue hue + underline)
# ---------------------------------------------------------------------------

def _link_boxes(img_bgr, gray, P) -> list:
    import cv2
    import numpy as np
    h, w = gray.shape
    boxes = []
    # D1: blue-hue text (HSV). OpenCV H: blue ~ 100-125.
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([95, 90, 60]), np.array([125, 255, 255]))
    edges = cv2.Canny(gray, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    blue_c = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(blue_c)
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if bh < 10 or bh > 70 or bw < 2 * bh or area < 300:
            continue
        if bw > w * 0.7:
            continue
        # filled-widget rejection: a flat blue BUTTON is blue too, but has
        # almost no interior edges; link TEXT has glyph-stroke edges.
        inner = edges[y:y + bh, x:x + bw]
        if inner.size and float(inner.mean()) / 255.0 < 0.04:
            continue
        boxes.append((x, y, bw, bh))
    # D2: blue underlines — thin horizontal blue segments with text above.
    # (Non-blue underlines are rare in modern UI; accepted gap.)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    horiz = cv2.morphologyEx(edges, cv2.MORPH_OPEN, hk)
    contours, _ = cv2.findContours(horiz, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 40 or bh > 6:
            continue
        # the segment itself must be blue (kills gray divider lines,
        # button top edges, and other non-link horizontals)
        seg_blue = blue[y:y + bh, x:x + bw]
        if seg_blue.size == 0 or float(seg_blue.mean()) / 255.0 < 0.15:
            continue
        y0 = max(0, y - 40)
        band = edges[y0:y, x:x + bw]
        if band.size == 0:
            continue
        if float(band.mean()) / 255.0 < 0.02:
            continue
        boxes.append((x, y0, bw, (y + bh) - y0))
    return boxes


# ---------------------------------------------------------------------------
# stage E: text-context suppression (label + geometry fusion)
#
# The edge/MSER stages over-detect on text-heavy screens: word fragments
# ("tn", "nt at") and whole text blocks get boxed as if they were widgets.
# Pure geometry cannot separate a word fragment from an icon, so we fuse in
# two cheap deterministic signals:
#   1. text-line bands (horizontal closing on the edge map) tell us where
#      body text lives;
#   2. the OCR label of a candidate box tells us whether it READS as text.
# Rules:
#   E1 container: a box holding >=3 text lines is a text block, not a widget.
#   E2 word fragment: a small box embedded in a text line whose crop OCRs as
#      text-like (>=2 alpha chars) is a piece of a word, not a widget.
#      Link-styled boxes (stage D) are exempt: links ARE text and ARE clickable.
# ---------------------------------------------------------------------------

_ALPHA_RE = None


def _is_text_like(label: str) -> bool:
    global _ALPHA_RE
    if _ALPHA_RE is None:
        import re
        _ALPHA_RE = re.compile(r"[A-Za-z]")
    return len(_ALPHA_RE.findall(label)) >= 2


def _text_line_boxes(gray) -> list:
    """Tight text-line bands: horizontal morphological closing on Canny edges
    joins words into lines; connected components give one box per line.
    The kernel (25,9) joins intra-line word gaps but NOT the wider gaps
    between side-by-side widgets, so a button row does not become one band."""
    import cv2
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    lines = []
    for i in range(1, n):
        x, y, bw, bh = (int(v) for v in stats[i][:4])
        if bh < 8 or bh > 90:
            continue
        if bw < 2.5 * bh:
            continue
        lines.append((x, y, bw, bh))
    return lines


def _suppress_text_fps(boxes: list, gray, link_boxes: list) -> list:
    lines = _text_line_boxes(gray)
    out = []
    for b in boxes:
        x, y, bw, bh = b
        cx, cy = x + bw / 2.0, y + bh / 2.0
        # E1: container — holds 3+ text lines -> text block, not a widget
        n_in = sum(1 for (lx, ly, lw, lh) in lines
                   if x <= lx + lw / 2 <= x + bw and y <= ly + lh / 2 <= y + bh)
        if n_in >= 3:
            continue
        # E2: word fragment — small box embedded MID-LINE (text continues well
        # past it on BOTH sides) whose crop OCRs as text-like. Icons and
        # thumbnails OCR as garbage/empty and are kept; a widget's own text
        # sits at the edge of its band (or is the whole band), so the
        # both-sides test lets buttons in merged rows survive. Link-styled
        # boxes are clickable text and are exempt.
        if bw * bh < 12000:
            drop = False
            for (lx, ly, lw, lh) in lines:
                if lx <= cx <= lx + lw and ly <= cy <= ly + lh:
                    m = max(8.0, bw * 0.25)
                    if lx < x - m and lx + lw > x + bw + m:
                        if any(_iou(b, lb) > 0.5 for lb in link_boxes):
                            break  # link-styled: keep
                        if _is_text_like(_ocr_crop(gray, b)):
                            drop = True
                        break
            if drop:
                continue
        out.append(b)
    return out


# ---------------------------------------------------------------------------
# public API (compatible with irix.elements)
# ---------------------------------------------------------------------------

def detect_boxes(image_or_path, profile="auto") -> list:
    """Clickable UI element boxes across interfaces. profile: auto/desktop/phone/web."""
    import cv2
    img = _imread(image_or_path)
    if img is None:
        return []
    if profile == "auto":
        profile = _auto_profile(img)
    P = PROFILES[profile]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    boxes = _edge_stage(gray, P)                       # A+B
    link_boxes = []
    if P["mser"]:
        boxes += _mser_boxes(gray, P)                  # C
    if P["links"] and img.ndim == 3:
        link_boxes = _link_boxes(img, gray, P)         # D
        boxes += link_boxes
    boxes = _dedup(boxes)
    boxes = _suppress_text_fps(boxes, gray, link_boxes)  # E
    # status bar carries no tappable UI widgets (phone profile only)
    if P is PROFILES["phone"]:
        boxes = [b for b in boxes if b[1] + b[3] >= 70]
    return _dedup(boxes)


def _ocr_crop(gray, box) -> str:
    """Same single-call inset OCR as irix.elements."""
    import cv2
    from PIL import Image
    x, y, bw, bh = box
    m = max(3, bh // 8)
    crop = gray[y + m:y + bh - m, x + m:x + bw - m]
    if crop.size == 0:
        return ""
    if crop.shape[0] < 48:
        crop = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                          interpolation=cv2.INTER_CUBIC)
        crop = cv2.addWeighted(crop, 1.8, cv2.GaussianBlur(crop, (0, 0), 2.0), -0.8, 0)
    txt = ocr.run_image(Image.fromarray(crop), timeout=8)
    return " ".join(txt.split()).strip()


def parse(image_or_path, profile="auto") -> list:
    """Every detected box with OCR label + click centre."""
    import cv2
    img = _imread(image_or_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    out = []
    for box in detect_boxes(img, profile=profile):
        x, y, bw, bh = box
        out.append({"label": _ocr_crop(gray, box),
                    "x": x, "y": y, "w": bw, "h": bh,
                    "cx": x + bw // 2, "cy": y + bh // 2})
    return out


def find(image_path: str, text: str, profile="auto") -> dict:
    """The element whose label best matches `text`. Deterministic scoring."""
    t = text.lower().strip()
    toks = [w for w in t.split() if len(w) > 1]
    best, best_score = None, 0
    for el in parse(image_path, profile=profile):
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


def mark(image_path: str, out_path: str | None = None, profile="auto") -> dict:
    """Number every detected box and draw it (inspection/debugging)."""
    import tempfile
    boxes = detect_boxes(image_path, profile=profile)
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
