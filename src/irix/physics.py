"""physics — model-free PHYSICAL-ATTRIBUTE extraction (Irix's half of semantic vision).

The breakthrough (Flouk 2026-07-01): a model-free eye can never *name* what it sees from
pixels — but it does not have to. It measures an object's PHYSICS (shape, whether it emits
or reflects light, colour temperature, where it sits, how its surface behaves over time),
and a REASONING mind (a language model, elsewhere — never here) deduces the identity from
world knowledge, the way a blind person names a TV from "a flat thing on the wall that lights
itself up and shows moving pictures". Irix stays 100% model-free: it describes, it never names.

This module extracts, per region, a rich attribute dict and renders a NEUTRAL physical
description (no object nouns). All deterministic OpenCV/NumPy — no model, ever.
"""
from __future__ import annotations

import numpy as np

_BRIGHT = 140            # gray level considered "lit" for blob finding
_EMISSIVE_RATIO = 1.8    # region brightness / surround brightness above which it is self-lit
_SPECULAR = 240          # near-white = glare / shiny highlight


def _gray(img):
    import cv2
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------- finding regions
def salient_regions(frame, *, max_n: int = 5, min_area_frac: float = 0.003) -> list[tuple]:
    """Candidate regions worth identifying: self-lit blobs (bright) AND contrast contours
    (CLAHE-lifted so a dark room still yields structure). Deduped, largest-first. Model-free."""
    import cv2
    g = _gray(frame)
    H, W = g.shape[:2]
    boxes: list[tuple] = []
    # 1) self-lit blobs
    _, th = cv2.threshold(g, _BRIGHT, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    for c in cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) >= min_area_frac * H * W:
            boxes.append(cv2.boundingRect(c))
    # 2) contrast contours on a CLAHE-normalised image (reveals structure in shadow)
    eq = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    ed = cv2.dilate(cv2.Canny(eq, 60, 160), np.ones((7, 7), np.uint8))
    for c in cv2.findContours(ed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) >= min_area_frac * H * W:
            boxes.append(cv2.boundingRect(c))

    def _inter(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix, iy = max(ax, bx), max(ay, by)
        iw, ih = min(ax + aw, bx + bw) - ix, min(ay + ah, by + bh) - iy
        return max(0, iw) * max(0, ih)

    def iou(a, b):
        i = _inter(a, b)
        u = a[2] * a[3] + b[2] * b[3] - i
        return i / u if u else 0.0

    def contained(b, k):                # fraction of b that sits inside k
        return _inter(b, k) / (b[2] * b[3]) if b[2] * b[3] else 0.0

    # Largest-first, so a smaller box NESTED inside a kept one (a screen's interior inside its
    # frame) is merged away — otherwise one object splits into several identifications.
    kept: list[tuple] = []
    for b in sorted(boxes, key=lambda b: -b[2] * b[3]):
        if any(iou(b, k) >= 0.5 or contained(b, k) > 0.6 for k in kept):
            continue
        kept.append(b)
        if len(kept) >= max_n:
            break
    return kept


# ---------------------------------------------------------------- attribute extraction
def _surround_brightness(g, box) -> float:
    """Mean brightness of a ring just outside the box (the object's local backdrop)."""
    import cv2  # noqa: F401
    H, W = g.shape[:2]
    x, y, w, h = box
    pad = max(6, (w + h) // 8)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    outer = g[y0:y1, x0:x1].astype(np.float32).sum()
    inner = g[y:y + h, x:x + w].astype(np.float32).sum()
    n = (y1 - y0) * (x1 - x0) - w * h
    return float((outer - inner) / n) if n > 0 else float(g.mean())


def _shape(crop_gray) -> tuple[str, int]:
    """Coarse shape from the region's own contour: (rectangular|round|irregular, n_corners).
    Otsu-thresholded so it's robust whether the region is a uniform bright block or a shape
    on a darker border."""
    import cv2
    cg = crop_gray if crop_gray.dtype == np.uint8 else crop_gray.astype(np.uint8)
    _, th = cv2.threshold(cg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not cnts:
        return "irregular", 0
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    if peri <= 0:
        return "irregular", 0
    n = len(cv2.approxPolyDP(c, 0.04 * peri, True))
    circ = 4 * np.pi * cv2.contourArea(c) / (peri * peri)
    if 4 <= n <= 6 and circ < 0.85:                 # a square is circ~0.79 — keep it rectangular
        return "rectangular", n
    if circ > 0.72:
        return "round", n
    return "irregular", n


def region_attributes(frame, box, *, prev_frames=None) -> dict:
    """Model-free physics of one region. `prev_frames` (same view, older) enables the temporal
    signature: a self-updating surface (interior changes, outline fixed) vs static vs moving."""
    import cv2
    g = _gray(frame)
    H, W = g.shape[:2]
    x, y, w, h = box
    crop = frame[y:y + h, x:x + w]
    cg = g[y:y + h, x:x + w]

    region_b = float(cg.mean())
    surround_b = _surround_brightness(g, box)
    emissive_ratio = region_b / max(1.0, surround_b)
    emissive = emissive_ratio >= _EMISSIVE_RATIO

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) if crop.ndim == 3 else None
    sat = float(hsv[:, :, 1].mean()) if hsv is not None else 0.0
    B, G, R = (float(crop[:, :, i].mean()) for i in range(3)) if crop.ndim == 3 else (0, 0, 0)
    warmth = R - B                                  # >0 warm (incandescent), <0 cool (daylight/LED)
    specular = float((cg > _SPECULAR).mean())
    texture = float(cv2.Laplacian(cg, cv2.CV_32F).var())   # busy vs smooth surface

    shape, corners = _shape(cg)
    aspect = round(w / max(1, h), 2)
    rectangularity = int(100 * (cg > max(30, cg.mean())).mean())   # how filled/solid

    cx, cy = x + w / 2, y + h / 2
    vfrac = cy / H
    height = ("overhead" if vfrac < 0.15 else "high on a wall" if vfrac < 0.45 else
              "at eye level" if vfrac < 0.65 else "on a surface / waist height" if vfrac < 0.85 else
              "low / on the floor")
    horiz = "left" if cx < W / 3 else "right" if cx > 2 * W / 3 else "centre"
    # is something solid directly below? (a support surface -> sits ON something; else wall/floating)
    bx0, bx1 = x, x + w
    below = g[min(H - 1, y + h):min(H, y + h + max(8, h // 4)), bx0:bx1]
    supported = bool(below.size and float(below.mean()) > surround_b * 1.3)

    # Temporal signature, aggregated over the whole window: a screen changes INTERMITTENTLY,
    # so use the PEAK interior change (mean under-counts it) and the brightness swing over time
    # (an emitting screen's output varies; a lamp/light is steady). This is what separates a
    # screen from a lamp when both are just "a bright rectangle" in one frame.
    temporal = "unknown"
    interior_change = None
    brightness_swing = None
    if prev_frames:
        chs, brights = [], [region_b]
        cur48 = cv2.resize(cg, (48, 48)).astype(np.float32)
        for pf in prev_frames:
            pg = _gray(pf)
            if pg.shape == g.shape:
                pc = pg[y:y + h, x:x + w]
                brights.append(float(pc.mean()))
                chs.append(float((np.abs(cv2.resize(pc, (48, 48)).astype(np.float32) - cur48) > 25).mean()))
        if chs:
            interior_change = float(np.max(chs))              # peak, not mean
            brightness_swing = round(max(brights) - min(brights))
            self_updating = interior_change > 0.05 or brightness_swing > 25
            temporal = ("a self-updating surface (its pattern and/or brightness change over time "
                        "while its outline stays fixed)" if self_updating else "static / unchanging")

    return {
        "aspect_ratio": aspect,
        "shape": shape,
        "corners": corners,
        "solidity_pct": rectangularity,
        "size_pct": round(100 * w * h / (W * H), 1),
        "emissive": bool(emissive),
        "emissive_ratio": round(float(emissive_ratio), 1),
        "brightness": round(region_b),
        "specular_frac": round(specular, 3),
        "saturation": round(sat),
        "warmth": round(warmth),
        "colour_tone": "warm" if warmth > 12 else "cool/daylight" if warmth < -12 else "neutral",
        "texture_var": round(texture),
        "height": height,
        "horizontal": horiz,
        "supported_below": supported,
        "temporal": temporal,
        "interior_change": None if interior_change is None else round(interior_change, 3),
        "brightness_swing": brightness_swing,
    }


# ---------------------------------------------------------------- neutral description
def physical_description(attrs: dict) -> str:
    """Render attributes as NEUTRAL physical prose — no object nouns. This is what a reasoning
    mind reads to deduce identity (it never sees pixels)."""
    lines = []
    if attrs.get("emissive"):
        lines.append(f"- It emits its own light — about {attrs['emissive_ratio']}x brighter than "
                     f"its surroundings.")
    else:
        lines.append(f"- It does not emit light (reflective); brightness {attrs['brightness']}/255.")
    lines.append(f"- Shape: {attrs['shape']}, about {attrs['aspect_ratio']}:1, "
                 f"{attrs['solidity_pct']}% solid, ~{attrs['size_pct']}% of the view.")
    lines.append(f"- Colour: {attrs['colour_tone']} tone, saturation {attrs['saturation']}/255"
                 + (", with bright glints/highlights" if attrs.get('specular_frac', 0) > 0.02 else "")
                 + (", busy/detailed surface" if attrs.get('texture_var', 0) > 400 else
                    ", smooth surface") + ".")
    place = f"- Position: {attrs['height']}, toward the {attrs['horizontal']}"
    place += (", resting on a surface below it" if attrs.get("supported_below")
              else ", with nothing supporting it below (mounted or floating)")
    lines.append(place + ".")
    if attrs.get("temporal") not in (None, "unknown"):
        lines.append(f"- Over time: {attrs['temporal']}.")
    return "\n".join(lines)
