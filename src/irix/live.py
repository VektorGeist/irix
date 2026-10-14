"""live — the REAL-TIME perception loop. Milliseconds per frame, no OCR in the hot
path. This is the point of Irix: a text brain can't afford per-frame OCR (seconds),
so the loop runs on deterministic primitives (downscale + colour blobs + a change
signature) that cost ~tens of ms, and reserves OCR for a single small region,
on demand, cached. All numpy/OpenCV, no model.

Measured on a 4K frame: signature ~5ms, colour blobs ~13ms, downscale ~20ms —
i.e. ~25 FPS — versus elements.parse (full-screen OCR) which does not finish in
two minutes. Different tool for a different job.
"""
from __future__ import annotations

import time


def signature(frame, size: int = 32) -> int:
    """Cheap perceptual hash of the frame — the cache key / change detector. ~5ms.
    Only re-run expensive perception (OCR, physics) when this changes."""
    import cv2
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return hash(cv2.resize(g, (size, size)).tobytes())


def blobs(frame, lo, hi, *, downscale: float = 0.5, min_area: int = 30,
          max_n: int = 32) -> list[dict]:
    """Fast colour-blob targets in HSV — the per-frame actionable state (units,
    minimap pings, resource sparkles, highlighted buttons). Returns click centres in
    FULL-frame pixel coords, largest first. ~13ms. No OCR, no model.

    `lo`/`hi` are HSV bounds (tuples). Detection runs on a downscaled copy for speed,
    centres are scaled back up so a click lands true."""
    import cv2
    import numpy as np
    s = cv2.resize(frame, (0, 0), fx=downscale, fy=downscale) if downscale != 1 else frame
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.asarray(lo, np.uint8), np.asarray(hi, np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    inv = 1.0 / downscale
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = cent[i]
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        out.append({"cx": int(cx * inv), "cy": int(cy * inv), "area": int(area * inv * inv),
                    "box": (int(x * inv), int(y * inv), int(w * inv), int(h * inv))})
    return sorted(out, key=lambda b: -b["area"])[:max_n]


def find_template(frame, template, *, region=None, threshold: float = 0.75) -> dict | None:
    """Locate a fixed sprite (a command-card button, an icon) by template match.
    ALWAYS pass a `region` (x,y,w,h) — matching a small template over a full 4K frame
    is ~1s; over a 200px region it is ~ms. Returns {cx,cy,score} in full-frame coords."""
    import cv2
    x, y, w, h = region if region else (0, 0, frame.shape[1], frame.shape[0])
    roi = frame[y:y + h, x:x + w]
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, mx, _, mloc = cv2.minMaxLoc(res)
    if mx < threshold:
        return None
    th, tw = template.shape[:2]
    return {"cx": x + mloc[0] + tw // 2, "cy": y + mloc[1] + th // 2, "score": round(float(mx), 3)}


class Perceiver:
    """Per-frame perception with a signature cache: expensive reads (OCR of a fixed
    region) only recompute when that region actually changed. Keeps the hot loop cheap."""

    def __init__(self):
        self._cache: dict = {}

    def read_region(self, frame, box, *, key: str) -> str:
        """OCR ONE small region, cached by the region's signature. Recomputes only when
        the region changes — so reading a resource counter every frame is near-free after
        the first read. (One OCR call, not four; region-limited.)"""
        import cv2
        from . import ocr
        x, y, w, h = box
        crop = frame[y:y + h, x:x + w]
        sig = signature(crop, 16)
        hit = self._cache.get(key)
        if hit and hit[0] == sig:
            return hit[1]
        from PIL import Image
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        txt = " ".join(ocr.run_image(Image.fromarray(rgb)).split()).strip()
        self._cache[key] = (sig, txt)
        return txt


def benchmark(frame, *, ranges=((40, 60, 60), (80, 255, 255))) -> dict:
    """Time the hot-path primitives on `frame` so the loop budget is measured, not assumed."""
    out = {}
    t = time.time(); signature(frame); out["signature_ms"] = round((time.time() - t) * 1000, 1)
    t = time.time(); b = blobs(frame, ranges[0], ranges[1]); out["blobs_ms"] = round((time.time() - t) * 1000, 1)
    out["n_blobs"] = len(b)
    out["hot_loop_ms"] = round(out["signature_ms"] + out["blobs_ms"], 1)
    return out
