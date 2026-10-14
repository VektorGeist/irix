"""fuse — the hybrid element map: accessibility tree WHERE apps expose it,
pixel CV everywhere else. The coverage play no pure-VLM screen parser makes.

UIA gives OS-exact rectangles + real names for free, but only for apps that
implement the accessibility tree — nothing for games, canvas/WebGL apps,
custom-drawn UIs, remote/streamed screens, or the VektOS target. Pixel CV
(`elements`) sees ALL of those but must detect + OCR. Fusing them:

  * every UIA element is kept (exact, trusted);
  * a pixel-CV box is added only where UIA has NOTHING (no overlap) — it fills
    the gaps UIA can't see;
  * where both see the same widget, the UIA label can REPAIR a shaky OCR read
    (and vice-versa: a pixel box with text confirms a nameless UIA node).

Deterministic, model-free, and it degrades gracefully: no UIA (non-Windows,
games) -> pure pixel CV; no pixels (a handle with no capture) -> pure UIA.
"""
from __future__ import annotations


def _iou(a, b) -> float:
    ax, ay, aw, ah = a["x"], a["y"], a["w"], a["h"]
    bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]
    ix, iy = max(ax, bx), max(ay, by)
    iw, ih = min(ax + aw, bx + bw) - ix, min(ay + ah, by + bh) - iy
    inter = max(0, iw) * max(0, ih)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _contains(outer, inner, frac: float = 0.75) -> bool:
    ix = max(outer["x"], inner["x"])
    iy = max(outer["y"], inner["y"])
    iw = min(outer["x"] + outer["w"], inner["x"] + inner["w"]) - ix
    ih = min(outer["y"] + outer["h"], inner["y"] + inner["h"]) - iy
    inter = max(0, iw) * max(0, ih)
    area = inner["w"] * inner["h"]
    return area > 0 and inter / area >= frac


def fuse(uia_els: list[dict], cv_els: list[dict], iou_thresh: float = 0.4) -> list[dict]:
    """Merge a UIA element list with a pixel-CV element list. UIA wins on
    geometry; pixel CV fills the gaps and can lend a label to a nameless UIA
    node. Returns one element map, each tagged with its `source`."""
    out = [dict(e) for e in uia_els]
    for u in out:
        if u.get("label"):
            continue
        for c in cv_els:                                # borrow a label for a
            if c.get("label") and _contains(u, c):      # nameless UIA node
                u["label"], u["labeled_by"] = c["label"], "cv"
                break
    for c in cv_els:
        best = max((u for u in uia_els), key=lambda u: _iou(u, c), default=None)
        if best is not None and (_iou(best, c) >= iou_thresh or _contains(best, c)):
            continue                                    # UIA already covers it
        out.append({**c, "source": c.get("source", "cv")})
    return out


def parse(image_or_path=None, hwnd: int | None = None,
          interactive_only: bool = False) -> list[dict]:
    """The best available element map of the current screen/window.

    Reads the UIA tree (if a window is available) AND pixel CV (if an image is
    given), and fuses them. Pass `image_or_path` for the pixel half; pass
    `hwnd` to target a specific window (default: the foreground window). With
    neither usable it returns whatever one source yields — never raises."""
    from . import uia
    uia_els = uia.elements(hwnd=hwnd, interactive_only=interactive_only) \
        if uia.available() else []
    cv_els = []
    if image_or_path is not None:
        try:
            from . import elements
            cv_els = [{**e, "source": "cv", "interactive": True}
                      for e in elements.parse(image_or_path)]
        except Exception:
            cv_els = []
    fused = fuse(uia_els, cv_els)
    return [e for e in fused if e.get("interactive", True)] if interactive_only else fused


def find(text: str, image_or_path=None, hwnd: int | None = None) -> dict:
    """Locate an element by label across the fused map. UIA-named elements make
    this exact where the tree exists; pixel CV covers the rest. Deterministic
    scoring: exact > substring > token-overlap; ties broken toward UIA."""
    t = text.lower().strip()
    toks = [w for w in t.split() if len(w) > 1]
    best, best_score = None, 0.0
    for el in parse(image_or_path, hwnd=hwnd):
        lab = (el.get("label") or "").lower().strip()
        if not lab:
            continue
        if t == lab:
            score = 4.0
        elif t in lab or lab in t:
            score = 3.0
        elif toks and any(tok in lab for tok in toks):
            score = 2.0
        else:
            continue
        score += 0.5 if el.get("source") == "uia" else 0.0     # trust exact geometry
        if score > best_score:
            best, best_score = el, score
    if best is None:
        return {"error": f"no element matching {text!r}"}
    return {"ok": True, "x": best["cx"], "y": best["cy"], "label": best["label"],
            "source": best.get("source"), "control": best.get("control"),
            "box": (best["x"], best["y"], best["w"], best["h"])}
