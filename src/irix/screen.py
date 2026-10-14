"""screen — perceive a SCREEN once and remember it, so repeats are instant.

The slow part of acting on a screen is UNDERSTANDING it, so we understand each
distinct screen once and cache the result keyed by a cheap visual signature; the
same query on an unchanged screen is then an instant recall.

Model-free: location comes from deterministic element parsing (OCR label match).
The old vision-model fallback was removed — Irix has no model.
"""
from __future__ import annotations

import hashlib

from . import elements

_CACHE: dict[str, dict] = {}   # screen signature -> { description: {ok,x,y,via} }


def signature(image_path: str) -> str:
    """A cheap, noise-tolerant fingerprint of the screen (32x32 grayscale -> hash).
    Same screen -> same signature -> cached answers are reused."""
    try:
        from PIL import Image
        im = Image.open(image_path).convert("L").resize((32, 32))
        return hashlib.sha1(im.tobytes()).hexdigest()
    except Exception:
        return ""


def clear() -> None:
    _CACHE.clear()


def locate(image_path: str, description: str) -> dict:
    """Where is `description` on this screen? INSTANT on a cache hit; otherwise
    parse the elements ONCE (deterministic OCR match) and remember it.
    Returns {ok, x, y, via, cached} or {error}."""
    sig = signature(image_path)
    cached = _CACHE.get(sig, {}).get(description)
    if cached:
        return {**cached, "cached": True}

    f = elements.find(image_path, description)
    if not f.get("ok"):
        return f
    ans = {"ok": True, "x": f["x"], "y": f["y"], "via": "elements", "label": f.get("label")}
    if sig:
        _CACHE.setdefault(sig, {})[description] = ans
    return {**ans, "cached": False}
