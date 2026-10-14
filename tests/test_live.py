"""Real-time perception primitives: change signature, colour-blob targets, and
region-limited template match. Millisecond hot path, no OCR, no model."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from irix import live


def _frame(w=400, h=300, bg=30):
    return np.full((h, w, 3), bg, np.uint8)


# --- signature: cache key / change detector -------------------------------

def test_signature_is_stable_and_changes():
    a = _frame()
    assert live.signature(a) == live.signature(a.copy())      # same frame -> same hash
    b = a.copy(); cv2.rectangle(b, (50, 50), (150, 150), (255, 255, 255), -1)
    assert live.signature(a) != live.signature(b)             # changed -> different


# --- blobs: fast colour targets in full-frame coords ----------------------

def test_blobs_finds_colour_target_at_true_centre():
    f = _frame()
    cv2.rectangle(f, (200, 100), (280, 180), (0, 255, 0), -1)  # green block, centre (240,140)
    out = live.blobs(f, (40, 60, 60), (80, 255, 255))
    assert out, "no blob found"
    top = out[0]
    assert abs(top["cx"] - 240) < 12 and abs(top["cy"] - 140) < 12, top   # centre recovered


def test_blobs_ignores_tiny_noise():
    f = _frame()
    cv2.rectangle(f, (10, 10), (13, 13), (0, 255, 0), -1)      # 3px speck < min_area
    assert live.blobs(f, (40, 60, 60), (80, 255, 255), min_area=50) == []


# --- find_template: region-limited sprite match ---------------------------

def test_find_template_locates_a_sprite_in_region():
    f = _frame()
    cv2.circle(f, (320, 90), 18, (0, 0, 255), -1)             # a red icon
    templ = f[70:110, 300:340].copy()
    hit = live.find_template(f, templ, region=(260, 40, 140, 120))
    assert hit is not None and hit["score"] > 0.9, hit
    assert abs(hit["cx"] - 320) < 12 and abs(hit["cy"] - 90) < 12, hit


def test_find_template_returns_none_below_threshold():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)   # textured (non-degenerate)
    templ = np.zeros((20, 20, 3), np.uint8)
    cv2.line(templ, (0, 0), (19, 19), (255, 255, 255), 2)     # a distinct diagonal, not in f
    assert live.find_template(f, templ, region=(0, 0, 200, 200), threshold=0.9) is None


# --- Perceiver.read_region: cached region OCR (mechanism only, no real OCR) --

def test_read_region_caches_by_region_signature(monkeypatch):
    calls = {"n": 0}
    import irix.ocr as ocr

    def fake_run_image(img, **k):
        calls["n"] += 1
        return "1200"
    monkeypatch.setattr(ocr, "run_image", fake_run_image)
    p = live.Perceiver()
    f = _frame()
    cv2.rectangle(f, (10, 10), (90, 40), (255, 255, 255), -1)
    box = (10, 10, 80, 30)
    assert p.read_region(f, box, key="minerals") == "1200"
    assert p.read_region(f, box, key="minerals") == "1200"    # unchanged region
    assert calls["n"] == 1, "should OCR once then serve from cache"
