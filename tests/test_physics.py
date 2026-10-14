"""Model-free physics extraction (Irix's half of semantic vision, 2026-07-01). Synthetic
frames only — no camera. Verifies attributes are measured and the description stays NEUTRAL
(no object names — naming is the reasoning mind's job, elsewhere)."""
import numpy as np
import pytest

pytest.importorskip("cv2")
import cv2

from irix import physics


def _dark_frame(h=240, w=320, level=20):
    return np.full((h, w, 3), level, np.uint8)


def _emissive_rect(f, box, level=245, interior=None):
    x, y, w, h = box
    f[y:y + h, x:x + w] = level
    if interior is not None:
        f[y + 6:y + h - 6, x + 6:x + w - 6] = interior
    return f


def test_salient_regions_finds_the_bright_object():
    f = _emissive_rect(_dark_frame(), (40, 30, 140, 80))     # bright rect, upper area
    boxes = physics.salient_regions(f)
    assert boxes, "should find at least the bright region"
    x, y, w, h = boxes[0]
    assert w * h >= 140 * 80 * 0.5                            # roughly the bright rectangle


def test_attributes_of_a_wall_screen_are_physical_and_correct():
    f = _emissive_rect(_dark_frame(), (40, 24, 160, 90))     # 16:9-ish, high up
    a = physics.region_attributes(f, (40, 24, 160, 90))
    assert a["emissive"] is True and a["emissive_ratio"] > 3
    assert a["shape"] == "rectangular"
    assert 1.5 <= a["aspect_ratio"] <= 2.1                   # ~16:9
    assert a["height"] in ("high on a wall", "overhead")
    assert a["supported_below"] is False                     # nothing bright under it -> mounted


def test_temporal_self_updating_surface():
    box = (40, 24, 160, 90)
    now = _emissive_rect(_dark_frame(), box, interior=(30, 30, 30))
    then = _emissive_rect(_dark_frame(), box, interior=(240, 240, 240))   # very different interior
    a = physics.region_attributes(now, box, prev_frames=[then])
    assert a["interior_change"] is not None and a["interior_change"] > 0.05
    assert a["brightness_swing"] is not None
    assert "self-updating surface" in a["temporal"]


def test_static_surface_reads_static():
    box = (40, 24, 160, 90)
    f = _emissive_rect(_dark_frame(), box, interior=(200, 200, 200))
    a = physics.region_attributes(f, box, prev_frames=[f.copy()])         # identical -> no change
    assert a["temporal"].startswith("static")


def test_description_is_neutral_no_object_names():
    f = _emissive_rect(_dark_frame(), (40, 24, 160, 90))
    desc = physics.physical_description(physics.region_attributes(f, (40, 24, 160, 90)))
    assert "emits its own light" in desc
    low = desc.lower()
    for noun in ("tv", "television", "screen", "monitor", "lamp", "window", "phone", "person"):
        assert noun not in low, f"description must not name objects, found {noun!r}"


def test_reflective_object_is_not_emissive():
    # a mid-grey block on a same-ish background -> reflective, not self-lit
    f = np.full((240, 320, 3), 90, np.uint8)
    f[100:150, 120:200] = 110
    a = physics.region_attributes(f, (120, 100, 80, 50))
    assert a["emissive"] is False
