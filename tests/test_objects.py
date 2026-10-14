"""General object gallery: deterministic descriptor discrimination, enroll/
identify round-trip, physics fusion, auto-enroll growth, zip persistence.
All synthetic — no model."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from irix import objects


def _scene(ball=True, box=True, radius=40, bright=210):
    """Dark room with a red ball (smooth) and a green striped block (textured)."""
    img = np.full((300, 300, 3), 25, np.uint8)
    if ball:
        cv2.circle(img, (80, 90), radius, (30, 30, bright), -1)
    if box:
        cv2.rectangle(img, (180, 180), (270, 260), (40, 170, 40), -1)
        for x in range(185, 270, 12):                 # stripes = texture
            cv2.line(img, (x, 182), (x, 258), (20, 90, 20), 3)
    return img


def _crop(img, x, y, w, h):
    return img[y:y + h, x:x + w]


def _ball_crop(radius=40, bright=210):
    # a different radius/brightness = a genuinely different VIEW of the object
    return _crop(_scene(box=False, radius=radius, bright=bright), 35, 45, 90, 90)


def _box_crop():
    return _crop(_scene(ball=False), 175, 175, 100, 90)


def _gal(tmp_path, name="obj.irix"):
    return objects.ObjectGallery(str(tmp_path / name))


# --- descriptor: same-object distance < cross-object distance ----------------

def test_descriptor_discriminates_objects():
    same = objects._dist(objects.descriptor(_ball_crop()),
                         objects.descriptor(_ball_crop(radius=34, bright=185)))
    cross = objects._dist(objects.descriptor(_ball_crop()),
                          objects.descriptor(_box_crop()))
    assert same < cross, (same, cross)
    assert same < 0.28 < cross, (same, cross)         # default threshold separates


def test_descriptor_is_deterministic():
    a = objects.descriptor(_ball_crop())
    b = objects.descriptor(_ball_crop())
    assert np.array_equal(a, b)


# --- enroll + match / identify -------------------------------------------------

def test_enroll_and_recognize_roundtrip(tmp_path):
    g = _gal(tmp_path)
    out = g.enroll(_scene(box=False), "red-ball")
    assert out["ok"], out
    r = g.match(_ball_crop(radius=34, bright=185))
    assert r["known"] and r["name"] == "red-ball", r


def test_identify_names_known_and_describes_unknown(tmp_path):
    g = _gal(tmp_path)
    assert g.enroll(_scene(box=False), "red-ball")["ok"]
    out = g.identify(_scene())                        # ball (known) + block (new)
    assert out["ok"] and len(out["objects"]) >= 2, out
    names = {o["name"] for o in out["objects"]}
    assert "red-ball" in names, names
    for o in out["objects"]:                          # physics rides along for all
        assert o["description"] and o["attributes"]["shape"] in (
            "round", "rectangular", "irregular")
    unknown = [o for o in out["objects"] if not o["known"]]
    assert unknown and unknown[0]["name"] is None


def test_identify_auto_enrolls_unknowns(tmp_path):
    g = _gal(tmp_path)
    out = g.identify(_scene(), auto_enroll_unknown="obj-")
    assert out["ok"] and all(o.get("auto_enrolled") for o in out["objects"]), out
    again = g.identify(_scene())                      # second look: now remembered
    assert all(o["known"] for o in again["objects"]), again


def test_gallery_persists_across_instances(tmp_path):
    g = _gal(tmp_path)
    g.enroll(_scene(box=False), "red-ball")
    fresh = _gal(tmp_path)                            # same zip, new instance
    assert fresh.stats()["exemplars"] == 1
    assert fresh.match(_ball_crop())["name"] == "red-ball"
