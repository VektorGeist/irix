"""Phase-2 metric heads: lane geometry (curvature/offset in m), drivable area
(clear-ahead in m), and ego-compensated movers — all in the BEV vector space.
An identity BEV (corners->corners) lets us feed top-down images directly. No model."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from irix import bev, scene


def _identity_bev(w=200, h=300, ppm=10.0):
    src = [[0, 0], [w, 0], [w, h], [0, h]]
    b = bev.BEV.from_points(src, width_m=w / ppm, length_m=h / ppm, px_per_m=ppm, near_m=0.0)
    assert (b.out_w, b.out_h) == (w, h)
    return b


def _lanes(xl, xr, w=200, h=300):
    img = np.full((h, w, 3), 30, np.uint8)
    cv2.line(img, (xl, 0), (xl, h), (255, 255, 255), 4)
    cv2.line(img, (xr, 0), (xr, h), (255, 255, 255), 4)
    return img


# --- lane_bev: metric lane geometry ---------------------------------------

def test_straight_centred_lane_reads_straight_zero_offset():
    b = _identity_bev()
    out = scene.lane_bev(_lanes(70, 130), b)              # symmetric about centre
    assert out["ok"] and out["direction"] == "STRAIGHT", out
    assert out["lanes"] == 2 and abs(out["ego_offset_m"]) < 0.6, out
    assert abs(out["lane_width_m"] - 6.0) < 1.0, out      # (130-70)/10 m


def test_ego_offset_sign_when_lane_shifts_right():
    b = _identity_bev()
    out = scene.lane_bev(_lanes(90, 150), b)              # lane centre right of ego
    assert out["ok"] and out.get("ego_offset_m") is not None
    assert out["ego_offset_m"] < -0.8, out                # ego is LEFT of lane centre


# --- freespace_bev: metric drivable area ----------------------------------

def test_clear_ahead_in_metres():
    b = _identity_bev()
    img = np.zeros((300, 200, 3), np.uint8)
    img[120:, :] = (100, 100, 100)                        # bottom 60% = road
    out = scene.freespace_bev(img, b)
    assert out["ok"] and out["clear_ahead_m"] > 12, out   # ~18 m of free corridor
    assert out["drivable_m2"] > 0, out


# --- flow_bev: ego-compensated independent motion -------------------------

def _scene_with_blob(bx, seed=3):
    rng = np.random.default_rng(seed)
    img = np.full((300, 200, 3), 40, np.uint8)
    for _ in range(30):                                   # static textured background
        x, y = int(rng.integers(0, 180)), int(rng.integers(0, 280))
        cv2.rectangle(img, (x, y), (x + 12, y + 12), (180, 180, 180), -1)
    cv2.rectangle(img, (bx, 150), (bx + 20, 175), (0, 0, 255), -1)   # the mover
    return img


def test_flow_bev_isolates_an_independent_mover():
    b = _identity_bev()
    a, c = _scene_with_blob(80), _scene_with_blob(120)    # blob moved right, bg static
    out = scene.flow_bev(a, c, b)
    assert out["ok"], out
    assert out["n_moving"] >= 1, out
    assert abs(out["ego_dy_m"]) < 1.0, out                # ego itself ~stationary


# --- describe_scene in metric mode ----------------------------------------

def test_describe_scene_metric_text():
    b = _identity_bev()
    out = scene.describe_scene(_lanes(70, 130), bev_t=b, include_ui=False)
    assert out["metric"] is True
    assert "LANE:" in out["text"], out["text"]


# --- Phase 5: per-obstacle velocity, min TTC, occupancy + world fusion -----

def _drivable_with_mover(bx, seed=5):
    rng = np.random.default_rng(seed)
    img = np.full((300, 200, 3), 110, np.uint8)           # drivable colour
    for _ in range(24):                                   # static texture
        x, y = int(rng.integers(0, 185)), int(rng.integers(0, 285))
        cv2.rectangle(img, (x, y), (x + 8, y + 8), (140, 140, 140), -1)
    cv2.rectangle(img, (bx, 140), (bx + 35, 175), (0, 0, 220), -1)
    return img


def test_describe_scene_movers_carry_velocity_and_occupancy():
    b = _identity_bev()
    prev, cur = _drivable_with_mover(90), _drivable_with_mover(102)
    out = scene.describe_scene(cur, prev, bev_t=b, include_ui=False, dt=0.1)
    assert out["metric"] and out["motion"]["ok"], out["motion"]
    assert "MOVERS:" in out["text"] and "OCC:" in out["text"], out["text"]
    assert out["occupancy"] is not None and "clear_ahead_m" in out["occupancy"]
    movers = out["motion"]["movers"]
    assert movers and "vx_m" in movers[0] and "speed_mps" in movers[0], movers


def test_describe_scene_observes_into_world_model():
    from irix.worldmodel import WorldModel
    b = _identity_bev()
    wm = WorldModel(b, cell_m=0.5)
    out = scene.describe_scene(_drivable_with_mover(90), bev_t=b,
                               include_ui=False, world=wm)
    assert out["world"] and out["world"]["ok"], out["world"]
    assert wm.frames == 1
    assert "WORLD: pose" in out["text"] and "mem-clear" in out["text"], out["text"]
