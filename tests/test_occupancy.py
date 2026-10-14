"""Phase-3: class-agnostic BEV occupancy grid, occupancy flow, the TTC/looming
reflex, and monocular distance. All synthetic + deterministic — no model."""
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from irix import bev, occupancy
from irix.occupancy import FREE, OCCUPIED, UNKNOWN


def _identity_bev(w=200, h=300, ppm=10.0):
    src = [[0, 0], [w, 0], [w, h], [0, h]]
    return bev.BEV.from_points(src, width_m=w / ppm, length_m=h / ppm, px_per_m=ppm, near_m=0.0)


def _road_with_obstacle(obst=True, w=200, h=300):
    """Grey drivable ground filling the frame, with a red block ahead-centre."""
    img = np.full((h, w, 3), 110, np.uint8)               # uniform drivable colour
    if obst:
        cv2.rectangle(img, (85, 90), (115, 130), (0, 0, 220), -1)   # obstacle
    return img


# --- occupancy_grid: class-agnostic FREE / OCCUPIED / UNKNOWN --------------

def test_grid_is_mostly_free_when_road_is_clear():
    g = occupancy.occupancy_grid(_road_with_obstacle(obst=False), _identity_bev())
    s = g.summary()
    assert s["ok"] and s["free_m2"] > s["occupied_m2"], s
    assert s["clear_ahead_m"] > 10, s                     # long clear corridor


def test_grid_flags_an_obstacle_ahead():
    g = occupancy.occupancy_grid(_road_with_obstacle(obst=True), _identity_bev())
    s = g.summary()
    assert s["n_occupied_cells"] >= 1, s
    assert s["nearest"] is not None and s["nearest"]["y_m"] > 0, s
    # obstacle centred -> small bearing; clear-ahead cut short vs the empty road
    assert abs(s["nearest"]["bearing_deg"]) < 20, s


def test_grid_cell_values_are_class_agnostic_labels():
    g = occupancy.occupancy_grid(_road_with_obstacle(), _identity_bev())
    vals = set(np.unique(g.cells).tolist())
    assert vals <= {UNKNOWN, FREE, OCCUPIED} and OCCUPIED in vals, vals


def test_clear_ahead_shrinks_with_a_closer_obstacle():
    b = _identity_bev()
    far = _road_with_obstacle(obst=False)
    near = _road_with_obstacle(obst=False)
    cv2.rectangle(near, (85, 250), (115, 290), (0, 0, 220), -1)      # obstacle near ego
    assert (occupancy.occupancy_grid(near, b).clear_ahead_m()
            < occupancy.occupancy_grid(far, b).clear_ahead_m())


# --- occupancy_flow: velocity on the occupied cells -----------------------

def _bg(seed=5):
    rng = np.random.default_rng(seed)
    img = np.full((300, 200, 3), 110, np.uint8)           # drivable
    for _ in range(24):                                   # static texture for ego-motion
        x, y = int(rng.integers(0, 185)), int(rng.integers(0, 285))
        cv2.rectangle(img, (x, y), (x + 8, y + 8), (140, 140, 140), -1)
    return img


def test_occupancy_flow_attaches_velocity_to_a_mover():
    b = _identity_bev()
    a, c = _bg(), _bg()
    # overlapping rightward translation (Farneback tracks motion, not appear/vanish)
    cv2.rectangle(a, (90, 140), (125, 175), (0, 0, 220), -1)        # mover start
    cv2.rectangle(c, (102, 140), (137, 175), (0, 0, 220), -1)       # moved right
    out = occupancy.occupancy_flow(a, c, b)
    assert out["ok"], out
    assert out["n_moving_cells"] >= 1, out
    assert np.median([m["vx_m"] for m in out["movers"]]) > 0, out   # rightward velocity


# --- ttc: looming / time-to-contact (scale-free) --------------------------

def _square(size, cx=150, cy=110, canvas=300):
    img = np.full((canvas, canvas, 3), 20, np.uint8)
    h = size // 2
    cv2.rectangle(img, (cx - h, cy - h), (cx + h, cy + h), (60, 200, 60), -1)
    cv2.circle(img, (cx, cy), max(2, h // 2), (200, 60, 60), -1)    # texture inside
    return img


def test_ttc_detects_looming_and_orders_by_approach_rate():
    roi = (150 - 20, 110 - 20, 40, 40)
    slow = occupancy.ttc(_square(40), _square(48), roi=roi)          # grows 1.2x
    fast = occupancy.ttc(_square(40), _square(64), roi=roi)          # grows 1.6x
    assert slow["ok"] and slow["looming"] and slow["ttc_frames"] is not None, slow
    assert fast["ok"] and fast["looming"], fast
    assert fast["ttc_frames"] < slow["ttc_frames"], (fast, slow)     # faster = sooner


def test_ttc_not_looming_when_receding():
    roi = (150 - 20, 110 - 20, 40, 40)
    out = occupancy.ttc(_square(48), _square(40), roi=roi)           # shrinking
    assert out["ok"] and out["matched"] and not out["looming"], out
    assert out["ttc_frames"] is None, out


def test_ttc_seconds_uses_dt():
    roi = (150 - 20, 110 - 20, 40, 40)
    out = occupancy.ttc(_square(40), _square(48), roi=roi, dt=0.5)
    assert out["ttc_s"] is not None and out["ttc_s"] > 0, out


# --- mono_distance: two deterministic routes ------------------------------

def test_mono_distance_ground_plane_matches_bev():
    b = bev.BEV.from_camera((1280, 720), height_m=1.5, pitch_deg=12.0,
                            hfov_deg=60.0, range_m=30, width_m=20, px_per_m=10)
    d = occupancy.mono_distance(bev_t=b, ground_px=(640, 600))
    assert d["ok"] and d["method"] == "ground_plane" and d["range_m"] > 0, d


def test_mono_distance_size_prior_formula():
    # f=800px, real width 0.5 m, apparent 100 px -> 4 m
    d = occupancy.mono_distance(focal_px=800, real_size_m=0.5, pixel_size=100)
    assert d["ok"] and d["method"] == "size_prior", d
    assert abs(d["range_m"] - 4.0) < 1e-6, d


def test_mono_distance_needs_valid_args():
    assert "error" in occupancy.mono_distance()
