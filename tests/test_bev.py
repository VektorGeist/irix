"""Tests for the BEV foundation: the calibrated metric anchor (distance grows up
the image, resolves monocular scale for ground contacts), the calibration-free
IPM warp, ego-motion recovery (the memory seam), and vanishing-point bootstrap.
All deterministic, synthetic — no camera, no model."""
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from irix import bev


# --- from_camera: the metric anchor ---------------------------------------

def _cam():
    return bev.BEV.from_camera((1280, 720), height_m=1.5, pitch_deg=12.0,
                               hfov_deg=60.0, range_m=30, width_m=20, px_per_m=10)


def test_distance_grows_higher_up_the_image():
    b = _cam()
    near = b.distance_bearing((640, 700))     # low in frame -> close
    far = b.distance_bearing((640, 430))      # higher up   -> far
    assert near and far
    assert near["range_m"] < far["range_m"], (near, far)


def test_center_column_is_straight_ahead():
    b = _cam()
    d = b.distance_bearing((640, 600))
    assert d and abs(d["bearing_deg"]) < 2.0, d        # u==cx -> bearing ~0


def test_right_of_centre_bears_right():
    b = _cam()
    d = b.distance_bearing((1000, 600))
    assert d and d["bearing_deg"] > 1.0, d             # +X -> +bearing (right)


def test_above_horizon_has_no_ground_distance():
    b = _cam()
    assert b.distance_bearing((640, 40)) is None       # sky / above horizon


def test_known_geometry_distance_is_physical():
    # ray through the principal point hits ground at ~height/tan(pitch)
    b = _cam()
    d = b.distance_bearing((640, 360))
    assert d and abs(d["y_m"] - 1.5 / math.tan(math.radians(12))) < 1.5, d


# --- from_points: calibration-free IPM warp -------------------------------

def test_warp_outputs_expected_metric_canvas():
    src = [[520, 430], [760, 430], [900, 690], [380, 690]]   # a ground trapezoid
    b = bev.BEV.from_points(src, width_m=4.0, length_m=8.0, px_per_m=20, near_m=2.0)
    assert (b.out_w, b.out_h) == (80, 200)
    img = np.full((720, 1280, 3), 200, np.uint8)
    out = b.warp(img)
    assert out is not None and out.shape == (200, 80, 3)


# --- ego_motion: the memory seam ------------------------------------------

def _textured(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((240, 200, 3), 60, np.uint8)
    for _ in range(40):
        x, y = int(rng.integers(0, 180)), int(rng.integers(0, 220))
        c = tuple(int(v) for v in rng.integers(80, 255, 3))
        cv2.rectangle(img, (x, y), (x + 14, y + 14), c, -1)
    return img


def test_ego_motion_recovers_forward_translation():
    a = _textured(1)
    b_img = np.roll(a, 20, axis=0)            # ground shifted 20px down (rows)
    m = bev.ego_motion(a, b_img, px_per_m=10.0)
    assert "error" not in m
    assert abs(abs(m["dy_m"]) - 2.0) < 0.5, m        # 20px / 10 px_per_m ~ 2.0 m
    assert abs(m["dx_m"]) < 0.5, m                   # no lateral component


def test_ego_motion_reports_method_and_conf():
    a = _textured(2)
    m = bev.ego_motion(a, np.roll(a, 8, axis=1), px_per_m=10.0)
    assert m.get("method") in ("orb", "phasecorr") and "conf" in m


# --- vanishing point: uncalibrated pitch bootstrap ------------------------

def test_vanishing_point_finds_the_convergence():
    img = np.zeros((400, 600, 3), np.uint8)
    vx, vy = 300, 140
    cv2.line(img, (60, 399), (vx, vy), (255, 255, 255), 3)
    cv2.line(img, (540, 399), (vx, vy), (255, 255, 255), 3)
    vp = bev.vanishing_point(img)
    assert vp is not None
    assert abs(vp[0] - vx) < 60 and abs(vp[1] - vy) < 60, vp
