"""Phase-4: persistent temporal BEV world-model — pose composition, evidence
voting, memory beyond the current view, the observe() stream, and persistence.
All synthetic + deterministic — no model."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from irix import bev
from irix.occupancy import FREE, OCCUPIED, UNKNOWN, OccGrid
from irix.worldmodel import WorldModel


def _identity_bev(w=200, h=300, ppm=10.0):
    src = [[0, 0], [w, 0], [w, h], [0, h]]
    return bev.BEV.from_points(src, width_m=w / ppm, length_m=h / ppm,
                               px_per_m=ppm, near_m=0.0)


def _wm(cell_m=1.0):
    return WorldModel(_identity_bev(), cell_m=cell_m)


# --- pose composition -------------------------------------------------------

def test_advance_composes_forward_then_turn():
    wm = _wm()
    wm.advance(0, 5)                              # forward 5 m
    assert (round(wm.x_m, 3), round(wm.y_m, 3)) == (0, 5)
    wm.advance(0, 0, 90)                          # turn in place
    wm.advance(0, 5)                              # forward again, now rotated
    assert abs(wm.x_m - (-5)) < 1e-6 and abs(wm.y_m - 5) < 1e-6, wm.pose()


def test_world_ego_transforms_roundtrip():
    wm = _wm()
    wm.advance(2, 7, 33)
    x, y = wm._ego_to_world(1.5, 4.0)
    xe, ye = wm._world_to_ego(x, y)
    assert abs(xe - 1.5) < 1e-9 and abs(ye - 4.0) < 1e-9


# --- integrate / state_at ----------------------------------------------------

def _grid_with_obstacle(rows=8, cols=4, cell_m=1.0, at=(2, 2)):
    cells = np.full((rows, cols), FREE, np.uint8)
    cells[at] = OCCUPIED
    return OccGrid(cells, cell_m)


def test_integrate_votes_states_into_world():
    wm = _wm()
    g = _grid_with_obstacle()
    wm.integrate(g)
    ox, oy = g.cell_metric(2, 2)                  # the obstacle's ego coords
    assert wm.state_at(ox, oy) == OCCUPIED
    fx, fy = g.cell_metric(7, 1)                  # a free cell near the ego
    assert wm.state_at(fx, fy) == FREE
    assert wm.state_at(500, 500) == UNKNOWN       # never observed


def test_map_remembers_what_scrolled_out_of_view():
    wm = _wm()
    g = _grid_with_obstacle()                     # obstacle ~5.5 m ahead
    ox, oy = g.cell_metric(2, 2)
    wm.integrate(g)
    wm.advance(0, 20)                             # drive well past the obstacle
    assert wm.state_at(ox, oy) == OCCUPIED        # world map still knows it


def test_vote_cap_lets_a_changed_scene_flip():
    wm = _wm()
    for _ in range(150):
        wm._vote(0.5, 0.5, FREE)
    assert wm.state_at(0.5, 0.5) == FREE
    for _ in range(150):
        wm._vote(0.5, 0.5, OCCUPIED)
    assert wm.state_at(0.5, 0.5) == OCCUPIED      # decay allowed the flip


# --- snapshot: memory rendered back as an ego-frame OccGrid ------------------

def test_snapshot_includes_remembered_obstacle():
    wm = _wm()
    g = _grid_with_obstacle()
    ox, oy = g.cell_metric(2, 2)
    wm.integrate(g)
    snap = wm.snapshot(range_m=8, width_m=4)
    s = snap.summary()
    assert s["n_occupied_cells"] >= 1, s
    near = s["nearest"]
    assert abs(near["x_m"] - ox) <= 1.0 and abs(near["y_m"] - oy) <= 1.0, s
    assert 0 < s["clear_ahead_m"] < oy + 1.0, s   # corridor stops at the obstacle


# --- observe(): the live stream ----------------------------------------------

def _ground(seed=3, w=200, h=300):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 110, np.uint8)       # uniform drivable colour
    for _ in range(30):                            # static texture for ego-motion
        x, y = int(rng.integers(0, w - 12)), int(rng.integers(0, h - 12))
        cv2.rectangle(img, (x, y), (x + 8, y + 8), (140, 140, 140), -1)
    return img


def test_observe_stream_advances_pose_and_grows_map():
    wm = WorldModel(_identity_bev(), cell_m=0.5)
    f1 = _ground()
    f2 = np.roll(f1, 20, axis=0)                  # ground streams 20px down = +2 m
    o1 = wm.observe(f1)
    o2 = wm.observe(f2)
    assert o1["ok"] and o2["ok"]
    assert wm.frames == 2 and o2["mapped_cells"] > 0
    assert abs(abs(wm.y_m) - 2.0) < 0.6, wm.pose()
    assert wm.summary()["free_m2"] > 0


# --- persistence ---------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    wm = _wm()
    wm.integrate(_grid_with_obstacle())
    wm.advance(1, 2, 15)
    wm.frames = 4
    path = str(tmp_path / "map.npz")
    assert wm.save(path)["ok"]
    back = WorldModel.load(path, _identity_bev())
    assert back._cells == wm._cells
    assert back.pose() == wm.pose()
    assert back.frames == 4 and back.cell_m == wm.cell_m
