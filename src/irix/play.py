"""play — the autonomous PLAY loop: perceive -> decide -> act on a MOVING world.

Closes the loop the scene heads were built for. Each tick: render/capture a
frame, read it with `describe_scene` (metric BEV + occupancy + world-model),
hand ONLY the structured output to a deterministic reflex policy (the brain
contract — the decider never touches pixels), act, repeat.

A built-in synthetic world (`SimWorld`: scrolling road, lane lines, a stalled
obstacle to swerve around) makes the loop runnable and testable with no game
installed; to play a real moving game, swap the render for a capture source
and route steer/brake to real inputs. All deterministic, no model.
"""
from __future__ import annotations

import time


class SimWorld:
    """A minimal moving world rendered on the BEV plane: grey road, dashed lane
    lines, scrolling ground speckles (so ego-motion is observable), and a
    stalled obstacle in the lane ahead. Ego is bottom-centre; steering shifts
    the world the other way."""
    W, H, PPM = 200, 300, 10          # frame px + px per metre

    def __init__(self, seed: int = 7):
        import numpy as np
        rng = np.random.default_rng(seed)
        self._speckles = [(int(rng.integers(10, self.W - 10)),
                           int(rng.integers(0, self.H))) for _ in range(26)]
        self.ego_x = 0.0              # lateral offset from road centre (m)
        self.speed = 2.0              # forward metres per tick
        self.obs_x, self.obs_y = 0.3, 22.0   # stalled obstacle (m, lateral/ahead)
        self.scroll = 0
        self.collided = self.off_road = False

    def step(self, steer_m: float = 0.0, brake: bool = False):
        v = self.speed * (0.5 if brake else 1.0)
        self.ego_x += steer_m
        self.obs_y -= v                       # world streams past the ego
        self.scroll = (self.scroll + int(v * self.PPM)) % self.H
        if -0.5 < self.obs_y <= 0.5 and abs(self.obs_x - self.ego_x) < 1.2:
            self.collided = True                  # only while actually alongside
        if abs(self.ego_x) > 4.0:
            self.off_road = True

    def render(self):
        import cv2
        import numpy as np
        img = np.full((self.H, self.W, 3), 30, np.uint8)          # off-road dark
        cx = self.W // 2 - int(self.ego_x * self.PPM)             # road centre px
        x0, x1 = max(0, cx - 60), min(self.W, cx + 60)
        img[:, x0:x1] = (110, 110, 110)                           # 12 m road band
        for sx, sy in self._speckles:                             # ground texture
            y = (sy + self.scroll) % self.H
            px = cx - self.W // 2 + sx
            if x0 <= px < x1 - 6:
                cv2.rectangle(img, (px, y), (px + 6, min(self.H - 1, y + 6)),
                              (130, 130, 130), -1)
        for lx in (cx - 55, cx + 55):                             # dashed lane lines
            for y in range(-self.H, self.H, 30):
                yy = y + self.scroll % 30
                cv2.line(img, (lx, yy), (lx, yy + 16), (255, 255, 255), 3)
        if 0 < self.obs_y * self.PPM < self.H:                    # the obstacle
            ox = cx + int(self.obs_x * self.PPM)
            oy = self.H - int(self.obs_y * self.PPM)
            cv2.rectangle(img, (ox - 8, oy - 8), (ox + 8, oy + 8), (0, 0, 220), -1)
        return img


class Reflex:
    """Deterministic reflex policy over describe_scene's STRUCTURED output only.

    Threat = the centre-corridor clear-ahead shrinking (edge cells of the road
    read as OCCUPIED too, so bearings alone mislead — the corridor signal
    doesn't). The swerve direction is chosen once from the blocking cell's
    lateral side and LATCHED until the corridor clears (an unlatched policy
    re-decides each tick and oscillates). While a passed-but-close blocker is
    beside the corridor, HOLD — lane-keeping back too early re-enters its path."""

    def __init__(self):
        self._dir = 0.0

    @staticmethod
    def _blockers(occ, y_max: float):
        return [o for o in (occ.get("occupied") or [])
                if 0 < o["y_m"] < y_max and abs(o["x_m"]) < 2.5]

    def __call__(self, out: dict) -> tuple[float, bool, str]:
        occ = out.get("occupancy") or {}
        clear = occ.get("clear_ahead_m")
        # threat = the corridor shrinking; TTC is reported but not a trigger here
        # (repetitive lane dashes strobe the optical flow into spurious TTCs)
        if clear is not None and clear < 9:
            if not self._dir:
                bl = self._blockers(occ, 12.0)
                self._dir = (-0.8 if bl[0]["x_m"] >= 0 else 0.8) if bl else -0.8
            return self._dir, clear < 7, (f"reflex: clear {clear}m -> "
                                          f"swerve {'L' if self._dir < 0 else 'R'}")
        self._dir = 0.0
        bl = self._blockers(occ, 6.0)
        if bl:
            return 0.0, False, f"hold: blocker {bl[0]['y_m']}m ahead, passing"
        off = (out.get("path") or {}).get("ego_offset_m")
        if off is not None:
            return max(-0.3, min(0.3, -0.25 * off)), False, "lane-keep"
        return 0.0, False, "cruise"


def run(ticks: int = 30, dt: float = 0.1, use_world_model: bool = True,
        verbose: bool = False) -> dict:
    """Run the closed loop on the synthetic moving world. Returns the outcome
    (survived / collided / off-road), measured FPS, and the per-tick log."""
    from . import bev as _bev, scene
    sim = SimWorld()
    b = _bev.BEV.from_points([[0, 0], [sim.W, 0], [sim.W, sim.H], [0, sim.H]],
                             width_m=sim.W / sim.PPM, length_m=sim.H / sim.PPM,
                             px_per_m=sim.PPM)
    wm = None
    if use_world_model:
        from . import worldmodel
        wm = worldmodel.WorldModel(b, cell_m=0.5)
    policy = Reflex()
    prev, log, done = None, [], 0
    t0 = time.time()
    for i in range(ticks):
        cur = sim.render()
        out = scene.describe_scene(cur, prev, bev_t=b, dt=dt, world=wm,
                                   include_ui=False)
        steer, brake, why = policy(out)
        sim.step(steer, brake)
        log.append({"tick": i, "why": why, "steer": round(steer, 2),
                    "brake": brake, "text": out["text"]})
        if verbose:
            print(f"[{i:02d}] {why:38s} | {out['text']}")
        prev, done = cur, i + 1
        if sim.collided or sim.off_road:
            break
    dt_wall = time.time() - t0
    return {"ok": not sim.collided and not sim.off_road, "ticks": done,
            "collided": sim.collided, "off_road": sim.off_road,
            "reflex_fired": any(t["brake"] for t in log),
            "fps": round(done / dt_wall, 1),
            "world_mapped_m2": (wm.summary()["mapped_m2"] if wm else None),
            "log": log}
