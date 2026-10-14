"""worldmodel — Phase 4: persistent temporal BEV world-model. Model-free.

Tesla's temporal layer remembers what the cameras saw a moment ago, so the car
still "knows" about the occluded child behind the parked van. Irix reproduces
that with MEMORY, not a net — the same principle as the face gallery: a
persistent metric occupancy MAP, accumulated over `bev.ego_motion`, that the
brain can query beyond the current field of view.

  * `WorldModel.observe(image)` — warp the frame to BEV, estimate the ego's own
    displacement since the last frame (ORB/phase-correlation, deterministic),
    advance the pose, and VOTE the frame's FREE/OCCUPIED cells into a sparse
    world-frame map.
  * `.snapshot()` — the remembered map re-rendered as an ego-frame `OccGrid`,
    so `.clear_ahead_m()`/`.summary()` work on MEMORY, including cells that
    scrolled out of view.
  * `.save()`/`.load()` — the map persists across sessions (npz), like the
    face gallery's zip.

Per-cell evidence is a pair of capped vote counters (free, occupied); at the
cap both halve, so old evidence decays and a changed scene can flip a cell.
Occupied wins ties — the safe default. All CPU geometry, no model.
"""
from __future__ import annotations

import math

from .occupancy import FREE, OCCUPIED, UNKNOWN, OccGrid, _grids

_VOTE_CAP = 100          # halve both counters at the cap -> bounded, recency-adaptive


class WorldModel:
    """A persistent metric occupancy map in the WORLD frame (= ego frame of the
    first observed frame: x right, y forward, metres). Sparse — grows unbounded
    like the exemplar gallery; memory instead of a learned world-model."""

    def __init__(self, bev_t, cell_m: float = 0.5, occ_frac: float = 0.35):
        self.bev = bev_t
        self.cell_m = cell_m
        self.occ_frac = occ_frac
        self.x_m = self.y_m = self.yaw_deg = 0.0
        self.frames = 0
        self._last_warp = None
        self._cells: dict[tuple[int, int], list[int]] = {}   # (i,j) -> [free, occ]

    # --- pose ---------------------------------------------------------------
    def pose(self) -> dict:
        return {"x_m": round(self.x_m, 2), "y_m": round(self.y_m, 2),
                "yaw_deg": round(self.yaw_deg, 1)}

    def advance(self, dx_m: float, dy_m: float, dyaw_deg: float = 0.0):
        """Compose an EGO-frame displacement (x right, y forward, yaw) onto the
        world pose. Public seam so pose math is testable without images."""
        th = math.radians(self.yaw_deg)
        c, s = math.cos(th), math.sin(th)
        self.x_m += c * dx_m - s * dy_m
        self.y_m += s * dx_m + c * dy_m
        self.yaw_deg = (self.yaw_deg + dyaw_deg + 180.0) % 360.0 - 180.0

    def _ego_to_world(self, x_e: float, y_e: float) -> tuple[float, float]:
        th = math.radians(self.yaw_deg)
        c, s = math.cos(th), math.sin(th)
        return self.x_m + c * x_e - s * y_e, self.y_m + s * x_e + c * y_e

    def _world_to_ego(self, x_w: float, y_w: float) -> tuple[float, float]:
        th = math.radians(self.yaw_deg)
        c, s = math.cos(th), math.sin(th)
        dx, dy = x_w - self.x_m, y_w - self.y_m
        return c * dx + s * dy, -s * dx + c * dy

    # --- integrate ------------------------------------------------------------
    def _vote(self, x_w: float, y_w: float, state: int):
        key = (math.floor(x_w / self.cell_m), math.floor(y_w / self.cell_m))
        v = self._cells.setdefault(key, [0, 0])
        v[0 if state == FREE else 1] += 1
        if max(v) >= _VOTE_CAP:                    # decay: keep ratio, allow flips
            v[0], v[1] = (v[0] + 1) // 2, (v[1] + 1) // 2

    def integrate(self, grid: OccGrid):
        """Vote an ego-frame OccGrid into the world map at the current pose."""
        for r in range(grid.rows):
            for c in range(grid.cols):
                state = int(grid.cells[r, c])
                if state == UNKNOWN:
                    continue
                self._vote(*self._ego_to_world(*grid.cell_metric(r, c)), state)

    def observe(self, image) -> dict:
        """One frame of the stream: ego-motion -> advance pose -> integrate the
        frame's occupancy. Returns pose + map size + this frame's grid summary."""
        from . import bev as _bev
        warp = self.bev.warp(image)
        if warp is None:
            return {"error": "no image"}
        if self._last_warp is not None:
            em = _bev.ego_motion(self._last_warp, warp, self.bev.px_per_m)
            if "error" not in em:
                self.advance(em["dx_m"], em["dy_m"], em.get("dyaw_deg", 0.0))
        cells, *_ = _grids(warp, self.bev, self.cell_m, self.occ_frac)
        grid = OccGrid(cells, self.cell_m)
        self.integrate(grid)
        self._last_warp = warp
        self.frames += 1
        return {"ok": True, "pose": self.pose(), "frames": self.frames,
                "mapped_cells": len(self._cells), "grid": grid.summary()}

    # --- query ----------------------------------------------------------------
    def state_at(self, x_w: float, y_w: float) -> int:
        """UNKNOWN/FREE/OCCUPIED at a WORLD-frame point. Occupied wins ties."""
        v = self._cells.get((math.floor(x_w / self.cell_m),
                             math.floor(y_w / self.cell_m)))
        if v is None or v == [0, 0]:
            return UNKNOWN
        return FREE if v[0] > v[1] else OCCUPIED

    def snapshot(self, range_m: float = 30.0, width_m: float = 20.0) -> OccGrid:
        """The REMEMBERED map rendered as an ego-frame OccGrid — includes cells
        currently out of view. Feed it to `.summary()`/`.clear_ahead_m()`."""
        import numpy as np
        rows = max(1, int(round(range_m / self.cell_m)))
        cols = max(1, int(round(width_m / self.cell_m)))
        cells = np.full((rows, cols), UNKNOWN, np.uint8)
        grid = OccGrid(cells, self.cell_m)
        for r in range(rows):
            for c in range(cols):
                cells[r, c] = self.state_at(*self._ego_to_world(*grid.cell_metric(r, c)))
        return grid

    def summary(self) -> dict:
        a = self.cell_m * self.cell_m
        free = sum(1 for v in self._cells.values() if v[0] > v[1])
        return {"ok": True, "pose": self.pose(), "frames": self.frames,
                "mapped_cells": len(self._cells),
                "mapped_m2": round(len(self._cells) * a, 2),
                "free_m2": round(free * a, 2),
                "occupied_m2": round((len(self._cells) - free) * a, 2)}

    # --- persistence ------------------------------------------------------------
    def save(self, path: str) -> dict:
        """Persist the map + pose to an .npz — the scene gallery on disk."""
        import numpy as np
        idx = np.array(list(self._cells.keys()) or np.empty((0, 2)), np.int64)
        votes = np.array(list(self._cells.values()) or np.empty((0, 2)), np.uint16)
        np.savez_compressed(path, idx=idx.reshape(-1, 2), votes=votes.reshape(-1, 2),
                            pose=np.array([self.x_m, self.y_m, self.yaw_deg]),
                            meta=np.array([self.cell_m, self.occ_frac, self.frames]))
        return {"ok": True, "path": path, "mapped_cells": len(self._cells)}

    @classmethod
    def load(cls, path: str, bev_t) -> "WorldModel":
        """Restore a saved map; pass the session's BEV transform."""
        import numpy as np
        z = np.load(path)
        wm = cls(bev_t, cell_m=float(z["meta"][0]), occ_frac=float(z["meta"][1]))
        wm.x_m, wm.y_m, wm.yaw_deg = (float(v) for v in z["pose"])
        wm.frames = int(z["meta"][2])
        wm._cells = {(int(i), int(j)): [int(f), int(o)]
                     for (i, j), (f, o) in zip(z["idx"], z["votes"])}
        return wm
