"""occupancy — class-agnostic BEV occupancy grid, occupancy flow, and the
collision reflexes (TTC / looming, monocular distance). Model-free.

Tesla's occupancy network answers one question without ever naming the object:
"is this volume of space FREE or OCCUPIED, and which way is it moving?" That is
class-agnostic by design — you don't need to know it's a car to not drive into
it. Irix reproduces the same representation with pure geometry:

  * `occupancy_grid()` — warp to the BEV ground plane, call the drivable-colour
    region FREE, everything else inside the sensed field OCCUPIED, and the
    unsensed border UNKNOWN. Block-reduce to a metric grid (ego at bottom-centre,
    forward = up). No class labels, no model.
  * `occupancy_flow()` — ego-compensate two grids and attach a per-cell velocity
    to the occupied cells (the moving-obstacle field).
  * `ttc()` — time-to-contact from looming (scale change of a region between
    frames). Scale-FREE: needs no metric calibration, only that the thing is
    growing in the image (Lee's tau). The reflex that fires before geometry.
  * `mono_distance()` — deterministic monocular range: ground-plane intersection
    when calibrated, or a remembered real-size prior (distance = f·size/px).

All CPU OpenCV/geometry. `cv2`/`numpy` imported lazily.
"""
from __future__ import annotations

import math

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


def _asimg(x, scale: float = 1.0):
    import cv2
    import numpy as np
    img = cv2.imread(x) if isinstance(x, str) else x
    if img is None:
        return None
    if isinstance(img, np.ndarray) and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if scale != 1.0:
        img = cv2.resize(img, (max(1, int(img.shape[1] * scale)),
                               max(1, int(img.shape[0] * scale))))
    return img


def _drivable(warp, sample_frac: float = 0.12):
    """Largest bottom-touching region of the colour just ahead of the ego =
    drivable/FREE ground. Same colour-homogeneity idea as scene.freespace, no model.
    Samples only SENSED pixels (a from_camera warp leaves a black unsensed border
    at the near edge — sampling that would call the whole ground 'not road')."""
    import cv2
    import numpy as np
    h, w = warp.shape[:2]
    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
    sensed = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY) > 0
    sh, sw = max(1, int(h * sample_frac)), max(1, int(w * sample_frac))
    box = slice(h - sh, h), slice(w // 2 - sw // 2, w // 2 + sw // 2 + 1)
    patch, pv = hsv[box].reshape(-1, 3), sensed[box].reshape(-1)
    if pv.any():                                          # ignore unsensed border px
        patch = patch[pv]
    elif sensed.any():                                    # near patch fully unsensed
        patch = hsv[sensed]
    med = np.median(patch, axis=0)
    lo = np.clip(med - (12, 60, 60), 0, 255).astype(np.uint8)
    hi = np.clip(med + (12, 60, 60), 0, 255).astype(np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best, best_area = 0, 0
    for i in range(1, n):
        _, y, _, hh, area = stats[i]
        if y + hh >= h - 2 and area > best_area:
            best, best_area = i, area
    return (lab == best).astype(np.uint8) if best else (mask > 0).astype(np.uint8)


class OccGrid:
    """A metric top-down occupancy grid. `cells[r, c]` is UNKNOWN/FREE/OCCUPIED;
    row 0 = farthest forward, last row = ego; column centre = straight ahead.
    Ready for Phase-4 temporal accumulation (integrate over `bev.ego_motion`)."""

    def __init__(self, cells, cell_m: float):
        self.cells = cells
        self.cell_m = cell_m
        self.rows, self.cols = cells.shape

    def cell_metric(self, r, c):
        """Cell (row, col) centre -> metric ego coords (x right, y forward), m."""
        x_m = (c + 0.5 - self.cols / 2) * self.cell_m
        y_m = (self.rows - 0.5 - r) * self.cell_m
        return float(x_m), float(y_m)

    def clear_ahead_m(self):
        """Forward distance of the FREE corridor straight ahead before the first
        OCCUPIED/UNKNOWN cell in the centre column."""
        c = self.cols // 2
        dist = 0.0
        for r in range(self.rows - 1, -1, -1):        # ego row upward = forward
            if self.cells[r, c] != FREE:
                break
            dist += self.cell_m
        return round(dist, 2)

    def occupied(self):
        """Occupied cells as {range_m, bearing_deg, x_m, y_m}, nearest first."""
        import numpy as np
        out = []
        for r, c in zip(*np.where(self.cells == OCCUPIED)):
            x_m, y_m = self.cell_metric(int(r), int(c))
            if y_m <= 0:
                continue
            out.append({"range_m": round(math.hypot(x_m, y_m), 2),
                        "bearing_deg": round(math.degrees(math.atan2(x_m, y_m)), 1),
                        "x_m": round(x_m, 2), "y_m": round(y_m, 2),
                        "row": int(r), "col": int(c)})
        return sorted(out, key=lambda o: o["range_m"])

    def summary(self):
        import numpy as np
        occ = self.occupied()
        a = self.cell_m * self.cell_m
        return {"ok": True, "rows": self.rows, "cols": self.cols,
                "cell_m": self.cell_m,
                "free_m2": round(int(np.count_nonzero(self.cells == FREE)) * a, 2),
                "occupied_m2": round(int(np.count_nonzero(self.cells == OCCUPIED)) * a, 2),
                "unknown_m2": round(int(np.count_nonzero(self.cells == UNKNOWN)) * a, 2),
                "clear_ahead_m": self.clear_ahead_m(),
                "n_occupied_cells": len(occ),
                "nearest": occ[0] if occ else None,
                "occupied": occ[:12]}


def _grids(warp, bev_t, cell_m, occ_frac):
    """Shared: build FREE/OCCUPIED/UNKNOWN full-res masks + block-reduce to cells."""
    import cv2
    import numpy as np
    h, w = warp.shape[:2]
    sensed = (cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    free = (_drivable(warp) & sensed).astype(np.uint8)
    occ = ((sensed > 0) & (free == 0)).astype(np.uint8)
    cell_px = max(1, int(round(cell_m * bev_t.px_per_m)))
    rows, cols = max(1, h // cell_px), max(1, w // cell_px)
    cells = np.full((rows, cols), UNKNOWN, np.uint8)
    for r in range(rows):
        for c in range(cols):
            ys, xs = slice(r * cell_px, (r + 1) * cell_px), slice(c * cell_px, (c + 1) * cell_px)
            s = int(sensed[ys, xs].sum())
            if s < cell_px * cell_px * 0.25:           # mostly unsensed
                continue
            cells[r, c] = OCCUPIED if occ[ys, xs].sum() >= s * occ_frac else FREE
    return cells, sensed, free, occ, cell_px


def occupancy_grid(image, bev_t, cell_m: float = 0.5, occ_frac: float = 0.35) -> OccGrid:
    """Class-agnostic BEV occupancy grid. Warp to the ground plane, call the
    drivable colour FREE and the rest of the sensed field OCCUPIED, block-reduce
    to `cell_m` metric cells. Returns an `OccGrid` (`.summary()` for the brain)."""
    warp = bev_t.warp(image)
    if warp is None:
        raise ValueError("no image / empty BEV warp")
    cells, *_ = _grids(warp, bev_t, cell_m, occ_frac)
    return OccGrid(cells, cell_m)


def occupancy_flow(prev_image, image, bev_t, cell_m: float = 0.5,
                   occ_frac: float = 0.35, dt: float | None = None,
                   move_thresh_m: float = 0.15) -> dict:
    """Occupancy flow: velocity attached to occupied cells. Ego-compensate the two
    BEV frames (cancel the world streaming past), take dense flow on the residual,
    and report the velocity of each OCCUPIED cell that is genuinely moving. No model."""
    try:
        import cv2
        import numpy as np
        from . import bev as _bev
        pw, cw = bev_t.warp(prev_image), bev_t.warp(image)
        if pw is None or cw is None:
            return {"error": "no image"}
        em = _bev.ego_motion(pw, cw, bev_t.px_per_m)
        h, w = cw.shape[:2]
        ppm = bev_t.px_per_m
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -em.get("dyaw_deg", 0), 1.0)
        M[0, 2] += -em.get("dx_m", 0) * ppm
        M[1, 2] += em.get("dy_m", 0) * ppm
        aligned = cv2.warpAffine(pw, M, (w, h))            # ego-compensate prev -> cur
        ga = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        gc = cv2.cvtColor(cw, cv2.COLOR_BGR2GRAY)
        # flow cur -> ego-comp prev, so it is indexed by the object's CURRENT cell;
        # the object's prev->cur displacement is the negative of that flow.
        flow = cv2.calcOpticalFlowFarneback(gc, ga, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        cells, _, _, occ, cell_px = _grids(cw, bev_t, cell_m, occ_frac)
        grid = OccGrid(cells, cell_m)
        movers = []
        for r, c in zip(*np.where(cells == OCCUPIED)):
            ys, xs = slice(r * cell_px, (r + 1) * cell_px), slice(c * cell_px, (c + 1) * cell_px)
            if ga[ys, xs].min() == 0:                      # touched the warp-in border
                continue
            fx, fy = float(flow[ys, xs, 0].mean()), float(flow[ys, xs, 1].mean())
            vx_m, vy_m = -fx / ppm, fy / ppm               # BEV v grows down = toward ego
            speed_m = math.hypot(vx_m, vy_m)
            if speed_m < move_thresh_m:
                continue
            x_m, y_m = grid.cell_metric(int(r), int(c))
            mv = {"range_m": round(math.hypot(x_m, y_m), 2),
                  "bearing_deg": round(math.degrees(math.atan2(x_m, y_m)), 1),
                  "vx_m": round(vx_m, 3), "vy_m": round(vy_m, 3),
                  "speed_m_per_frame": round(speed_m, 3)}
            if dt:
                mv["speed_mps"] = round(speed_m / dt, 2)
                mv["ttc_s"] = round(y_m / (vy_m / -dt), 2) if vy_m < 0 else None
            movers.append(mv)
        movers.sort(key=lambda m: m["range_m"])
        return {"ok": True, "ego_dx_m": em.get("dx_m"), "ego_dy_m": em.get("dy_m"),
                "ego_dyaw_deg": em.get("dyaw_deg"), "n_moving_cells": len(movers),
                "movers": movers[:12], "grid": grid.summary()}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _roi(shape, roi):
    h, w = shape[:2]
    if roi is None:                                        # central lower box
        return int(w * 0.3), int(h * 0.4), int(w * 0.4), int(h * 0.45)
    return tuple(int(v) for v in roi)


def ttc(prev_image, image, roi=None, dt: float | None = None,
        scales=None, min_score: float = 0.3) -> dict:
    """Time-to-contact from LOOMING — scale change of a region between two frames.
    Scale-free (Lee's tau): needs no metric calibration, only that the thing is
    growing. Templates the ROI from the previous frame, finds the scale that best
    matches it in the current frame; a scale ratio r>1 means it is looming and
    tau = 1/(r-1) frames. Deterministic template matching, no model.

    `roi`=(x,y,w,h) of the object in the previous frame (defaults to the central
    lower region). `dt`=seconds/frame to also report ttc in seconds."""
    try:
        import cv2
        import numpy as np
        a, b = _asimg(prev_image), _asimg(image)
        if a is None or b is None:
            return {"error": "no image"}
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        x, y, w, h = _roi(ga.shape, roi)
        templ = ga[y:y + h, x:x + w]
        if templ.size == 0 or min(templ.shape) < 4:
            return {"error": "roi too small"}
        H, W = gb.shape[:2]
        if scales is None:
            scales = [round(s, 3) for s in np.linspace(0.75, 1.35, 13)]
        best = {"score": -1.0, "scale": 1.0}
        for s in scales:
            tw, th = int(round(w * s)), int(round(h * s))
            if tw < 4 or th < 4 or tw > W or th > H:
                continue
            res = cv2.matchTemplate(gb, cv2.resize(templ, (tw, th)), cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score > best["score"]:
                best = {"score": score, "scale": float(s)}
        if best["score"] < min_score:
            return {"ok": True, "matched": False, "score": round(best["score"], 3),
                    "looming": False, "ttc_frames": None, "ttc_s": None}
        r = best["scale"]
        looming = r > 1.0 + 1e-3
        ttc_frames = round(1.0 / (r - 1.0), 2) if looming else None
        return {"ok": True, "matched": True, "score": round(best["score"], 3),
                "scale_ratio": round(r, 3), "looming": looming,
                "ttc_frames": ttc_frames,
                "ttc_s": (round(ttc_frames * dt, 2) if (ttc_frames and dt) else None)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def mono_distance(bev_t=None, ground_px=None, focal_px: float | None = None,
                  real_size_m: float | None = None, pixel_size: float | None = None) -> dict:
    """Deterministic monocular distance — two model-free routes:

    * GROUND-CONTACT (calibrated): pass `bev_t` + `ground_px` (the pixel where the
      object meets the ground). Ray/ground-plane intersection resolves scale.
    * REMEMBERED-SIZE prior: pass `focal_px`, the object's known `real_size_m`, and
      its apparent `pixel_size`. distance = focal·size/px. (The size prior is what
      a face/object gallery remembers — memory instead of a learned scale head.)"""
    if bev_t is not None and ground_px is not None:
        d = bev_t.distance_bearing(ground_px)
        if d is None:
            return {"ok": True, "range_m": None, "method": "ground_plane",
                    "note": "pixel at/above horizon"}
        return {"ok": True, "method": "ground_plane", **d}
    if focal_px and real_size_m and pixel_size:
        rng = focal_px * real_size_m / pixel_size
        return {"ok": True, "method": "size_prior", "range_m": round(rng, 2),
                "focal_px": focal_px, "real_size_m": real_size_m, "pixel_size": pixel_size}
    return {"error": "need (bev_t, ground_px) or (focal_px, real_size_m, pixel_size)"}
