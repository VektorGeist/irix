"""bev — bird's-eye / "vector space" foundation for scene perception. Model-free.

Tesla's edge is the REPRESENTATION, not the net: perception is done in a top-down
metric world (their "vector space"), not in image pixels. That move is pure
geometry — an Inverse-Perspective-Mapping (IPM) homography of the ground plane —
so Irix gets the same representation with zero models.

Two anchors make it metric and memory-ready:
  * `from_camera()` — a calibrated ground-plane homography (camera height + pitch +
    field-of-view). This is the metric ANCHOR that resolves monocular scale: a
    pixel touching the ground has a real distance, derived by ray/ground-plane
    intersection. No net needed, no scale ambiguity for ground contacts.
  * `from_points()` — calibration-free: 4 image points that bound a real-world
    rectangle on the ground (a road patch, a board) + its size in metres.
  * `ego_motion()` — rigid displacement between two BEV frames = the ego's own
    motion. This is the seam a persistent occupancy map integrates over time
    (the "remember the scene" / temporal world-model layer).

Everything here is CPU OpenCV/geometry. `cv2`/`numpy` are imported lazily so
`import irix` never needs them.
"""
from __future__ import annotations

import math


def _asimg(x, scale: float = 1.0):
    """Accept an image PATH or an ndarray; return a BGR ndarray (or None)."""
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


def _ray_ground(K_inv, R, height_m, u, v):
    """Intersect the ray through pixel (u,v) with the ground plane Z=0.

    World: X=right, Y=forward (horizontal), Z=up; camera at (0,0,height). Returns
    metric (X, Y) on the ground, or None if the ray points at/above the horizon.
    """
    import numpy as np
    d = R @ (K_inv @ np.array([u, v, 1.0]))      # ray direction in world frame
    if d[2] >= -1e-6:                            # not pointing down -> no ground hit
        return None
    t = height_m / -d[2]                         # camera Z=height, plane Z=0
    return float(t * d[0]), float(t * d[1])      # (X right, Y forward) in metres


def _rot_pitch(theta):
    """cam->world rotation for a camera pitched DOWN by theta (about world X).

    Zero pitch looks along world +Y (forward, horizontal); image-down -> world -Z.
    """
    import numpy as np
    r0 = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])   # cam->world at 0 pitch
    c, s = math.cos(-theta), math.sin(-theta)                 # tilt forward axis down
    rx = np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])
    return rx @ r0


class BEV:
    """A ground-plane bird's-eye transform with a metric scale.

    Holds `H` (image px -> BEV image px, for warping) and `Hg` (image px ->
    metric ground (X,Y) in metres, for distance/bearing). Ego sits at the bottom
    centre of the BEV image, forward pointing up.
    """

    def __init__(self, H, Hg, out_w, out_h, px_per_m):
        self.H, self.Hg = H, Hg
        self.out_w, self.out_h, self.px_per_m = out_w, out_h, px_per_m

    # --- constructors -----------------------------------------------------
    @classmethod
    def from_points(cls, src_px, width_m, length_m, px_per_m=20.0, near_m=0.0):
        """Calibration-free IPM from 4 image points that bound a ground rectangle.

        `src_px` = 4 image points (top-left, top-right, bottom-right, bottom-left)
        of a real rectangle on the ground that is `width_m` wide and `length_m`
        long. `near_m` = forward distance of the rectangle's near (bottom) edge.
        """
        import cv2
        import numpy as np
        src = np.asarray(src_px, np.float32)
        far = near_m + length_m
        # ground metres for the 4 corners: X right (centred), Y forward
        ground = np.array([[-width_m / 2, far], [width_m / 2, far],
                           [width_m / 2, near_m], [-width_m / 2, near_m]], np.float32)
        out_w = int(round(width_m * px_per_m))
        out_h = int(round(far * px_per_m))
        bev = np.array([cls._g2bev(g, out_w, out_h, px_per_m) for g in ground], np.float32)
        H = cv2.getPerspectiveTransform(src, bev)
        Hg = cv2.getPerspectiveTransform(src, ground)
        return cls(H, Hg, out_w, out_h, px_per_m)

    @classmethod
    def from_camera(cls, img_size, height_m, pitch_deg, hfov_deg,
                    range_m=30.0, width_m=20.0, px_per_m=12.0):
        """Calibrated ground-plane homography from camera geometry — the metric
        anchor. `img_size`=(w,h), `height_m` above ground, `pitch_deg` below
        horizontal, `hfov_deg` horizontal field of view. BEV covers `range_m`
        forward and `width_m` across at `px_per_m` resolution."""
        import cv2
        import numpy as np
        w, h = img_size
        fx = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
        K_inv = np.linalg.inv(np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]]))
        R = _rot_pitch(math.radians(pitch_deg))
        out_w, out_h = int(round(width_m * px_per_m)), int(round(range_m * px_per_m))
        src, gnd, bev = [], [], []
        for v in np.linspace(h * 0.55, h * 0.98, 4):        # sample below the horizon
            for u in np.linspace(w * 0.05, w * 0.95, 4):
                g = _ray_ground(K_inv, R, height_m, u, v)
                if g is None or not (0 < g[1] <= range_m) or abs(g[0]) > width_m / 2:
                    continue
                src.append([u, v]); gnd.append(list(g))
                bev.append(cls._g2bev(g, out_w, out_h, px_per_m))
        if len(src) < 4:
            raise ValueError("camera geometry yields no ground samples; check pitch/fov")
        src, gnd, bev = np.array(src, np.float32), np.array(gnd, np.float32), np.array(bev, np.float32)
        H, _ = cv2.findHomography(src, bev)
        Hg, _ = cv2.findHomography(src, gnd)
        return cls(H, Hg, out_w, out_h, px_per_m)

    @staticmethod
    def _g2bev(g, out_w, out_h, px_per_m):
        """Metric ground (X right, Y forward) -> BEV image px (ego bottom-centre)."""
        return [out_w / 2 + g[0] * px_per_m, out_h - g[1] * px_per_m]

    # --- use --------------------------------------------------------------
    def warp(self, image):
        """Top-down BEV image of the ground plane. Path or ndarray in."""
        import cv2
        img = _asimg(image)
        if img is None:
            return None
        return cv2.warpPerspective(img, self.H, (self.out_w, self.out_h))

    def ground(self, px):
        """Image pixel(s) -> metric ground (X right, Y forward) in metres."""
        import cv2
        import numpy as np
        pts = np.asarray(px, np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.Hg).reshape(-1, 2)
        return out[0] if out.shape[0] == 1 else out

    def distance_bearing(self, px):
        """Metric distance + bearing to a GROUND-CONTACT pixel (e.g. where an
        obstacle meets the road). The deterministic monocular-depth anchor:
        range in metres, bearing in degrees (+right). Returns None above horizon."""
        x, y = self.ground(px)
        if y <= 0:                                   # at/above the horizon
            return None
        return {"range_m": round(math.hypot(x, y), 2),
                "bearing_deg": round(math.degrees(math.atan2(x, y)), 1),
                "x_m": round(float(x), 2), "y_m": round(float(y), 2)}


def ego_motion(prev_bev, cur_bev, px_per_m, max_features=400):
    """Ego displacement between two BEV frames — the seam a persistent occupancy
    map integrates over. Ground features move rigidly opposite the ego, so the
    inverse of their motion is the ego's. Feature-based (ORB + partial-affine),
    falling back to phase correlation for low-texture frames.

    Returns {dx_m, dy_m (forward+), dyaw_deg, conf}. Memory-first: this is what
    Phase 4 accumulates to "remember the scene".
    """
    import cv2
    import numpy as np
    a, b = _asimg(prev_bev), _asimg(cur_bev)
    if a is None or b is None:
        return {"error": "no image"}
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(max_features)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is not None and db is not None and len(ka) >= 6 and len(kb) >= 6:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(da, db), key=lambda m: m.distance)[:80]
        if len(matches) >= 6:
            pa = np.float32([ka[m.queryIdx].pt for m in matches])
            pb = np.float32([kb[m.trainIdx].pt for m in matches])
            M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC)
            if M is not None and inl is not None and int(inl.sum()) >= 4:
                # ground moved by (M dx, M dy); ego moved the opposite way
                dx_px, dy_px = float(M[0, 2]), float(M[1, 2])
                dyaw = math.degrees(math.atan2(M[1, 0], M[0, 0]))
                return {"dx_m": round(-dx_px / px_per_m, 3),
                        "dy_m": round(dy_px / px_per_m, 3),   # BEV up=forward, v grows down
                        "dyaw_deg": round(-dyaw, 2),
                        "conf": round(int(inl.sum()) / max(1, len(matches)), 2),
                        "method": "orb"}
    (sx, sy), resp = cv2.phaseCorrelate(np.float32(ga), np.float32(gb))
    return {"dx_m": round(-sx / px_per_m, 3), "dy_m": round(sy / px_per_m, 3),
            "dyaw_deg": 0.0, "conf": round(float(resp), 2), "method": "phasecorr"}


def vanishing_point(image, roi_top: float = 0.4):
    """Deterministic vanishing point from converging Hough lines — bootstraps
    pitch when the camera is uncalibrated (no model). Returns (vx, vy) in image
    pixels, or None. pitch ~= atan2(cy - vy, fy)."""
    import cv2
    import numpy as np
    img = _asimg(image)
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60,
                            minLineLength=int(h * 0.12), maxLineGap=40)
    if lines is None:
        return None
    inter = []
    segs = [s[0] for s in lines if abs(s[0][2] - s[0][0]) > 1]
    for i in range(len(segs)):
        x1, y1, x2, y2 = segs[i]
        m1 = (y2 - y1) / (x2 - x1)
        if abs(m1) < 0.2:                            # ignore near-horizontal
            continue
        b1 = y1 - m1 * x1
        for j in range(i + 1, len(segs)):
            x3, y3, x4, y4 = segs[j]
            m2 = (y4 - y3) / (x4 - x3)
            if abs(m2 - m1) < 1e-3:
                continue
            b2 = y3 - m2 * x3
            xv = (b2 - b1) / (m1 - m2)
            yv = m1 * xv + b1
            if 0 <= xv <= w and 0 <= yv <= h * (1 - roi_top):
                inter.append((xv, yv))
    if not inter:
        return None
    arr = np.array(inter)
    return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
