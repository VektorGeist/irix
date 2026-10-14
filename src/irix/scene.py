"""scene — deterministic eyes for a MOVING world (road / track / game scene),
where there are no UI boxes+text to read. Classical CV only (Canny/Hough/
optical-flow/colour) — NO model. Each head emits structured TEXT; the consuming
brain reasons on that text. Runs on CPU, real-time, no VRAM.
"""
from __future__ import annotations


def _imread(image_path, scale=1.0):
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return None
    if scale != 1.0:
        img = cv2.resize(img, (max(1, int(img.shape[1] * scale)),
                               max(1, int(img.shape[0] * scale))))
    return img


def path(image_path: str, roi_top: float = 0.45) -> dict:
    """Where the road/track goes — classical lane detection (Canny + Hough) on the
    lower ROI. Returns a STEER signal {direction, angle_deg, offset}. Pure geometry
    of the detected road edges; no model."""
    try:
        import cv2
        import numpy as np
        img = _imread(image_path)
        if img is None:
            return {"error": "no image"}
        h, w = img.shape[:2]
        roi = img[int(h * roi_top):h, :]
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        edges = cv2.Canny(gray, 60, 160)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50,
                                minLineLength=int(h * 0.10), maxLineGap=40)
        if lines is None:
            return {"ok": True, "direction": "UNKNOWN", "n_lines": 0, "confidence": 0.0}
        left, right = [], []
        for x1, y1, x2, y2 in lines[:, 0, :]:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:                  # near-horizontal -> not a lane edge
                continue
            (left if slope < 0 else right).append((x1 + x2) / 2)
        cx = w / 2
        if left and right:
            center = (np.mean(left) + np.mean(right)) / 2
        elif left:
            center = np.mean(left) + w * 0.25
        elif right:
            center = np.mean(right) - w * 0.25
        else:
            return {"ok": True, "direction": "STRAIGHT", "n_lines": int(len(lines)),
                    "confidence": 0.2}
        offset = center - cx                       # +right, -left of frame centre
        frac = offset / (w / 2)
        direction = "STRAIGHT" if abs(frac) < 0.12 else ("RIGHT" if frac > 0 else "LEFT")
        angle = float(np.degrees(np.arctan2(offset, h * (1 - roi_top))))
        return {"ok": True, "direction": direction, "offset": int(offset),
                "angle_deg": round(angle, 1), "frac": round(float(frac), 2),
                "n_lines": int(len(lines)), "left_edges": len(left),
                "right_edges": len(right),
                "confidence": round(min(1.0, (len(left) + len(right)) / 8), 2)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def freespace(image_path: str, sample_frac: float = 0.12) -> dict:
    """Drivable area by colour homogeneity (no model): sample the patch just ahead
    of the player (bottom-centre = ground under us), grow the region of similar
    colour, report how open each direction is."""
    try:
        import cv2
        import numpy as np
        img = _imread(image_path, scale=0.5)
        if img is None:
            return {"error": "no image"}
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sh, sw = int(h * sample_frac), int(w * sample_frac)
        patch = hsv[h - sh:h, w // 2 - sw // 2: w // 2 + sw // 2].reshape(-1, 3)
        med = np.median(patch, axis=0)
        tol = np.array([12, 60, 60])               # hue tight, sat/val loose
        lo = np.clip(med - tol, 0, 255).astype(np.uint8)
        hi = np.clip(med + tol, 0, 255).astype(np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        best, best_area = 0, 0
        for i in range(1, n):                      # largest blob touching the bottom
            _, y, _, hh, area = stats[i]
            if y + hh >= h - 2 and area > best_area:
                best, best_area = i, area
        road = (lab == best).astype(np.uint8) if best else (mask > 0).astype(np.uint8)
        bands = {}
        for name, a, b in (("LEFT", 0, w // 3), ("CENTER", w // 3, 2 * w // 3),
                           ("RIGHT", 2 * w // 3, w)):
            bands[name] = round(float(road[:, a:b].mean()), 2)
        open_direction = max(bands, key=bands.get)
        center_col = road[:, w // 3:2 * w // 3].max(axis=1)
        rows = np.where(center_col > 0)[0]
        clear_ahead = round(float((h - rows.min()) / h), 2) if len(rows) else 0.0
        return {"ok": True, "coverage": round(float(road.mean()), 2),
                "open_direction": open_direction, "clear_ahead": clear_ahead,
                "bands": bands}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def motion(prev_image_path: str, image_path: str) -> dict:
    """What's moving between two frames — dense optical flow (Farneback, no model).
    Overall motion (speed proxy), lateral drift, focus-of-expansion."""
    try:
        import cv2
        import numpy as np
        a = _imread(prev_image_path, scale=0.25)
        b = _imread(image_path, scale=0.25)
        if a is None or b is None:
            return {"error": "no image"}
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx, fy = flow[..., 0], flow[..., 1]
        speed = float(np.sqrt(fx * fx + fy * fy).mean())
        mean_fx = float(fx.mean())
        drift = "STEADY" if abs(mean_fx) < 0.3 else ("RIGHT" if mean_fx > 0 else "LEFT")
        h, w = ga.shape[:2]
        sign = np.sign(fx.mean(axis=0))            # forward: flow diverges from FOE
        foe = next((i for i in range(1, len(sign)) if sign[i - 1] < 0 <= sign[i]), None)
        return {"ok": True, "speed": round(speed, 2), "drift": drift,
                "mean_fx": round(mean_fx, 2),
                "foe_x_frac": round(foe / w, 2) if foe is not None else None,
                "moving": speed > 0.4}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ====================================================================== #
# BEV / "vector space" heads — Phase 2. Same deterministic algorithms, run
# in the top-down METRIC world (bev.BEV) so outputs are real metres/curvature,
# not image pixels. No model. Image-space heads above stay as the no-calibration
# fallback.
# ====================================================================== #

def _lane_mask(bev_img):
    """Binary lane-pixel mask in BEV: vertical-gradient (Sobel-x) OR bright lines."""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)
    sx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sx = (sx / (sx.max() + 1e-6) * 255).astype(np.uint8)
    grad = sx > 40
    bright = gray > max(180, int(np.percentile(gray, 95)))
    return ((grad | bright).astype(np.uint8)) * 255


def _sliding_lane(mask, nwin=9, margin=40, minpix=25):
    """Sliding-window lane-pixel collection from a histogram base. Returns
    [left_pts, right_pts] where each is (xs, ys) in BEV px, or None per side."""
    import numpy as np
    h, w = mask.shape
    hist = (mask[h // 2:] > 0).sum(axis=0)
    mid = w // 2
    out = []
    for half, base in (("L", int(np.argmax(hist[:mid]))),
                       ("R", mid + int(np.argmax(hist[mid:])))):
        if hist[base] == 0:
            out.append(None); continue
        cur, xs, ys, win_h = base, [], [], h // nwin
        for i in range(nwin):
            ylo, yhi = h - (i + 1) * win_h, h - i * win_h
            xlo, xhi = max(0, cur - margin), min(w, cur + margin)
            yy, xx = np.nonzero(mask[ylo:yhi, xlo:xhi])
            if len(xx) > minpix:
                cur = int(xlo + xx.mean())
            xs += list(xlo + xx); ys += list(ylo + yy)
        out.append((np.array(xs), np.array(ys)) if len(xs) > minpix else None)
    return out


def _fit_metric(pts, w, ppm):
    """Fit a lane's pixels as x_m = a*y_m^2 + b*y_m + c in METRIC ego coords
    (x right, y forward, origin at ego). Returns (a, b, c) or None."""
    import numpy as np
    xs, ys = pts
    h = ys.max() + 1
    ym = (h - 1 - ys) / ppm                    # forward metres
    xm = (xs - w / 2) / ppm                    # lateral metres (ego centre = 0)
    if len(set(ys.tolist())) < 3:
        return None
    return tuple(np.polyfit(ym, xm, 2))


def lane_bev(image, bev_t) -> dict:
    """Lane geometry in the metric BEV: curvature radius (m), heading, and ego
    offset from lane centre (m). The real upgrade over image-space `path`."""
    try:
        import numpy as np
        warp = bev_t.warp(image)
        if warp is None:
            return {"error": "no image"}
        sides = _sliding_lane(_lane_mask(warp))
        ppm, w = bev_t.px_per_m, bev_t.out_w
        fits = [(_fit_metric(p, w, ppm) if p is not None else None) for p in sides]
        lf, rf = fits
        present = [f for f in fits if f is not None]
        if not present:
            return {"ok": True, "direction": "UNKNOWN", "lanes": 0, "confidence": 0.0}
        a = float(np.mean([f[0] for f in present]))       # avg curvature term
        b = float(np.mean([f[1] for f in present]))       # heading term at ego
        radius = float((1 + b * b) ** 1.5 / abs(2 * a)) if abs(a) > 1e-6 else float("inf")
        direction = ("STRAIGHT" if radius > 80 or radius == float("inf")
                     else ("RIGHT" if a > 0 else "LEFT"))
        out = {"ok": True, "lanes": len(present),
               "curve_radius_m": (round(radius, 1) if radius != float("inf") else None),
               "direction": direction,
               "heading_deg": round(float(np.degrees(np.arctan(b))), 1),
               "confidence": round(len(present) / 2, 2)}
        if lf is not None and rf is not None:             # both lanes -> ego offset
            centre = (lf[2] + rf[2]) / 2                   # lane-centre x at ego (m)
            out["lane_width_m"] = round(abs(rf[2] - lf[2]), 2)
            out["ego_offset_m"] = round(-centre, 2)        # +right of lane centre
        return out
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def freespace_bev(image, bev_t, sample_frac: float = 0.12) -> dict:
    """Drivable area in the metric BEV: clear-ahead distance (m), drivable area
    (m^2), and per-band openness. Metric upgrade over image-space `freespace`."""
    try:
        import cv2
        import numpy as np
        warp = bev_t.warp(image)
        if warp is None:
            return {"error": "no image"}
        h, w = warp.shape[:2]
        hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
        sh, sw = int(h * sample_frac), int(w * sample_frac)
        patch = hsv[h - sh:h, w // 2 - sw // 2: w // 2 + sw // 2].reshape(-1, 3)
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
        road = (lab == best).astype(np.uint8) if best else (mask > 0).astype(np.uint8)
        ppm = bev_t.px_per_m
        col = road[:, w // 3:2 * w // 3].max(axis=1)       # centre corridor
        free_rows = np.where(col > 0)[0]
        clear_ahead_m = round((h - free_rows.min()) / ppm, 2) if len(free_rows) else 0.0
        bands = {nm: round(float(road[:, a:b].mean()), 2) for nm, a, b in
                 (("LEFT", 0, w // 3), ("CENTER", w // 3, 2 * w // 3), ("RIGHT", 2 * w // 3, w))}
        return {"ok": True, "clear_ahead_m": clear_ahead_m,
                "drivable_m2": round(float(road.sum()) / (ppm * ppm), 2),
                "open_direction": max(bands, key=bands.get), "bands": bands}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def flow_bev(prev_image, image, bev_t, dt: float | None = None,
             move_thresh: float = 1.2) -> dict:
    """Ego-COMPENSATED motion in BEV: estimate ego displacement, cancel the
    world streaming past, and report only INDEPENDENT movers (the occupancy-flow
    idea). No model."""
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
        dxg, dyg = -em.get("dx_m", 0) * ppm, em.get("dy_m", 0) * ppm   # ground px motion
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -em.get("dyaw_deg", 0), 1.0)
        M[0, 2] += dxg; M[1, 2] += dyg
        aligned = cv2.warpAffine(pw, M, (w, h))            # ego-compensate prev -> cur
        ga = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        gc = cv2.cvtColor(cw, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(ga, gc, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt((flow * flow).sum(axis=2))
        movers = (mag > move_thresh).astype(np.uint8)
        movers[ga == 0] = 0                                # ignore the warp-in border
        nlab, lab, stats, cent = cv2.connectedComponentsWithStats(movers, 8)
        objs = []
        for i in range(1, nlab):
            if stats[i, cv2.CC_STAT_AREA] < max(20, h * w // 2000):
                continue
            cx, cy = cent[i]
            x_m, y_m = (cx - w / 2) / ppm, (h - cy) / ppm
            objs.append({"range_m": round(float(np.hypot(x_m, y_m)), 2),
                         "bearing_deg": round(float(np.degrees(np.arctan2(x_m, y_m))), 1),
                         "speed_px": round(float(mag[lab == i].mean()), 2)})
        ego_speed = (round(abs(em.get("dy_m", 0)) / dt, 2) if dt else None)
        return {"ok": True, "ego_dx_m": em.get("dx_m"), "ego_dy_m": em.get("dy_m"),
                "ego_dyaw_deg": em.get("dyaw_deg"), "ego_speed_mps": ego_speed,
                "n_moving": len(objs),
                "movers": sorted(objs, key=lambda o: o["range_m"])[:6]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def describe_scene(image_path: str, prev_image_path: str | None = None,
                   include_ui: bool = True, bev_t=None, dt: float | None = None,
                   world=None) -> dict:
    """Fuse the deterministic heads into ONE structured-text scene, zero models.

    Pass a `bev.BEV` as `bev_t` to read the scene in the metric "vector space"
    (lane curvature/offset in m, clear-ahead in m, ego-compensated movers with
    range+bearing). Without it, falls back to the image-space heads.

    Phase 5: with `bev_t` + a prev frame, movers come from the occupancy flow —
    per-obstacle VELOCITY, min TTC (with `dt`), and the occupancy-grid summary
    are fused into the text. Pass a `worldmodel.WorldModel` as `world` to also
    observe the frame into the persistent map and report remembered space."""
    if world is not None and bev_t is None:
        bev_t = world.bev
    metric = bev_t is not None
    p = lane_bev(image_path, bev_t) if metric else path(image_path)
    fs = freespace_bev(image_path, bev_t) if metric else freespace(image_path)
    mo, occ = None, None
    if prev_image_path is not None:
        if metric:
            from . import occupancy as _occ
            mo = _occ.occupancy_flow(prev_image_path, image_path, bev_t, dt=dt)
            occ = mo.get("grid") if mo.get("ok") else None
        else:
            mo = motion(prev_image_path, image_path)
    wm = None
    if world is not None:
        wm = world.observe(image_path)
        if occ is None and wm.get("ok"):
            occ = wm["grid"]
    ui = []
    if include_ui:
        try:
            from . import elements
            ui = [e["label"] for e in elements.parse(image_path) if e.get("label")]
        except Exception:
            ui = []
    parts = []
    if p.get("ok") and p.get("direction") not in (None, "UNKNOWN"):
        if metric:
            seg = f"LANE: {p['direction']}"
            if p.get("curve_radius_m"):
                seg += f" R={p['curve_radius_m']}m"
            if p.get("ego_offset_m") is not None:
                seg += f" off {p['ego_offset_m']}m"
            parts.append(seg + f" (conf {p.get('confidence', 0)})")
        else:
            parts.append(f"PATH: {p['direction']} ~{p.get('angle_deg', 0)}° "
                         f"(off {p.get('offset', 0)}px, conf {p.get('confidence', 0)})")
    if fs.get("ok"):
        if metric:
            parts.append(f"FREE: ahead {fs['clear_ahead_m']}m, most-open "
                         f"{fs['open_direction']} ({fs['drivable_m2']}m²)")
        else:
            parts.append(f"FREE: ahead {fs['clear_ahead']}, most-open {fs['open_direction']} "
                         f"(cover {fs['coverage']})")
    min_ttc = None
    if mo and mo.get("ok"):
        if metric:
            seg = f"MOVERS: {mo['n_moving_cells']}"
            if dt and mo.get("ego_dy_m") is not None:
                seg += f", ego {round(abs(mo['ego_dy_m']) / dt, 2)}m/s"
            if mo["movers"]:
                near = mo["movers"][0]
                seg += (f", nearest {near['range_m']}m @ {near['bearing_deg']}° "
                        f"v({near['vx_m']},{near['vy_m']})m/f")
                ttcs = [m["ttc_s"] for m in mo["movers"] if m.get("ttc_s")]
                if ttcs:
                    min_ttc = min(ttcs)
                    seg += f", minTTC {min_ttc}s"
            parts.append(seg)
        else:
            parts.append(f"MOTION: speed {mo['speed']}, drift {mo['drift']}")
    if occ:
        parts.append(f"OCC: clear {occ['clear_ahead_m']}m, "
                     f"free {occ['free_m2']}m²/occ {occ['occupied_m2']}m²")
    if wm and wm.get("ok"):
        snap = world.snapshot().summary()
        pose = wm["pose"]
        parts.append(f"WORLD: pose ({pose['x_m']},{pose['y_m']},{pose['yaw_deg']}°), "
                     f"mapped {world.summary()['mapped_m2']}m², "
                     f"mem-clear {snap['clear_ahead_m']}m")
    if ui:
        parts.append("UI: [" + ", ".join(ui[:8]) + "]")
    return {"ok": True, "text": " | ".join(parts) or "no scene read",
            "path": p, "freespace": fs, "motion": mo, "occupancy": occ,
            "world": wm, "min_ttc_s": min_ttc, "ui": ui, "metric": metric}
