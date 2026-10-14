"""track — multi-cue DETERMINISTIC tracking + the fast reflex loop ("spider-sense").

No model in the loop. A target is tracked by whichever cheap cue fits:
  - colour     : numpy distance to a target BGR colour (fastest; great for markers)
  - template   : normalized cross-correlation of a locked patch (matchTemplate)
  - orb        : ORB keypoint match (robust to scale/rotation; re-acquire after loss)
Each tick searches only a small ROI around the last position (fast), and on loss
re-acquires over the full frame. Optical-flow gives a one-step motion PREDICT so the
ROI is centered where the target is heading, not where it was.

Returns positions; actuation (moving a cursor / pressing a key) is the consumer's
job — Irix is eyes, not hands. Faster than any model can run, because there is no
model: the whole tick is array arithmetic.
"""
from __future__ import annotations


class Tracker:
    """Track a target across frames. Construct with ONE cue, then call step(frame)
    per frame. `pos` is the last (x, y) center in frame coords, or None if lost."""

    def __init__(self, *, color=None, template=None, roi: int = 140, tol: int = 80):
        self.roi = roi
        self.tol = tol
        self.color = tuple(int(c) for c in color) if color is not None else None
        self.template = template            # BGR patch (numpy) for template/orb cues
        self.pos: tuple[int, int] | None = None
        self._vel = (0, 0)                  # last motion vector (predict)
        self._orb = None
        self._tmpl_kp = None
        if template is not None:
            self._init_template(template)

    # --- cue: colour ------------------------------------------------------------
    def _find_color(self, frame, ox, oy):
        import numpy as np
        d = np.abs(frame.astype("int16") - np.array(self.color, dtype="int16")).sum(axis=2)
        ys, xs = np.where(d < self.tol)
        if len(xs) == 0:
            return None
        return int(xs.mean()) + ox, int(ys.mean()) + oy

    # --- cue: template (NCC) ----------------------------------------------------
    def _find_template(self, frame, ox, oy):
        import cv2
        if self.template is None or frame.shape[0] < self.template.shape[0] \
                or frame.shape[1] < self.template.shape[1]:
            return None
        res = cv2.matchTemplate(frame, self.template, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv < 0.5:
            return None
        th, tw = self.template.shape[:2]
        return maxloc[0] + tw // 2 + ox, maxloc[1] + th // 2 + oy

    def _init_template(self, patch):
        import cv2
        self._orb = cv2.ORB_create(500)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        self._tmpl_kp = self._orb.detectAndCompute(gray, None)

    # --- cue: ORB (scale/rotation robust re-acquire) ----------------------------
    def _find_orb(self, frame):
        import cv2
        import numpy as np
        if self._orb is None or self._tmpl_kp[1] is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        kp2, des2 = self._orb.detectAndCompute(gray, None)
        if des2 is None:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self._tmpl_kp[1], des2)
        if len(matches) < 6:
            return None
        pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        return int(pts[:, 0].mean()), int(pts[:, 1].mean())

    # --- the tick ---------------------------------------------------------------
    def _roi_box(self, frame):
        if self.pos is None:
            return None
        px = self.pos[0] + self._vel[0]        # predict where it's heading
        py = self.pos[1] + self._vel[1]
        r = self.roi
        h, w = frame.shape[:2]
        x0, y0 = max(0, px - r), max(0, py - r)
        x1, y1 = min(w, px + r), min(h, py + r)
        return x0, y0, x1, y1

    def step(self, frame):
        """One perceive tick. Returns (x, y) or None."""
        box = self._roi_box(frame)
        if box:
            x0, y0, x1, y1 = box
            sub = frame[y0:y1, x0:x1]
            ox, oy = x0, y0
        else:
            sub, ox, oy = frame, 0, 0

        found = None
        if self.color is not None:
            found = self._find_color(sub, ox, oy)
        elif self.template is not None:
            found = self._find_template(sub, ox, oy)

        # lost in ROI -> re-acquire over the full frame
        if found is None and box is not None:
            if self.color is not None:
                found = self._find_color(frame, 0, 0)
            elif self.template is not None:
                found = self._find_orb(frame) or self._find_template(frame, 0, 0)

        if found is not None and self.pos is not None:
            self._vel = (found[0] - self.pos[0], found[1] - self.pos[1])
        self.pos = found
        return found


def reflex_loop(source, tracker, on_target, max_frames: int | None = None):
    """The fast reflex: read frames from a capture source, track, and fire a
    deterministic callback `on_target(x, y, frame)` each time the target is seen.
    No model anywhere in the loop. Returns the number of frames processed."""
    n = 0
    with source as src:
        while max_frames is None or n < max_frames:
            frame = src.read()
            if frame is None:
                break
            pos = tracker.step(frame)
            if pos is not None:
                on_target(pos[0], pos[1], frame)
            n += 1
    return n
