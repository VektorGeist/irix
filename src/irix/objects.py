"""objects — model-free GENERAL object recognition by remembered exemplars.

The face gallery's "no ceiling" principle (Flouk 2026-06-29) extended to
everything Irix sees: recognition is MEMORY, not inference. No detector class
is needed — `physics.salient_regions` finds candidate regions, a deterministic
descriptor (LBP texture grid + HSV colour histogram — pure arithmetic)
fingerprints each one, and a zip-backed gallery remembers it.

`identify()` closes the semantic-vision loop: for every salient region, physics
DESCRIBES it (neutral prose, no nouns — the blind-person's account) and the
gallery NAMES it if it has been seen before. The brain gets both: a name when
memory has one, a physical description to deduce from when it doesn't. More
exemplars (angles, lighting) -> strictly better matching; never a model.

Zip entries are identical to the face gallery's (<id>.npy/.png/.json), so one
container format serves both organs.
"""
from __future__ import annotations

import os

from . import face as _face
from . import physics as _physics

_OBJ_SIZE = 128           # normalized object crop
_H_BINS, _S_BINS = 32, 8  # HSV colour histogram bins
_TEX_W, _COL_W = 0.5, 0.5
_DEFAULT_DIST = float(os.environ.get("IRIX_OBJ_MATCH_DIST", "0.28"))
# 0.28 = placeholder like IRIX_FACE_MATCH_DIST: synthetic same-object ~0.02,
# different-object ~0.35; calibrate on real captures before trusting in the wild.


def _imread(image_or_path):
    import cv2
    return cv2.imread(image_or_path) if isinstance(image_or_path, str) else image_or_path


def _norm(crop_bgr):
    """Normalize any crop to a fixed BGR canvas so descriptors are comparable."""
    import cv2
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    if crop_bgr.ndim == 2:
        crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
    return cv2.resize(crop_bgr, (_OBJ_SIZE, _OBJ_SIZE))


def _fg_mask(n_bgr):
    """The object's own pixels inside its crop. A salient crop always carries
    backdrop; histogramming it would fingerprint the ROOM, not the object.
    Otsu splits the crop in two; the side denser in the central half (the
    salient box centres the object) is the foreground. Deterministic."""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(n_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    q = _OBJ_SIZE // 4
    inner = slice(q, -q)
    m = th > 0
    if m[inner, inner].mean() < (~m)[inner, inner].mean():
        m = ~m
    return m if 0.05 < m.mean() < 0.98 else np.ones_like(m)   # degenerate -> all


def descriptor(crop_bgr):
    """Deterministic object fingerprint: the face module's LBP texture grid on
    the equalized gray + an HSV colour histogram over the FOREGROUND pixels only
    (faces are colour-normalized, objects are largely colour-DEFINED). Both
    L1-normalized, concatenated."""
    import cv2
    import numpy as np
    n = _norm(crop_bgr)
    gray = cv2.equalizeHist(cv2.cvtColor(n, cv2.COLOR_BGR2GRAY))
    tex = _face.descriptor(gray)
    hsv = cv2.cvtColor(n, cv2.COLOR_BGR2HSV)
    m = _fg_mask(n)
    hh, _ = np.histogram(hsv[:, :, 0][m], bins=_H_BINS, range=(0, 180))
    sh, _ = np.histogram(hsv[:, :, 1][m], bins=_S_BINS, range=(0, 256))
    col = np.concatenate([hh, sh]).astype(np.float32)
    s = col.sum()
    return np.concatenate([tex, col / s if s else col])


def _dist(a, b) -> float:
    """Weighted chi-square over the texture and colour parts of the descriptor."""
    k = _H_BINS + _S_BINS
    return _TEX_W * _face._chi2(a[:-k], b[:-k]) + _COL_W * _face._chi2(a[-k:], b[-k:])


class ObjectGallery(_face.Gallery):
    """Zip-backed exemplar store for ARBITRARY objects. Storage container and
    append/stat mechanics come from the face gallery; only the descriptor,
    the distance, and the detect seam differ."""

    def _default_dist(self) -> float:
        return _DEFAULT_DIST

    def remember(self, crop_bgr, name: str, src: str = "") -> str:
        """Store one object crop (any size, BGR) under `name`. Returns entry id."""
        import time
        import uuid
        n = _norm(crop_bgr)
        eid = uuid.uuid4().hex
        meta = {"name": name, "ts": time.time(), "src": src,
                "w": int(crop_bgr.shape[1]), "h": int(crop_bgr.shape[0])}
        self._append(eid, descriptor(n), n, meta)
        return eid

    def match(self, crop_bgr, threshold: float | None = None) -> dict:
        """Nearest remembered exemplar for one crop — minimum weighted chi-square
        over the whole gallery. Memory retrieval, no model."""
        thr = self._default_dist() if threshold is None else threshold
        q = descriptor(crop_bgr)
        best_name, best_dist = None, 1e9
        for e in self._load().values():
            d = _dist(q, e["desc"])
            if d < best_dist:
                best_dist, best_name = d, e["name"]
        known = best_name is not None and best_dist <= thr
        return {"name": best_name if known else None,
                "dist": round(best_dist, 4) if best_name is not None else None,
                "known": known, "nearest": best_name}

    def enroll(self, image_or_path, name: str, box=None, src: str = "") -> dict:
        """Remember an object from an image: crop `box` (x,y,w,h) if given, else
        the most salient region physics finds."""
        img = _imread(image_or_path)
        if img is None:
            return {"error": "no image"}
        boxes = [tuple(box)] if box else _physics.salient_regions(img, max_n=1)
        if not boxes:
            return {"error": "no salient region"}
        x, y, w, h = boxes[0]
        eid = self.remember(img[y:y + h, x:x + w], name, src)
        return {"ok": True, "enrolled": [eid], "n": 1, "box": list(boxes[0])}

    def identify(self, image_or_path, prev_frames=None, threshold: float | None = None,
                 auto_enroll_unknown: str | None = None, max_n: int = 5) -> dict:
        """The semantic-vision loop: every salient region gets its PHYSICS read
        (neutral description for the brain) and a gallery lookup (a NAME, if this
        eye has seen it before). `auto_enroll_unknown` remembers new objects on
        sight under a generated name — the gallery grows by looking."""
        img = _imread(image_or_path)
        if img is None:
            return {"error": "no image"}
        out = []
        for box in _physics.salient_regions(img, max_n=max_n):
            x, y, w, h = box
            crop = img[y:y + h, x:x + w]
            r = self.match(crop, threshold)
            attrs = _physics.region_attributes(img, box, prev_frames=prev_frames)
            entry = {"box": list(box), "name": r["name"], "dist": r["dist"],
                     "known": r["known"], "attributes": attrs,
                     "description": _physics.physical_description(attrs)}
            if not r["known"] and auto_enroll_unknown:
                new_name = f"{auto_enroll_unknown}{len(self._identities()) + 1}"
                self.remember(crop, new_name, src="auto")
                entry["name"], entry["auto_enrolled"] = new_name, True
            out.append(entry)
        return {"ok": True, "objects": out, "gallery_size": len(self._load())}
