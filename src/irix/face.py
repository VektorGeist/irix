"""face — model-free face recognition by REMEMBERED EXEMPLARS.

No trained model. Irix detects a face (classical Haar cascade), computes a
DETERMINISTIC descriptor (Local Binary Pattern histogram — pure arithmetic, no
weights), and stores it — plus the cropped face frame — into a zip file that acts
as a growing storage container (the "gallery"). Recognition is nearest-neighbor
retrieval over that gallery, not inference.

Why there is no ceiling: the gallery grows. Every new exemplar of a person (a new
angle, new lighting) is another stored sample; recognition takes the *minimum*
distance over all of that person's exemplars, so accuracy only improves as Irix
sees more. Memory + a deterministic protocol = capability — the same principle as
a search index.

Each gallery entry in the zip:
  <id>.npy   the LBP descriptor (float32)
  <id>.png   the cropped, normalized face frame (the remembered picture)
  <id>.json  {name, ts, src, w, h}
The index is rebuilt by listing the zip — nothing central to rewrite on append.
"""
from __future__ import annotations

import io
import json
import os
import time
import uuid
import zipfile
from pathlib import Path

# --- tunables -----------------------------------------------------------------
_FACE_SIZE = 128          # normalized face crop
_GRID = 8                 # LBP histogram grid (GxG cells)
_BINS = 256               # LBP codes
_DEFAULT_DIST = float(os.environ.get("IRIX_FACE_MATCH_DIST", "0.55"))
# 0.55 calibrated on real webcam captures 2026-07-01 (tools/calibrate_face.py):
# within-person chi2 ran 0.04-0.48 (p95 0.475) across poses, so the old 0.42
# placeholder REJECTED genuine matches. 0.55 clears the measured within-person
# max; between-person margin still unmeasured (needs a second face) — revisit
# with a two-person capture before trusting against strangers.


def _cascade():
    import cv2
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(path)


def detect(image_or_path) -> list[tuple]:
    """Face bounding boxes [(x,y,w,h)] via Haar cascade. Accepts a path or a BGR
    numpy frame. Classical detector that ships with OpenCV — no deep model."""
    import cv2
    img = cv2.imread(image_or_path) if isinstance(image_or_path, str) else image_or_path
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.equalizeHist(gray)
    faces = _cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                        minSize=(48, 48))
    return [tuple(int(v) for v in f) for f in faces]


def _crop_norm(img, box):
    """Crop a face, grayscale, resize, histogram-equalize -> a stable canvas so the
    descriptor is comparable across captures."""
    import cv2
    x, y, w, h = box
    face = img[max(0, y):y + h, max(0, x):x + w]
    if face.size == 0:
        return None
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if face.ndim == 3 else face
    gray = cv2.resize(gray, (_FACE_SIZE, _FACE_SIZE))
    return cv2.equalizeHist(gray)


def _lbp_code(gray):
    """8-neighbour Local Binary Pattern, vectorized. Pure arithmetic, deterministic."""
    import numpy as np
    g = gray.astype(np.int16)
    center = g[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint8)
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    H, W = g.shape
    for i, (dy, dx) in enumerate(neigh):
        shifted = g[1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx]
        code |= ((shifted >= center).astype(np.uint8) << i)
    return code


def descriptor(gray):
    """LBP descriptor: grid of per-cell histograms, concatenated + L1-normalized.
    Deterministic feature vector — the fingerprint we store and match."""
    import numpy as np
    code = _lbp_code(gray)
    ch, cw = code.shape[0] // _GRID, code.shape[1] // _GRID
    feats = []
    for gy in range(_GRID):
        for gx in range(_GRID):
            cell = code[gy * ch:(gy + 1) * ch, gx * cw:(gx + 1) * cw]
            hist, _ = np.histogram(cell, bins=_BINS, range=(0, 256))
            feats.append(hist.astype(np.float32))
    v = np.concatenate(feats)
    s = v.sum()
    return v / s if s else v


def _chi2(a, b) -> float:
    """Chi-square distance between two L1-normalized histograms. 0 = identical."""
    import numpy as np
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-10)))


class Gallery:
    """A zip-backed face gallery — the storage container. Append-only on enroll;
    descriptors are cached in memory for fast matching."""

    def __init__(self, path: str):
        base = os.environ.get("IRIX_GALLERY_DIR")
        self.path = str(Path(base) / path) if base and not os.path.isabs(path) else path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict | None = None     # {id: {"name","desc",...}}

    # --- storage ----------------------------------------------------------------
    def _load(self) -> dict:
        if self._cache is not None:
            return self._cache
        import numpy as np
        out: dict = {}
        if Path(self.path).exists():
            with zipfile.ZipFile(self.path, "r") as z:
                names = z.namelist()
                for n in names:
                    if not n.endswith(".json"):
                        continue
                    eid = n[:-5]
                    meta = json.loads(z.read(n))
                    npy = eid + ".npy"
                    if npy in names:
                        desc = np.load(io.BytesIO(z.read(npy)))
                        out[eid] = {**meta, "desc": desc}
        self._cache = out
        return out

    def _append(self, eid: str, desc, crop, meta: dict) -> None:
        import cv2
        import numpy as np
        nb = io.BytesIO()
        np.save(nb, desc.astype(np.float32))
        ok, png = cv2.imencode(".png", crop)
        mode = "a" if Path(self.path).exists() else "w"
        with zipfile.ZipFile(self.path, mode, zipfile.ZIP_DEFLATED) as z:
            z.writestr(eid + ".npy", nb.getvalue())
            if ok:
                z.writestr(eid + ".png", png.tobytes())
            z.writestr(eid + ".json", json.dumps(meta))
        if self._cache is not None:
            self._cache[eid] = {**meta, "desc": desc}

    # --- core: detection-free recognition (the testable model-free seam) ---------
    def remember(self, crop_gray, name: str, src: str = "") -> str:
        """Store one already-normalized (gray, _FACE_SIZE²) face crop under `name`.
        Returns the entry id. This is the pure 'remember an exemplar' step."""
        desc = descriptor(crop_gray)
        eid = uuid.uuid4().hex
        meta = {"name": name, "ts": time.time(), "src": src,
                "w": int(crop_gray.shape[1]), "h": int(crop_gray.shape[0])}
        self._append(eid, desc, crop_gray, meta)
        return eid

    def match(self, crop_gray, threshold: float | None = None) -> dict:
        """Nearest-neighbor recognition for one normalized crop. No model — just the
        minimum chi-square distance over every remembered exemplar."""
        thr = self._default_dist() if threshold is None else threshold
        q = descriptor(crop_gray)
        best_name, best_dist = None, 1e9
        for e in self._load().values():
            d = _chi2(q, e["desc"])
            if d < best_dist:
                best_dist, best_name = d, e["name"]
        known = best_name is not None and best_dist <= thr
        return {"name": best_name if known else None,
                "dist": round(best_dist, 4) if best_name is not None else None,
                "known": known, "nearest": best_name}

    # --- public API: detect + (remember | match) --------------------------------
    def enroll(self, image_or_path, name: str, src: str = "") -> dict:
        """Detect every face in the image and remember each under `name`."""
        import cv2
        img = cv2.imread(image_or_path) if isinstance(image_or_path, str) else image_or_path
        if img is None:
            return {"error": "no image"}
        ids = []
        for box in detect(img):
            crop = _crop_norm(img, box)
            if crop is None:
                continue
            ids.append(self.remember(crop, name, src))
        return {"ok": True, "enrolled": ids, "n": len(ids)} if ids \
            else {"error": "no face detected"}

    def recognize(self, image_or_path, threshold: float | None = None,
                  auto_enroll_unknown: str | None = None) -> dict:
        """Recognize every face in the image. If `auto_enroll_unknown` is a name
        prefix, unknown faces are remembered as new identities on sight."""
        import cv2
        img = cv2.imread(image_or_path) if isinstance(image_or_path, str) else image_or_path
        if img is None:
            return {"error": "no image"}
        results = []
        for box in detect(img):
            crop = _crop_norm(img, box)
            if crop is None:
                continue
            r = self.match(crop, threshold)
            entry = {"box": list(box), "name": r["name"], "dist": r["dist"],
                     "known": r["known"]}
            if not r["known"] and auto_enroll_unknown:
                new_name = f"{auto_enroll_unknown}{len(self._identities()) + 1}"
                self.remember(crop, new_name, src="auto")
                entry["name"], entry["auto_enrolled"] = new_name, True
            results.append(entry)
        return {"ok": True, "faces": results, "gallery_size": len(self._load())}

    def _identities(self) -> set:
        return {e["name"] for e in self._load().values()}

    def _default_dist(self) -> float:
        return _DEFAULT_DIST

    def stats(self) -> dict:
        gal = self._load()
        by = {}
        for e in gal.values():
            by[e["name"]] = by.get(e["name"], 0) + 1
        return {"exemplars": len(gal), "identities": len(by), "by_name": by,
                "path": self.path}
