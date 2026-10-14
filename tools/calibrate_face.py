"""Calibrate IRIX_FACE_MATCH_DIST on REAL webcam captures.

Sits on the webcam for a few seconds, detects the person's face in each frame,
and measures the WITHIN-person descriptor distances (same face, different
instants/angles). The match threshold must sit above the within-person spread
— recommended = p95(within) * 1.3 — and stats are printed so the margin can be
judged against any second identity already in a gallery.

Usage:  python tools/calibrate_face.py [N_FRAMES] [GALLERY_ZIP]
Needs a person in front of the camera; reports plainly if it sees no face.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from irix import capture, face


def main(n_frames: int = 20, gallery_path: str | None = None) -> int:
    crops = []
    with capture.webcam(0) as cam:
        t0 = time.time()
        while len(crops) < n_frames and time.time() - t0 < 20:
            frame = cam.read()
            if frame is None:
                continue
            boxes = face.detect(frame)
            if boxes:
                big = max(boxes, key=lambda b: b[2] * b[3])
                crop = face._crop_norm(frame, big)
                if crop is not None:
                    crops.append(crop)
            time.sleep(0.25)                       # spread captures over time
    if len(crops) < 4:
        print(f"BLOCKED: only {len(crops)} face captures in 20s — need a person "
              f"in front of the webcam (got a frame but no detectable face)."
              if crops or True else "")
        return 1
    descs = [face.descriptor(c) for c in crops]
    within = [face._chi2(descs[i], descs[j])
              for i in range(len(descs)) for j in range(i + 1, len(descs))]
    w = np.array(within)
    stats = {"captures": len(crops), "pairs": len(within),
             "min": round(float(w.min()), 4), "median": round(float(np.median(w)), 4),
             "p95": round(float(np.percentile(w, 95)), 4),
             "max": round(float(w.max()), 4)}
    proposed = round(float(np.percentile(w, 95)) * 1.3, 2)
    print("within-person distances:", stats)
    print(f"current threshold: {face._DEFAULT_DIST}  (env IRIX_FACE_MATCH_DIST)")
    print(f"PROPOSED threshold: {proposed}  (p95 * 1.3)")
    if gallery_path:
        g = face.Gallery(gallery_path)
        if g.stats()["exemplars"]:
            between = [face._chi2(descs[0], e["desc"]) for e in g._load().values()]
            print("distances to existing gallery exemplars:",
                  {"min": round(min(between), 4), "median": round(float(np.median(between)), 4)})
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    gal = sys.argv[2] if len(sys.argv) > 2 else None
    raise SystemExit(main(n, gal))
