"""preprocess — deterministic image conditioning before OCR. No model.

Otsu binarize (auto-polarity), skew correction, perspective rectify, denoise. The
deterministic equivalents of what an LSTM OCR partly absorbs implicitly — done in
the open so the reader sees clean, axis-aligned text.
"""
from __future__ import annotations


def to_gray(image_or_path):
    import cv2
    if isinstance(image_or_path, str):
        return cv2.imread(image_or_path, cv2.IMREAD_GRAYSCALE)
    img = image_or_path
    if img is not None and img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def binarize(gray):
    """Otsu threshold with auto-polarity: text -> white (255) on black, so the
    sparser ink is always the foreground (handles dark- and light-mode)."""
    import cv2
    import numpy as np
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(th) > th.size / 2:
        th = 255 - th
    return th


def denoise(bw):
    """Drop speckle: morphological open with a tiny kernel."""
    import cv2
    import numpy as np
    k = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)


def _row_sharpness(bw) -> float:
    """Variance of the horizontal ink projection — high when text rows are
    axis-aligned (ink concentrated in bands), low when tilted (smeared)."""
    import numpy as np
    return float(np.var((bw > 0).sum(axis=1).astype(np.float64)))


def deskew(gray, max_angle: float = 20.0):
    """Estimate text skew (minAreaRect over the ink cloud) and rotate flat —
    but VERIFY before applying: the rotation must sharpen the horizontal
    projection, else the estimate was an artifact (minAreaRect tilts to hug
    the ascenders/descenders of a single short line) and rotating would smear
    perfectly straight text. Deterministic; bounded to ±max_angle."""
    import cv2
    import numpy as np
    bw = binarize(gray)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 50:
        return gray, 0.0
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    if abs(angle) > max_angle or abs(angle) < 0.2:
        return gray, 0.0
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    bw_rot = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST)
    if _row_sharpness(bw_rot) <= _row_sharpness(bw) * 1.05:   # no real gain
        return gray, 0.0
    rot = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    return rot, round(float(angle), 2)


def rectify(gray):
    """If a dominant quadrilateral (a document/sign) is present, warp it to a
    front-on rectangle. Returns the rectified gray, or the original if no quad."""
    import cv2
    import numpy as np
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray
    h, w = gray.shape[:2]
    big = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(big) < 0.25 * h * w:
        return gray
    peri = cv2.arcLength(big, True)
    quad = cv2.approxPolyDP(big, 0.02 * peri, True)
    if len(quad) != 4:
        return gray
    pts = quad.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    src = np.array([tl, tr, br, bl], np.float32)
    ow = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    oh = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if ow < 10 or oh < 10:
        return gray
    dst = np.array([[0, 0], [ow, 0], [ow, oh], [0, oh]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (ow, oh))


def condition(image_or_path, do_deskew: bool = True, do_rectify: bool = False):
    """Full deterministic conditioning -> a clean binarized image for the reader.
    Returns {bw, gray, skew}."""
    gray = to_gray(image_or_path)
    if gray is None:
        return {"error": "no image"}
    if do_rectify:
        gray = rectify(gray)
    skew = 0.0
    if do_deskew:
        gray, skew = deskew(gray)
    return {"bw": denoise(binarize(gray)), "gray": gray, "skew": skew}
