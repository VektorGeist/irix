"""ocr — the text-reading shim the rest of Irix calls.

Default engine is Irix's OWN model-free OCR (see text.py) — no external binary,
no neural net. tesseract remains available as an optional legacy engine only if a
caller forces it (IRIX_OCR=tesseract) and it's installed.

  run(path|image, psm=...) -> str          (psm kept for signature compat; ignored
  run_image(pil_image)     -> str           by the Irix engine)
  available()              -> bool
  engine()                 -> 'irix' | 'tesseract' | 'none'
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path


# --- engine selection ---------------------------------------------------------
def _forced() -> str:
    return os.environ.get("IRIX_OCR", "irix").lower()


@lru_cache(maxsize=1)
def _tesseract_bin() -> str | None:
    cand = os.environ.get("IRIX_TESSERACT") or shutil.which("tesseract")
    return cand if cand and Path(cand).exists() else None


def _irix_ready() -> bool:
    try:
        from . import text
        return text.default_gallery().stats()["exemplars"] > 0
    except Exception:
        return False


def engine() -> str:
    if _forced() == "tesseract":
        return "tesseract" if _tesseract_bin() else "none"
    if _irix_ready():
        return "irix"
    return "tesseract" if _tesseract_bin() else "none"


def available() -> bool:
    return engine() != "none"


# --- tesseract (legacy) -------------------------------------------------------
def _tess_run(image_path: str, psm: str, timeout: int) -> str:
    exe = _tesseract_bin()
    if not exe:
        return ""
    try:
        r = subprocess.run([exe, image_path, "stdout", "--psm", psm],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout or ""
    except Exception:
        return ""


# --- public API ---------------------------------------------------------------
def _irix_multiscale(arr) -> str:
    """Irix OCR at 1x, 2x and 3x, merged by word union.

    Small/thin glyphs (web body text) can vanish in binarization at 1x; at
    higher scales the strokes survive and the exemplar matcher sees them.
    Each scale up roughly halves the misread rate on small text (measured
    2026-09-15: 11px DejaVu went from total garbage at 1x to near-clean at
    3x). The 1x read keeps line structure; tokens found only at 2x/3x are
    appended. Deterministic. ~3x OCR cost per image.
    """
    from . import text
    base = text.read(arr, do_deskew=False).get("text", "") or ""
    seen = set(base.split())
    extra: list[str] = []
    try:
        import cv2
        h, w = arr.shape[:2]
        for scale in (2, 3):
            big = cv2.resize(arr, (w * scale, h * scale),
                             interpolation=cv2.INTER_CUBIC)
            up = text.read(big, do_deskew=False).get("text", "") or ""
            for tok in up.split():
                if tok not in seen:
                    seen.add(tok)
                    extra.append(tok)
    except Exception:
        pass
    return base + ("\n" + " ".join(extra) if extra else "")


def run(image_path: str, psm: str = "6", timeout: int = 15) -> str:
    eng = engine()
    if eng == "irix":
        import numpy as np
        from PIL import Image
        try:
            arr = np.array(Image.open(image_path).convert("L"))
        except Exception:
            return ""
        return _irix_multiscale(arr)
    if eng == "tesseract":
        return _tess_run(image_path, psm, timeout)
    return ""


def run_image(img, psm: str = "6", timeout: int = 15) -> str:
    eng = engine()
    if eng == "irix":
        import numpy as np
        return _irix_multiscale(np.array(img.convert("L")))
    tmp = str(Path(tempfile.gettempdir()) / "irix_ocr.png")
    try:
        img.save(tmp)
    except Exception:
        return ""
    return _tess_run(tmp, psm, timeout)
