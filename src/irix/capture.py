"""capture — cross-platform frame sources. Irix reads frames; it never assumes one OS.

Sources:
  webcam(index)         -> FrameSource over a camera (OpenCV VideoCapture)
  screen(region)        -> FrameSource over the screen (mss if installed; grim on
                           Wayland/sway as a fallback for the VektOS target)
A FrameSource yields BGR numpy frames via .read(); use .close() when done.
All deterministic; no model.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


class FrameSource:
    def read(self):
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class _Webcam(FrameSource):
    def __init__(self, index: int = 0):
        import cv2
        self.cap = cv2.VideoCapture(index)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()


class _MSS(FrameSource):
    def __init__(self, region=None):
        import mss
        import numpy as np
        self._np = np
        self.sct = mss.mss()
        self.mon = (region or self.sct.monitors[0])
        if isinstance(self.mon, (tuple, list)):
            x, y, w, h = self.mon
            self.mon = {"left": x, "top": y, "width": w, "height": h}

    def read(self):
        import cv2
        shot = self.sct.grab(self.mon)
        arr = self._np.asarray(shot)             # BGRA
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self.sct.close()


class _PILGrab(FrameSource):
    """Windows/macOS desktop grab via Pillow ImageGrab (GDI BitBlt on Windows).
    No extra deps beyond Pillow; works on the interactive WinSta0 desktop."""

    def __init__(self, region=None):
        from PIL import ImageGrab  # noqa: F401
        self.bbox = None
        if region:
            x, y, w, h = region
            self.bbox = (x, y, x + w, y + h)

    def read(self):
        import cv2
        import numpy as np
        from PIL import ImageGrab
        im = ImageGrab.grab(bbox=self.bbox)
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


class _Grim(FrameSource):
    """Wayland/sway screen grab — the VektOS target. Spawns grim per frame."""

    def __init__(self, region=None):
        self.region = region
        self.tmp = str(Path(tempfile.gettempdir()) / "irix_grim.png")

    def read(self):
        import cv2
        args = ["grim"]
        if self.region:
            x, y, w, h = self.region
            args += ["-g", f"{x},{y} {w}x{h}"]
        args += [self.tmp]
        try:
            subprocess.run(args, capture_output=True, timeout=5)
        except Exception:
            return None
        return cv2.imread(self.tmp)


def webcam(index: int = 0) -> FrameSource:
    return _Webcam(index)


def screen(region=None) -> FrameSource:
    """Best available screen source: mss if installed, else Pillow ImageGrab on
    Windows/macOS, else grim (Wayland/sway)."""
    try:
        import mss  # noqa: F401
        return _MSS(region)
    except Exception:
        pass
    try:
        from PIL import ImageGrab  # noqa: F401
        if sys.platform in ("win32", "darwin"):
            return _PILGrab(region)
    except Exception:
        pass
    if shutil.which("grim"):
        return _Grim(region)
    raise RuntimeError("no screen capture backend: pip install mss "
                       "(or Pillow on Windows/macOS, or grim on Wayland)")
