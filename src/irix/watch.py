"""watch — the irix-watch capture daemon.

Watches the screen and remembers what changed. Every `interval` seconds it grabs
a frame; when the pixels differ enough from the last remembered frame it stores
the screenshot in episodic visual memory (auto-captioned from the Irix element
parse). Quiet periods store nothing — the memory holds episodes, not footage.

Deterministic, model-free, local. No network, no models.

Usage:
    irix-watch                          # watch primary screen, 5s interval
    irix-watch --interval 10 --region 0,0,1920,1080
    irix-watch --frames 12              # stop after 12 captures (or Ctrl-C)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Mean per-pixel absdiff (0-255) on a 64x36 thumbnail that counts as "changed".
DEFAULT_THRESHOLD = 8.0
DEFAULT_INTERVAL = 5.0


def _thumbnail(frame):
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)


def frame_changed(prev_thumb, frame, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """True when `frame` differs enough from the previous thumbnail."""
    import cv2
    import numpy as np
    cur = _thumbnail(frame)
    diff = cv2.absdiff(prev_thumb, cur)
    return float(np.mean(diff)) >= threshold


def _loop(source, mem, interval: float, threshold: float,
          max_frames: int | None, quiet: bool, log) -> dict:
    """Capture loop over an already-open frame source. Returns stats."""
    import cv2
    stored = 0
    skipped = 0
    prev_thumb = None
    frames = 0
    try:
        while max_frames is None or frames < max_frames:
            frame = source.read()
            frames += 1
            if frame is None:
                time.sleep(interval)
                continue
            if prev_thumb is not None and not frame_changed(
                    prev_thumb, frame, threshold):
                skipped += 1
                time.sleep(interval)
                continue
            prev_thumb = _thumbnail(frame)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            shot = Path(mem.episodes_dir) / f"watch-{stamp}.png"
            cv2.imwrite(str(shot), frame)
            try:
                res = mem.remember(str(shot), source="watch")
            except Exception as ex:
                log({"event": "remember_failed", "error": f"{type(ex).__name__}: {ex}"})
                skipped += 1
                continue
            stored += 1
            if not quiet:
                log({"event": "remembered", "id": res["id"],
                     "duplicate_of": res["duplicate_of"], "file": str(shot)})
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    return {"stored": stored, "skipped": skipped, "frames": frames}


def run(interval: float = DEFAULT_INTERVAL, threshold: float = DEFAULT_THRESHOLD,
        region=None, memory_dir: str = "~/.irix/memory",
        max_frames: int | None = None, quiet: bool = False) -> dict:
    """Watch the screen and remember changes. Returns stats dict."""
    import json
    from . import capture
    from .memory import MemoryStore

    def log(obj):
        print(json.dumps(obj), flush=True)

    mem = MemoryStore(memory_dir)
    try:
        source = capture.screen(region)
    except RuntimeError as ex:
        log({"event": "no_capture_backend", "error": str(ex)})
        return {"stored": 0, "skipped": 0, "frames": 0, "error": str(ex)}
    with source:
        if not quiet:
            log({"event": "watch_start", "interval": interval,
                 "threshold": threshold, "memory": str(mem.db_path)})
        stats = _loop(source, mem, interval, threshold, max_frames, quiet, log)
    if not quiet:
        log({"event": "watch_stop", **stats})
    return stats


def _parse_region(s: str):
    x, y, w, h = (int(v) for v in s.split(","))
    return (x, y, w, h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="irix-watch", description=__doc__.splitlines()[0])
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                    help="seconds between frame grabs")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="mean pixel change (0-255) that counts as changed")
    ap.add_argument("--region", default=None,
                    help="x,y,w,h capture region (default: primary screen)")
    ap.add_argument("--memory-dir", default="~/.irix/memory")
    ap.add_argument("--frames", type=int, default=None,
                    help="stop after this many grabs (default: run until Ctrl-C)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    region = _parse_region(a.region) if a.region else None
    run(interval=a.interval, threshold=a.threshold, region=region,
        memory_dir=a.memory_dir, max_frames=a.frames, quiet=a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
