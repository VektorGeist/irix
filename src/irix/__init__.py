"""Irix — deterministic, model-free eyes.

Turns pixels into structured TEXT and recognizes the world by remembered
exemplars, never by a trained model. CPU-only, no VRAM, pixel-exact.

Submodules:
  elements   — UI element parsing + find + set-of-mark (deterministic)
  bev        — bird's-eye "vector space": IPM homography + metric ground distance + ego-motion
  scene      — moving-world perception (path/freespace/motion/describe_scene)
  screen     — perceive-a-screen-once cache (instant recall)
  face       — model-free face gallery (zip storage container)
  text       — Irix's own OCR: DP segmentation-free reader + glyph gallery
  preprocess — binarize / deskew / perspective-rectify / denoise
  textcorr   — deterministic dictionary/confusion context correction
  inject     — prompt-injection screen (reads hidden text before any model)
  track      — multi-cue deterministic tracker + fast reflex loop
  capture    — cross-platform frame sources (webcam / screen)
  ocr        — text-reading shim (Irix engine by default; tesseract optional)
"""
from __future__ import annotations

try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    try:
        __version__ = _pkg_version("irix-eyes")
    except PackageNotFoundError:
        __version__ = "0.0.0+local"
except Exception:
    __version__ = "0.0.0+local"

__all__ = ["elements", "bev", "scene", "screen", "face", "text", "preprocess",
           "textcorr", "inject", "track", "capture", "ocr"]
