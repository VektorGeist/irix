#!/usr/bin/env python3
"""Eidetrix tiny brain: the learned fallback behind the deterministic eye.

Product shape (Flouk's cut): the gallery/eye reads first; the tiny brain
arbitrates. Two rules, both measured on the study battery (see
~/workspace/eidetrix-study/tinybrain/eval_hybrid.py):

  ARBITER   if the reader is very sure (confidence >= C) and disagrees with
            the gallery's best label, the reader overrules. This catches the
            gallery's confident-but-wrong confusions (measured: 'Wi-Fj' ->
            'Wi-Fi', +1 recall@0.8 on the battery, zero regressions).
  FALLBACK  if the gallery's best cost exceeds T (or the gallery has no
            answer), the reader's ranking is used. This covers degraded /
            out-of-gallery glyphs, where the reader beats the gallery
            0.668 to 0.441 at glyph level.

A plain uncertainty fallback alone does NOT improve the battery (ties the
eye at 0.895; aggressive thresholds hurt). The arbiter is what earns the
wiring: 102/114 -> 103/114 @0.8, no regressions.

Pure numpy, CPU-only, deterministic. Weights (136 KB) ship as package data.
"""
import os
from pathlib import Path

import numpy as np

from .text import _CHARS, _square

N = 24            # reader input size (matches the eye's _square)
T = 0.20          # gallery-cost fallback threshold (calibrated)
C = 0.99          # reader-confidence arbiter threshold (calibrated)


def _im2col(x, k):
    N_, C_, H, W = x.shape
    OH, OW = H - k + 1, W - k + 1
    sN, sC, sH, sW = x.strides
    cols = np.lib.stride_tricks.as_strided(
        x, shape=(N_, OH, OW, C_, k, k),
        strides=(sN, sH, sW, sC, sH, sW))
    return cols.reshape(N_ * OH * OW, C_ * k * k)


def _conv(x, W, b, k):
    N_, _, H, W_ = x.shape
    col = _im2col(x, k)
    out = col @ W.reshape(W.shape[0], -1).T + b
    OH, OW = H - k + 1, W_ - k + 1
    return out.reshape(N_, OH, OW, W.shape[0]).transpose(0, 3, 1, 2)


def _relu(x):
    return np.maximum(x, 0.0)


def _maxpool(x):
    N_, C_, H, W = x.shape
    return x.reshape(N_, C_, H // 2, 2, W // 2, 2).max(axis=(3, 5))


def _affine(x, W, b):
    return x.reshape(x.shape[0], -1) @ W + b


def _forward(p, x):
    x = _relu(_conv(x, p["c1W"], p["c1b"], 5))
    x = _maxpool(x)
    x = _relu(_conv(x, p["c2W"], p["c2b"], 3))
    x = _maxpool(x)
    x = _relu(_affine(x, p["a1W"], p["a1b"]))
    return _affine(x, p["a2W"], p["a2b"])


def _load_params(path):
    d = np.load(path)
    return {
        "c1W": d["L00_Conv2d_W"].astype(np.float32),
        "c1b": d["L00_Conv2d_b"].astype(np.float32),
        "c2W": d["L03_Conv2d_W"].astype(np.float32),
        "c2b": d["L03_Conv2d_b"].astype(np.float32),
        "a1W": d["L06_Affine_W"].astype(np.float32),
        "a1b": d["L06_Affine_b"].astype(np.float32),
        "a2W": d["L09_Affine_W"].astype(np.float32),
        "a2b": d["L09_Affine_b"].astype(np.float32),
    }


class TinyBrain:
    """The Kaggle-trained glyph reader (tb_read_v1, 33k params).

    Reads the same 24x24 _square windows the gallery matcher sees.
    predict(window) -> (char, confidence) or None for blank windows.
    """

    def __init__(self, path=None):
        if path is None:
            base = os.environ.get("IRIX_BRAIN_DIR")
            path = str(Path(base) / "tinybrain.npz") if base else str(
                Path(__file__).resolve().parent / "tinybrain.npz")
        self.path = path
        self._p = None

    def _ensure(self):
        if self._p is None:
            self._p = _load_params(self.path)
        return self._p

    def predict(self, window):
        sq, _ = _square(window)
        if sq is None:
            return None
        p = self._ensure()
        x = sq.astype(np.float32)[None, None, :, :]
        logits = _forward(p, x)[0]
        z = logits - logits.max()
        e = np.exp(z)
        probs = e / e.sum()
        i = int(probs.argmax())
        return _CHARS[i], float(probs[i])

    def rank(self, window, m=3):
        sq, _ = _square(window)
        if sq is None:
            return None
        p = self._ensure()
        x = sq.astype(np.float32)[None, None, :, :]
        logits = _forward(p, x)[0]
        z = logits - logits.max()
        e = np.exp(z)
        probs = e / e.sum()
        top = np.argsort(-probs, kind="stable")[:m]
        return [(_CHARS[int(i)], float(probs[int(i)])) for i in top]


_brain = None


def default_brain() -> TinyBrain:
    global _brain
    if _brain is None:
        _brain = TinyBrain()
    return _brain


class HybridGallery:
    """Drop-in for GlyphGallery: gallery first, tiny brain arbitrates.

    Same match_multi/match interface; the eye's DP code is untouched.
    """

    def __init__(self, gallery, brain=None, t=T, c=C):
        self.gallery = gallery
        self.brain = brain or default_brain()
        self.t = t
        self.c = c

    def _reader_ranking(self, window, m):
        r = self.brain.rank(window, m)
        if r is None:
            return None
        # map confidence -> cost in [t, 1]: the DP keeps preferring
        # gallery-confident windows; among reader windows, confident first.
        sq, asp = _square(window)
        mapped = [(lab, self.t + (1.0 - conf) * (1.0 - self.t))
                  for lab, conf in r]
        return mapped, asp

    def match_multi(self, window, m=3):
        g_res = self.gallery.match_multi(window, m=m)
        r = self.brain.rank(window, m)
        if g_res is None:
            return self._reader_ranking(window, m)
        ranked, asp = g_res
        g_lab, g_cost = ranked[0]
        if r is not None:
            r_lab, r_conf = r[0]
            if r_conf >= self.c and r_lab != g_lab:
                return self._reader_ranking(window, m)
        if g_cost <= self.t:
            return g_res
        return self._reader_ranking(window, m) or g_res

    def match(self, window):
        res = self.match_multi(window)
        if res is None:
            return None
        ranked, asp = res
        return ranked[0][0], ranked[0][1], asp

    def add(self, squares, aspects, labels):
        return self.gallery.add(squares, aspects, labels)

    def stats(self):
        s = self.gallery.stats()
        s["brain"] = Path(self.brain.path).name
        return s
