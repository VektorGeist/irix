"""text — Irix's OWN OCR. Model-free, the gallery way. Aims to beat LSTM-tesseract
on printed/rendered text via three things tesseract can't do:

  1. EXACT-FONT MEMORY — render any font once and Irix has its literal glyph
     shapes; it matches exact shapes instead of generalizing, so on a font it has
     seen it can be *perfect* where a trained net interpolates and errs.
  2. SEGMENTATION-FREE DP READER — tesseract's real trick is it never cuts glyphs
     (its LSTM reads a whole line via CTC). We do the deterministic equivalent:
     dynamic programming finds the best tiling of the WHOLE line out of glyph
     templates, so touching/kerned text doesn't break a column-cutter.
  3. GROWS WITH USE — every new font seen is more exemplars; no ceiling.

Each glyph (template AND a candidate window) is normalized to a fixed SQUARE by
its own ink bbox, with SIZE carried separately as an aspect ratio. Shape match is
then size/baseline-invariant; CASE is recovered afterwards from the line's own
typographic frame (each glyph's measured ink height vs the line's cap height
decides c/C, o/O, ... ; bar glyphs l/I/| by word context). Honest scope:
printed/rendered/screen text, not handwriting.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

from . import preprocess

H = 36                       # normalized line height (px)
N = 24                       # square glyph canvas (N×N)
_PEN = 0.04                  # per-glyph penalty (discourages over-segmentation)
_GAP_PEN = 0.30              # per blank-gap a glyph window spans: glyph ink is
                             # horizontally connected ('tti' with 2 gaps can't be
                             # one 'm' or '@')
_GAP_OK_PEN = 0.10           # reduced rate for glyphs whose parts ARE separated
_GAP_OK = set('"%')
_TOP_ONLY = set("'\"`")      # live in the line's TOP: a full-height match is a
                             # blurred letter-pair being eaten, not a quote
_BOT_ONLY = set(".,_")       # live at the BOTTOM: never span the upper zone
_ASPECT_W = 0.35             # weight of the aspect-ratio term in glyph cost
_SPACE_FRAC = 0.33           # gap > this * H between glyphs => a space
_INVERT_FRAC = 0.65          # strip ink above this => inverted-polarity island
                             # (dark button / light modal on an opposite-polarity
                             # page): global Otsu picked the page's polarity, so
                             # this band is flipped and re-segmented. Legit text
                             # strips sit well below (measured max ~0.51).
_SPLIT_FRAC = 1.5            # blank column-run wider than this * band height =>
                             # separate text items sharing a row (status-bar ends,
                             # nav labels, icon captions) get their own lines.
_SMALL_H = 20                # bands shorter than this are re-binarized from an
_UPSCALE = 3                 # upscaled gray crop: global Otsu (tuned for the
                             # image's dominant text) starves tiny gray glyphs;
                             # a local threshold on the 3x crop recovers them.
_WMIN, _WMAX = 3, int(1.8 * H)
_CHARS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
          "abcdefghijklmnopqrstuvwxyz"
          "0123456789.,:;!?@#%&()[]{}+-*/=<>_'\"|\\$")


# --- runs / normalization -----------------------------------------------------
def _runs(mask, gap=0):
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    if gap > 0 and runs:
        merged = [runs[0]]
        for s, e in runs[1:]:
            if s - merged[-1][1] <= gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        runs = merged
    return runs


def _square(window):
    """Crop a binary window to its ink bbox and resize to N×N. Returns (sq, aspect)
    where aspect = inkw/inkh (size cue kept out of the shape)."""
    import cv2
    import numpy as np
    ys, xs = np.where(window > 0)
    if len(xs) == 0:
        return None, 0.0
    crop = window[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    sq = cv2.resize((crop * 255).astype(np.uint8), (N, N),
                    interpolation=cv2.INTER_AREA)
    return (sq > 127).astype(np.float32), float(w) / float(h)


# --- glyph gallery (zip storage container) ------------------------------------
class GlyphGallery:
    """Square glyph shapes + aspect ratios + labels, in a zip. Born by rendering
    fonts; grows when Irix meets a new font."""

    def __init__(self, path: str = "glyphs.irix"):
        base = os.environ.get("IRIX_GALLERY_DIR")
        self.path = str(Path(base) / path) if base and not os.path.isabs(path) else path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._sq = None          # (M, N, N) float
        self._asp = None         # (M,) float
        self._labels: list[str] = []

    def _load(self):
        if self._sq is not None:
            return
        import numpy as np
        if Path(self.path).exists():
            with zipfile.ZipFile(self.path, "r") as z:
                names = set(z.namelist())
                if {"sq.npy", "asp.npy", "labels.json"} <= names:
                    self._sq = np.load(io.BytesIO(z.read("sq.npy")))
                    self._asp = np.load(io.BytesIO(z.read("asp.npy")))
                    self._labels = json.loads(z.read("labels.json"))
                    self._index()
                    return
        self._sq = np.zeros((0, N, N), np.float32)
        self._asp = np.zeros((0,), np.float32)
        self._labels = []
        self._index()

    def _index(self):
        """Derived matching structures: the flat (M,N*N) matrix for the exact
        stage and a coarse 6x6 pyramid for the prefilter stage."""
        import cv2
        import numpy as np
        if len(self._labels):
            self._flat = self._sq.reshape(len(self._labels), -1).astype(np.float32)
            self._coarse = np.stack([cv2.resize(s, (6, 6), interpolation=cv2.INTER_AREA)
                                     for s in self._sq]).reshape(len(self._labels), -1)
        else:
            self._flat = np.zeros((0, N * N), np.float32)
            self._coarse = np.zeros((0, 36), np.float32)
        self._cache: dict = {}

    def _save(self):
        import numpy as np
        a, b = io.BytesIO(), io.BytesIO()
        np.save(a, self._sq.astype(np.float32))
        np.save(b, self._asp.astype(np.float32))
        with zipfile.ZipFile(self.path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("sq.npy", a.getvalue())
            z.writestr("asp.npy", b.getvalue())
            z.writestr("labels.json", json.dumps(self._labels))

    def add(self, squares: list, aspects: list, labels: list[str]) -> None:
        import numpy as np
        self._load()
        if not squares:
            return
        self._sq = np.vstack([self._sq, np.array(squares, np.float32)])
        self._asp = np.concatenate([self._asp, np.array(aspects, np.float32)])
        self._labels.extend(labels)
        self._index()
        self._save()

    def match_multi(self, window, m: int = 3):
        """Top-m DISTINCT labels [(label, cost)] + the window aspect, or None.
        Two-stage, deterministic: a coarse 6x6 shape + aspect prefilter keeps
        the top candidates, the exact N×N distance ranks them. Cached by window
        content (screen text repeats). The runner-ups let the reader reject an
        implausible best match without losing the window."""
        import cv2
        import numpy as np
        self._load()
        if len(self._labels) == 0:
            return None
        sq, asp = _square(window)
        if sq is None:
            return None
        # Cache key must include the aspect: the match cost adds an aspect
        # term, so the same squared shape at a different aspect ranks
        # differently. Keying on shape alone returned stale results for
        # same-shape/different-aspect windows (order-dependent OCR).
        key = (hash(sq.tobytes()), round(float(asp), 4))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        adiff = _ASPECT_W * np.abs(np.log1p(self._asp) - np.log1p(asp))
        q6 = cv2.resize(sq, (6, 6), interpolation=cv2.INTER_AREA).reshape(-1)
        d6 = np.abs(self._coarse - q6).mean(axis=1) + adiff
        k = min(48, len(self._labels))
        cand = np.argpartition(d6, k - 1)[:k]
        cost = np.abs(self._flat[cand] - sq.reshape(-1)).mean(axis=1) + adiff[cand]
        ranked, seen = [], set()
        for i in np.argsort(cost, kind="stable"):
            lab = self._labels[int(cand[i])]
            if lab in seen:
                continue
            seen.add(lab)
            ranked.append((lab, float(cost[i])))
            if len(ranked) >= m:
                break
        res = (ranked, asp)
        if len(self._cache) > 20000:
            self._cache.clear()
        self._cache[key] = res
        return res

    def match(self, window):
        """Best (label, cost, aspect_of_window) for a binary window — see
        match_multi. None if the window is blank or the gallery empty."""
        res = self.match_multi(window)
        if res is None:
            return None
        ranked, asp = res
        return ranked[0][0], ranked[0][1], asp

    def stats(self) -> dict:
        self._load()
        by = {}
        for c in self._labels:
            by[c] = by.get(c, 0) + 1
        return {"exemplars": len(self._labels), "chars": len(by), "path": self.path}

    # --- bootstrap by rendering fonts -----------------------------------------
    def seed_fonts(self, fonts: list[str], chars: str = _CHARS,
                   render_px: int = 64) -> dict:
        """Render each char on its own and store its square shape + aspect. Each
        glyph is normalized by its OWN bbox, identical to how a read window is
        normalized — so training and reading share one frame."""
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        used, squares, aspects, labels = [], [], [], []
        for fpath in fonts:
            try:
                font = ImageFont.truetype(fpath, render_px)
            except Exception:
                continue
            used.append(fpath)
            for ch in chars:
                if ch == " ":
                    continue
                cw = max(2, int(font.getlength(ch)))
                im = Image.new("L", (cw + render_px, render_px * 2), 0)
                ImageDraw.Draw(im).text((render_px // 2, render_px // 2), ch,
                                        fill=255, font=font)
                arr = (np.array(im) > 127).astype(np.float32)
                sq, asp = _square(arr)
                if sq is None:
                    continue
                squares.append(sq)
                aspects.append(asp)
                labels.append(ch)
        self.add(squares, aspects, labels)
        return {"ok": True, "fonts_used": used, "glyphs_added": len(labels),
                **self.stats()}

    def seed_fonts_small(self, fonts: list[str], chars: str = _CHARS,
                         pixel_sizes=(13, 17), aa_blur: float = 0.6) -> dict:
        """Remember what each glyph looks like at REAL UI sizes: rendered small,
        anti-aliased, then passed through the exact same conditioning the reader
        applies to a small screen crop (blur ~ AA, 3x cubic upscale, binarize).
        Small anti-aliased shapes are systematically different from crisp 64px
        renders — the gallery must hold BOTH (memory, not normalization).
        Measured: closes most of the crisp-vs-anti-aliased accuracy gap."""
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        used, squares, aspects, labels = [], [], [], []
        for fpath in fonts:
            for px in pixel_sizes:
                try:
                    font = ImageFont.truetype(fpath, px)
                except Exception:
                    break
                for ch in chars:
                    if ch == " ":
                        continue
                    cw = max(2, int(font.getlength(ch)))
                    im = Image.new("L", (cw + px, px * 2), 0)
                    ImageDraw.Draw(im).text((px // 2, px // 4), ch, fill=255, font=font)
                    a = np.array(im)
                    a = cv2.GaussianBlur(a, (3, 3), aa_blur)
                    a = cv2.resize(a, (a.shape[1] * 3, a.shape[0] * 3),
                                   interpolation=cv2.INTER_CUBIC)
                    sq, asp = _square((a > 127).astype(np.float32))
                    if sq is None:
                        continue
                    squares.append(sq)
                    aspects.append(asp)
                    labels.append(ch)
            else:
                used.append(fpath)
        self.add(squares, aspects, labels)
        return {"ok": True, "fonts_used": used, "glyphs_added": len(labels),
                **self.stats()}


    def seed_fonts_pipeline(self, fonts: list[str], chars: str = _CHARS,
                            pixel_sizes=(12, 16, 20, 24, 30, 36, 44)) -> dict:
        """Remember what each glyph looks like AFTER THE READER'S OWN PIPELINE:
        rendered at real UI sizes, then blur + Otsu + denoise exactly as
        preprocess.condition does. The older small-size seeding used a fixed
        127 threshold, which renders thinner strokes than Otsu produces on
        anti-aliased screen text — so small 'e'/'o' matched '*' better than
        their own thin exemplars (measured: 'e' window cost 0.288 to '*' vs
        0.310 to 'e'). Seeding through the real pipeline closes the
        stroke-weight gap ('e' cost 0.075 after). Deterministic."""
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        used, squares, aspects, labels = [], [], [], []
        for fpath in fonts:
            for px in pixel_sizes:
                try:
                    font = ImageFont.truetype(fpath, px)
                except Exception:
                    break
                for ch in chars:
                    if ch == " ":
                        continue
                    cw = max(2, int(font.getlength(ch)))
                    im = Image.new("L", (cw + px, px * 2), 0)
                    ImageDraw.Draw(im).text((px // 2, px // 2), ch,
                                            fill=255, font=font)
                    arr = np.array(im)
                    bw = preprocess.denoise(preprocess.binarize(arr))
                    sq, asp = _square((bw > 0).astype(np.float32))
                    if sq is None:
                        continue
                    squares.append(sq)
                    aspects.append(asp)
                    labels.append(ch)
            else:
                used.append(fpath)
        self.add(squares, aspects, labels)
        return {"ok": True, "fonts_used": used, "glyphs_added": len(labels),
                **self.stats()}


# --- line segmentation --------------------------------------------------------
def _local_binarize(gray_band):
    """Contrast-stretch a band's gray pixels (robust percentiles), then local
    Otsu + denoise. Global Otsu picks one threshold for the whole image, which
    starves low-contrast regions (white text on a gray card keeps only the
    brightest stroke cores -> thin broken glyphs the matcher can't read).
    A per-band stretch + threshold recovers the stroke weight. Bright = ink,
    matching the global auto-polarity convention. Returns None if the band is
    essentially flat (nothing to read). Deterministic."""
    import cv2
    import numpy as np
    lo, hi = np.percentile(gray_band, 2), np.percentile(gray_band, 98)
    if hi - lo < 8:
        return None
    st = np.clip((gray_band.astype(np.float32) - lo) / (hi - lo) * 255,
                 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(st, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return preprocess.denoise(th)


def _to_strip(crop, y0: int, y1: int, x0: int, xs: int = 1):
    """Resize a band crop to the H-tall strip frame.
    Returns (y0, y1, x0, x1, strip); x in original image px (xs = crop's
    scale relative to original, for upscaled small-band crops)."""
    import cv2
    import numpy as np
    scale = H / crop.shape[0]
    w = max(1, int(crop.shape[1] * scale))
    strip = cv2.resize(crop, (w, H), interpolation=cv2.INTER_AREA)
    return (int(y0), int(y1), int(x0), int(x0 + crop.shape[1] // xs),
            (strip > 127).astype(np.float32))


def _split_wide(crop, y0: int, y1: int, x0: int, xs: int = 1):
    """Split one band's crop into separate strips at wide blank column-runs:
    distinct text items that share a row (status-bar left/right, nav labels,
    icon captions) become their own lines instead of one merged line."""
    import numpy as np
    bh = crop.shape[0]
    thresh = _SPLIT_FRAC * bh
    out, cur = [], []
    for s, e in _runs(crop.sum(axis=0) > 0):
        if cur and s - cur[-1][1] > thresh:
            gx0, gx1 = cur[0][0], cur[-1][1]
            out.append(_to_strip(crop[:, gx0:gx1], y0, y1,
                                 x0 + gx0 // xs, xs))
            cur = []
        cur.append((s, e))
    if cur:
        gx0, gx1 = cur[0][0], cur[-1][1]
        out.append(_to_strip(crop[:, gx0:gx1], y0, y1, x0 + gx0 // xs, xs))
    return out


def _line_strips(bw, gray=None):
    import cv2
    import numpy as np
    rows = bw.sum(axis=1) > 0
    out = []
    for y0, y1 in _runs(rows, gap=max(1, H // 5)):
        band = bw[y0:y1, :]
        cols = np.where(band.sum(axis=0) > 0)[0]
        if len(cols) == 0:
            continue
        x0, x1 = cols.min(), cols.max() + 1
        xs = 1
        if gray is not None:
            # Re-binarize this band from gray: per-band contrast stretch +
            # local Otsu recovers low-contrast text the global threshold
            # thinned out. Tiny bands are 3x-upscaled first (cubic) so small
            # gray glyphs regain stroke weight. Boxes stay in original px.
            g = gray[y0:y1, x0:x1]
            if (y1 - y0) < _SMALL_H:
                g = cv2.resize(g, (g.shape[1] * _UPSCALE, g.shape[0] * _UPSCALE),
                               interpolation=cv2.INTER_CUBIC)
                xs = _UPSCALE
            loc = _local_binarize(g)
            if loc is None:
                continue
            # Polarity anchor: the global crop (despite being thin) inherits
            # the full-image auto-polarity, so it knows which mode is text.
            # Pick the local polarity (as-is or inverted) that agrees with it.
            # Pure majority-vote fails here: bold text can be >50% ink locally.
            glo = (band[:, x0:x1] > 0)
            if xs != 1:
                glo = cv2.resize(glo.astype(np.uint8),
                                 (loc.shape[1], loc.shape[0]),
                                 interpolation=cv2.INTER_NEAREST) > 0
            loc_b = loc > 0
            agree = (loc_b == glo).mean()
            agree_inv = ((~loc_b) == glo).mean()
            crop = (255 - loc) if agree_inv > agree else loc
        else:
            crop = band[:, x0:x1]
        if (crop > 0).mean() > _INVERT_FRAC:
            # Inverted-polarity island: a dark button / light modal on a page
            # whose global polarity Otsu chose the other way. Flip this band
            # (x-tight: the page bg outside the island must NOT flip) and
            # re-segment its rows. A solid divider inverts to nothing and
            # correctly yields no strip.
            inv = 255 - crop
            for sy0, sy1 in _runs(inv.sum(axis=1) > 0, gap=max(1, H // 5)):
                sub = inv[sy0:sy1, :]
                scols = np.where(sub.sum(axis=0) > 0)[0]
                if len(scols) == 0:
                    continue
                sx0, sx1 = scols.min(), scols.max() + 1
                out.extend(_split_wide(sub[:, sx0:sx1],
                                       y0 + sy0 // xs, y0 + (sy1 + xs - 1) // xs,
                                       x0 + sx0 // xs, xs))
            continue
        out.extend(_split_wide(crop, y0, y1, x0, xs))
    return out


# --- case recovery from the line's typographic frame --------------------------
_XH_PAIRS = set("cosuvwxz")          # upper/lower shapes differ only by SIZE
_BARS = ("l", "I", "|")              # pixel-identical once square-normalized


def _case_recover(glyphs) -> None:
    """Recover case for the size-only pairs (c/C, o/O, ...) from the line's OWN
    typographic frame. Matching was size-blind (square-normalized), but each
    glyph's measured ink height vs the line's cap height tells the case. Needs
    the line to exhibit BOTH height classes; a uniform line (e.g. 'cocoon'
    alone) has no frame to measure against and is left as matched.

    `glyphs` = mutable [label, ink_top, ink_bottom] rows in strip px."""
    import numpy as np
    letters = [g for g in glyphs if g[0] != " "]
    if len(letters) < 2:
        return
    baseline = float(np.median([b for _, _, b in letters]))
    sitting = [b - t for _, t, b in letters if abs(b - baseline) <= H * 0.15]
    if len(sitting) < 2:
        return
    cap_h = max(sitting)
    if cap_h < H * 0.3 or min(sitting) / cap_h > 0.78:   # one height class only
        return
    for g in letters:
        ch, t, b = g
        if ch.lower() in _XH_PAIRS and abs(b - baseline) <= H * 0.15:
            g[0] = ch.upper() if (b - t) / cap_h > 0.78 else ch.lower()


def _fix_bars(txt: str) -> str:
    """l / I / | share one shape; WORD CONTEXT decides deterministically: inside
    a word, a bar flanked by a lowercase letter reads 'l'; flanked by uppercase
    reads 'I'. Word-initial bars stay as matched ('Ice' vs 'lce' is a dictionary
    call — textcorr's job)."""
    out = list(txt)
    for i, ch in enumerate(out):
        if ch not in _BARS:
            continue
        prev = out[i - 1] if i else " "
        nxt = out[i + 1] if i + 1 < len(out) else " "
        if not prev.isalpha():                            # word-initial: leave it
            continue
        if prev.islower() or nxt.islower():
            out[i] = "l"
        elif prev.isupper() and (nxt.isupper() or not nxt.isalpha()):
            out[i] = "I"
    return "".join(out)


# --- the DP segmentation-free line reader (the differentiator) ----------------
def _cut_positions(strip):
    """Candidate glyph boundaries: ink-run edges + interior INK VALLEYS of wide
    runs (where touching glyphs meet). The old DP tried every (position, width)
    window — tens of thousands of gallery matches per line; real glyph
    boundaries only ever sit at these candidates, so the DP shrinks ~50x with
    no loss (and mid-glyph windows that used to false-match wide punctuation
    stop existing). Deterministic."""
    import numpy as np
    W = strip.shape[1]
    col = strip.sum(axis=0)
    empty = col < (H * 0.04)
    pos = {0, W}
    for s, e in _runs(~empty):
        pos.add(int(s)); pos.add(int(e))
        if e - s > int(0.9 * H):                       # wide run: maybe touching
            thr = max(H * 0.15, float(np.percentile(col[s:e], 30)))
            for i in range(s + _WMIN, e - _WMIN):
                if col[i] <= thr and col[i] == col[max(s, i - 2):min(e, i + 3)].min():
                    pos.add(int(i))
    return sorted(pos), empty


def _remove_underline(strip):
    """Zero out solid horizontal rules (link underlines) in the bottom half of
    a strip: a row whose longest ink run spans >70% of the strip width is a
    rule, not a glyph (dashes and crossbars are short; 'T' tops are upper).
    Deterministic."""
    import numpy as np
    b = (strip > 0)
    h, w = b.shape
    out = strip.copy()
    for r in range(h // 2, h):
        row = b[r]
        best = cur = 0
        for v in row:
            if v:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        if best > 0.7 * w:
            out[r, :] = 0
    return out


def _read_strip(strip, gallery) -> tuple[str, float]:
    import numpy as np
    strip = _remove_underline(strip)
    W = strip.shape[1]
    pos, empty = _cut_positions(strip)
    ink_runs = _runs(~empty)
    seps = [(a[1], b[0]) for a, b in zip(ink_runs, ink_runs[1:])]
    n = len(pos)
    INF = 1e18
    dp = [INF] * n
    back = [None] * n
    dp[0] = 0.0
    for j in range(1, n):
        x = pos[j]
        for i in range(j - 1, -1, -1):
            px = pos[i]
            if x - px > _WMAX:                         # positions sorted: done
                break
            if dp[i] >= INF:
                continue
            if bool(empty[px:x].all()):                # blank span: free skip
                if dp[i] < dp[j]:
                    dp[j], back[j] = dp[i], (i, None, 0.0)
                continue
            window = strip[:, px:x]
            if x - px < _WMIN or window.sum() < H * 0.06:
                continue
            res = gallery.match_multi(window)
            if res is None:
                continue
            ranked, _asp = res
            rows = np.where(window.sum(axis=1) > 0)[0]
            label = cost = None
            for lab, c in ranked:                      # best PLAUSIBLE candidate:
                if len(rows) and (                     # quotes live in the top of
                        (lab in _TOP_ONLY and rows.max() > 0.62 * H) or   # the line,
                        (lab in _BOT_ONLY and rows.min() < 0.38 * H)):    # dots at the
                    continue                                              # bottom
                label, cost = lab, c
                break
            if label is None:
                continue
            cand = dp[i] + cost + _PEN
            rate = _GAP_OK_PEN if label in _GAP_OK else _GAP_PEN
            cand += rate * sum(1 for s, e in seps if px < s and e < x)
            if cand < dp[j]:
                dp[j], back[j] = cand, (i, label, cost)
    segs, j = [], n - 1
    while j > 0 and back[j] is not None:
        i, label, cost = back[j]
        segs.append((pos[i], pos[j], label, cost))
        j = i
    segs.reverse()
    glyphs, costs, gap = [], [], 0     # [label, ink_top, ink_bottom] per glyph
    for px, ex, label, cost in segs:
        if label is None:
            gap += (ex - px)
            continue
        if gap > _SPACE_FRAC * H and glyphs:
            glyphs.append([" ", 0, 0])
        gap = 0
        ys = np.where(strip[:, px:ex].sum(axis=1) > 0)[0]
        glyphs.append([label, int(ys.min()), int(ys.max()) + 1] if len(ys)
                      else [label, 0, H])
        costs.append(cost)
    _case_recover(glyphs)
    txt = _fix_bars("".join(g[0] for g in glyphs).strip())
    return txt, (float(np.mean(costs)) if costs else 1.0)


def read(image_or_path, gallery: "GlyphGallery | None" = None,
         do_deskew: bool = True, do_rectify: bool = False, correct: bool = True) -> dict:
    """Read text from an image. Deterministic: the model-free gallery reads
    first, the tiny-brain reader (bundled weights, CPU numpy) arbitrates —
    it overrules only when very sure the gallery is wrong, and covers
    glyphs the gallery can't match. Pass gallery=... to override; use
    default_gallery(hybrid=False) for the bare gallery.
    Returns {text, lines:[{text, box, confidence}], confidence, skew}."""
    g = gallery or default_gallery()
    cond = preprocess.condition(image_or_path, do_deskew=do_deskew, do_rectify=do_rectify)
    if "error" in cond:
        return cond
    lines = []
    for y0, y1, x0, x1, strip in _line_strips(cond["bw"], cond.get("gray")):
        txt, cost = _read_strip(strip, g)
        if correct:
            from . import textcorr
            txt = textcorr.correct(txt)
        lines.append({"text": txt, "box": [x0, y0, x1 - x0, y1 - y0],
                      "confidence": round(max(0.0, 1.0 - cost), 3)})
    text = "\n".join(ln["text"] for ln in lines)
    conf = round(sum(l["confidence"] for l in lines) / len(lines), 3) if lines else 0.0
    return {"ok": True, "text": text, "lines": lines, "confidence": conf,
            "skew": cond["skew"]}


def read_frames(images: list, gallery: "GlyphGallery | None" = None, **kw) -> dict:
    """MULTI-FRAME FUSION — read the same (static) text across several video frames
    and majority-vote each line. A single-image OCR can't do this; an eye that sees
    video can."""
    reads = [r for r in (read(im, gallery=gallery, **kw) for im in images) if r.get("ok")]
    if not reads:
        return {"error": "no readable frames"}
    n_lines = max(len(r["lines"]) for r in reads)
    fused, agree = [], []
    for i in range(n_lines):
        votes = {}
        for r in reads:
            if i < len(r["lines"]):
                t = r["lines"][i]["text"]
                votes[t] = votes.get(t, 0) + 1
        best = max(votes, key=votes.get)
        fused.append(best)
        agree.append(round(votes[best] / len(reads), 2))
    return {"ok": True, "text": "\n".join(fused), "lines": fused,
            "agreement": agree, "frames": len(reads)}


# --- default gallery, seeded from system fonts on first use -------------------
_DEFAULT = None
_DEFAULT_HYBRID = None


def system_fonts() -> list[str]:
    cands = [
        r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\verdana.ttf", r"C:\Windows\Fonts\calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    return [c for c in cands if Path(c).exists()]


def default_gallery(hybrid: bool = True):
    """The BUNDLED gallery (next to this module) unless IRIX_GALLERY_DIR says
    otherwise — a bare relative path would silently give every working
    directory its own gallery.

    hybrid=True (default): gallery first, tiny-brain arbiter behind it
    (Eidetrix product shape). hybrid=False: the bare deterministic gallery.
    """
    global _DEFAULT, _DEFAULT_HYBRID
    if _DEFAULT is None:
        base = os.environ.get("IRIX_GALLERY_DIR") or str(Path(__file__).parent)
        _DEFAULT = GlyphGallery(str(Path(base) / "glyphs.irix"))
        if _DEFAULT.stats()["exemplars"] == 0:
            fonts = system_fonts()
            bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if Path(bold).exists() and bold not in fonts:
                fonts = fonts + [bold]
            _DEFAULT.seed_fonts(fonts)
            _DEFAULT.seed_fonts_small(fonts)
            _DEFAULT.seed_fonts_pipeline(fonts)
    if not hybrid:
        return _DEFAULT
    if _DEFAULT_HYBRID is None:
        from .tinybrain import HybridGallery
        _DEFAULT_HYBRID = HybridGallery(_DEFAULT)
    return _DEFAULT_HYBRID
