"""Tests for Irix's model-free OCR: textcorr correction + the glyph-gallery reader.

textcorr cases are exact (deterministic). Case is recovered from the line's own
typographic frame (glyph ink height vs cap height) + word context for the bar
glyphs — exact-case round-trips are asserted on same-font renders.
"""
from __future__ import annotations

import difflib
import os

import numpy as np
import pytest

from irix import textcorr, text


# --- textcorr (deterministic, no image) --------------------------------------
def test_bundled_dictionary_loads():
    assert textcorr.available()
    assert len(textcorr._dictionary()) > 100


@pytest.mark.parametrize("bad,good", [
    ("0pen", "open"),       # 0/O
    ("fi1e", "file"),       # 1/l
    ("Cance1", "Cancel"),   # 1/l, case preserved from context
    ("Sett1ngs", "Settings"),  # 1/i — the l/I/1 family
    ("dovvn", "down"),      # vv/w
])
def test_confusion_repairs(bad, good):
    assert textcorr.correct(bad) == good


def test_l_I_cross_case_repair():
    # l<->1<->I (two confusion edits); case is ambiguous for a caseless source, so
    # the repair is deterministic but lower-cased — assert content, not case.
    assert textcorr.correct("lcon").lower() == "icon"


def test_correction_is_deterministic():
    # the bug this guards: set-ordered candidates made the result hash-seed dependent
    assert len({textcorr.correct("0pen") for _ in range(50)}) == 1


@pytest.mark.parametrize("token", ["Vektorgeist", "xq7z", "Irix", "zzqx"])
def test_novel_tokens_pass_through(token):
    # an unknown token with no in-dictionary confusion-neighbour is never mangled
    assert textcorr.correct(token) == token


def test_correct_is_whitespace_preserving():
    assert textcorr.correct("0pen   fi1e") == "open   file"


# --- glyph gallery -----------------------------------------------------------
def _fonts():
    fonts = text.system_fonts()
    if not fonts:
        pytest.skip("no system TrueType fonts available")
    return fonts


def _gallery(tmp_path, fonts):
    os.environ["IRIX_GALLERY_DIR"] = str(tmp_path)
    g = text.GlyphGallery("test_glyphs.irix")
    g.seed_fonts(fonts[:1])
    return g


def test_gallery_seeds_and_matches(tmp_path):
    fonts = _fonts()
    g = _gallery(tmp_path, fonts)
    assert g.stats()["exemplars"] > 0
    # match a freshly-rendered glyph against its own font's exemplars
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(fonts[0], 64)
    im = Image.new("L", (96, 128), 0)
    ImageDraw.Draw(im).text((20, 20), "A", fill=255, font=font)
    res = g.match((np.array(im) > 127).astype(np.float32))
    assert res is not None
    label, cost, _asp = res
    assert label == "A"
    assert cost < 0.2


# --- end-to-end reader round-trip --------------------------------------------
def _render(msg, font_path, px=40):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_path, px)
    w = int(font.getlength(msg)) + 20
    im = Image.new("L", (w, px + 24), 0)
    ImageDraw.Draw(im).text((10, 10), msg, fill=255, font=font)
    return np.array(im)


@pytest.mark.parametrize("msg", ["Cancel", "Download", "Open File"])
def test_read_roundtrip_case_insensitive(tmp_path, msg):
    fonts = _fonts()
    g = _gallery(tmp_path, fonts)
    out = text.read(_render(msg, fonts[0]), gallery=g, do_deskew=False)
    assert out["ok"]
    got = out["text"].lower().replace(" ", "")
    want = msg.lower().replace(" ", "")
    ratio = difflib.SequenceMatcher(None, got, want).ratio()
    assert ratio >= 0.75, f"{msg!r} read as {out['text']!r} (ratio {ratio:.2f})"


@pytest.mark.parametrize("msg", ["Cancel", "Open File", "close ALL", "CO co"])
def test_read_recovers_exact_case_from_line_frame(tmp_path, msg):
    # the l/L-o/O-c/C fix: measured ink height vs the line's cap height decides
    # case; bars (l/I/|) resolve by word context. Same-font render -> exact.
    fonts = _fonts()
    g = _gallery(tmp_path, fonts)
    # px=48: below that the H=36 strip downsample can over-segment glyphs
    # (a segmentation artifact, separate from the case recovery under test)
    out = text.read(_render(msg, fonts[0], px=48), gallery=g, do_deskew=False)
    assert out["ok"] and out["text"] == msg, out["text"]


def test_uniform_height_line_is_left_as_matched(tmp_path):
    # 'cocoon' alone has ONE height class -> no frame to measure against; the
    # case pass must not invent uppercase out of it.
    fonts = _fonts()
    g = _gallery(tmp_path, fonts)
    out = text.read(_render("cocoon", fonts[0]), gallery=g, do_deskew=False)
    assert out["ok"] and out["text"] == "cocoon", out["text"]


def test_fix_bars_word_context():
    assert text._fix_bars("CanceI") == "Cancel"
    assert text._fix_bars("cIose") == "close"
    assert text._fix_bars("FlLE") == "FILE"
    assert text._fix_bars("Ice") == "Ice"          # word-initial bar untouched


def test_read_returns_structured_lines(tmp_path):
    fonts = _fonts()
    g = _gallery(tmp_path, fonts)
    out = text.read(_render("Save", fonts[0]), gallery=g, do_deskew=False)
    assert set(out) >= {"ok", "text", "lines", "confidence", "skew"}
    assert out["lines"] and "box" in out["lines"][0]
    assert 0.0 <= out["confidence"] <= 1.0


def test_read_on_blank_image_is_safe(tmp_path):
    fonts = _fonts()
    g = _gallery(tmp_path, fonts)
    out = text.read(np.zeros((40, 120), np.uint8), gallery=g, do_deskew=False)
    assert out["ok"]
    assert out["text"] == ""
