"""UI element detection on a synthetic widget, and a tolerant smoke of face.detect
(it must run and return a list on any image, with or without a face present)."""
import numpy as np
from PIL import Image, ImageDraw

from irix import elements, face


def _button_image(path):
    im = Image.new("RGB", (400, 300), (240, 240, 240))
    d = ImageDraw.Draw(im)
    # a flat-filled button with a crisp border -> detectable widget
    d.rectangle([120, 130, 280, 175], fill=(200, 200, 205), outline=(40, 40, 40), width=2)
    im.save(path)


def test_detect_boxes_finds_a_widget(tmp_path):
    p = str(tmp_path / "ui.png")
    _button_image(p)
    boxes = elements.detect_boxes(p)
    assert any(110 < x < 130 and 120 < y < 140 for (x, y, w, h) in boxes), boxes


def test_face_detect_returns_list_without_crashing(tmp_path):
    p = str(tmp_path / "noise.png")
    Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8)).save(p)
    out = face.detect(p)
    assert isinstance(out, list)   # no face -> empty list, never an exception
