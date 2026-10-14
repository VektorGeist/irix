"""Hybrid element map: UIA + pixel-CV fusion logic (synthetic, cross-platform)
plus a live UIA smoke that self-skips where UIA is unavailable. No model."""
import sys

import pytest

from irix import fuse, uia


def _el(x, y, w, h, label="", source="uia", interactive=True):
    return {"label": label, "x": x, "y": y, "w": w, "h": h,
            "cx": x + w // 2, "cy": y + h // 2, "source": source,
            "interactive": interactive, "control": "button"}


# --- fusion logic (deterministic, no OS) -------------------------------------

def test_uia_elements_all_kept():
    uia_els = [_el(0, 0, 50, 20, "OK"), _el(60, 0, 50, 20, "Cancel")]
    out = fuse.fuse(uia_els, [])
    assert len(out) == 2 and all(e["source"] == "uia" for e in out)


def test_cv_fills_gaps_uia_missed():
    uia_els = [_el(0, 0, 50, 20, "OK")]
    cv_els = [_el(200, 200, 60, 24, "Play", source="cv")]     # far from any UIA box
    out = fuse.fuse(uia_els, cv_els)
    labels = {(e["label"], e["source"]) for e in out}
    assert ("OK", "uia") in labels and ("Play", "cv") in labels


def test_cv_overlapping_uia_is_dropped_not_duplicated():
    uia_els = [_el(0, 0, 100, 40, "Submit")]
    cv_els = [_el(3, 3, 96, 36, "Submit", source="cv")]        # same widget
    out = fuse.fuse(uia_els, cv_els)
    assert len(out) == 1 and out[0]["source"] == "uia"


def test_cv_label_repairs_nameless_uia_node():
    uia_els = [_el(0, 0, 100, 40, "")]                          # UIA node, no name
    cv_els = [_el(10, 8, 60, 20, "Login", source="cv")]        # OCR read inside it
    out = fuse.fuse(uia_els, cv_els)
    assert len(out) == 1
    assert out[0]["label"] == "Login" and out[0]["labeled_by"] == "cv"


def test_find_prefers_uia_on_a_tie():
    uia_els = [_el(0, 0, 80, 30, "Export")]
    cv_els = [_el(300, 300, 80, 30, "Export", source="cv")]
    # both match 'Export' exactly; UIA's +0.5 trust bonus wins
    import irix.fuse as f
    orig = f.parse
    f.parse = lambda *a, **k: fuse.fuse(uia_els, cv_els)
    try:
        r = fuse.find("Export")
    finally:
        f.parse = orig
    assert r["ok"] and r["source"] == "uia"


# --- live UIA smoke (Windows only) -------------------------------------------

@pytest.mark.skipif(not uia.available(), reason="UIA not available on this OS")
def test_uia_reads_the_taskbar():
    import ctypes
    hwnd = ctypes.windll.user32.FindWindowW("Shell_TrayWnd", None)
    if not hwnd:
        pytest.skip("no taskbar window")
    els = uia.elements(hwnd=hwnd)
    assert len(els) > 0
    assert any(e["interactive"] and e["label"] for e in els)
    for e in els:                                              # exact geometry
        assert e["w"] > 0 and e["h"] > 0 and e["source"] == "uia"


def test_uia_available_is_false_off_windows():
    if not sys.platform.startswith("win"):
        assert uia.available() is False
        assert uia.elements() == []                            # fails open
