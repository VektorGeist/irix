"""uia — Windows UI Automation tree reader, pure ctypes, ZERO dependencies.

Where an app exposes its accessibility tree, the OS already knows every widget's
exact rectangle, name, and role — for free, pixel-exact, no OCR, no model. Irix
reads it directly through the UIAutomationCore COM API via ctypes (no comtypes,
no pywin32), and `fuse.py` merges it with the pixel-CV `elements` so we get the
best of both: a11y where apps expose it, pixels where they don't (games, canvas
apps, custom-drawn UIs, and the whole VektOS target).

Deliberately minimal + defensive: the ONLY element method used is
`GetCurrentPropertyValue` (vtable slot 10, right above the Find methods), so
there is no hand-counting of the ~80 property-getter slots that could crash the
process on an off-by-one. Every call is guarded; any failure (non-Windows, COM
denied, headless) returns [] — this never raises into the host.
"""
from __future__ import annotations

import sys

# --- property + control-type ids (stable UIA constants) ----------------------
_PROP_BOUNDING_RECT = 30001
_PROP_CONTROL_TYPE = 30003
_PROP_NAME = 30005
_PROP_IS_OFFSCREEN = 30022
_PROP_IS_ENABLED = 30010

_TREESCOPE_SUBTREE = 7
_VT_BSTR = 8
_VT_ARRAY = 0x2000
_VT_R8 = 5

# control types worth clicking (name -> id); others are still returned, flagged
_CTL = {50000: "button", 50002: "checkbox", 50003: "combobox", 50004: "edit",
        50005: "hyperlink", 50007: "listitem", 50011: "menuitem",
        50013: "radiobutton", 50019: "tabitem", 50024: "splitbutton",
        50025: "text", 50029: "hyperlink"}
_INTERACTIVE = {50000, 50002, 50003, 50004, 50005, 50007, 50011, 50013, 50019, 50024}


def available() -> bool:
    """True only on Windows with UIAutomationCore reachable."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        ctypes.oledll.ole32  # noqa: B018
        ctypes.windll.UIAutomationCore  # noqa: B018 - present on all modern Windows
        return True
    except Exception:
        return False


def _guid(s: str):
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]
    g = GUID()
    ctypes.oledll.ole32.CLSIDFromString(ctypes.c_wchar_p(s), ctypes.byref(g))
    return g


def _vt_call(this, index, restype, *argtypes):
    """Bind vtable method `index` on COM interface pointer `this`."""
    import ctypes
    vtbl = ctypes.cast(this, ctypes.POINTER(ctypes.c_void_p))[0]
    fn = ctypes.cast(vtbl, ctypes.POINTER(ctypes.c_void_p))[index]
    proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return proto(fn)


def _prop(elem, prop_id):
    """One element property via GetCurrentPropertyValue (vtable 10). Returns a
    python value decoded from the VARIANT (str, float-tuple, int, bool) or None."""
    import ctypes
    from ctypes import wintypes

    class VARIANT(ctypes.Structure):                # 24 bytes on x64
        _fields_ = [("vt", wintypes.WORD), ("r1", wintypes.WORD),
                    ("r2", wintypes.WORD), ("r3", wintypes.WORD),
                    ("val", ctypes.c_void_p), ("val2", ctypes.c_void_p)]
    var = VARIANT()
    hr = _vt_call(elem, 10, ctypes.c_long, ctypes.c_int,
                  ctypes.POINTER(VARIANT))(elem, prop_id, ctypes.byref(var))
    if hr != 0:
        return None
    vt = var.vt
    base = ctypes.addressof(var) + 8            # union starts past vt + 3 reserved words
    out = None
    try:
        if vt == _VT_BSTR:
            p = ctypes.cast(base, ctypes.POINTER(ctypes.c_void_p))[0]
            if p:
                out = ctypes.wstring_at(p)
        elif vt == (_VT_ARRAY | _VT_R8):                  # SAFEARRAY of 4 doubles
            psa = ctypes.cast(base, ctypes.POINTER(ctypes.c_void_p))[0]
            pv = ctypes.c_void_p()
            if psa and ctypes.oledll.oleaut32.SafeArrayAccessData(
                    ctypes.c_void_p(psa), ctypes.byref(pv)) == 0:
                arr = ctypes.cast(pv, ctypes.POINTER(ctypes.c_double * 4))[0]
                out = (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
                ctypes.oledll.oleaut32.SafeArrayUnaccessData(ctypes.c_void_p(psa))
        elif vt in (2, 3):                                 # I2 / I4
            out = int(ctypes.cast(base, ctypes.POINTER(ctypes.c_int))[0])
        elif vt == 11:                                     # VARIANT_BOOL (0/-1 short)
            out = ctypes.cast(base, ctypes.POINTER(ctypes.c_short))[0] != 0
    finally:
        ctypes.oledll.oleaut32.VariantClear(ctypes.byref(var))
    return out


def elements(hwnd: int | None = None, max_n: int = 400,
             interactive_only: bool = False) -> list[dict]:
    """The UIA element map of a window (default: the foreground window), as
    [{label, x, y, w, h, cx, cy, control, interactive, source:'uia'}]. Exact
    rectangles + names from the OS. Fails open to [] on any error."""
    if not available():
        return []
    import ctypes
    ole = ctypes.oledll.ole32
    ole.CoInitializeEx(None, 0x2)                          # apartment-threaded
    try:
        clsid = _guid("{FF48DBA4-60EF-4201-AA87-54103EEF594E}")   # CUIAutomation
        iid = _guid("{30CBE57D-D9D0-452A-AB13-7AC5AC4825EE}")     # IUIAutomation
        uia = ctypes.c_void_p()
        if ole.CoCreateInstance(ctypes.byref(clsid), None, 1,
                                ctypes.byref(iid), ctypes.byref(uia)) != 0:
            return []
        try:
            if hwnd is None:
                hwnd = ctypes.windll.user32.GetForegroundWindow()
            root = ctypes.c_void_p()
            # ElementFromHandle = IUIAutomation vtable slot 6
            if _vt_call(uia, 6, ctypes.c_long, ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_void_p))(
                    uia, ctypes.c_void_p(hwnd), ctypes.byref(root)) != 0 or not root:
                return []
            cond = ctypes.c_void_p()
            # CreateTrueCondition = slot 21
            if _vt_call(uia, 21, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p))(
                    uia, ctypes.byref(cond)) != 0:
                return []
            found = ctypes.c_void_p()
            # IUIAutomationElement::FindAll = slot 6
            if _vt_call(root, 6, ctypes.c_long, ctypes.c_int, ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_void_p))(
                    root, _TREESCOPE_SUBTREE, cond, ctypes.byref(found)) != 0 \
                    or not found:
                return []
            length = ctypes.c_int(0)
            # IUIAutomationElementArray::get_Length = slot 3
            _vt_call(found, 3, ctypes.c_long, ctypes.POINTER(ctypes.c_int))(
                found, ctypes.byref(length))
            out = []
            for i in range(min(length.value, max_n)):
                el = ctypes.c_void_p()
                # get_Element = slot 4
                if _vt_call(found, 4, ctypes.c_long, ctypes.c_int,
                            ctypes.POINTER(ctypes.c_void_p))(
                        found, i, ctypes.byref(el)) != 0 or not el:
                    continue
                try:
                    if _prop(el, _PROP_IS_OFFSCREEN) is True:
                        continue
                    rect = _prop(el, _PROP_BOUNDING_RECT)
                    if not rect or rect[2] <= 0 or rect[3] <= 0:
                        continue
                    ctype = _prop(el, _PROP_CONTROL_TYPE) or 0
                    inter = ctype in _INTERACTIVE
                    if interactive_only and not inter:
                        continue
                    x, y, w, h = (int(round(v)) for v in rect)
                    out.append({"label": (_prop(el, _PROP_NAME) or "").strip(),
                                "x": x, "y": y, "w": w, "h": h,
                                "cx": x + w // 2, "cy": y + h // 2,
                                "control": _CTL.get(ctype, str(ctype)),
                                "interactive": inter, "source": "uia"})
                finally:
                    _vt_call(el, 2, ctypes.c_long)(el)     # Release
            _vt_call(found, 2, ctypes.c_long)(found)
            return out
        finally:
            _vt_call(uia, 2, ctypes.c_long)(uia)           # Release IUIAutomation
    except Exception:
        return []
    finally:
        ole.CoUninitialize()
