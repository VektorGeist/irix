"""inject — the eyes' SECURITY screen.

Read ALL of an image's text — including text hidden by being nearly the same
colour as its background, the classic invisible-injection trick — at multiple
contrast AND scale levels, then flag prompt-injection signatures BEFORE any model
sees the pixels. Deterministic; no model.

Hardened over the original: more signature families, multi-scale OCR (catches tiny
hidden text), per-match severity, a hidden-text ratio, and a redacted-text helper
so a caller can quarantine the offending spans before passing text downstream.
"""
from __future__ import annotations

import re

from . import ocr

# Signature families, each with a severity. CRITICAL = direct attempt to override
# the agent or run a destructive/exfil action; HIGH = role/identity hijack;
# MEDIUM = suspicious framing. Patterns are deliberately broad (recall over
# precision) — this is a screen, not a verdict.
_SIGNATURES: list[tuple[str, str]] = [
    # --- CRITICAL: instruction override / destructive / exfiltration ---
    (r"ignore\s+(the\s+|all\s+|your\s+|any\s+|these\s+)?(previous|prior|above|earlier|"
     r"preceding|all)\s+(instruction|prompt|rule|message|context|direction)s?", "CRITICAL"),
    (r"disregard\s+(the\s+|all\s+|your\s+|any\s+|everything)", "CRITICAL"),
    (r"forget\s+(everything|all|your\s+(instructions|rules|prompt))", "CRITICAL"),
    (r"override\s+(your\s+)?(safety|rules|guard|system|instructions)", "CRITICAL"),
    (r"(delete|rm\s+-rf|format|wipe|drop\s+table|shutdown|exfiltrate|send\s+(it\s+)?to)\b",
     "CRITICAL"),
    (r"(reveal|print|show|output|repeat|leak|dump)\s+(me\s+)?(your\s+|the\s+)?"
     r"(system\s+|initial\s+|hidden\s+)?(prompt|instructions|rules|secret|api[\s_-]?key|"
     r"token|password|credential)s?", "CRITICAL"),
    (r"\b(curl|wget|powershell|bash|/bin/sh|base64\s+-d|invoke-webrequest)\b", "CRITICAL"),
    # --- HIGH: role / identity hijack ---
    (r"you\s+are\s+now\b", "HIGH"),
    (r"\bact\s+as\b", "HIGH"),
    (r"pretend\s+(to\s+be|you('?re)?)", "HIGH"),
    (r"new\s+(instruction|rule|task|directive|persona|role|system\s+prompt)s?", "HIGH"),
    (r"\bjailbreak\b|\bDAN\s+mode\b|developer\s+mode", "HIGH"),
    (r"</?(system|instruction|prompt|im_start|im_end)>", "HIGH"),
    (r"\[/?(INST|SYS|system|assistant)\]", "HIGH"),
    (r"\b(system|assistant|developer)\s*[:>\]]", "HIGH"),
    # --- MEDIUM: suspicious framing ---
    (r"do\s+not\s+(tell|inform|alert|warn|mention|notify|reveal)", "MEDIUM"),
    (r"without\s+(telling|informing|alerting|asking)", "MEDIUM"),
    (r"this\s+is\s+(not\s+)?(a\s+)?(test|drill|hypothetical)", "MEDIUM"),
    (r"\buser\s*[:>\]]", "MEDIUM"),
]

_SEVERITY_RANK = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, None: 0}


def _read_variants(image_path: str) -> str:
    """Pull text out at several contrast AND scale levels, so near-invisible text
    (same colour as background) and tiny text are both surfaced."""
    from PIL import Image, ImageOps
    base = Image.open(image_path).convert("L")
    variants = [
        base,
        ImageOps.autocontrast(base, cutoff=0),
        ImageOps.equalize(base),
        ImageOps.invert(ImageOps.autocontrast(base, cutoff=0)),
    ]
    # add 2x upscale of each — tiny hidden text becomes legible
    scaled = [v.resize((v.width * 2, v.height * 2)) for v in variants]
    chunks = []
    for v in variants + scaled:
        for psm in ("6", "11"):     # 6 = block, 11 = sparse text anywhere
            chunks.append(ocr.run_image(v, psm=psm))
    return "\n".join(chunks)


def _plain_read(image_path: str) -> str:
    """A single naive read — what a human (or a careless OCR) would see."""
    return ocr.run(image_path, psm="6")


def scan(image_path: str) -> dict:
    """Scan an image for prompt-injection. Returns:
      {flagged, severity, matches:[{pattern_severity, text}], hidden_text_ratio,
       hidden_text_found, sample, ocr_available}.
    `severity` is the max over all matches. If tesseract is unavailable, returns a
    clearly-marked unscanned result rather than a false 'clean'."""
    if not ocr.available():
        return {"flagged": False, "severity": None, "matches": [],
                "hidden_text_ratio": 0.0, "hidden_text_found": False,
                "sample": "", "ocr_available": False,
                "note": "tesseract unavailable — image NOT scanned"}

    full = _read_variants(image_path)
    flat = " ".join(full.split())
    matches = []
    seen = set()
    for pat, sev in _SIGNATURES:
        for m in re.finditer(pat, flat, re.IGNORECASE):
            span = m.group(0)[:80]
            key = (sev, span.lower())
            if key in seen:
                continue
            seen.add(key)
            matches.append({"severity": sev, "text": span})

    plain = " ".join(_plain_read(image_path).split())
    ratio = (len(flat) / max(1, len(plain))) if plain else (2.0 if flat else 0.0)
    hidden = ratio > 1.4

    severity = max((m["severity"] for m in matches),
                   key=lambda s: _SEVERITY_RANK[s], default=None)
    return {"flagged": bool(matches), "severity": severity, "matches": matches,
            "hidden_text_ratio": round(ratio, 2), "hidden_text_found": hidden,
            "sample": flat[:400], "ocr_available": True}


def redact(text: str, replacement: str = "[REDACTED-INJECTION]") -> str:
    """Quarantine helper: blank out any injection span in a text blob before it is
    passed downstream. Deterministic — same patterns as scan()."""
    out = text
    for pat, _sev in _SIGNATURES:
        out = re.sub(pat, replacement, out, flags=re.IGNORECASE)
    return out


def safe(image_path: str, block_at: str = "HIGH") -> bool:
    """True if the image is safe to pass to a downstream consumer. Blocks at the
    given severity or above (default: HIGH). Fails CLOSED if OCR is unavailable
    only when block_at is CRITICAL-strict? No — fails OPEN-with-warning is the
    caller's choice via scan(); here we treat unscanned as unsafe to be safe."""
    r = scan(image_path)
    if not r["ocr_available"]:
        return False
    if not r["flagged"]:
        return True
    return _SEVERITY_RANK[r["severity"]] < _SEVERITY_RANK[block_at]
