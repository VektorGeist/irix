"""textcorr — deterministic context correction. The model-free stand-in for the
language understanding an LSTM OCR absorbs implicitly.

Two stored-memory signals, no model:
  - a DICTIONARY (word list) — an unknown token is repaired to the nearest known
    word, but ONLY across the classic OCR confusion pairs (0/O, 1/l/I, rn/m, …),
    so we never "autocorrect" a genuinely-novel string into a wrong word.
  - the memory is just data: point IRIX_WORDLIST at a bigger list and correction
    gets better — same 'grows with use' principle as the glyph gallery.

If no dictionary is available, correction is a safe no-op (returns input unchanged).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

# Bidirectional OCR confusions — only these substitutions are tried when repairing.
_CONFUSE = [
    ("0", "O"), ("0", "o"), ("1", "l"), ("1", "I"), ("1", "i"),
    ("5", "S"), ("8", "B"), ("6", "G"), ("2", "Z"), ("rn", "m"),
    ("vv", "w"), ("cl", "d"), ("nn", "m"), ("|", "l"), ("ii", "u"),
]


@lru_cache(maxsize=1)
def _dictionary() -> frozenset:
    """Load a word list. IRIX_WORDLIST wins; else common OS locations; else empty."""
    cands = [os.environ.get("IRIX_WORDLIST"),
             "/usr/share/dict/words", "/usr/share/dict/american-english",
             str(Path(__file__).with_name("words.txt"))]
    for c in cands:
        if c and Path(c).exists():
            try:
                with open(c, encoding="utf-8", errors="ignore") as f:
                    return frozenset(w.strip().lower() for w in f if w.strip().isalpha())
            except Exception:
                continue
    return frozenset()


def _candidates(token: str):
    """Yield strings reachable from token by confusion substitutions (both
    directions), bounded to 2 edits. Yielded in EDIT-DISTANCE order, sorted within
    each distance — so the first dictionary hit is the minimal, deterministic
    repair (no hash-order dependence)."""
    seen = {token}
    frontier = [token]
    for _ in range(2):                       # up to 2 confusion edits
        nxt = []
        for s in frontier:
            for a, b in _CONFUSE:
                for x, y in ((a, b), (b, a)):
                    i = s.find(x)
                    while i != -1:
                        cand = s[:i] + y + s[i + len(x):]
                        if cand not in seen:
                            seen.add(cand)
                            nxt.append(cand)
                        i = s.find(x, i + 1)
        for cand in sorted(nxt):             # deterministic within this distance
            yield cand
        frontier = nxt
        if not frontier:
            break


def _recase(cand: str, token: str) -> str:
    """Re-apply the token's case to a same-length candidate. A caseless source char
    (a digit/symbol that was misread for a letter) follows the token's overall case:
    upper if the token's letters are all caps, else lower."""
    if len(cand) != len(token):
        return cand
    letters = [c for c in token if c.isalpha()]
    upper_ctx = bool(letters) and all(c.isupper() for c in letters)
    out = []
    for c, o in zip(cand, token):
        if o.isupper():
            out.append(c.upper())
        elif o.islower():
            out.append(c.lower())
        else:
            out.append(c.upper() if upper_ctx else c.lower())
    return "".join(out)


def correct_word(token: str) -> str:
    d = _dictionary()
    if not d or not token.isalnum():
        return token
    if token.lower() in d:
        return token
    for cand in _candidates(token):
        if cand.lower() in d:
            return _recase(cand, token)
    return token


def correct(text: str) -> str:
    """Repair OCR-confusion errors token-by-token against the dictionary. Safe
    no-op when no dictionary is loaded."""
    if not _dictionary():
        return text
    return re.sub(r"[A-Za-z0-9]+", lambda m: correct_word(m.group(0)), text)


def available() -> bool:
    return bool(_dictionary())
