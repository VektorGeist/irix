"""irix.memory — episodic visual memory. Deterministic, no models.

An agent remembers what it SAW. Screenshots are stored with timestamp, source,
caption, and an Irix deterministic element parse, retrievable by text query.
Recall returns the image path so the agent looks with its own eyes instead of
confabulating from a text summary.

Storage: SQLite + FTS5 (stdlib). Dedup: perceptual dHash (Pillow, already a dep).
No network, no models, no API keys.

Usage:
    from irix.memory import MemoryStore
    mem = MemoryStore("~/.irix/memory")
    eid = mem.remember("shot.png", source="agent", caption="login screen")
    for ep in mem.recall("login prompt"):
        print(ep["id"], ep["image_path"])
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes(
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  source TEXT NOT NULL,
  image_path TEXT NOT NULL,
  caption TEXT NOT NULL,
  ocr_text TEXT NOT NULL,
  ocr_norm TEXT NOT NULL DEFAULT '',
  elements_json TEXT NOT NULL,
  phash TEXT NOT NULL,
  bytes INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
  USING fts5(caption, ocr_text, ocr_norm, content='episodes', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
  INSERT INTO episodes_fts(rowid, caption, ocr_text, ocr_norm)
  VALUES (new.id, new.caption, new.ocr_text, new.ocr_norm);
END;
CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
  INSERT INTO episodes_fts(episodes_fts, rowid, caption, ocr_text, ocr_norm)
  VALUES ('delete', old.id, old.caption, old.ocr_text, old.ocr_norm);
END;
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts_vocab
  USING fts5vocab(episodes_fts, 'row');
"""

# Hamming distance <= this => near-duplicate frame, store once.
# Calibrated 2026-09-15: 9x8 dHash could not distinguish same-layout screens
# (different screens ham<=1, near-dups ham=8 — inverted). 17x16 separates
# cleanly: realistic near-dups ham=0, different screens ham>=6.
DEDUP_THRESHOLD = 4
DEDUP_HASH_SIZE = 17


# --- OCR-error-tolerant recall ------------------------------------------------
# Irix's exemplar OCR misreads characters the way all template matchers do:
# 'invoice'->'lnvoice', 'Error'->'Ermr', '0'<->'O', '1'<->'l'. Folding confusable
# characters to one canonical form at index AND query time makes those
# misreadings match. Deterministic, no models, no dictionary.

_CONFUSE_MULTI = (("vv", "w"), ("rn", "m"), ("cl", "d"))
_CONFUSE_SINGLE = {
    "0": "o", "1": "l", "i": "l", "|": "l", "!": "l",
    "5": "s", "8": "b", "2": "z", "6": "g", "9": "g",
    "*": "",  # the engine emits '*' for a glyph it could not match: pure noise
    "-": "",  # hyphens usually mark a mis-split word at small sizes ('tim-ut')
}


def _fold_confusables(text: str) -> str:
    t = text.lower()
    for a, b in _CONFUSE_MULTI:
        t = t.replace(a, b)
    return "".join(_CONFUSE_SINGLE.get(ch, ch) for ch in t)


def _lev(a: str, b: str, cap: int = 2) -> int:
    """Levenshtein distance with early exit past cap. Deterministic."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        rowmin = i
        for j, cb in enumerate(b, 1):
            c = prev[j - 1] if ca == cb else 1 + min(prev[j - 1], prev[j],
                                                     cur[j - 1])
            cur.append(c)
            if c < rowmin:
                rowmin = c
        if rowmin > cap:
            return cap + 1
        prev = cur
    return prev[-1]


# --- paraphrase-tolerant recall ------------------------------------------------
# Keyword recall misses when the agent paraphrases: the screen says "login",
# the agent asks for "sign in". A small hand-built synonym map (spaceless keys,
# UI/agent domain) expands query terms deterministically. No embeddings, no
# models — a word list, checked in.

_SYNONYMS: dict[str, tuple[str, ...]] = {
    "signin": ("login", "log in", "authenticate", "authentication"),
    "login": ("signin", "sign in", "authenticate", "authentication"),
    "signout": ("logout", "log out"),
    "logout": ("signout", "sign out"),
    "password": ("passwd", "passcode", "credential", "credentials"),
    "username": ("username", "userid", "user id", "account name"),
    "settings": ("preferences", "config", "configuration", "options", "setup"),
    "preferences": ("settings", "config", "options"),
    "error": ("failure", "bug", "crash", "exception", "traceback", "oops"),
    "warning": ("alert", "caution", "warn"),
    "success": ("done", "complete", "completed", "ok", "finished"),
    "loading": ("progress", "wait", "waiting", "spinner", "busy"),
    "photo": ("picture", "image", "pic", "snapshot"),
    "picture": ("photo", "image", "pic"),
    "video": ("movie", "clip", "film"),
    "music": ("song", "songs", "audio", "track", "tracks"),
    "search": ("find", "lookup", "look up", "query", "seek"),
    "delete": ("remove", "trash", "erase", "del"),
    "remove": ("delete", "trash", "erase"),
    "save": ("store", "keep"),
    "file": ("document", "doc"),
    "folder": ("directory", "dir"),
    "email": ("mail", "inbox", "message", "letter"),
    "mail": ("email", "inbox", "message"),
    "chat": ("conversation", "thread", "im", "messages", "messenger"),
    "calendar": ("agenda", "schedule", "events", "planner"),
    "map": ("navigation", "directions", "gps", "location"),
    "weather": ("forecast", "temperature", "conditions"),
    "news": ("article", "articles", "headline", "headlines", "press"),
    "contacts": ("addressbook", "address book", "people", "phonebook"),
    "task": ("todo", "to-do", "checklist", "reminder"),
    "todo": ("task", "checklist", "to-do"),
    "note": ("memo", "notepad", "notes"),
    "notes": ("note", "memo", "notepad"),
    "game": ("play", "score", "scores", "gaming"),
    "browser": ("web", "internet", "surf"),
    "download": ("dl", "get", "fetch"),
    "upload": ("share", "send", "publish", "post"),
    "payment": ("checkout", "billing", "purchase", "buy", "cart", "order", "pay"),
    "checkout": ("payment", "cart", "buy", "purchase", "order"),
    "code": ("source", "script", "program", "editor"),
    "terminal": ("shell", "console", "commandline", "command line", "prompt"),
    "shell": ("terminal", "console", "command line"),
    "dashboard": ("overview", "stats", "statistics", "charts", "metrics", "home"),
    "profile": ("account", "user", "me"),
    "account": ("profile", "user"),
    "help": ("support", "faq", "about", "assist", "guide"),
    "info": ("information", "details", "about"),
    "update": ("upgrade", "refresh"),
    "upgrade": ("update",),
    "refresh": ("reload", "update", "sync"),
    "reload": ("refresh",),
    "close": ("exit", "quit", "dismiss"),
    "exit": ("close", "quit"),
    "open": ("launch", "start", "run"),
    "launch": ("open", "start", "run"),
    "edit": ("modify", "change", "alter"),
    "copy": ("duplicate", "clone"),
    "duplicate": ("copy", "clone"),
    "print": ("pdf", "export"),
    "zoom": ("magnify", "enlarge"),
    "wifi": ("network", "internet", "connection", "online"),
    "network": ("wifi", "internet", "connection"),
    "offline": ("disconnected", "nointernet", "no internet"),
    "battery": ("power", "charge", "charging"),
    "camera": ("webcam", "photo"),
    "gallery": ("photos", "pictures", "images", "albums"),
    "clock": ("time", "alarm", "timer", "stopwatch"),
    "calculator": ("calc", "math"),
    "spreadsheet": ("sheet", "excel", "table", "budget"),
    "document": ("doc", "file", "paper", "pdf"),
    "pdf": ("document", "doc", "paper"),
    "slides": ("presentation", "deck", "powerpoint"),
    "presentation": ("slides", "deck"),
    "drawing": ("canvas", "sketch", "paint", "art"),
    "canvas": ("drawing", "sketch"),
    "social": ("feed", "timeline", "posts"),
    "feed": ("timeline", "social", "posts", "stream"),
}


def _quote_fts(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def _query_variants(query: str) -> set[str]:
    """All FTS match variants for a query: raw terms, synonyms, the joined
    phrase, and the confusable-fold of each. Deterministic."""
    terms = [t for t in query.split() if t]
    variants: set[str] = set()
    for t in terms:
        variants.add(t)
        variants.update(_SYNONYMS.get(t.lower(), ()))
    joined = "".join(query.lower().split())
    if joined:
        variants.update(_SYNONYMS.get(joined, ()))
        variants.add(joined)
    phrase = " ".join(terms)
    if phrase:
        variants.add(phrase)
    folded = {_fold_confusables(v) for v in list(variants)}
    variants.update(folded)
    return {v for v in variants if v}


def _migrate_fts(con: sqlite3.Connection) -> None:
    """Add the ocr_norm column to pre-0.3.1 databases and rebuild the FTS
    index from the episodes table. New databases already have it."""
    cols = [r[1] for r in con.execute("PRAGMA table_info(episodes_fts)")]
    if "ocr_norm" in cols:
        return
    con.executescript("""
        DROP TRIGGER IF EXISTS episodes_ai;
        DROP TRIGGER IF EXISTS episodes_ad;
        DROP TABLE IF EXISTS episodes_fts;
    """)
    ecol = [r[1] for r in con.execute("PRAGMA table_info(episodes)")]
    if "ocr_norm" not in ecol:
        con.execute("ALTER TABLE episodes ADD COLUMN ocr_norm TEXT NOT NULL DEFAULT ''")
    con.executescript("""
        CREATE VIRTUAL TABLE episodes_fts
          USING fts5(caption, ocr_text, ocr_norm, content='episodes', content_rowid='id');
        CREATE TRIGGER episodes_ai AFTER INSERT ON episodes BEGIN
          INSERT INTO episodes_fts(rowid, caption, ocr_text, ocr_norm)
          VALUES (new.id, new.caption, new.ocr_text, new.ocr_norm);
        END;
        CREATE TRIGGER episodes_ad AFTER DELETE ON episodes BEGIN
          INSERT INTO episodes_fts(episodes_fts, rowid, caption, ocr_text, ocr_norm)
          VALUES ('delete', old.id, old.caption, old.ocr_text, old.ocr_norm);
        END;
    """)
    for r in con.execute("SELECT id, ocr_text FROM episodes"):
        con.execute("UPDATE episodes SET ocr_norm=? WHERE id=?",
                    (_fold_confusables(r["ocr_text"] or ""), r["id"]))
    con.execute(
        "INSERT INTO episodes_fts(rowid, caption, ocr_text, ocr_norm)"
        " SELECT id, caption, ocr_text, ocr_norm FROM episodes")
    con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts_vocab"
                " USING fts5vocab(episodes_fts, 'row')")
    con.commit()


def _gentle_token(tok: str) -> str:
    """Lowercase alphanumeric skeleton. Unlike _norm_token, no confusable
    folding: the fold maps i->l etc., which distorts ordinary words and
    breaks edit-distance comparison ('timeout' vs 'timut')."""
    return "".join(ch for ch in tok.lower() if ch.isalnum())


def _fuzzy_extra_terms(con: sqlite3.Connection, query: str) -> list[str]:
    """Vocabulary terms within edit distance of a query term's gentle form.
    Last-resort fallback for OCR-corrupted tokens ('timut' vs 'timeout').
    Runs only when exact/variant matching under-delivers."""
    qterms = {_gentle_token(t) for t in query.split() if len(t) >= 4}
    qterms = {t for t in qterms if len(t) >= 4}
    if not qterms:
        return []
    found: set[str] = set()
    try:
        rows = con.execute("SELECT term FROM episodes_fts_vocab").fetchall()
    except sqlite3.Error:
        return []
    for (term,) in rows:
        nt = _gentle_token(term)
        if len(nt) < 4 or nt in qterms:
            continue
        for q in qterms:
            cap = 2 if max(len(nt), len(q)) < 7 else 3
            if abs(len(nt) - len(q)) <= cap and _lev(nt, q, cap) <= cap:
                found.add(term)
                break
    return sorted(found)


def dhash(path: str | Path, size: int = DEDUP_HASH_SIZE) -> str:
    """Difference hash of an image -> hex string. Pure Pillow."""
    from PIL import Image

    img = Image.open(path).convert("L").resize((size, size - 1), Image.BILINEAR)
    px = list(img.getdata())
    bits = 0
    for r in range(size - 1):
        for c in range(size - 1):
            bits = (bits << 1) | (1 if px[r * size + c] > px[r * size + c + 1] else 0)
    width = (size * (size - 1) + 3) // 4
    return f"{bits:0{width}x}"


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _parse_image(path: Path):
    """Irix deterministic parse -> (searchable_text, elements). Soft-fails.

    Combines element labels AND full-screen OCR text so an episode is
    retrievable by any visible text, not just UI element labels.
    """
    els = []
    try:
        from . import elements
        els = elements.parse(str(path)) or []
    except Exception:
        els = []
    labels: list[str] = []
    for el in els:
        lab = (el.get("label") or "").strip()
        if lab and lab not in labels:
            labels.append(lab)
    ocr_text = ""
    try:
        from . import ocr as _ocr
        if _ocr.available():
            ocr_text = (_ocr.run(str(path)) or "").strip()
    except Exception:
        ocr_text = ""
    # element labels first (higher signal), then raw OCR text
    combined = " ".join(labels)
    if ocr_text:
        combined = (combined + " " + " ".join(ocr_text.split())).strip()
    return combined, els


def _auto_caption(ocr_text: str, n_labels: int = 8) -> str:
    words = ocr_text.split()
    if not words:
        return "screenshot (no readable text)"
    return " ".join(words[:60])


def _same_content(a: str, b: str) -> bool:
    """True if two OCR texts describe the same screen content.

    OCR is deterministic, so true duplicates give identical text. If either
    side has no readable text, fall back to trusting the pixel signal.
    """
    na, nb = " ".join(a.lower().split()), " ".join(b.lower().split())
    if not na or not nb:
        return True
    return na == nb


class MemoryStore:
    """A persistent episodic visual memory rooted at a directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self.episodes_dir = self.root / "episodes"
        self.db_path = self.root / "memory.db"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        con = self._db()
        con.executescript(SCHEMA)
        con.commit()
        _migrate_fts(con)
        con.close()

    def _db(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def remember(self, image: str | Path, source: str,
                 caption: str | None = None, ts: str | None = None) -> dict:
        """Store a screenshot episode. Returns {id, duplicate_of}.

        Dedup is two-signal: pixel-similar (dHash) AND same OCR text content.
        Same layout with different text is a different episode, not a dup.
        """
        src = Path(image)
        if not src.is_file():
            raise FileNotFoundError(f"no such file: {src}")
        ph = dhash(src)
        # parse first: content comparison needs the OCR text
        ocr_text, els = _parse_image(src)
        con = self._db()
        dup = con.execute(
            "SELECT id, phash, ocr_text FROM episodes"
        ).fetchall()
        for row in dup:
            if (_hamming(ph, row["phash"]) <= DEDUP_THRESHOLD
                    and _same_content(ocr_text, row["ocr_text"] or "")):
                con.close()
                return {"id": row["id"], "duplicate_of": row["id"]}

        digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
        dest = self.episodes_dir / f"{digest}{src.suffix.lower()}"
        if not dest.exists():
            shutil.copy2(src, dest)
        cap = caption or _auto_caption(ocr_text)
        norm = _fold_confusables(ocr_text)
        stamp = ts or datetime.now(timezone.utc).isoformat(timespec="seconds")
        cur = con.execute(
            "INSERT INTO episodes(ts, source, image_path, caption, ocr_text,"
            " ocr_norm, elements_json, phash, bytes) VALUES (?,?,?,?,?,?,?,?,?)",
            (stamp, source, str(dest), cap, ocr_text, norm,
             json.dumps(els), ph, dest.stat().st_size),
        )
        eid = cur.lastrowid
        con.commit()
        con.close()
        return {"id": eid, "duplicate_of": None,
                "ocr_chars": len(ocr_text), "elements": len(els)}

    def recall(self, query: str, n: int = 5) -> list[dict]:
        # OR semantics between terms: a query is a set of hints, not a filter.
        # AND (FTS5's default) returns nothing when any single hint word misses,
        # which is the common case with OCR misreadings and paraphrase.
        # Variants add synonym expansion (paraphrase) and confusable folding
        # (OCR character errors) on top of the raw terms.
        variants = _query_variants(query)
        if not variants:
            return []
        match = " OR ".join(_quote_fts(v) for v in sorted(variants))
        con = self._db()
        rows = con.execute(
            """SELECT e.id, e.ts, e.source, e.image_path, e.caption,
                      snippet(episodes_fts, 1, '[', ']', '...', 24) AS snip
               FROM episodes_fts f JOIN episodes e ON e.id = f.rowid
               WHERE episodes_fts MATCH ? ORDER BY rank LIMIT ?""",
            (match, n),
        ).fetchall()
        out = [dict(r) for r in rows]
        # Fuzzy fallback: if OCR corruption left too few hits, pull in vocab
        # terms within edit distance 2 of the query terms.
        if len(out) < n:
            have = {r["id"] for r in out}
            extra = _fuzzy_extra_terms(con, query)
            if extra:
                match2 = " OR ".join(_quote_fts(t) for t in extra)
                rows2 = con.execute(
                    """SELECT e.id, e.ts, e.source, e.image_path, e.caption,
                              snippet(episodes_fts, 1, '[', ']', '...', 24) AS snip
                       FROM episodes_fts f JOIN episodes e ON e.id = f.rowid
                       WHERE episodes_fts MATCH ? ORDER BY rank LIMIT ?""",
                    (match2, n),
                ).fetchall()
                for r in rows2:
                    if r["id"] not in have:
                        out.append(dict(r))
                        have.add(r["id"])
                    if len(out) >= n:
                        break
        con.close()
        return out

    def list(self, n: int = 10) -> list[dict]:
        con = self._db()
        rows = con.execute(
            "SELECT id, ts, source, caption FROM episodes"
            " ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        con.close()
        return [dict(r) for r in rows]

    def forget(self, eid: int) -> bool:
        con = self._db()
        cur = con.execute("DELETE FROM episodes WHERE id = ?", (eid,))
        con.commit()
        con.close()
        return cur.rowcount > 0

    def prune(self, max_episodes: int | None = None,
              max_age_days: int | None = None,
              max_bytes: int | None = None) -> dict:
        """Evict oldest-first until caps hold. Returns counts removed."""
        removed = 0
        freed = 0
        con = self._db()
        if max_age_days is not None:
            cutoff = time.time() - max_age_days * 86400
            rows = con.execute(
                "SELECT id, ts, image_path, bytes FROM episodes").fetchall()
            for r in rows:
                try:
                    ets = datetime.fromisoformat(r["ts"]).timestamp()
                except Exception:
                    continue
                if ets < cutoff:
                    if self._delete(con, r["id"], r["image_path"]):
                        removed += 1
                        freed += r["bytes"] or 0
        if max_episodes is not None:
            ids = [r["id"] for r in con.execute(
                "SELECT id FROM episodes ORDER BY id DESC").fetchall()]
            for eid in ids[max_episodes:]:
                row = con.execute(
                    "SELECT image_path, bytes FROM episodes WHERE id=?",
                    (eid,)).fetchone()
                if row and self._delete(con, eid, row["image_path"]):
                    removed += 1
                    freed += row["bytes"] or 0
        if max_bytes is not None:
            total = con.execute("SELECT COALESCE(SUM(bytes),0) s FROM episodes"
                                ).fetchone()["s"]
            ids = [r["id"] for r in con.execute(
                "SELECT id FROM episodes ORDER BY id").fetchall()]
            for eid in ids:
                if total <= max_bytes:
                    break
                row = con.execute(
                    "SELECT image_path, bytes FROM episodes WHERE id=?",
                    (eid,)).fetchone()
                if row and self._delete(con, eid, row["image_path"]):
                    removed += 1
                    freed += row["bytes"] or 0
                    total -= row["bytes"] or 0
        con.commit()
        con.close()
        return {"removed": removed, "bytes_freed": freed}

    @staticmethod
    def _delete(con: sqlite3.Connection, eid: int, image_path: str) -> bool:
        cur = con.execute("DELETE FROM episodes WHERE id = ?", (eid,))
        if cur.rowcount:
            try:
                Path(image_path).unlink(missing_ok=True)
            except OSError:
                pass
            return True
        return False
