"""Memory recall robustness: OCR-error folding, synonym expansion, FTS migration.

Deterministic unit tests. Recall is exercised through a real SQLite FTS5
index; rows are inserted directly so tests don't depend on the OCR engine.
"""
import sqlite3

from irix.memory import (
    MemoryStore,
    _fold_confusables,
    _query_variants,
)


def _row(caption="login screen", ocr_text="lnvoice report"):
    return {
        "ts": "2026-09-15T00:00:00+00:00",
        "source": "test",
        "image_path": "/tmp/x.png",
        "caption": caption,
        "ocr_text": ocr_text,
        "elements_json": "[]",
        "phash": "0" * 68,
        "bytes": 10,
    }


def _insert(con, **kw):
    r = _row(**kw)
    from irix.memory import _fold_confusables
    con.execute(
        "INSERT INTO episodes(ts, source, image_path, caption, ocr_text,"
        " ocr_norm, elements_json, phash, bytes)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (r["ts"], r["source"], r["image_path"], r["caption"], r["ocr_text"],
         _fold_confusables(r["ocr_text"]), r["elements_json"], r["phash"],
         r["bytes"]),
    )
    con.commit()
    return con.execute("SELECT last_insert_rowid()").fetchone()[0]


def test_fold_merges_ocr_confusables():
    assert _fold_confusables("lnvoice") == _fold_confusables("invoice")
    assert _fold_confusables("ERR0R") == _fold_confusables("error")
    assert _fold_confusables("quantum") == "quantum"  # untouched words stable
    assert _fold_confusables("Qu*ntum") == "quntum"  # '*' is a failed glyph
    assert _fold_confusables("tim-ut") == "tlmut"  # hyphen marks a mis-split


def test_query_variants_include_synonyms_and_folds():
    v = _query_variants("signin")
    assert "login" in v
    assert "log in" in v
    v2 = _query_variants("sign in")
    assert "login" in v2  # joined-phrase synonym lookup
    v3 = _query_variants("invoice")
    assert _fold_confusables("invoice") in v3


def test_recall_finds_ocr_misread(tmp_path):
    mem = MemoryStore(tmp_path)
    con = mem._db()
    eid = _insert(con, caption="monthly billing", ocr_text="lnvoice report")
    con.close()
    got = [r["id"] for r in mem.recall("invoice")]
    assert eid in got


def test_recall_finds_paraphrase(tmp_path):
    mem = MemoryStore(tmp_path)
    con = mem._db()
    eid = _insert(con, caption="login screen", ocr_text="welcome back")
    con.close()
    got = [r["id"] for r in mem.recall("signin")]
    assert eid in got


def test_recall_exact_still_works(tmp_path):
    mem = MemoryStore(tmp_path)
    con = mem._db()
    eid = _insert(con, caption="weather widget", ocr_text="sunny skies")
    con.close()
    got = [r["id"] for r in mem.recall("weather")]
    assert eid in got


def test_recall_fuzzy_fallback_for_corrupted_tokens(tmp_path):
    # Simulates small-text OCR output: readable but character-corrupted.
    mem = MemoryStore(tmp_path)
    con = mem._db()
    eid = _insert(con, caption="misc screen",
                  ocr_text="quntum timut connktion dollm")
    con.close()
    assert eid in [r["id"] for r in mem.recall("quantum")]
    assert eid in [r["id"] for r in mem.recall("timeout")]
    assert eid in [r["id"] for r in mem.recall("connection")]
    assert eid in [r["id"] for r in mem.recall("dollars")]


def _old_schema_db(path):
    """Build a pre-0.3.1 database: no ocr_norm anywhere."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE episodes(
          id INTEGER PRIMARY KEY,
          ts TEXT NOT NULL, source TEXT NOT NULL, image_path TEXT NOT NULL,
          caption TEXT NOT NULL, ocr_text TEXT NOT NULL,
          elements_json TEXT NOT NULL, phash TEXT NOT NULL, bytes INTEGER NOT NULL
        );
        CREATE VIRTUAL TABLE episodes_fts
          USING fts5(caption, ocr_text, content='episodes', content_rowid='id');
        CREATE TRIGGER episodes_ai AFTER INSERT ON episodes BEGIN
          INSERT INTO episodes_fts(rowid, caption, ocr_text)
          VALUES (new.id, new.caption, new.ocr_text);
        END;
        CREATE TRIGGER episodes_ad AFTER DELETE ON episodes BEGIN
          INSERT INTO episodes_fts(episodes_fts, rowid, caption, ocr_text)
          VALUES ('delete', old.id, old.caption, old.ocr_text);
        END;
        INSERT INTO episodes(ts, source, image_path, caption, ocr_text,
                             elements_json, phash, bytes)
        VALUES ('2026-09-15T00:00:00+00:00','test','/tmp/x.png',
                'billing page','lnvoice total','[]','%s',10);
    """ % ("0" * 68))
    con.commit()
    con.close()


def test_migrate_old_db_adds_ocr_norm(tmp_path):
    _old_schema_db(tmp_path / "memory.db")
    mem = MemoryStore(tmp_path)  # must migrate, not crash
    con = mem._db()
    cols = [r[1] for r in con.execute("PRAGMA table_info(episodes_fts)")]
    assert "ocr_norm" in cols
    row = con.execute("SELECT ocr_norm FROM episodes").fetchone()
    assert row[0] == _fold_confusables("lnvoice total")
    con.close()
    got = [r["id"] for r in mem.recall("invoice")]
    assert got, "migrated DB must recall through the folded column"
