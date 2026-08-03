import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from src.core.config import settings

DB_PATH = Path(settings.data_dir) / "registry.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (collection_id) REFERENCES collections(id)
);
"""


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def create_collection(name: str) -> dict:
    cid, ts = str(uuid.uuid4()), time.time()
    with get_conn() as conn:
        conn.execute("INSERT INTO collections (id, name, created_at) VALUES (?, ?, ?)", (cid, name, ts))
    return {"id": cid, "name": name, "created_at": ts}


def get_collection(collection_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM collections WHERE id = ?", (collection_id,)).fetchone()
        return dict(row) if row else None


def list_collections() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM collections ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def register_document(collection_id: str, filename: str, chunk_count: int) -> dict:
    did, ts = str(uuid.uuid4()), time.time()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO documents (id, collection_id, filename, chunk_count, created_at) VALUES (?, ?, ?, ?, ?)",
            (did, collection_id, filename, chunk_count, ts),
        )
    return {"id": did, "collection_id": collection_id, "filename": filename, "chunk_count": chunk_count}


def list_documents(collection_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE collection_id = ? ORDER BY created_at", (collection_id,)
        ).fetchall()
        return [dict(r) for r in rows]
