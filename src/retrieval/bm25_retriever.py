import json
from pathlib import Path

from langchain_community.retrievers import BM25Retriever

from src.core.config import settings


def _store_path(collection_id: str) -> Path:
    return Path(settings.data_dir) / collection_id / "bm25_chunks.json"


def build_bm25_retriever(chunks: list[dict], collection_id: str, top_k: int = 5):
    """Builds a BM25 retriever and persists raw chunks so the index can be rebuilt on reload."""
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    retriever = BM25Retriever.from_texts(texts=texts, metadatas=metadatas)
    retriever.k = max(top_k * 6, 36)
    _save_chunks(chunks, collection_id)
    return retriever


def load_bm25_retriever(collection_id: str, top_k: int = 5):
    """Rebuilds the BM25 retriever from persisted chunks. None if nothing's been ingested yet."""
    path = _store_path(collection_id)
    if not path.exists():
        return None
    chunks = json.loads(path.read_text())
    return build_bm25_retriever(chunks, collection_id, top_k=top_k)


def add_bm25_retriever(new_chunks: list[dict], collection_id: str, top_k: int = 5):
    """
    Merges new chunks with persisted ones and rebuilds.
    Replaces existing chunks for the same source filename to prevent duplicate chunk bloat.
    """
    path = _store_path(collection_id)
    existing = json.loads(path.read_text()) if path.exists() else []
    new_sources = {c["metadata"].get("source") for c in new_chunks if c.get("metadata")}
    retained = [c for c in existing if c.get("metadata", {}).get("source") not in new_sources]
    merged = retained + new_chunks
    return build_bm25_retriever(merged, collection_id, top_k=top_k)



def _save_chunks(chunks: list[dict], collection_id: str):
    path = _store_path(collection_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks))
