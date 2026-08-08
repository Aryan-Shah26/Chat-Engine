import pytest
from langchain_core.documents import Document

from src.ingestion.parser import clean_extracted_text
from src.ingestion.chunker import chunk
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.bm25_retriever import build_bm25_retriever, add_bm25_retriever, load_bm25_retriever
from src.store import db


def test_clean_extracted_text():
    raw = "\uf0b7 Project 1: Built an AI\u202fengine.\n\uf0a7 Features: auto-\nmation.\n\n\n\nDetails here."
    cleaned = clean_extracted_text(raw)
    assert "\uf0b7" not in cleaned
    assert "\uf0a7" not in cleaned
    assert "\u202f" not in cleaned
    assert "• Project 1: Built an AI engine." in cleaned
    assert "automation." in cleaned
    assert "\n\n\n\n" not in cleaned


def test_chunker_sizes_and_separators():
    doc = {
        "text": "Header Section\n\n• Bullet 1: " + "A" * 600 + "\n\n• Bullet 2: " + "B" * 600,
        "metadata": {"source": "test.pdf", "page": 1}
    }
    chunks = chunk([doc], chunk_size=1000, chunk_overlap=200)
    assert len(chunks) >= 2
    assert all("source" in c["metadata"] for c in chunks)
    assert all(c["metadata"]["page"] == 1 for c in chunks)
    assert chunks[0]["metadata"]["chunk"] == 0
    assert chunks[1]["metadata"]["chunk"] == 1


def test_rrf_content_deduplication():
    doc1 = Document(page_content="Identical project description across uploads", metadata={"source": "resume1.pdf", "page": 1, "chunk": 0})
    doc2 = Document(page_content="Identical project description across uploads", metadata={"source": "resume2.pdf", "page": 1, "chunk": 0})
    doc3 = Document(page_content="Unique skills section: Python, PyTorch, FastAPI", metadata={"source": "resume1.pdf", "page": 1, "chunk": 1})

    dense_list = [doc1, doc2, doc3]
    sparse_list = [doc2, doc1, doc3]

    fused = reciprocal_rank_fusion(dense_list, sparse_list)
    # The duplicate content between doc1 and doc2 should be merged into a single fused candidate
    assert len(fused) == 2
    contents = [d.page_content for d in fused]
    assert "Identical project description across uploads" in contents
    assert "Unique skills section: Python, PyTorch, FastAPI" in contents


def test_db_register_document_upsert(tmp_path, monkeypatch):
    test_db_path = tmp_path / "registry.db"
    monkeypatch.setattr("src.store.db.DB_PATH", test_db_path)
    db.init_db()

    coll = db.create_collection("Test Coll")
    cid = coll["id"]

    d1 = db.register_document(cid, "resume.pdf", 10)
    docs1 = db.list_documents(cid)
    assert len(docs1) == 1
    assert docs1[0]["chunk_count"] == 10

    # Re-registering the same filename should update, not create a duplicate row
    d2 = db.register_document(cid, "resume.pdf", 12)
    docs2 = db.list_documents(cid)
    assert len(docs2) == 1
    assert docs2[0]["chunk_count"] == 12
    assert docs2[0]["id"] == d1["id"]
