from pathlib import Path

from src.ingestion.chunker import chunk
from src.ingestion.parser import extract_tables, parse_file
from src.retrieval.hybrid_retriever import add_to_hybrid_index, build_hybrid_index


def ingest_document(file_path: str | Path, collection_id: str, vectorstore=None, top_k: int = 5):
    """
    Single entrypoint for ingestion: parse -> chunk -> (PDF) extract tables ->
    index into the collection's hybrid store. Pass an existing `vectorstore`
    to add to an already-initialized collection; omit it to create a new one.

    Returns (chunk_count, dense_retriever, bm25_retriever, vectorstore).
    dense_retriever is None on the "add to existing" path since the caller
    already holds a live retriever from the cache.
    """
    file_path = Path(file_path)
    pages = parse_file(file_path)
    chunks = chunk(pages)

    if file_path.suffix.lower() == ".pdf":
        chunks += extract_tables(file_path)  # table chunks are now actually searchable

    if vectorstore is None:
        dense_r, bm25_r, vs = build_hybrid_index(chunks, collection_id, top_k=top_k)
        return len(chunks), dense_r, bm25_r, vs

    bm25_r = add_to_hybrid_index(chunks, vectorstore, collection_id, top_k=top_k)
    return len(chunks), None, bm25_r, vectorstore
