from src.retrieval.retriever import build_chroma_retriever, load_chroma_retriever, add_chroma_retriever, rerank
from src.retrieval.bm25_retriever import build_bm25_retriever, load_bm25_retriever, add_bm25_retriever
from src.retrieval.fusion import reciprocal_rank_fusion


def build_hybrid_index(chunks: list[dict], collection_id: str, top_k: int = 5):
    """Builds both the dense (Chroma) and sparse (BM25) indexes for a collection from the same chunks."""
    dense_retriever, vectorstore = build_chroma_retriever(chunks, collection_id, top_k=top_k)
    bm25_retriever = build_bm25_retriever(chunks, collection_id, top_k=top_k)
    return dense_retriever, bm25_retriever, vectorstore


def load_hybrid_index(collection_id: str, top_k: int = 5):
    """Loads both indexes from disk. bm25_retriever is None if nothing's been ingested yet."""
    dense_retriever, vectorstore = load_chroma_retriever(collection_id, top_k=top_k)
    bm25_retriever = load_bm25_retriever(collection_id, top_k=top_k)
    return dense_retriever, bm25_retriever, vectorstore


def add_to_hybrid_index(new_chunks: list[dict], vectorstore, collection_id: str, top_k: int = 5):
    """Adds new chunks to both indexes. Returns the refreshed bm25_retriever."""
    add_chroma_retriever(new_chunks, vectorstore)
    return add_bm25_retriever(new_chunks, collection_id, top_k=top_k)


def hybrid_search(query: str, dense_retriever, bm25_retriever, top_k: int = 5) -> list:
    """Dense + sparse retrieval, fused via RRF, then reranked with the cross-encoder."""
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = bm25_retriever.invoke(query) if bm25_retriever else []
    fused = reciprocal_rank_fusion(dense_docs, sparse_docs)
    return rerank(query, fused, top_k=top_k)


def multi_query_hybrid_search(queries: list[str], dense_retriever, bm25_retriever, top_k: int = 5) -> list:
    """Runs dense+sparse+RRF per query variant, fuses all lists via a second RRF pass, reranks once."""
    per_query_fused = []
    for q in queries:
        dense_docs = dense_retriever.invoke(q)
        sparse_docs = bm25_retriever.invoke(q) if bm25_retriever else []
        per_query_fused.append(reciprocal_rank_fusion(dense_docs, sparse_docs))
    fused = reciprocal_rank_fusion(*per_query_fused)
    return rerank(queries[0], fused, top_k=top_k)


def multi_source_search(query: str, dense_retriever, bm25_retriever, sources: list[str], top_k_per_source: int = 3) -> list:
    """
    For cross-document questions ("compare X across all sources"). Runs hybrid
    search once per source filename so every document gets guaranteed
    representation instead of the top-k being dominated by one document's chunks.
    """
    all_docs = []
    for source in sources:
        dense_docs = (
            dense_retriever.vectorstore.similarity_search(query, k=top_k_per_source * 4, filter={"source": source})
            if hasattr(dense_retriever, "vectorstore") else dense_retriever.invoke(query)
        )
        sparse_raw = bm25_retriever.invoke(query) if bm25_retriever else []
        sparse_docs = [d for d in sparse_raw if d.metadata.get("source") == source]
        fused = reciprocal_rank_fusion(dense_docs, sparse_docs)
        all_docs.extend(rerank(query, fused, top_k=top_k_per_source))
    return all_docs
