from src.retrieval.retriever import build_chroma_retriever, load_chroma_retriever, add_chroma_retriever, rerank, rerank_with_scores
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


def _compute_overlap(dense_docs, sparse_docs) -> float:
    """Fraction of docs appearing in both dense and sparse result sets."""
    if not dense_docs or not sparse_docs:
        return 0.0
    dense_keys = {(d.metadata.get("source"), d.metadata.get("page"), d.metadata.get("chunk")) for d in dense_docs}
    sparse_keys = {(d.metadata.get("source"), d.metadata.get("page"), d.metadata.get("chunk")) for d in sparse_docs}
    union = dense_keys | sparse_keys
    intersection = dense_keys & sparse_keys
    return round(len(intersection) / len(union), 4) if union else 0.0


def hybrid_search(query: str, dense_retriever, bm25_retriever, top_k: int = 5, metrics=None) -> list:
    """Dense + sparse retrieval, fused via RRF, then reranked with the cross-encoder."""
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = bm25_retriever.invoke(query) if bm25_retriever else []
    fused = reciprocal_rank_fusion(dense_docs, sparse_docs)

    if metrics is not None:
        metrics.record("dense_sparse_overlap", _compute_overlap(dense_docs, sparse_docs))
        docs, scores = rerank_with_scores(query, fused, top_k=top_k)
        metrics.record("reranker_scores", scores)
        if scores:
            metrics.record("avg_reranker_score", round(sum(scores) / len(scores), 4))
            metrics.record("min_reranker_score", min(scores))
            metrics.record("max_reranker_score", max(scores))
        metrics.record("chunks_retrieved", len(docs))
        metrics.record("unique_sources", len({d.metadata.get("source") for d in docs}))
        return docs

    return rerank(query, fused, top_k=top_k)


def multi_query_hybrid_search(queries: list[str], dense_retriever, bm25_retriever, top_k: int = 5, metrics=None) -> list:
    """Runs dense+sparse+RRF per query variant, fuses all lists via a second RRF pass, reranks once."""
    per_query_fused = []
    all_dense = []
    all_sparse = []
    for q in queries:
        dense_docs = dense_retriever.invoke(q)
        sparse_docs = bm25_retriever.invoke(q) if bm25_retriever else []
        all_dense.extend(dense_docs)
        all_sparse.extend(sparse_docs)
        per_query_fused.append(reciprocal_rank_fusion(dense_docs, sparse_docs))
    fused = reciprocal_rank_fusion(*per_query_fused)

    if metrics is not None:
        metrics.record("dense_sparse_overlap", _compute_overlap(all_dense, all_sparse))
        docs, scores = rerank_with_scores(queries[0], fused, top_k=top_k)
        metrics.record("reranker_scores", scores)
        if scores:
            metrics.record("avg_reranker_score", round(sum(scores) / len(scores), 4))
            metrics.record("min_reranker_score", min(scores))
            metrics.record("max_reranker_score", max(scores))
        metrics.record("chunks_retrieved", len(docs))
        metrics.record("unique_sources", len({d.metadata.get("source") for d in docs}))
        return docs

    return rerank(queries[0], fused, top_k=top_k)


def multi_source_search(query: str, dense_retriever, bm25_retriever, sources: list[str], top_k_per_source: int = 2, max_total: int = 8, metrics=None) -> list:
    """
    For cross-document questions ("compare X across all sources"). Runs hybrid
    search once per source filename so every document gets guaranteed
    representation, then globally reranks candidates to provide the best context.
    """
    all_candidates = []
    all_dense = []
    all_sparse = []
    # Execute BM25 once and filter per-source from cached results
    all_sparse_raw = bm25_retriever.invoke(query) if bm25_retriever else []
    for source in sources:
        dense_docs = (
            dense_retriever.vectorstore.similarity_search(query, k=max(top_k_per_source * 4, 12), filter={"source": source})
            if hasattr(dense_retriever, "vectorstore") else dense_retriever.invoke(query)
        )
        sparse_docs = [d for d in all_sparse_raw if d.metadata.get("source") == source]
        all_dense.extend(dense_docs)
        all_sparse.extend(sparse_docs)
        fused = reciprocal_rank_fusion(dense_docs, sparse_docs)
        per_source_docs = rerank(query, fused, top_k=top_k_per_source)
        all_candidates.extend(per_source_docs)

    # Deduplicate any duplicate candidates across sources
    unique_candidates = reciprocal_rank_fusion(all_candidates)
    target_k = min(len(unique_candidates), max_total)

    if metrics is not None:
        docs, scores = rerank_with_scores(query, unique_candidates, top_k=target_k)
        metrics.record("dense_sparse_overlap", _compute_overlap(all_dense, all_sparse))
        if scores:
            metrics.record("reranker_scores", [round(float(s), 4) for s in scores])
            metrics.record("avg_reranker_score", round(sum(scores) / len(scores), 4))
            metrics.record("min_reranker_score", round(min(scores), 4))
            metrics.record("max_reranker_score", round(max(scores), 4))
        metrics.record("chunks_retrieved", len(docs))
        metrics.record("unique_sources", len({d.metadata.get("source") for d in docs}))
        return docs

    return rerank(query, unique_candidates, top_k=target_k)

